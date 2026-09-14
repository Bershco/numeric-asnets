#!/usr/bin/env python3
"""Build validation-led, RQ-separated advisor follow-up artifacts.

This script deliberately excludes terminal-led rows from primary inference.
Completed terminal-led evidence is retained in a separate archive index.
"""

from __future__ import annotations

import csv
import html
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT = TRACK / "advisor_followup_20260910"
OUT.mkdir(parents=True, exist_ok=True)
DOMAINS = ["block_grouping", "drone", "fo_counters", "rover", "counters"]
MCTS_DOMAINS = [*DOMAINS, "mprime"]
LABELS = {
    "block_grouping": "Block Grouping",
    "drone": "Drone",
    "fo_counters": "FO Counters",
    "rover": "Rover",
    "counters": "Counters",
    "mprime": "MPrime",
}

FO_STAGE2_PARTIAL_PROVENANCE = (
    "experiment_tracking/stage2_policy_mcts_comparison_by_branch_latest.csv;"
    "experiment_tracking/stage2_mcts_historical_log_audit_20260902.csv;"
    "experiment_tracking/advisor_followup_20260910/"
    "fo_stage2_validation_partial_recovery_manifest.csv"
)
FO_STAGE2_VH_ON_EXACT = (
    OUT / "fo_stage2_validation_vh_on_exact_seed_results_20260912.csv"
)
FO_STAGE2_VH_OFF_EXACT = (
    OUT / "fo_stage2_validation_vh_off_exact_seed_results_20260913.csv"
)
FO_STAGE2_VH_ON_EXACT_PROVENANCE = (
    "experiment_tracking/advisor_followup_20260910/"
    "fo_stage2_validation_vh_on_exact_seed_results_20260912.csv"
)
FO_STAGE2_VH_OFF_EXACT_PROVENANCE = (
    "experiment_tracking/advisor_followup_20260910/"
    "fo_stage2_validation_vh_off_exact_seed_results_20260913.csv"
)
CAPACITY = {domain: 20 for domain in DOMAINS}
CAPACITY["counters"] = 59
CAPACITY["mprime"] = 20
T95 = {10: 2.262157, 9: 2.306004, 8: 2.364624, 7: 2.446912,
       6: 2.570582, 5: 2.776445, 4: 3.182446, 3: 4.302653,
       2: 12.706205}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: list[float]) -> tuple[float, float]:
    center = statistics.mean(values)
    if len(values) < 2 or statistics.stdev(values) == 0:
        return center, center
    half = T95.get(len(values), 1.96) * statistics.stdev(values) / math.sqrt(len(values))
    return center - half, center + half


def signflip(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    extreme = sum(
        abs(statistics.mean(value * sign for value, sign in zip(values, signs)))
        >= observed - 1e-12
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    )
    return extreme / (2 ** len(values))


def add_holm(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("raw_p") != "":
            groups[(str(row["rq"]), str(row["stage"]), str(row["cutoff"]),
                    str(row["estimand"]))].append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: float(rows[index]["raw_p"]))
        running = 0.0
        for rank, index in enumerate(ordered):
            adjusted = (len(indices) - rank) * float(rows[index]["raw_p"])
            running = max(running, adjusted)
            rows[index]["holm_p"] = min(1.0, running)


def result_row(*, rq: str, stage: str, estimand: str, domain: str,
               cutoff: str, values: list[float], baseline_mean: float,
               comparison_mean: float, provenance: str) -> dict[str, object]:
    low, high = ci95(values)
    capacity = CAPACITY[domain]
    return {
        "rq": rq,
        "stage": stage,
        "estimand": estimand,
        "domain": domain,
        "cutoff": cutoff,
        "n": len(values),
        "capacity": capacity,
        "baseline_mean": round(baseline_mean, 6),
        "comparison_mean": round(comparison_mean, 6),
        "effect_solved": round(statistics.mean(values), 6),
        "ci95_low_solved": round(low, 6),
        "ci95_high_solved": round(high, 6),
        "effect_percentage_points": round(statistics.mean(values) / capacity * 100, 6),
        "ci95_low_percentage_points": round(low / capacity * 100, 6),
        "ci95_high_percentage_points": round(high / capacity * 100, 6),
        "raw_p": round(signflip(values), 9),
        "holm_p": "",
        "evidence_status": "complete_paired_inference",
        "row_level_provenance": provenance,
    }


def lower_bound_row(*, rq: str, estimand: str, cutoff: str,
                    baseline_mean: float, comparison_lower_bound: float) -> dict[str, object]:
    """Represent censored FO Stage-2 evidence without inventing inference."""
    effect = comparison_lower_bound - baseline_mean
    return {
        "rq": rq, "stage": "Stage 2", "estimand": estimand,
        "domain": "fo_counters", "cutoff": cutoff, "n": 10,
        "capacity": 20, "baseline_mean": baseline_mean,
        "comparison_mean": comparison_lower_bound,
        "effect_solved": round(effect, 6),
        "ci95_low_solved": "", "ci95_high_solved": "",
        "effect_percentage_points": round(effect / 20 * 100, 6),
        "ci95_low_percentage_points": "", "ci95_high_percentage_points": "",
        "raw_p": "", "holm_p": "", "evidence_status": "partial_lower_bound",
        "row_level_provenance": FO_STAGE2_PARTIAL_PROVENANCE,
    }


def build_rq_rows() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    policy = [row for row in read_csv(TRACK / "policy_paired_seed_results.csv")
              if row["experiment_id"] == "MAIN-VAL"]
    policy_map = {(row["domain"], row["value_head"], row["seed"]): row for row in policy}

    for domain in DOMAINS:
        off = [row for row in policy if row["domain"] == domain and row["value_head"] == "off"]
        on = [row for row in policy if row["domain"] == domain and row["value_head"] == "on"]
        off = sorted(off, key=lambda row: row["seed"])
        on = sorted(on, key=lambda row: row["seed"])
        output.append(result_row(
            rq="RQ1", stage="Stage 2", estimand="VH-off: Stage 2 policy - Stage 1 policy",
            domain=domain, cutoff="endpoint",
            values=[float(row["difference"]) for row in off],
            baseline_mean=statistics.mean(float(row["before_score"]) for row in off),
            comparison_mean=statistics.mean(float(row["after_score"]) for row in off),
            provenance="experiment_tracking/policy_paired_seed_results.csv",
        ))
        output.append(result_row(
            rq="RQ3", stage="Stage 2", estimand="VH-on direct: Stage 2 policy - Stage 1 policy",
            domain=domain, cutoff="endpoint",
            values=[float(row["difference"]) for row in on],
            baseline_mean=statistics.mean(float(row["before_score"]) for row in on),
            comparison_mean=statistics.mean(float(row["after_score"]) for row in on),
            provenance="experiment_tracking/policy_paired_seed_results.csv",
        ))
        seeds = sorted(set(row["seed"] for row in off) & set(row["seed"] for row in on))
        off_delta = {seed: float(policy_map[(domain, "off", seed)]["difference"]) for seed in seeds}
        on_delta = {seed: float(policy_map[(domain, "on", seed)]["difference"]) for seed in seeds}
        output.append(result_row(
            rq="RQ3", stage="Stage 2", estimand="VH interaction: VH-on refinement - VH-off refinement",
            domain=domain, cutoff="endpoint",
            values=[on_delta[seed] - off_delta[seed] for seed in seeds],
            baseline_mean=statistics.mean(off_delta.values()),
            comparison_mean=statistics.mean(on_delta.values()),
            provenance="experiment_tracking/policy_paired_seed_results.csv",
        ))

    for stage, path, branch in (
        ("Stage 1", TRACK / "stage1_policy_mcts_seed_cutoffs_latest.csv", None),
        ("Stage 2", TRACK / "stage2_policy_mcts_seed_cutoffs_latest.csv", "validation_led"),
    ):
        rows = read_csv(path)
        if stage == "Stage 1":
            mprime_path = (
                TRACK / "mprime_phase_b_a_stage1_mcts_20260913" /
                "results_20260914" / "per_seed_results.csv"
            )
            rows += [
                {
                    "domain": "mprime",
                    "value_head": row["value_head"],
                    "seed": row["seed"],
                    "policy_score": row["policy_score"],
                    "mcts_30m": row["mcts_30m"],
                    "mcts_2h": row["mcts_2h"],
                    "mcts_6h": row["mcts_6h"],
                }
                for row in read_csv(mprime_path)
            ]
        if branch:
            rows = [row for row in rows if row.get("stage2_branch") == branch]
        mapping = {(row["domain"], row["value_head"], row["seed"]): row for row in rows}
        available_domains = [domain for domain in MCTS_DOMAINS
                             if any(row["domain"] == domain for row in rows)]
        for domain in available_domains:
            for cutoff in ("30m", "2h", "6h"):
                off = sorted([row for row in rows if row["domain"] == domain and row["value_head"] == "off"], key=lambda row: row["seed"])
                on = sorted([row for row in rows if row["domain"] == domain and row["value_head"] == "on"], key=lambda row: row["seed"])
                if off:
                    off_values = [float(row[f"mcts_{cutoff}"]) - float(row["policy_score"]) for row in off]
                    output.append(result_row(
                        rq="RQ2", stage=stage,
                        estimand="VH-off direct: MCTS - same-checkpoint policy",
                        domain=domain, cutoff=cutoff, values=off_values,
                        baseline_mean=statistics.mean(float(row["policy_score"]) for row in off),
                        comparison_mean=statistics.mean(float(row[f"mcts_{cutoff}"]) for row in off),
                        provenance=str(path.relative_to(ROOT)).replace("\\", "/"),
                    ))
                if on:
                    on_values = [float(row[f"mcts_{cutoff}"]) - float(row["policy_score"]) for row in on]
                    output.append(result_row(
                        rq="RQ4", stage=stage,
                        estimand="VH-on direct: MCTS - same-checkpoint policy",
                        domain=domain, cutoff=cutoff, values=on_values,
                        baseline_mean=statistics.mean(float(row["policy_score"]) for row in on),
                        comparison_mean=statistics.mean(float(row[f"mcts_{cutoff}"]) for row in on),
                        provenance=str(path.relative_to(ROOT)).replace("\\", "/"),
                    ))
                off_seeds = {row["seed"] for row in off}
                on_seeds = {row["seed"] for row in on}
                seeds = sorted(off_seeds & on_seeds)
                if seeds:
                    # Mandatory absolute comparator requested after the
                    # advisor meeting: show whether VH-on MCTS also exceeds the
                    # parallel VH-off policy, not only its weaker/stronger VH-on
                    # policy checkpoint.  This is a cross-cell comparison, not an
                    # interaction estimate, and is therefore labelled separately.
                    output.append(result_row(
                        rq="RQ4", stage=stage,
                        estimand="Cross-cell level: VH-on MCTS - parallel VH-off policy",
                        domain=domain, cutoff=cutoff,
                        values=[float(mapping[(domain, "on", seed)][f"mcts_{cutoff}"])
                                - float(mapping[(domain, "off", seed)]["policy_score"])
                                for seed in seeds],
                        baseline_mean=statistics.mean(
                            float(mapping[(domain, "off", seed)]["policy_score"])
                            for seed in seeds),
                        comparison_mean=statistics.mean(
                            float(mapping[(domain, "on", seed)][f"mcts_{cutoff}"])
                            for seed in seeds),
                        provenance=str(path.relative_to(ROOT)).replace("\\", "/"),
                    ))
                    off_benefit = {
                        seed: float(mapping[(domain, "off", seed)][f"mcts_{cutoff}"])
                              - float(mapping[(domain, "off", seed)]["policy_score"])
                        for seed in seeds
                    }
                    on_benefit = {
                        seed: float(mapping[(domain, "on", seed)][f"mcts_{cutoff}"])
                              - float(mapping[(domain, "on", seed)]["policy_score"])
                        for seed in seeds
                    }
                    output.append(result_row(
                        rq="RQ4", stage=stage,
                        estimand="VH interaction: VH-on MCTS benefit - VH-off MCTS benefit",
                        domain=domain, cutoff=cutoff,
                        values=[on_benefit[seed] - off_benefit[seed] for seed in seeds],
                        baseline_mean=statistics.mean(off_benefit.values()),
                        comparison_mean=statistics.mean(on_benefit.values()),
                        provenance=str(path.relative_to(ROOT)).replace("\\", "/"),
                    ))

    # Both FO modes are now exact after minimal instance recoveries.  The last
    # VH-off opportunity (job 21219947) consumed its full six-hour instance
    # budget without a plan, leaving the mean at 6.1 but closing the cell.
    fo_off = read_csv(FO_STAGE2_VH_OFF_EXACT)
    fo_on = read_csv(FO_STAGE2_VH_ON_EXACT)
    for cutoff in ("30m", "2h", "6h"):
        off_comparison = [float(row[f"mcts_{cutoff}"]) for row in fo_off]
        off_policy = [float(row["policy_score"]) for row in fo_off]
        output.append(result_row(
            rq="RQ2", stage="Stage 2",
            estimand="VH-off direct: MCTS - same-checkpoint policy",
            domain="fo_counters", cutoff=cutoff,
            values=[mcts - policy for mcts, policy in zip(off_comparison, off_policy)],
            baseline_mean=statistics.mean(off_policy),
            comparison_mean=statistics.mean(off_comparison),
            provenance=FO_STAGE2_VH_OFF_EXACT_PROVENANCE,
        ))
        comparison = [float(row[f"mcts_{cutoff}"]) for row in fo_on]
        policy_on = [float(row["policy_vh_on"]) for row in fo_on]
        policy_off = [float(row["policy_vh_off"]) for row in fo_on]
        output.append(result_row(
            rq="RQ4", stage="Stage 2",
            estimand="VH-on direct: MCTS - same-checkpoint policy",
            domain="fo_counters", cutoff=cutoff,
            values=[mcts - policy for mcts, policy in zip(comparison, policy_on)],
            baseline_mean=statistics.mean(policy_on),
            comparison_mean=statistics.mean(comparison),
            provenance=FO_STAGE2_VH_ON_EXACT_PROVENANCE,
        ))
        off_by_seed = {row["seed"]: row for row in fo_off}
        on_by_seed = {row["seed"]: row for row in fo_on}
        seeds = sorted(off_by_seed.keys() & on_by_seed.keys(), key=int)
        output.append(result_row(
            rq="RQ4", stage="Stage 2",
            estimand="VH interaction: VH-on MCTS benefit - VH-off MCTS benefit",
            domain="fo_counters", cutoff=cutoff,
            values=[
                (float(on_by_seed[seed][f"mcts_{cutoff}"]) - float(on_by_seed[seed]["policy_vh_on"]))
                - (float(off_by_seed[seed][f"mcts_{cutoff}"]) - float(off_by_seed[seed]["policy_score"]))
                for seed in seeds
            ],
            baseline_mean=statistics.mean(
                float(off_by_seed[seed][f"mcts_{cutoff}"]) - float(off_by_seed[seed]["policy_score"])
                for seed in seeds
            ),
            comparison_mean=statistics.mean(
                float(on_by_seed[seed][f"mcts_{cutoff}"]) - float(on_by_seed[seed]["policy_vh_on"])
                for seed in seeds
            ),
            provenance=f"{FO_STAGE2_VH_OFF_EXACT_PROVENANCE};{FO_STAGE2_VH_ON_EXACT_PROVENANCE}",
        ))
        output.append(result_row(
            rq="RQ4", stage="Stage 2",
            estimand="Cross-cell level: VH-on MCTS - parallel VH-off policy",
            domain="fo_counters", cutoff=cutoff,
            values=[mcts - policy for mcts, policy in zip(comparison, policy_off)],
            baseline_mean=statistics.mean(policy_off),
            comparison_mean=statistics.mean(comparison),
            provenance=FO_STAGE2_VH_ON_EXACT_PROVENANCE,
        ))

    add_holm(output)
    for row in output:
        if row["stage"] == "Stage 2" and row["rq"] in {"RQ2", "RQ4"}:
            final_five_domain_family = row["evidence_status"] == "complete_paired_inference"
            row["multiplicity_status"] = (
                "withheld_partial_cell" if row["evidence_status"] == "partial_lower_bound"
                else "final_declared_family" if final_five_domain_family
                else "incomplete_family"
            )
        else:
            row["multiplicity_status"] = "final_declared_family"
    return output


def plot_rows(rows: list[dict[str, object]], rq: str, output_name: str, title: str,
              estimand_prefix: str | None = None) -> None:
    selected = [row for row in rows if row["rq"] == rq]
    if estimand_prefix is not None:
        selected = [row for row in selected if str(row["estimand"]).startswith(estimand_prefix)]
    estimands = list(dict.fromkeys(str(row["estimand"]) for row in selected))
    # Keep a generous right margin for the exact effect and adjusted-p labels.
    # These labels were previously clipped in the PNG rendering.
    width = 1800
    panel_height = 105 + max(
        len([row for row in selected if row["estimand"] == estimand]) * 34
        for estimand in estimands
    )
    height = 85 + panel_height * len(estimands)
    left, right = 435, 1435
    axis_low, axis_high = -50.0, 50.0

    def esc(value: object) -> str:
        return html.escape(str(value))

    def scale(value: float) -> float:
        bounded = max(axis_low, min(axis_high, value))
        return left + (bounded - axis_low) / (axis_high - axis_low) * (right - left)

    family_note = (
        " Six fixed-search domains are exact at Stage 1; five are exact at Stage 2."
        if rq in {"RQ2", "RQ4"} else ""
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.title{font-size:23px;font-weight:700}.sub{font-size:12px;fill:#536273}.label{font-size:11px}.head{font-size:13px;font-weight:700}</style>',
        f'<text x="28" y="34" class="title">{esc(title)}</text>',
        f'<text x="28" y="57" class="sub">Validation-led primary analysis only. Effects are paired by seed; lines are 95% t-intervals; pH is Holm-adjusted within each RQ/stage/cutoff/estimand family.{family_note}</text>',
    ]
    panel_top = 78
    for estimand in estimands:
        data = [row for row in selected if row["estimand"] == estimand]
        panel_bottom = panel_top + panel_height - 20
        parts += [
            f'<rect x="20" y="{panel_top}" width="{width - 40}" height="{panel_height - 10}" rx="5" fill="#fbfcfd" stroke="#d9e0e7"/>',
            f'<text x="36" y="{panel_top + 25}" class="head">{esc(estimand)}</text>',
        ]
        for tick in (-50, -25, 0, 25, 50):
            x = scale(tick)
            color = "#17212b" if tick == 0 else "#d9e0e7"
            line_width = 1.4 if tick == 0 else 1
            parts += [
                f'<line x1="{x:.1f}" y1="{panel_top + 37}" x2="{x:.1f}" y2="{panel_bottom - 23}" stroke="{color}" stroke-width="{line_width}"/>',
                f'<text x="{x:.1f}" y="{panel_bottom - 7}" text-anchor="middle" class="sub">{tick:+d} pp</text>',
            ]
        y = panel_top + 55
        for row in data:
            label = f"{LABELS[str(row['domain'])]} / {row['stage']} / {row['cutoff']}"
            center = float(row["effect_percentage_points"])
            partial = row.get("evidence_status") == "partial_lower_bound"
            color = "#238b45" if center >= 0 else "#c43c39"
            if partial:
                arrow_end = scale(min(center + 5, axis_high))
                parts += [
                    f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" class="label">{esc(label)}</text>',
                    f'<circle cx="{scale(center):.1f}" cy="{y}" r="5" fill="{color}"/>',
                    f'<line x1="{scale(center):.1f}" y1="{y}" x2="{arrow_end:.1f}" y2="{y}" stroke="{color}" stroke-width="2"/>',
                    f'<text x="{arrow_end + 3:.1f}" y="{y + 4}" class="label">&#x25B6;</text>',
                    f'<text x="{right + 10}" y="{y + 4}" class="sub">&#x2265;{center:+.1f} pp; partial, inference withheld</text>',
                ]
                y += 34
                continue
            low = float(row["ci95_low_percentage_points"])
            high = float(row["ci95_high_percentage_points"])
            marker = " *" if float(row["holm_p"]) < .05 else ""
            parts += [
                f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" class="label">{esc(label)}</text>',
                f'<line x1="{scale(low):.1f}" y1="{y}" x2="{scale(high):.1f}" y2="{y}" stroke="#536273" stroke-width="2"/>',
                f'<line x1="{scale(low):.1f}" y1="{y - 4}" x2="{scale(low):.1f}" y2="{y + 4}" stroke="#536273"/>',
                f'<line x1="{scale(high):.1f}" y1="{y - 4}" x2="{scale(high):.1f}" y2="{y + 4}" stroke="#536273"/>',
                f'<circle cx="{scale(center):.1f}" cy="{y}" r="4.5" fill="{color}"/>',
                f'<text x="{right + 10}" y="{y + 4}" class="sub">{center:+.1f} pp; pH={float(row["holm_p"]):.3g}{marker}</text>',
            ]
            y += 34
        panel_top += panel_height
    parts += [
        f'<text x="{left}" y="{height - 12}" class="sub">Left: lower coverage. Right: higher coverage. * Holm-adjusted p &lt; .05.</text>',
        '</svg>',
    ]
    (OUT / f"{output_name}.svg").write_text("".join(parts), encoding="utf-8")


def build_raw_mean_rows(rq_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rq2: list[dict[str, object]] = []
    rq4: list[dict[str, object]] = []
    for stage, path, branch in (
        ("Stage 1", TRACK / "stage1_policy_mcts_seed_cutoffs_latest.csv", None),
        ("Stage 2", TRACK / "stage2_policy_mcts_seed_cutoffs_latest.csv", "validation_led"),
    ):
        source = read_csv(path)
        source_path = path
        if stage == "Stage 1":
            mprime_path = (
                TRACK / "mprime_phase_b_a_stage1_mcts_20260913" /
                "results_20260914" / "per_seed_results.csv"
            )
            source += [
                {
                    "domain": "mprime",
                    "value_head": row["value_head"],
                    "seed": row["seed"],
                    "policy_score": row["policy_score"],
                    "mcts_30m": row["mcts_30m"],
                    "mcts_2h": row["mcts_2h"],
                    "mcts_6h": row["mcts_6h"],
                    "evidence_status": "complete",
                }
                for row in read_csv(mprime_path)
            ]
        if branch:
            source = [row for row in source if row.get("stage2_branch") == branch]
        for domain in MCTS_DOMAINS:
            for vh in ("off", "on"):
                group = [row for row in source if row["domain"] == domain and row["value_head"] == vh]
                if not group:
                    continue
                complete = all(row.get("evidence_status", "complete") != "partial_lower_bound" for row in group)
                raw_row = {
                    "rq": "RQ4",
                    "stage": stage,
                    "domain": domain,
                    "value_head": vh,
                    "n": len(group),
                    "capacity": CAPACITY[domain],
                    "policy_mean": statistics.mean(float(row["policy_score"]) for row in group),
                    "mcts_30m_mean": statistics.mean(float(row["mcts_30m"]) for row in group),
                    "mcts_2h_mean": statistics.mean(float(row["mcts_2h"]) for row in group),
                    "mcts_6h_mean": statistics.mean(float(row["mcts_6h"]) for row in group),
                    "evidence_status": "complete" if complete else "partial_lower_bound",
                    "seed_level_source": (
                        "experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913/"
                        "results_20260914/per_instance_results.csv"
                        if domain == "mprime" else
                        str(source_path.relative_to(ROOT)).replace("\\", "/")
                    ),
                    "job_log_columns": (
                        "source_job_id;source_attempt_log;source_completion_record;source_training_job_id;source_policy_job_id"
                        if domain == "mprime" else
                        "mcts_job_id;source_policy_log;source_mcts_log;source_completion_ledger"
                    ),
                }
                rq4.append(raw_row)
                if vh == "off":
                    rq2.append({**raw_row, "rq": "RQ2"})

    # Replace older FO Stage-2 rows with the exact post-recovery aggregates.
    rq2 = [row for row in rq2 if not (row["stage"] == "Stage 2" and row["domain"] == "fo_counters")]
    rq4 = [row for row in rq4 if not (row["stage"] == "Stage 2" and row["domain"] == "fo_counters")]
    fo_off = read_csv(FO_STAGE2_VH_OFF_EXACT)
    rq2.append({
        "rq": "RQ2", "stage": "Stage 2", "domain": "fo_counters", "value_head": "off",
        "n": 10, "capacity": 20,
        "policy_mean": statistics.mean(float(row["policy_score"]) for row in fo_off),
        "mcts_30m_mean": statistics.mean(float(row["mcts_30m"]) for row in fo_off),
        "mcts_2h_mean": statistics.mean(float(row["mcts_2h"]) for row in fo_off),
        "mcts_6h_mean": statistics.mean(float(row["mcts_6h"]) for row in fo_off),
        "evidence_status": "complete",
        "seed_level_source": FO_STAGE2_VH_OFF_EXACT_PROVENANCE,
        "job_log_columns": "source_training_job_id;source_policy_job_id;source_policy_log;source_mcts_logs;source_completion_ledgers",
    })
    rq4.append({**rq2[-1], "rq": "RQ4"})
    rq4.append({
        "rq": "RQ4", "stage": "Stage 2", "domain": "fo_counters", "value_head": "on",
        "n": 10, "capacity": 20, "policy_mean": 3.1, "mcts_30m_mean": 5.2,
        "mcts_2h_mean": 5.4, "mcts_6h_mean": 5.4, "evidence_status": "complete",
        "seed_level_source": FO_STAGE2_VH_ON_EXACT_PROVENANCE,
        "job_log_columns": "source_job_id;recovery_job_id;source_evaluation_log;source_completion_ledger",
    })

    policy = [row for row in read_csv(TRACK / "policy_paired_seed_results.csv") if row["experiment_id"] == "MAIN-VAL"]
    rq3: list[dict[str, object]] = []
    effects = {(str(row["domain"]), str(row["estimand"])): row for row in rq_rows if row["rq"] == "RQ3"}
    for domain in DOMAINS:
        item: dict[str, object] = {"rq": "RQ3", "domain": domain, "capacity": CAPACITY[domain]}
        for vh in ("off", "on"):
            group = [row for row in policy if row["domain"] == domain and row["value_head"] == vh]
            item[f"vh_{vh}_stage1_mean"] = statistics.mean(float(row["before_score"]) for row in group)
            item[f"vh_{vh}_stage2_mean"] = statistics.mean(float(row["after_score"]) for row in group)
        interaction = effects[(domain, "VH interaction: VH-on refinement - VH-off refinement")]
        item.update({
            "interaction_effect_solved": interaction["effect_solved"],
            "interaction_ci95_low_solved": interaction["ci95_low_solved"],
            "interaction_ci95_high_solved": interaction["ci95_high_solved"],
            "interaction_holm_p": interaction["holm_p"],
            "seed_level_source": "experiment_tracking/policy_paired_seed_results.csv",
            "job_log_columns": "before_training_job;before_evaluation_job;before_log;after_training_job;after_evaluation_job;after_log",
        })
        rq3.append(item)
    return rq2, rq3, rq4


def _raw_plot_header(title: str, subtitle: str, width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.title{font-size:23px;font-weight:700}.sub{font-size:12px;fill:#536273}.label{font-size:11px}.head{font-size:13px;font-weight:700}.value{font-size:10px;font-weight:600}</style>',
        f'<text x="28" y="34" class="title">{html.escape(title)}</text>',
        f'<text x="28" y="56" class="sub">{html.escape(subtitle)}</text>',
    ]


def plot_rq2_raw(rows: list[dict[str, object]]) -> None:
    width, height = 1750, 760
    parts = _raw_plot_header(
        "RQ2 raw coverage — VH-off policy versus MCTS at both stages",
        "Stage 2 uses validation-led checkpoints. BG/Counters use narrow 5/20; other cells use normal 20/70. MPrime Stage 1 is final; Stage 2 awaits valid retraining.",
        width, height,
    )
    colors = {"Policy": "#e68632", "30m": "#9ecae1", "2h": "#4292c6", "6h": "#08519c"}
    left, top, panel_w, panel_h = 85, 105, 790, 530
    for panel_index, stage in enumerate(("Stage 1", "Stage 2")):
        x0 = left + panel_index * 840
        parts += [
            f'<rect x="{x0 - 12}" y="68" width="{panel_w + 24}" height="24" rx="6" fill="#eef2f6"/>',
            f'<text x="{x0 + panel_w / 2}" y="85" text-anchor="middle" class="head">{stage}</text>',
        ]
        for tick in range(0, 101, 20):
            y = top + panel_h - tick / 100 * panel_h
            parts += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="#e4e9ee"/>']
            if panel_index == 0:
                parts += [f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" class="sub">{tick}%</text>']
        data = sorted([row for row in rows if row["stage"] == stage], key=lambda row: MCTS_DOMAINS.index(str(row["domain"])))
        group_w = panel_w / len(data)
        bar_w = 25
        for di, row in enumerate(data):
            center = x0 + (di + .5) * group_w
            values = [("Policy", float(row["policy_mean"])), ("30m", float(row["mcts_30m_mean"])), ("2h", float(row["mcts_2h_mean"])), ("6h", float(row["mcts_6h_mean"]))]
            for vi, (label, value) in enumerate(values):
                x = center + (vi - 1.5) * (bar_w + 4) - bar_w / 2
                pct = value / float(row["capacity"]) * 100
                y = top + panel_h - pct / 100 * panel_h
                partial = row["evidence_status"] == "partial_lower_bound" and label != "Policy"
                parts += [
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + panel_h - y:.1f}" fill="{colors[label]}" opacity="{0.65 if partial else 0.95}" stroke="{"#17212b" if partial else colors[label]}" stroke-dasharray="{"4 2" if partial else "none"}"/>',
                    f'<text x="{x + bar_w/2:.1f}" y="{max(top + 10, y - 5):.1f}" text-anchor="middle" class="value">{"≥" if partial else ""}{value:.1f}</text>',
                ]
            parts += [f'<text x="{center:.1f}" y="{top + panel_h + 22}" text-anchor="middle" class="label">{html.escape(LABELS[str(row["domain"])])}</text>']
    lx = 610
    for label in ("Policy", "30m", "2h", "6h"):
        parts += [f'<rect x="{lx}" y="692" width="14" height="14" fill="{colors[label]}"/><text x="{lx + 20}" y="704" class="sub">{label}</text>']
        lx += 110
    parts += ['<text x="28" y="738" class="sub">All bars are exact declared-budget means. CIs and Holm-adjusted tests are reported in the companion effect plot/table.</text>', '</svg>']
    (OUT / "rq2_raw_means_by_stage.svg").write_text("".join(parts), encoding="utf-8")


def plot_rq3_raw(rows: list[dict[str, object]]) -> None:
    width, height = 1750, 650
    parts = _raw_plot_header(
        "RQ3 raw policy coverage and value-head interaction",
        "Each line is Stage 1 → validation-led Stage 2. Labels show solved instances; interaction is the difference between the VH-on and VH-off changes.",
        width, height,
    )
    left, top, chart_w, chart_h = 85, 100, 1580, 430
    for tick in range(0, 101, 20):
        y = top + chart_h - tick / 100 * chart_h
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e4e9ee"/><text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="sub">{tick}%</text>']
    group_w = chart_w / len(rows)
    for di, row in enumerate(rows):
        center = left + (di + .5) * group_w
        capacity = float(row["capacity"])
        for vh, color, offset in (("off", "#4c78a8", -42), ("on", "#d95f02", 42)):
            start = float(row[f"vh_{vh}_stage1_mean"])
            end = float(row[f"vh_{vh}_stage2_mean"])
            y1 = top + chart_h - (start / capacity * 100) / 100 * chart_h
            y2 = top + chart_h - (end / capacity * 100) / 100 * chart_h
            x1, x2 = center + offset - 20, center + offset + 20
            parts += [
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="5" fill="white" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="5" fill="{color}"/>',
                f'<text x="{x1:.1f}" y="{y1 - 9:.1f}" text-anchor="middle" class="value">{start:.1f}</text>',
                f'<text x="{x2:.1f}" y="{y2 - 9:.1f}" text-anchor="middle" class="value">{end:.1f}</text>',
            ]
        effect = float(row["interaction_effect_solved"])
        low, high = float(row["interaction_ci95_low_solved"]), float(row["interaction_ci95_high_solved"])
        ph = float(row["interaction_holm_p"])
        parts += [
            f'<text x="{center:.1f}" y="{top + chart_h + 24}" text-anchor="middle" class="label">{html.escape(LABELS[str(row["domain"])])}</text>',
            f'<text x="{center:.1f}" y="{top + chart_h + 43}" text-anchor="middle" class="sub">DiD {effect:+.1f} [{low:+.1f},{high:+.1f}], pH={ph:.3g}</text>',
        ]
    parts += [
        '<line x1="650" y1="595" x2="690" y2="595" stroke="#4c78a8" stroke-width="3"/><text x="700" y="599" class="sub">VH-off</text>',
        '<line x1="820" y1="595" x2="860" y2="595" stroke="#d95f02" stroke-width="3"/><text x="870" y="599" class="sub">VH-on</text>',
        '<text x="28" y="628" class="sub">Open marker: Stage 1. Filled marker: Stage 2. The displayed DiD includes its 95% CI and Holm-adjusted p-value.</text>',
        '</svg>',
    ]
    (OUT / "rq3_raw_means_and_interaction.svg").write_text("".join(parts), encoding="utf-8")


def plot_rq4_raw(rq2_rows: list[dict[str, object]], rq4_rows: list[dict[str, object]], effects: list[dict[str, object]]) -> None:
    width, height = 1750, 760
    parts = _raw_plot_header(
        "RQ4 raw policy and six-hour MCTS coverage by value-head mode",
        "Stage 2 uses validation-led checkpoints. BG/Counters use narrow 5/20; other cells use normal 20/70. MPrime Stage 1 is final; Stage 2 awaits valid retraining.",
        width, height,
    )
    effect_map = {(str(row["stage"]), str(row["domain"])): row for row in effects if row["rq"] == "RQ4" and row["cutoff"] == "6h" and str(row["estimand"]).startswith("VH interaction")}
    left, top, panel_w, panel_h = 85, 105, 790, 530
    for panel_index, stage in enumerate(("Stage 1", "Stage 2")):
        x0 = left + panel_index * 840
        parts += [
            f'<rect x="{x0 - 12}" y="68" width="{panel_w + 24}" height="24" rx="6" fill="#eef2f6"/>',
            f'<text x="{x0 + panel_w / 2}" y="85" text-anchor="middle" class="head">{stage}</text>',
        ]
        for tick in range(0, 101, 20):
            y = top + panel_h - tick / 100 * panel_h
            parts += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="#e4e9ee"/>']
            if panel_index == 0:
                parts += [f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" class="sub">{tick}%</text>']
        data_domains = [
            domain for domain in MCTS_DOMAINS
            if any(row["stage"] == stage and row["domain"] == domain for row in rq4_rows)
        ]
        group_w = panel_w / len(data_domains)
        for di, domain in enumerate(data_domains):
            center = x0 + (di + .5) * group_w
            off = next(row for row in rq4_rows if row["stage"] == stage and row["domain"] == domain and row["value_head"] == "off")
            on = next(row for row in rq4_rows if row["stage"] == stage and row["domain"] == domain and row["value_head"] == "on")
            for row, color, offset in ((off, "#4c78a8", -38), (on, "#d95f02", 38)):
                cap = float(row["capacity"])
                start, end = float(row["policy_mean"]), float(row["mcts_6h_mean"])
                y1 = top + panel_h - start / cap * panel_h
                y2 = top + panel_h - end / cap * panel_h
                x1, x2 = center + offset - 18, center + offset + 18
                partial = row["evidence_status"] == "partial_lower_bound"
                parts += [
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="3" stroke-dasharray="{"4 2" if partial else "none"}"/>',
                    f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="5" fill="white" stroke="{color}" stroke-width="3"/>',
                    f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="5" fill="{color}" opacity="{0.65 if partial else 1}"/>',
                    f'<text x="{x1:.1f}" y="{y1 - 9:.1f}" text-anchor="middle" class="value">{start:.1f}</text>',
                    f'<text x="{x2:.1f}" y="{y2 - 9:.1f}" text-anchor="middle" class="value">{"≥" if partial else ""}{end:.1f}</text>',
                ]
            effect = effect_map.get((stage, domain))
            annotation = "DiD pending" if effect is None else f'DiD {float(effect["effect_solved"]):+.1f}, pH={float(effect["holm_p"]):.3g}'
            parts += [
                f'<text x="{center:.1f}" y="{top + panel_h + 22}" text-anchor="middle" class="label">{html.escape(LABELS[domain])}</text>',
                f'<text x="{center:.1f}" y="{top + panel_h + 39}" text-anchor="middle" class="sub">{html.escape(annotation)}</text>',
            ]
    parts += ['<line x1="900" y1="68" x2="900" y2="667" stroke="#9aa7b4" stroke-width="2"/>']
    parts += [
        '<line x1="630" y1="699" x2="670" y2="699" stroke="#4c78a8" stroke-width="3"/><text x="680" y="703" class="sub">VH-off</text>',
        '<line x1="800" y1="699" x2="840" y2="699" stroke="#d95f02" stroke-width="3"/><text x="850" y="703" class="sub">VH-on</text>',
        '<text x="28" y="738" class="sub">Open marker: policy. Filled marker: exact six-hour MCTS mean. Stage 1 includes final MPrime; Stage 2 remains the five completed validation-led domains.</text>',
        '</svg>',
    ]
    (OUT / "rq4_raw_means_6h_by_stage.svg").write_text("".join(parts), encoding="utf-8")


def terminal_archive() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    policy = [row for row in read_csv(TRACK / "policy_paired_seed_results.csv")
              if row["experiment_id"] == "MAIN-TERM"]
    for (domain, vh), group in sorted(_group(policy, "domain", "value_head").items()):
        rows.append({
            "evidence_type": "policy_refinement",
            "domain": domain, "value_head": vh, "n": len(group),
            "status_from_20260910": "archived_sensitivity_only",
            "new_training": "no", "new_policy_evaluation": "no", "new_mcts_evaluation": "no",
            "primary_tables_or_plots": "exclude",
            "source": "experiment_tracking/policy_paired_seed_results.csv",
        })
    mcts = [row for row in read_csv(TRACK / "stage2_policy_mcts_seed_cutoffs_latest.csv")
            if row["stage2_branch"] == "terminal_led"]
    for (domain, vh), group in sorted(_group(mcts, "domain", "value_head").items()):
        rows.append({
            "evidence_type": "mcts_inference",
            "domain": domain, "value_head": vh, "n": len(group),
            "status_from_20260910": "archived_sensitivity_only",
            "new_training": "no", "new_policy_evaluation": "no", "new_mcts_evaluation": "no",
            "primary_tables_or_plots": "exclude",
            "source": "experiment_tracking/stage2_policy_mcts_seed_cutoffs_latest.csv",
        })
    return rows


def _group(rows: list[dict[str, str]], *fields: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def main() -> None:
    rows = build_rq_rows()
    write_csv(OUT / "rq_primary_validation_led.csv", rows)
    rq2_raw, rq3_raw, rq4_raw = build_raw_mean_rows(rows)
    write_csv(OUT / "rq2_raw_means_validation_led.csv", rq2_raw)
    write_csv(OUT / "rq3_raw_means_validation_led.csv", rq3_raw)
    write_csv(OUT / "rq4_raw_means_validation_led.csv", rq4_raw)
    write_csv(OUT / "terminal_led_archive_index.csv", terminal_archive())
    plot_rows(rows, "RQ1", "rq1_stage2_training_vh_off", "RQ1 — Does Stage-2 training improve policy coverage without a value head?")
    plot_rows(rows, "RQ2", "rq2_mcts_vh_off", "RQ2 — Does inference-time MCTS improve VH-off coverage?")
    plot_rows(rows, "RQ3", "rq3_value_head_training", "RQ3 — Does the value head change Stage-2 refinement?")
    plot_rows(rows, "RQ4", "rq4_value_head_mcts", "RQ4 — Does the value head change the benefit of MCTS inference?")
    plot_rows(rows, "RQ4", "rq4_direct", "RQ4a — MCTS benefit within VH-on checkpoints",
              "VH-on direct")
    plot_rows(rows, "RQ4", "rq4_cross_cell", "RQ4b — VH-on MCTS versus parallel VH-off policy",
              "Cross-cell level")
    plot_rows(rows, "RQ4", "rq4_interaction", "RQ4c — Value-head interaction with MCTS benefit",
              "VH interaction")
    plot_rq2_raw(rq2_raw)
    plot_rq3_raw(rq3_raw)
    plot_rq4_raw(rq2_raw, rq4_raw, rows)


if __name__ == "__main__":
    main()
