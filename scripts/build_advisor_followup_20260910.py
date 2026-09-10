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
LABELS = {
    "block_grouping": "Block Grouping",
    "drone": "Drone",
    "fo_counters": "FO Counters",
    "rover": "Rover",
    "counters": "Counters",
}
CAPACITY = {domain: 20 for domain in DOMAINS}
CAPACITY["counters"] = 59
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
        "row_level_provenance": provenance,
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
        if branch:
            rows = [row for row in rows if row.get("stage2_branch") == branch]
        mapping = {(row["domain"], row["value_head"], row["seed"]): row for row in rows}
        available_domains = [domain for domain in DOMAINS
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

    add_holm(output)
    return output


def plot_rows(rows: list[dict[str, object]], rq: str, output_name: str, title: str) -> None:
    selected = [row for row in rows if row["rq"] == rq]
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

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.title{font-size:23px;font-weight:700}.sub{font-size:12px;fill:#536273}.label{font-size:11px}.head{font-size:13px;font-weight:700}</style>',
        f'<text x="28" y="34" class="title">{esc(title)}</text>',
        '<text x="28" y="57" class="sub">Validation-led primary analysis only. Effects are paired by seed; lines are 95% t-intervals; pH is Holm-adjusted within each stage/cutoff family.</text>',
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
            low = float(row["ci95_low_percentage_points"])
            high = float(row["ci95_high_percentage_points"])
            color = "#238b45" if center >= 0 else "#c43c39"
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
    write_csv(OUT / "terminal_led_archive_index.csv", terminal_archive())
    plot_rows(rows, "RQ1", "rq1_stage2_training_vh_off", "RQ1 — Does MCTS-guided Stage-2 training improve policy coverage?")
    plot_rows(rows, "RQ2", "rq2_mcts_vh_off", "RQ2 — Does inference-time MCTS improve VH-off coverage?")
    plot_rows(rows, "RQ3", "rq3_value_head_training", "RQ3 — Does the value head change Stage-2 refinement?")
    plot_rows(rows, "RQ4", "rq4_value_head_mcts", "RQ4 — Does the value head change the benefit of MCTS inference?")


if __name__ == "__main__":
    main()
