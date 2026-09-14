#!/usr/bin/env python3
"""Build RQ2/RQ4 views of the final ten-seed PW70 evidence."""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_ten_seed_results_latest.csv"
OUT = ROOT / "experiment_tracking/advisor_followup_20260910"
CUTOFFS = ("30m", "2h", "6h")


def read() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def interval(diffs: list[float]) -> tuple[float, float]:
    mean = statistics.mean(diffs)
    if len(diffs) < 2:
        return mean, mean
    # Two-sided 95% Student-t critical value for the confirmatory n=10 cells.
    # Keep this aligned with the canonical PW statistics builder.
    critical = 2.2621571627409915 if len(diffs) == 10 else 1.96
    half = critical * statistics.stdev(diffs) / math.sqrt(len(diffs))
    return mean - half, mean + half


def signflip(diffs: list[float]) -> float:
    observed = abs(statistics.mean(diffs))
    count = 0
    for signs in itertools.product((-1, 1), repeat=len(diffs)):
        if abs(statistics.mean(a * b for a, b in zip(diffs, signs))) >= observed - 1e-12:
            count += 1
    return count / (2 ** len(diffs))


def holm(rows: list[dict[str, object]], group_fields: tuple[str, ...]) -> None:
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[tuple(row[field] for field in group_fields)].append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: float(rows[index]["raw_p"]))
        running = 0.0
        total = len(ordered)
        for rank, index in enumerate(ordered):
            running = max(running, (total - rank) * float(rows[index]["raw_p"]))
            rows[index]["holm_p"] = min(1.0, running)


def stats(diffs: list[float]) -> tuple[float, float, float, float]:
    low, high = interval(diffs)
    return statistics.mean(diffs), low, high, signflip(diffs)


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = read()
    cells = {(domain, vh): sorted(
        [row for row in rows if row["domain"] == domain and row["value_head"] == vh],
        key=lambda row: int(row["seed"]),
    ) for domain in ("fo_counters", "rover") for vh in ("off", "on")}
    rq2 = []
    rq4 = []
    provenance = "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_ten_seed_results_latest.csv"
    for domain in ("fo_counters", "rover"):
        off = cells[(domain, "off")]
        on = cells[(domain, "on")]
        assert len(off) == len(on) == 10
        assert len({r["seed"] for r in off}) == len({r["seed"] for r in on}) == 10
        assert [r["seed"] for r in off] == [r["seed"] for r in on]
        for cutoff in CUTOFFS:
            off_policy = [float(row["policy_score"]) for row in off]
            on_policy = [float(row["policy_score"]) for row in on]
            off_fixed = [float(row[f"fixed_{cutoff}"]) for row in off]
            on_fixed = [float(row[f"fixed_{cutoff}"]) for row in on]
            off_pw = [float(row[f"pw70_{cutoff}"]) for row in off]
            on_pw = [float(row[f"pw70_{cutoff}"]) for row in on]
            effect, low, high, raw = stats([a - b for a, b in zip(off_pw, off_policy)])
            rq2.append({
                "rq": "RQ2", "domain": domain, "cutoff": cutoff, "n": 10,
                "vh_off_policy_mean": statistics.mean(off_policy),
                "vh_off_fixed_mcts_mean": statistics.mean(off_fixed),
                "vh_off_pw70_mean": statistics.mean(off_pw),
                "pw70_minus_policy": effect, "ci95_low": low, "ci95_high": high,
                "raw_p": raw, "holm_p": "", "status": "complete_declared_budget",
                "seed_level_provenance": provenance,
            })
            direct = stats([a - b for a, b in zip(on_pw, on_policy)])
            cross = stats([a - b for a, b in zip(on_pw, off_policy)])
            interaction = stats([(a - b) - (c - d) for a, b, c, d in
                                 zip(on_pw, on_policy, off_pw, off_policy)])
            for estimand, result in (
                ("VH-on PW70 - VH-on policy", direct),
                ("VH-on PW70 - parallel VH-off policy", cross),
                ("(VH-on PW70-policy benefit) - (VH-off PW70-policy benefit)", interaction),
            ):
                effect, low, high, raw = result
                rq4.append({
                    "rq": "RQ4", "domain": domain, "cutoff": cutoff, "n": 10,
                    "estimand": estimand,
                    "vh_off_policy_mean": statistics.mean(off_policy),
                    "vh_on_policy_mean": statistics.mean(on_policy),
                    "vh_off_fixed_mcts_mean": statistics.mean(off_fixed),
                    "vh_on_fixed_mcts_mean": statistics.mean(on_fixed),
                    "vh_off_pw70_mean": statistics.mean(off_pw),
                    "vh_on_pw70_mean": statistics.mean(on_pw),
                    "effect": effect, "ci95_low": low, "ci95_high": high,
                    "raw_p": raw, "holm_p": "", "status": "complete_declared_budget",
                    "seed_level_provenance": provenance,
                })
    holm(rq2, ("cutoff",))
    holm(rq4, ("cutoff", "estimand"))
    return rq2, rq4


def plot(rq2: list[dict[str, object]], rq4: list[dict[str, object]]) -> None:
    rows = [row for row in rq4 if row["cutoff"] == "6h" and
            row["estimand"] == "VH-on PW70 - VH-on policy"]
    width, height = 1360, 650
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:26px;font-weight:700}.sub{font-size:16px}.label{font-size:18px;font-weight:700}.value{font-size:15px}</style>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="36" y="42" class="title">PW70 from Stage-1 validation-selected checkpoints: ten-seed evidence</text>',
             '<text x="36" y="72" class="sub">Mean solved /20 at 6h; PW uses Kmin=3, max width 20 and 70 simulations; full 30m/2h/6h results are in the CSVs</text>']
    colors = {"off_policy": "#e8953d", "off_fixed": "#7396b8", "off_pw": "#2a9d8f",
              "on_policy": "#c66b2d", "on_fixed": "#496f93", "on_pw": "#176f65"}
    legend = [("VH-off policy", colors["off_policy"]), ("VH-off fixed 20/70", colors["off_fixed"]),
              ("VH-off PW70", colors["off_pw"]), ("VH-on policy", colors["on_policy"]),
              ("VH-on fixed 20/70", colors["on_fixed"]), ("VH-on PW70", colors["on_pw"])]
    for i, (label, color) in enumerate(legend):
        x = 42 + i * 215
        parts += [f'<rect x="{x}" y="88" width="16" height="16" fill="{color}"/>',
                  f'<text x="{x+23}" y="101" class="sub">{label}</text>']
    scale = 36
    base_y = 445
    for panel, row in enumerate(rows):
        x0 = 105 + panel * 650
        values = [float(row["vh_off_policy_mean"]), float(row["vh_off_fixed_mcts_mean"]),
                  float(row["vh_off_pw70_mean"]), float(row["vh_on_policy_mean"]),
                  float(row["vh_on_fixed_mcts_mean"]), float(row["vh_on_pw70_mean"])]
        keys = ["off_policy", "off_fixed", "off_pw", "on_policy", "on_fixed", "on_pw"]
        for j, (value, key) in enumerate(zip(values, keys)):
            x = x0 + j * 78
            h = value * scale
            parts += [f'<rect x="{x}" y="{base_y-h:.1f}" width="58" height="{h:.1f}" fill="{colors[key]}" rx="3"/>',
                      f'<text x="{x+29}" y="{base_y-h-7:.1f}" class="value" text-anchor="middle">{value:.1f}</text>']
        divider_x = x0 + 224
        parts += [
            f'<line x1="{divider_x}" y1="122" x2="{divider_x}" y2="{base_y}" stroke="#657080" stroke-width="2" stroke-dasharray="7 7"/>',
        ]
        name = "FO Counters" if row["domain"] == "fo_counters" else "Rover"
        off = next(item for item in rq2 if item["domain"] == row["domain"] and item["cutoff"] == "6h")
        cross = next(item for item in rq4 if item["domain"] == row["domain"] and item["cutoff"] == "6h" and
                     item["estimand"] == "VH-on PW70 - parallel VH-off policy")
        interaction = next(item for item in rq4 if item["domain"] == row["domain"] and item["cutoff"] == "6h" and
                           str(item["estimand"]).startswith("(VH-on"))
        parts += [f'<line x1="{x0-18}" y1="{base_y}" x2="{x0+448}" y2="{base_y}" stroke="#657080"/>',
                  f'<text x="{x0+220}" y="478" class="label" text-anchor="middle">{name}</text>',
                  f'<text x="{x0+220}" y="508" class="sub" text-anchor="middle">RQ2 off PW−policy: {float(off["pw70_minus_policy"]):+.1f} [{float(off["ci95_low"]):+.1f}, {float(off["ci95_high"]):+.1f}], Holm p={float(off["holm_p"]):.3g}</text>',
                  f'<text x="{x0+220}" y="537" class="sub" text-anchor="middle">RQ4 on PW−policy: {float(row["effect"]):+.1f} [{float(row["ci95_low"]):+.1f}, {float(row["ci95_high"]):+.1f}], Holm p={float(row["holm_p"]):.3g}</text>',
                  f'<text x="{x0+220}" y="566" class="sub" text-anchor="middle">Cross-cell: {float(cross["effect"]):+.1f}; interaction: {float(interaction["effect"]):+.1f} [{float(interaction["ci95_low"]):+.1f}, {float(interaction["ci95_high"]):+.1f}]</text>']
    parts += ['<text x="680" y="620" class="sub" text-anchor="middle">Holm correction is within each cutoff/estimand across the two evaluated domains.</text>']
    parts += ['</svg>']
    (OUT / "rq2_rq4_pw70_final.svg").write_text("".join(parts), encoding="utf-8")


def main() -> None:
    rq2, rq4 = build()
    write(OUT / "rq2_pw70_branch_latest.csv", rq2)
    write(OUT / "rq4_pw70_branch_latest.csv", rq4)
    plot(rq2, rq4)


if __name__ == "__main__":
    main()
