#!/usr/bin/env python3
"""Build four deliberately different advisor-facing plot candidates as SVG."""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT = TRACK / "learning_curves" / "candidates_20260909"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    "block_grouping", "drone", "fo_counters", "rover", "counters",
    "mprime", "delivery", "tpp", "zenotravel",
]
LABELS = {
    "block_grouping": "Block Grouping", "drone": "Drone",
    "fo_counters": "FO Counters", "rover": "Rover", "counters": "Counters",
    "mprime": "MPrime", "delivery": "Delivery", "tpp": "TPP",
    "zenotravel": "Zenotravel",
}
BLUE = "#1769aa"
ORANGE = "#e07a1f"
GREEN = "#238b45"
RED = "#c43c39"
INK = "#17212b"
GRID = "#d9e0e7"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def esc(value: object) -> str:
    return html.escape(str(value))


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#52616f}.label{font-size:12px}.small{font-size:10px;fill:#52616f}.value{font-size:12px;font-weight:700}</style>',
        f'<text x="36" y="38" class="title">{esc(title)}</text>',
        f'<text x="36" y="61" class="sub">{esc(subtitle)}</text>',
    ]


def text(x: float, y: float, value: object, cls: str = "label", anchor: str = "start", fill: str | None = None) -> str:
    extra = f' fill="{fill}"' if fill else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{extra}>{esc(value)}</text>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", opacity: float = 1.0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1.0, dash: str = "") -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def circle(cx: float, cy: float, radius: float, fill: str, stroke: str = "white", opacity: float = 1.0) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'


def polyline(points: list[tuple[float, float]], stroke: str, width: float = 2.0, fill: str = "none") -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def change_color(value: float) -> str:
    magnitude = min(abs(value) / 5.0, 1.0)
    base = (196, 60, 57) if value < 0 else (23, 105, 170)
    rgb = tuple(round(245 + (component - 245) * magnitude) for component in base)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


# Candidate A: all-domain endpoint outcome map.
changes: dict[tuple[str, str, str], float] = {}
for row in read(TRACK / "experiment_statistics.csv"):
    if row["record_type"] != "paired" or "-> S2 validation-selected" not in row["comparison"]:
        continue
    if row["experiment_id"] in {"MAIN-VAL", "MAIN-TERM"}:
        branch = "validation" if row["experiment_id"] == "MAIN-VAL" else "terminal"
        changes[(row["domain"], branch, row["value_head"])] = float(row["mean_difference"])
for row in read(TRACK / "four_domain_preservation" / "stable_domain_stage2_seed_pairs_20260902.csv"):
    changes[(row["domain"], "validation", row["value_head"])] = changes.get(
        (row["domain"], "validation", row["value_head"]), 0.0
    ) + float(row["change"])
stable_counts = defaultdict(int)
for row in read(TRACK / "four_domain_preservation" / "stable_domain_stage2_seed_pairs_20260902.csv"):
    stable_counts[(row["domain"], "validation", row["value_head"])] += 1
for key, count in stable_counts.items():
    changes[key] /= count
for row in read(TRACK / "preserve3_terminal_led" / "terminal_led_selected_seed_results_20260903.csv"):
    key = (row["domain"], "terminal", row["value_head"])
    changes[key] = changes.get(key, 0.0) + float(row["paired_change"])
terminal_counts = defaultdict(int)
for row in read(TRACK / "preserve3_terminal_led" / "terminal_led_selected_seed_results_20260903.csv"):
    terminal_counts[(row["domain"], "terminal", row["value_head"])] += 1
for key, count in terminal_counts.items():
    changes[key] /= count
for filename, branch in (
    ("validation_led_stage2_selected_statistics_20260903.csv", "validation"),
    ("terminal_led_stage2_selected_statistics_20260902.csv", "terminal"),
):
    for row in read(TRACK / "mprime_validation_ipc_scale_v1" / filename):
        changes[("mprime", branch, row["value_head"])] = float(row["paired_change"])

heat_rows = []
columns = [("validation", "off"), ("validation", "on"), ("terminal", "off"), ("terminal", "on")]
parts = svg_start(980, 590, "Candidate A — all-domain Stage-2 outcome map", "Mean policy change from matched Stage 1; blue helps, red hurts. MPrime remains provisional pending Phase B.")
x0, y0, cw, rh = 270, 112, 150, 45
for index, (branch, vh) in enumerate(columns):
    parts.append(text(x0 + index * cw + cw / 2, 91, f"{branch.title()} / VH-{vh}", "label", "middle"))
for row_index, domain in enumerate(DOMAINS):
    y = y0 + row_index * rh
    parts.append(text(x0 - 18, y + 28, LABELS[domain], "label", "end"))
    for col_index, (branch, vh) in enumerate(columns):
        value = changes[(domain, branch, vh)]
        x = x0 + col_index * cw
        parts.append(rect(x, y, cw - 6, rh - 6, change_color(value), "white"))
        parts.append(text(x + (cw - 6) / 2, y + 27, f"{value:+.1f}", "value", "middle", "white" if abs(value) >= 2.8 else INK))
        heat_rows.append({"domain": domain, "branch": branch, "value_head": vh, "mean_change": value, "source": "experiment_statistics.csv or branch-specific stable/MPrime seed ledger"})
parts += [text(270, 548, "Interpretation: this is the broadest map. It answers where Stage 2 helps or hurts, but not how the trajectory evolved.", "sub"), "</svg>"]
(OUT / "candidate_a_all_domain_outcome_map.svg").write_text("".join(parts), encoding="utf-8")
write(OUT / "candidate_a_all_domain_outcome_map.csv", heat_rows)


# Candidate B: the six imperfect-domain learning dynamics.
curve = read(TRACK / "learning_curves" / "latest" / "rq1_rq3_five_domain_learning_curve_aggregates.csv")
mprime = read(TRACK / "mprime_validation_ipc_scale_v1" / "corrected_learning_curve_aggregate.csv")
parts = svg_start(1180, 740, "Candidate B — six imperfect-domain training dynamics", "Mean Stage-2 test coverage with full observed min–max envelope; MPrime is shown separately from the five original RQ domains.")
panel_domains = ["block_grouping", "drone", "fo_counters", "rover", "counters", "mprime"]
for panel, domain in enumerate(panel_domains):
    col, row_index = panel % 3, panel // 3
    px, py, pw, ph = 58 + col * 375, 100 + row_index * 310, 330, 235
    parts.append(rect(px, py, pw, ph, "#fbfcfd", GRID))
    parts.append(text(px + 8, py + 20, LABELS[domain], "value"))
    for tick in range(0, 101, 25):
        yy = py + ph - 25 - tick / 100 * (ph - 55)
        parts.append(line(px + 38, yy, px + pw - 8, yy, GRID))
        parts.append(text(px + 32, yy + 4, tick, "small", "end"))
    parts.append(line(px + 38, py + ph - 25, px + pw - 8, py + ph - 25, INK))
    parts.append(line(px + 38, py + 30, px + 38, py + ph - 25, INK))
    datasets = []
    if domain != "mprime":
        for rq, color, label in (("RQ1", BLUE, "VH-off"), ("RQ3", ORANGE, "VH-on")):
            rows = [r for r in curve if r["domain"] == domain and r["research_question"] == rq]
            baseline = float(rows[0]["stage1_baseline_percent"]) if rows else math.nan
            datasets.append((rows, color, label, "epoch", "mean_percent", "minimum_percent", "maximum_percent", baseline))
    else:
        for vh, color in (("off", BLUE), ("on", ORANGE)):
            rows = [r for r in mprime if r["value_head"] == vh]
            baseline = 75.0 if vh == "off" else 73.0
            datasets.append((rows, color, f"VH-{vh}", "epoch", "test_mean_pct", "test_min_pct", "test_max_pct", baseline))
    for rows, color, label, epoch_col, mean_col, min_col, max_col, baseline in datasets:
        rows.sort(key=lambda r: int(r[epoch_col]))
        if not rows:
            continue
        maximum_epoch = max(int(r[epoch_col]) for r in rows)
        def sx(epoch: float) -> float:
            return px + 38 + epoch / max(maximum_epoch, 1) * (pw - 50)
        def sy(score: float) -> float:
            return py + ph - 25 - score / 100 * (ph - 55)
        upper = [(sx(float(r[epoch_col])), sy(float(r[max_col]))) for r in rows]
        lower = [(sx(float(r[epoch_col])), sy(float(r[min_col]))) for r in reversed(rows)]
        polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
        parts.append(f'<polygon points="{polygon}" fill="{color}" opacity="0.10"/>')
        parts.append(line(px + 38, sy(baseline), px + pw - 8, sy(baseline), color, 1.2, "5 4"))
        points = [(sx(float(r[epoch_col])), sy(float(r[mean_col]))) for r in rows]
        parts.append(polyline(points, color, 2.3))
    parts.append(text(px + 38, py + ph - 7, "Stage-2 epoch", "small"))
parts += [
    line(70, 710, 105, 710, BLUE, 3), text(112, 714, "VH-off mean", "small"),
    line(230, 710, 265, 710, ORANGE, 3), text(272, 714, "VH-on mean", "small"),
    text(420, 714, "Solid = Stage-2 mean; dashed = matched Stage-1 baseline; shading = full min–max envelope.", "small"),
    "</svg>",
]
(OUT / "candidate_b_six_imperfect_learning_dynamics.svg").write_text("".join(parts), encoding="utf-8")


# Candidate C: PRESERVE-3 paired-seed robustness.
preserve_points = []
for row in read(TRACK / "four_domain_preservation" / "stable_domain_stage2_seed_pairs_20260902.csv"):
    preserve_points.append({"branch": "validation", "domain": row["domain"], "value_head": row["value_head"], "seed": row["seed"], "stage1": row["stage1_selected"], "stage2": row["stage2_selected"], "change": row["change"], "stage1_log": row["stage1_evaluation_log"], "stage2_log": row["stage2_evaluation_log"]})
for row in read(TRACK / "preserve3_terminal_led" / "terminal_led_selected_seed_results_20260903.csv"):
    preserve_points.append({"branch": "terminal", "domain": row["domain"], "value_head": row["value_head"], "seed": row["seed"], "stage1": row["stage1_final_score"], "stage2": row["stage2_selected_score"], "change": row["paired_change"], "stage1_log": row["stage1_evaluation_log"], "stage2_log": row["stage2_evaluation_log"]})
write(OUT / "candidate_c_preserve3_seed_changes.csv", preserve_points)
parts = svg_start(1280, 650, "Candidate C — PRESERVE-3 robustness, seed by seed", "Each dot is one matched seed's Stage-2 minus Stage-1 policy change. The isolated TPP/off collapse is visible instead of hidden in a mean.")
cells = [(domain, vh) for domain in ("delivery", "tpp", "zenotravel") for vh in ("off", "on")]
for panel, branch in enumerate(("validation", "terminal")):
    px = 155 + panel * 610
    py, pw, ph = 112, 420, 455
    parts.append(text(px + pw / 2, 92, f"{branch.title()}-led", "value", "middle"))
    for tick in (-12, -8, -4, 0, 4, 8):
        xx = px + (tick + 12) / 20 * pw
        parts.append(line(xx, py, xx, py + ph, RED if tick == 0 else GRID, 1.6 if tick == 0 else 1))
        parts.append(text(xx, py + ph + 22, f"{tick:+d}", "small", "middle"))
    for index, (domain, vh) in enumerate(cells):
        yy = py + 30 + index * 66
        parts.append(text(px - 10, yy + 4, f"{LABELS[domain]} / {vh}", "label", "end"))
        rows = [r for r in preserve_points if r["branch"] == branch and r["domain"] == domain and r["value_head"] == vh]
        values = [float(r["change"]) for r in rows]
        mean = sum(values) / len(values)
        for j, value in enumerate(values):
            xx = px + (value + 12) / 20 * pw
            jitter = ((j % 5) - 2) * 3
            color = RED if value <= -8 else (BLUE if value > 0 else "#7b8794")
            parts.append(circle(xx, yy + jitter, 4.5, color, "white", .9))
        mean_x = px + (mean + 12) / 20 * pw
        parts.append(line(mean_x, yy - 12, mean_x, yy + 12, INK, 3))
        parts.append(text(mean_x + 5, yy - 15, f"{mean:+.1f}", "small"))
parts += [text(155, 620, "Best for: defending 'preserved on average' while explicitly exposing rare catastrophic regressions.", "sub"), "</svg>"]
(OUT / "candidate_c_preserve3_seed_robustness.svg").write_text("".join(parts), encoding="utf-8")


# Candidate D: MPrime validation adequacy failure modes.
stage1 = read(TRACK / "mprime_validation_ipc_scale_v1" / "validation_adequacy_phase_a_stage1_seeds_20260903.csv")
stage2 = read(TRACK / "mprime_validation_ipc_scale_v1" / "validation_adequacy_phase_a_stage2_seeds_20260903.csv")
diagnostic_rows = []
for row in stage1:
    diagnostic_rows.append({"stage": "stage1", "branch": "corrected", "value_head": row["value_head"], "seed": row["seed"], "spearman": row["spearman_validation_test"], "selected_regret": row["selected_test_regret"], "validation_unique_scores": row["validation_unique_scores"], "validation_max_fraction": row["validation_max_fraction"], "source": row["source_checkpoint_audit"]})
for row in stage2:
    diagnostic_rows.append({"stage": "stage2", "branch": row["branch"], "value_head": row["value_head"], "seed": row["seed"], "spearman": row["spearman_validation_test"], "selected_regret": row["selected_test_regret"], "validation_unique_scores": row["validation_unique_scores"], "validation_max_fraction": row["validation_max_fraction"], "source": row["checkpoint_ledger"]})
write(OUT / "candidate_d_mprime_validation_diagnostic.csv", diagnostic_rows)
parts = svg_start(1060, 570, "Candidate D — why MPrime validation needed Phase B", "Stage 1 has weak rank agreement and nonzero regret; Stage 2 is completely saturated, so it cannot rank checkpoints.")
px, py, pw, ph = 90, 120, 410, 330
parts.append(text(px + pw / 2, 96, "Stage 1: rank agreement vs selection regret", "value", "middle"))
for tick in (-0.5, 0, 0.5, 1.0):
    xx = px + (tick + .5) / 1.5 * pw
    parts.append(line(xx, py, xx, py + ph, GRID))
    parts.append(text(xx, py + ph + 22, tick, "small", "middle"))
for tick in range(0, 9, 2):
    yy = py + ph - tick / 8 * ph
    parts.append(line(px, yy, px + pw, yy, GRID))
    parts.append(text(px - 10, yy + 4, tick, "small", "end"))
for row in stage1:
    x = px + (float(row["spearman_validation_test"]) + .5) / 1.5 * pw
    y = py + ph - float(row["selected_test_regret"]) / 8 * ph
    parts.append(circle(x, y, 6, BLUE if row["value_head"] == "off" else ORANGE))
parts.append(text(px + pw / 2, py + ph + 43, "Spearman(validation, test)", "small", "middle"))
parts.append(text(px, py - 10, "Selection regret (plans)", "small"))

px2, py2, pw2, ph2 = 600, 120, 360, 330
parts.append(text(px2 + pw2 / 2, 96, "Stage 2: fraction of checkpoints at max validation", "value", "middle"))
groups = [(branch, vh) for branch in ("validation_led", "terminal_led") for vh in ("off", "on")]
for index, (branch, vh) in enumerate(groups):
    vals = [float(r["validation_max_fraction"]) for r in stage2 if r["branch"] == branch and r["value_head"] == vh]
    mean = sum(vals) / len(vals)
    x = px2 + 35 + index * 82
    height = mean * ph2
    parts.append(rect(x, py2 + ph2 - height, 52, height, BLUE if vh == "off" else ORANGE))
    parts.append(text(x + 26, py2 + ph2 + 19, f"{branch.split('_')[0]}/{vh}", "small", "middle"))
    parts.append(text(x + 26, py2 + ph2 - height - 8, f"{mean:.0%}", "value", "middle"))
parts.append(text(px2 + pw2 / 2, py2 + ph2 + 49, "All four groups = 100% saturation", "sub", "middle"))
parts += [text(90, 525, "Best for: explaining why the original corrected validation set was better than the old one but still inadequate for Stage-2 selection.", "sub"), "</svg>"]
(OUT / "candidate_d_mprime_validation_diagnostic.svg").write_text("".join(parts), encoding="utf-8")


readme = """# Advisor plot candidates — 9 September 2026

These are deliberately different analytical views, not style variants.

1. `candidate_a_all_domain_outcome_map.svg`: broad all-nine-domain endpoint map.
2. `candidate_b_six_imperfect_learning_dynamics.svg`: actual Stage-2 trajectories for the six imperfect domains.
3. `candidate_c_preserve3_seed_robustness.svg`: paired-seed stability and catastrophic outliers for PRESERVE-3.
4. `candidate_d_mprime_validation_diagnostic.svg`: why the MPrime selector requires Phase B.

Every source CSV is colocated here. Candidate B reuses the locally cached aggregate
CSV and the MPrime aggregate; the other three have dedicated companion CSVs.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print(OUT)
