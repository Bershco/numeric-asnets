#!/usr/bin/env python3
"""Build revised advisor-facing figures requested on 9 September 2026."""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT = TRACK / "learning_curves" / "advisor_v2_20260909"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    "block_grouping", "drone", "fo_counters", "rover", "counters",
    "mprime", "delivery", "tpp", "zenotravel",
]
LABEL = {
    "block_grouping": "Block Grouping", "drone": "Drone",
    "fo_counters": "FO Counters", "rover": "Rover", "counters": "Counters",
    "mprime": "MPrime", "delivery": "Delivery", "tpp": "TPP",
    "zenotravel": "Zenotravel",
}
TOTAL = defaultdict(lambda: 20, {"counters": 59})
BLUE, ORANGE, RED, GREEN = "#1769aa", "#e07a1f", "#c43c39", "#238b45"
INK, MUTED, GRID = "#17212b", "#536273", "#d9e0e7"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def esc(value: object) -> str:
    return html.escape(str(value))


def start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.title{font-size:25px;font-weight:700}.sub{font-size:13px;fill:#536273}.label{font-size:12px}.small{font-size:10px;fill:#536273}.value{font-size:12px;font-weight:700}.cell{font-size:11px;font-weight:600}</style>',
        f'<text x="32" y="38" class="title">{esc(title)}</text>',
        f'<text x="32" y="62" class="sub">{esc(subtitle)}</text>',
    ]


def text(x: float, y: float, value: object, cls: str = "label", anchor: str = "start", fill: str | None = None) -> str:
    extra = f' style="fill:{fill}"' if fill else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{extra}>{esc(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1, dash: str = "") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{extra}/>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", radius: float = 0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def circle(x: float, y: float, r: float, fill: str, stroke: str = "white") -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}"/>'


def poly(points: list[tuple[float, float]], color: str, width: float = 2.3) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def fmt_p(value: float | None, adjusted: bool) -> str:
    if value is None or math.isnan(value):
        return "p unavailable"
    label = "Holm p" if adjusted else "raw p"
    return f"{label}={value:.3f}"


# A. Full all-domain endpoint table: absolute results, normalized changes, CIs and tests.
records: dict[tuple[str, str, str], dict[str, object]] = {}
for row in read(TRACK / "experiment_statistics.csv"):
    if row["record_type"] != "paired" or "S2 validation-selected" not in row["comparison"]:
        continue
    if row["experiment_id"] not in {"MAIN-VAL", "MAIN-TERM"}:
        continue
    branch = "validation" if row["experiment_id"] == "MAIN-VAL" else "terminal"
    records[(row["domain"], row["value_head"], branch)] = {
        "n": int(row["n"]), "before": float(row["mean_before"]), "after": float(row["mean_after"]),
        "change": float(row["mean_difference"]), "ci_low": float(row["ci95_low"]),
        "ci_high": float(row["ci95_high"]), "p": float(row["holm_p"]), "adjusted": True,
        "provisional": False, "source": "experiment_tracking/experiment_statistics.csv",
    }
for row in read(TRACK / "four_domain_preservation" / "stable_domain_stage2_statistics_20260902.csv"):
    records[(row["domain"], row["value_head"], "validation")] = {
        "n": int(row["n"]), "before": float(row["stage1_selected_mean"]),
        "after": float(row["stage2_selected_mean_all10"]), "change": float(row["mean_change"]),
        "ci_low": float(row["ci95_low"]), "ci_high": float(row["ci95_high"]),
        "p": float(row["holm_p"]), "adjusted": True, "provisional": False,
        "source": row["seed_ledger"],
    }
for row in read(TRACK / "preserve3_terminal_led" / "terminal_led_selected_summary_20260903.csv"):
    records[(row["domain"], row["value_head"], "terminal")] = {
        "n": int(row["matched_n"]), "before": float(row["stage1_final_mean"]),
        "after": float(row["stage2_selected_mean"]), "change": float(row["mean_change"]),
        "ci_low": float(row["ci95_low"]), "ci_high": float(row["ci95_high"]),
        "p": float(row["raw_signflip_p"]), "adjusted": False, "provisional": False,
        "source": row["row_level_companion"],
    }
for filename, branch in (
    ("validation_led_stage2_selected_statistics_20260903.csv", "validation"),
    ("terminal_led_stage2_selected_statistics_20260902.csv", "terminal"),
):
    for row in read(TRACK / "mprime_validation_ipc_scale_v1" / filename):
        records[("mprime", row["value_head"], branch)] = {
            "n": int(row["n"]), "before": float(row["stage1_mean"]), "after": float(row["stage2_mean"]),
            "change": float(row["paired_change"]), "ci_low": float(row["ci95_low"]),
            "ci_high": float(row["ci95_high"]), "p": float(row["raw_signflip_p"]),
            "adjusted": False, "provisional": True, "source": row["seed_ledger"],
        }

table_rows: list[dict[str, object]] = []
parts = start(1420, 1040, "A — all-domain Stage-2 endpoint evidence", "Absolute scores prevent large-denominator domains from looking artificially dominant. Change and CI are normalized to percentage points.")
parts += [text(265, 95, "Validation-led: S1 selected → S2 selected", "value"), text(850, 95, "Terminal-led: S1 final → S2 selected", "value")]
x_left, col_w, y0, row_h = 255, 565, 112, 48
for idx, (domain, vh) in enumerate((d, v) for d in DOMAINS for v in ("off", "on")):
    y = y0 + idx * row_h
    if idx % 2 == 0:
        parts.append(rect(20, y - 3, 1375, row_h, "#f7f9fb"))
    parts.append(text(238, y + 17, f"{LABEL[domain]} / VH-{vh}", "label", "end"))
    for bidx, branch in enumerate(("validation", "terminal")):
        rec = records[(domain, vh, branch)]
        total = TOTAL[domain]
        x = x_left + bidx * (col_w + 30)
        delta_pp = 100 * float(rec["change"]) / total
        lo_pp = 100 * float(rec["ci_low"]) / total
        hi_pp = 100 * float(rec["ci_high"]) / total
        significant = float(rec["p"]) < .05
        shade = "#e8f4ea" if significant and delta_pp > 0 else ("#fdebea" if significant and delta_pp < 0 else "#ffffff")
        parts.append(rect(x, y, col_w, 40, shade, GRID, 4))
        suffix = " *provisional" if rec["provisional"] else ""
        parts.append(text(x + 10, y + 15, f"{rec['before']:.1f} → {rec['after']:.1f} / {total}  (n={rec['n']}){suffix}", "cell"))
        parts.append(text(x + 10, y + 32, f"Δ {delta_pp:+.1f} pp [95% CI {lo_pp:+.1f}, {hi_pp:+.1f}]; {fmt_p(float(rec['p']), bool(rec['adjusted']))}", "small"))
        table_rows.append({
            "domain": domain, "value_head": vh, "branch": branch, "n": rec["n"], "total": total,
            "stage1_mean": rec["before"], "stage2_mean": rec["after"], "mean_change_plans": rec["change"],
            "change_percentage_points": delta_pp, "ci95_low_percentage_points": lo_pp,
            "ci95_high_percentage_points": hi_pp, "p_value": rec["p"],
            "p_adjustment": "Holm" if rec["adjusted"] else "raw", "provisional": rec["provisional"],
            "source_seed_or_statistics_ledger": rec["source"],
        })
parts += [
    rect(255, 992, 14, 14, "#e8f4ea", GRID), text(276, 1004, "significant gain", "small"),
    rect(380, 992, 14, 14, "#fdebea", GRID), text(401, 1004, "significant loss", "small"),
    text(540, 1004, "MPrime is provisional until Phase B freezes replacement validation-selected checkpoints.", "sub"), "</svg>",
]
(OUT / "a_all_domain_endpoint_evidence.svg").write_text("".join(parts), encoding="utf-8")
write(OUT / "a_all_domain_endpoint_evidence.csv", table_rows)


# B. Six imperfect domains: explicit Stage-1 endpoint left of a dotted Stage boundary, Stage-2 curve right.
curve = read(TRACK / "learning_curves" / "latest" / "rq1_rq3_five_domain_learning_curve_aggregates.csv")
mprime_cp = read(TRACK / "mprime_validation_ipc_scale_v1" / "validation_adequacy_phase_a_stage2_checkpoints_20260903.csv")
mprime_groups: dict[tuple[str, int], list[float]] = defaultdict(list)
for row in mprime_cp:
    if row["branch"] == "validation_led":
        mprime_groups[(row["value_head"], int(row["epoch"]))].append(100 * float(row["test_success"]) / 20)
parts = start(1320, 790, "B — Stage-1 endpoint → Stage-2 learning dynamics", "The dotted midpoint is the training-stage boundary. Left marker is the matched Stage-1 selected endpoint; right side is the observed Stage-2 test curve.")
panel_domains = ["block_grouping", "drone", "fo_counters", "rover", "counters", "mprime"]
for panel, domain in enumerate(panel_domains):
    col, prow = panel % 3, panel // 3
    px, py, pw, ph = 55 + col * 430, 105 + prow * 320, 390, 245
    parts.append(rect(px, py, pw, ph, "#fbfcfd", GRID, 4))
    parts.append(text(px + 10, py + 22, LABEL[domain], "value"))
    left, right, top, bottom = px + 42, px + pw - 10, py + 38, py + ph - 30
    boundary = left + 70
    for tick in (0, 25, 50, 75, 100):
        yy = bottom - tick / 100 * (bottom - top)
        parts.append(line(left, yy, right, yy, GRID))
        parts.append(text(left - 7, yy + 4, tick, "small", "end"))
    parts.append(line(boundary, top, boundary, bottom, INK, 1.4, "4 4"))
    parts.append(text(left + 22, bottom + 18, "S1", "small", "middle"))
    parts.append(text((boundary + right) / 2, bottom + 18, "S2 epoch 0 → 100", "small", "middle"))
    for vh, color, rq in (("off", BLUE, "RQ1"), ("on", ORANGE, "RQ3")):
        if domain == "mprime":
            grouped = [(epoch, values) for (gvh, epoch), values in mprime_groups.items() if gvh == vh]
            grouped.sort()
            rows = [{"epoch": e, "mean": sum(v) / len(v), "min": min(v), "max": max(v)} for e, v in grouped]
            baseline = 75.0 if vh == "off" else 73.0
        else:
            raw = [r for r in curve if r["domain"] == domain and r["research_question"] == rq]
            rows = [{"epoch": int(r["epoch"]), "mean": float(r["mean_percent"]), "min": float(r["minimum_percent"]), "max": float(r["maximum_percent"])} for r in raw]
            baseline = float(raw[0]["stage1_baseline_percent"])
        if not rows:
            continue
        max_epoch = max(int(r["epoch"]) for r in rows)
        sx = lambda epoch: boundary + 8 + epoch / max(max_epoch, 1) * (right - boundary - 8)
        sy = lambda score: bottom - score / 100 * (bottom - top)
        upper = [(sx(r["epoch"]), sy(r["max"])) for r in rows]
        lower = [(sx(r["epoch"]), sy(r["min"])) for r in reversed(rows)]
        polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
        parts.append(f'<polygon points="{polygon}" fill="{color}" opacity="0.10"/>')
        parts.append(circle(left + 22, sy(baseline), 5, color))
        parts.append(line(left + 22, sy(baseline), sx(rows[0]["epoch"]), sy(rows[0]["mean"]), color, 1.4, "3 3"))
        parts.append(poly([(sx(r["epoch"]), sy(r["mean"])) for r in rows], color))
parts += [
    line(60, 760, 95, 760, BLUE, 3), text(102, 764, "VH-off", "small"),
    line(175, 760, 210, 760, ORANGE, 3), text(217, 764, "VH-on", "small"),
    text(310, 764, "Shading = observed min–max; the connecting dotted segment is not an inferred Stage-1 curve.", "small"), "</svg>",
]
(OUT / "b_stage1_to_stage2_learning_dynamics.svg").write_text("".join(parts), encoding="utf-8")


# C. More readable PRESERVE-3 seed robustness view.
points = read(TRACK / "learning_curves" / "candidates_20260909" / "candidate_c_preserve3_seed_changes.csv")
parts = start(1420, 720, "C — PRESERVE-3 robustness by seed", "Each dot is a matched seed. Labels identify every ≥5-plan regression; bold ticks are cell means. Positive values mean Stage 2 solved more instances.")
cells = [(d, vh) for d in ("delivery", "tpp", "zenotravel") for vh in ("off", "on")]
for panel, branch in enumerate(("validation", "terminal")):
    px, py, pw, ph = 180 + panel * 665, 125, 455, 455
    parts.append(text(px + pw / 2, 100, f"{branch.title()}-led", "value", "middle"))
    for tick in (-12, -8, -4, 0, 4, 8, 12):
        xx = px + (tick + 12) / 24 * pw
        parts.append(line(xx, py, xx, py + ph, RED if tick == 0 else GRID, 1.6 if tick == 0 else 1))
        parts.append(text(xx, py + ph + 22, f"{tick:+d}", "small", "middle"))
    for i, (domain, vh) in enumerate(cells):
        yy = py + 35 + i * 69
        parts.append(text(px - 14, yy + 4, f"{LABEL[domain]} / VH-{vh}", "label", "end"))
        rows = [r for r in points if r["branch"] == branch and r["domain"] == domain and r["value_head"] == vh]
        values = [float(r["change"]) for r in rows]
        mean = sum(values) / len(values)
        for j, row in enumerate(rows):
            value = float(row["change"])
            xx = px + (value + 12) / 24 * pw
            jitter = ((j % 5) - 2) * 3
            color = RED if value <= -5 else (GREEN if value > 0 else "#7b8794")
            parts.append(circle(xx, yy + jitter, 5, color))
            if value <= -5:
                label = f"seed {row['seed']}: {float(row['stage1']):.0f}→{float(row['stage2']):.0f}"
                parts.append(text(xx + 7, yy + jitter - 7, label, "small"))
        mx = px + (mean + 12) / 24 * pw
        parts.append(line(mx, yy - 16, mx, yy + 16, INK, 3.2))
        parts.append(text(mx + 5, yy + 20, f"mean {mean:+.1f}", "small"))
parts += [
    text(180, 640, "Stage-2 − Stage-1 solved instances", "value"),
    circle(535, 636, 5, GREEN), text(547, 640, "gain", "small"),
    circle(600, 636, 5, "#7b8794"), text(612, 640, "tie / small loss", "small"),
    circle(730, 636, 5, RED), text(742, 640, "large regression", "small"),
    text(180, 682, "Takeaway: mean preservation can coexist with rare, severe seed-level failures; TPP/off validation-led is the clearest case.", "sub"), "</svg>",
]
(OUT / "c_preserve3_seed_robustness.svg").write_text("".join(parts), encoding="utf-8")
write(OUT / "c_preserve3_seed_robustness.csv", points)


# D. Meaningful MPrime selection-cost chart: selected test score vs oracle-best observed checkpoint.
s1 = read(TRACK / "mprime_validation_ipc_scale_v1" / "validation_adequacy_phase_a_stage1_seeds_20260903.csv")
s2 = read(TRACK / "mprime_validation_ipc_scale_v1" / "validation_adequacy_phase_a_stage2_seeds_20260903.csv")
groups: list[tuple[str, list[dict[str, str]]]] = []
for vh in ("off", "on"):
    groups.append((f"Stage 1 / VH-{vh}", [r for r in s1 if r["value_head"] == vh]))
for branch in ("validation_led", "terminal_led"):
    for vh in ("off", "on"):
        groups.append((f"S2 {branch.split('_')[0]} / VH-{vh}", [r for r in s2 if r["branch"] == branch and r["value_head"] == vh]))
summary = []
for label, rows in groups:
    selected = [float(r["selected_test_score"]) for r in rows]
    best = [float(r["observed_test_best_score"]) for r in rows]
    unique = [float(r["validation_unique_scores"]) for r in rows]
    maxfrac = [float(r["validation_max_fraction"]) for r in rows]
    summary.append({
        "group": label, "n": len(rows), "selected_test_mean": sum(selected) / len(selected),
        "oracle_observed_test_best_mean": sum(best) / len(best),
        "mean_selection_regret": sum(b - a for a, b in zip(selected, best)) / len(rows),
        "mean_validation_unique_scores": sum(unique) / len(unique),
        "mean_fraction_at_validation_max": sum(maxfrac) / len(maxfrac),
        "source": rows[0].get("source_checkpoint_audit", rows[0].get("checkpoint_ledger", "")),
    })
write(OUT / "d_mprime_validation_selection_cost.csv", summary)
parts = start(1320, 650, "D — what MPrime validation selection actually costs", "Dumbbells compare the checkpoint selected by validation with the best test checkpoint observed retrospectively. The right panel shows why Stage 2 cannot rank snapshots.")
px, py, pw = 285, 125, 500
for tick in range(8, 21, 2):
    xx = px + (tick - 8) / 12 * pw
    parts.append(line(xx, py, xx, py + 360, GRID))
    parts.append(text(xx, py + 382, tick, "small", "middle"))
for i, row in enumerate(summary):
    yy = py + 35 + i * 55
    sel, best = float(row["selected_test_mean"]), float(row["oracle_observed_test_best_mean"])
    sx = px + (sel - 8) / 12 * pw
    bx = px + (best - 8) / 12 * pw
    parts.append(text(px - 15, yy + 4, row["group"], "label", "end"))
    parts.append(line(sx, yy, bx, yy, "#8b98a5", 4))
    parts.append(circle(sx, yy, 7, ORANGE))
    parts.append(circle(bx, yy, 7, BLUE))
    parts.append(text(max(sx, bx) + 10, yy + 4, f"regret {float(row['mean_selection_regret']):.1f}", "small"))
parts.append(text(px + pw / 2, py + 410, "Mean solved test instances / 20", "value", "middle"))
parts.append(circle(290, 570, 6, ORANGE)); parts.append(text(303, 574, "validation-selected", "small"))
parts.append(circle(430, 570, 6, BLUE)); parts.append(text(443, 574, "retrospective observed best", "small"))

rx, ry, rw = 870, 140, 245
parts.append(text(rx + rw / 2, 110, "Validation discriminative power", "value", "middle"))
for i, row in enumerate(summary):
    yy = ry + i * 65
    frac = float(row["mean_fraction_at_validation_max"])
    parts.append(text(rx, yy, row["group"], "small"))
    parts.append(rect(rx, yy + 9, rw, 16, "#edf1f5", GRID, 3))
    parts.append(rect(rx, yy + 9, rw * frac, 16, RED if frac > .95 else GREEN, "none", 3))
    parts.append(text(rx + rw + 8, yy + 22, f"{frac:.0%} at max", "small"))
parts += [
    text(870, 555, "Stage 2: every checkpoint is at the validation maximum.", "value"),
    text(870, 576, "Therefore its selected checkpoint is effectively a tie-break,", "sub"),
    text(870, 595, "not evidence of best generalization.", "sub"),
    text(32, 626, "Takeaway: Phase B is needed because the current set creates measurable selection regret at Stage 1 and total ranking failure at Stage 2.", "sub"), "</svg>",
]
(OUT / "d_mprime_validation_selection_cost.svg").write_text("".join(parts), encoding="utf-8")


(OUT / "README.md").write_text(
    """# Revised advisor figures — 9 September 2026

These replace the first candidate set after user review.

- `a_all_domain_endpoint_evidence`: absolute S1/S2 scores, denominators, normalized changes, 95% CIs and p-values.
- `b_stage1_to_stage2_learning_dynamics`: explicit dotted S1/S2 boundary, Stage-1 endpoint marker, Stage-2 curve and min–max envelope.
- `c_preserve3_seed_robustness`: readable labels and explicit catastrophic-seed callouts.
- `d_mprime_validation_selection_cost`: direct selection-regret interpretation rather than an abstract diagnostic.

Each evidence-heavy figure has a colocated CSV. Candidate B is reproduced from the cached curve sources in `learning_curves/latest/` and the MPrime Phase-A checkpoint ledger; no cluster read is needed to restyle it.
""",
    encoding="utf-8",
)
print(OUT)
