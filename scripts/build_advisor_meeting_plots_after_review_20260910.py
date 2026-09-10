#!/usr/bin/env python3
"""Build advisor figures revised after a context-free visual review.

The input is the colocated CSV data from the immutable before-review figures,
so the visual revision cannot silently change any scientific result.
"""

from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiment_tracking" / "advisor_meeting_20260910"
BEFORE = PACKAGE / "before_review"
OUT = PACKAGE / "after_review"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, RED = "#1769aa", "#e07a1f", "#238b45", "#c43c39"
PURPLE, INK, MUTED, GRID = "#7b4ab5", "#17212b", "#536273", "#d9e0e7"
LABEL = {
    "block_grouping": "Block Grouping", "drone": "Drone",
    "fo_counters": "FO Counters", "rover": "Rover", "counters": "Counters",
    "mprime": "MPrime", "delivery": "Delivery", "tpp": "TPP",
    "zenotravel": "Zenotravel",
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def esc(value: object) -> str:
    return html.escape(str(value))


def start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17212b}.title{font-size:25px;font-weight:700}.sub{font-size:13px;fill:#536273}.label{font-size:12px}.small{font-size:10px;fill:#536273}.value{font-size:12px;font-weight:700}.big{font-size:15px;font-weight:700}</style>',
        f'<text x="34" y="39" class="title">{esc(title)}</text>',
        f'<text x="34" y="63" class="sub">{esc(subtitle)}</text>',
    ]


def text(x: float, y: float, value: object, cls: str = "label", anchor: str = "start", fill: str | None = None) -> str:
    style = f' style="fill:{fill}"' if fill else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{style}>{esc(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1, dash: str = "") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{extra}/>'


def rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "none", radius: float = 0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def circle(x: float, y: float, radius: float, fill: str, stroke: str = "white") -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}"/>'


def poly(points: list[tuple[float, float]], color: str, width: float = 2.2) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def save(name: str, parts: list[str]) -> None:
    (OUT / f"{name}.svg").write_text("".join(parts + ["</svg>"]), encoding="utf-8")


# 1. VH-separated scorecard. The baseline is defined explicitly as the mean
# test score of each seed's validation-selected Stage-1 checkpoint.
rows = read(OUT / "01_domain_scorecard_by_vh.csv")
parts = start(
    1840, 790, "1 — Stage-1 baseline and best observed coverage, separated by value-head mode",
    "Stage-1 baseline = mean test coverage of each seed's validation-selected Stage-1 checkpoint (n=10 per cell). Best observed may use a different method per cell.",
)
for panel, value_head in enumerate(("off", "on")):
    px, py, pw, ph = 45 + panel * 900, 105, 850, 595
    x0, x1, y0, row_height = px + 145, px + 650, py + 55, 58
    parts += [rect(px, py, pw, ph, "#fbfcfd", GRID, 5), text(px + 18, py + 30, f"VH-{value_head}", "big")]
    for tick in (0, 25, 50, 75, 100):
        x = x0 + tick / 100 * (x1 - x0)
        parts += [line(x, y0 - 22, x, y0 + 9 * row_height - 18, GRID), text(x, y0 - 29, f"{tick}%", "small", "middle")]
    panel_rows = [row for row in rows if row["value_head"] == value_head]
    for i, row in enumerate(panel_rows):
        y = y0 + i * row_height
        total = int(row["capacity"])
        baseline = float(row["stage1_validation_selected"])
        best = float(row["best_observed"])
        published = float(row["paper_reported"])
        to_percent = lambda score: score / total * 100
        parts += [
            text(x0 - 12, y + 22, LABEL[row["domain"]], "small", "end"),
            rect(x0, y, (x1 - x0) * to_percent(published) / 100, 9, "#9ca3af", radius=2),
            rect(x0, y + 12, (x1 - x0) * to_percent(baseline) / 100, 9, ORANGE, radius=2),
            rect(x0, y + 24, (x1 - x0) * to_percent(best) / 100, 9, GREEN, radius=2),
            text(x0 + (x1 - x0) * to_percent(published) / 100 + 5, y + 8, f"paper {published:g}/{total}", "small"),
            text(x0 + (x1 - x0) * to_percent(baseline) / 100 + 5, y + 20, f"S1 {baseline:g}/{total}", "small"),
            text(x0 + (x1 - x0) * to_percent(best) / 100 + 5, y + 32, f"best {best:g}/{total}", "small"),
            text(x0, y + 48, f"best method: {row['best_configuration']}", "small"),
        ]
parts += [
    rect(70, 735, 18, 9, ORANGE), text(96, 744, "validation-selected Stage-1 policy baseline", "small"),
    rect(365, 735, 18, 9, GREEN), text(391, 744, "best observed configuration", "small"),
    rect(615, 735, 18, 9, "#9ca3af"), text(641, 744, "published domain mean (exact score shown per row)", "small"),
    text(980, 744, "Exact solved/total labels are means across matched seeds; Counters uses 59 instances, all other domains use 20.", "sub"),
]
save("01_domain_scorecard", parts)


# 2. True Stage-1 and Stage-2 curves with explicit stage-local epoch axes.
curves = read(BEFORE / "02_two_stage_learning_dynamics.csv")
paper_rows = read(BEFORE / "01_domain_scorecard.csv")
paper_by_domain = {row["domain"]: (float(row["paper_reported"]), int(row["capacity"])) for row in paper_rows}
parts = start(
    1600, 940, "2 — full Stage-1 and validation-led Stage-2 policy learning curves",
    "Six imperfect domains; lines are seed means and translucent envelopes are full seed ranges. Stage 2 starts from each seed's validation-selected Stage-1 checkpoint.",
)
parts.append(text(62, 97, "y-axis: test coverage (%)", "small"))
for panel, domain in enumerate(["block_grouping", "drone", "fo_counters", "rover", "counters", "mprime"]):
    col, grid_row = panel % 3, panel // 3
    px, py, pw, ph = 62 + col * 510, 110 + grid_row * 375, 468, 315
    left, right, top, bottom = px + 47, px + pw - 14, py + 42, py + ph - 43
    boundary = (left + right) / 2
    parts += [rect(px, py, pw, ph, "#fbfcfd", GRID, 4), text(px + 12, py + 25, LABEL[domain], "big")]
    for tick in (0, 25, 50, 75, 100):
        y = bottom - tick / 100 * (bottom - top)
        parts += [line(left, y, right, y, GRID), text(left - 7, y + 4, tick, "small", "end")]
    parts += [
        line(boundary, top, boundary, bottom, INK, 1.4, "4 4"),
        text(boundary, top - 7, "Stage boundary", "small", "middle"),
    ]
    paper_score, paper_total = paper_by_domain[domain]
    paper_y = bottom - (paper_score / paper_total * 100) / 100 * (bottom - top)
    parts += [line(left, paper_y, right, paper_y, INK, 1.4, "7 4"), text(right - 3, paper_y - 4, f"paper {paper_score:g}/{paper_total}", "small", "end")]
    for stage, xa, xb in (("stage1", left, boundary - 10), ("stage2", boundary + 10, right)):
        stage_rows = [r for r in curves if r["domain"] == domain and r["stage"] == stage]
        epochs = [int(r["epoch"]) for r in stage_rows]
        if not epochs:
            continue
        emin, emax = min(epochs), max(epochs)
        for fraction in (0, .5, 1):
            x = xa + fraction * (xb - xa)
            epoch = round(emin + fraction * (emax - emin))
            parts += [line(x, bottom, x, bottom + 4, INK), text(x, bottom + 17, epoch, "small", "middle")]
        parts.append(text((xa + xb) / 2, bottom + 33, "Stage 1 epoch" if stage == "stage1" else "Stage 2 epoch", "small", "middle"))
    for value_head, color in (("off", BLUE), ("on", ORANGE)):
        for stage, xa, xb in (("stage1", left, boundary - 10), ("stage2", boundary + 10, right)):
            data = sorted(
                [r for r in curves if r["domain"] == domain and r["value_head"] == value_head and r["stage"] == stage],
                key=lambda r: int(r["epoch"]),
            )
            if not data:
                continue
            emin, emax = int(data[0]["epoch"]), int(data[-1]["epoch"])
            sx = lambda epoch: xa + (epoch - emin) / max(emax - emin, 1) * (xb - xa)
            sy = lambda score: bottom - score / 100 * (bottom - top)
            upper = [(sx(int(r["epoch"])), sy(float(r["max"]))) for r in data]
            lower = [(sx(int(r["epoch"])), sy(float(r["min"]))) for r in reversed(data)]
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
            parts.append(f'<polygon points="{points}" fill="{color}" opacity="0.09"/>')
            parts.append(poly([(sx(int(r["epoch"])), sy(float(r["mean"]))) for r in data], color))
parts += [
    line(70, 890, 105, 890, BLUE, 3), text(113, 894, "VH-off mean", "small"),
    line(230, 890, 265, 890, ORANGE, 3), text(273, 894, "VH-on mean", "small"),
    line(430, 890, 465, 890, INK, 1.4, "7 4"), text(473, 894, "paper result", "small"),
    text(590, 894, "Stage 1: n=10 per cell. Stage 2: available validation-led refinement trajectories; min–max bands show heterogeneity, not a confidence interval.", "sub"),
]
save("02_two_stage_learning_dynamics", parts)


# 3a/3b. Split the dense MCTS plot by stage and spell out the scientific axes.
forest: list[dict[str, str]] = []
for stage, path in (
    ("Stage 1", ROOT / "experiment_tracking" / "stage1_policy_mcts_all_cutoff_statistics_latest.csv"),
    ("Stage 2", ROOT / "experiment_tracking" / "stage2_policy_mcts_all_cutoff_statistics_latest.csv"),
):
    for source in read(path):
        branch = source.get("stage2_branch", source.get("experiment_id", "selected"))
        for cutoff in ("30m", "2h", "6h"):
            forest.append({
                "stage": stage,
                "branch": branch,
                "domain": source["domain"],
                "value_head": source["value_head"],
                "search": source["search"],
                "cutoff": cutoff,
                "policy_mean": source["policy_mean"],
                "mcts_mean": source[f"mcts_mean_{cutoff}"],
                "change": source[f"change_{cutoff}"],
                "ci95_low": source[f"ci95_low_{cutoff}"],
                "ci95_high": source[f"ci95_high_{cutoff}"],
                "raw_p": source[f"raw_p_{cutoff}"],
                "holm_p": source.get(f"holm_p_{cutoff}", source[f"raw_p_{cutoff}"]),
                "row_level_provenance": source["row_level_provenance"],
            })


def forest_plot(stage: str, filename: str, title: str) -> None:
    rows = [r for r in forest if r["stage"] == stage]
    identities: list[tuple[str, str, str, str]] = []
    for r in rows:
        key = (r["domain"], r["value_head"], r["branch"], r["search"])
        if key not in identities:
            identities.append(key)
    height = 150 + len(identities) * 48
    parts = start(
        1540, height, title,
        "Paired unit is the training seed (n=10 unless labelled otherwise). Points are mean changes in solved test instances; lines are 95% CIs.",
    )
    left, right, top = 455, 1465, 112
    lo_axis, hi_axis = -30, 30
    scale = lambda value: left + (value - lo_axis) / (hi_axis - lo_axis) * (right - left)
    for tick in (-30, -20, -10, 0, 10, 20, 30):
        x = scale(tick)
        parts += [line(x, top, x, height - 55, RED if tick == 0 else GRID, 1.5 if tick == 0 else 1), text(x, 96, f"{tick:+d}", "small", "middle")]
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["domain"], r["value_head"], r["branch"], r["search"])].append(r)
    y = top + 15
    for identity in identities:
        domain, value_head, branch, search = identity
        method = "narrow fixed 5/20" if "5/20" in search or "width5" in search else "normal fixed 20/70"
        branch_text = "" if stage == "Stage 1" else (" / validation-led" if "validation" in branch or "selected" in branch else " / terminal-led")
        parts.append(text(left - 15, y + 4, f"{LABEL[domain]} / VH-{value_head}{branch_text} / {method}", "small", "end"))
        for r in grouped[identity]:
            color = {"30m": PURPLE, "2h": ORANGE, "6h": GREEN}[r["cutoff"]]
            dy = {"30m": -7, "2h": 0, "6h": 7}[r["cutoff"]]
            low, high, change = float(r["ci95_low"]), float(r["ci95_high"]), float(r["change"])
            xlow, xhigh = scale(max(low, lo_axis)), scale(min(high, hi_axis))
            parts += [line(xlow, y + dy, xhigh, y + dy, color, 2), circle(scale(max(lo_axis, min(hi_axis, change))), y + dy, 4.2, color)]
            if low < lo_axis:
                parts.append(text(left + 2, y + dy + 3, "←", "value", fill=color))
            if high > hi_axis:
                parts.append(text(right - 2, y + dy + 3, "→", "value", "end", fill=color))
            if float(r["holm_p"]) < .05:
                parts.append(text(scale(max(lo_axis, min(hi_axis, change))) + 7, y + dy + 3, "*", "value", fill=color))
        y += 48
    parts += [
        line(65, height - 32, 95, height - 32, PURPLE, 3), text(102, height - 28, "30m", "small"),
        line(160, height - 32, 190, height - 32, ORANGE, 3), text(197, height - 28, "2h", "small"),
        line(250, height - 32, 280, height - 32, GREEN, 3), text(287, height - 28, "6h", "small"),
        text(455, height - 28, "Right helps; left hurts. * Holm-adjusted p<.05 within this stage/cutoff family.", "sub"),
        text(1040, height - 28, "x-axis: Δ test instances solved (MCTS − policy)", "sub"),
    ]
    save(filename, parts)


forest_plot("Stage 1", "03a_stage1_mcts_cutoff_forest", "3a — Stage-1 policy versus fixed MCTS at 30m, 2h and 6h")
forest_plot("Stage 2", "03b_stage2_mcts_cutoff_forest", "3b — Stage-2 policy versus fixed MCTS at 30m, 2h and 6h")


# 4. Validation-led PRESERVE-3 only, with stacked zero counts and an explicit
# definition of the experiment and of preservation.
preserve = read(BEFORE / "04_preserve3_validation_seed_robustness.csv")
parts = start(
    1400, 710, "4 — PRESERVE-3 seed robustness",
    "Delivery, TPP and Zenotravel began near ceiling. Each point is one matched seed's change in solved test instances from Stage 1 to Stage 2.",
)
left, right, top = 300, 1140, 135
scale = lambda value: left + (value + 12) / 20 * (right - left)
for tick in (-12, -8, -4, 0, 4, 8):
    x = scale(tick)
    parts += [line(x, top, x, 570, RED if tick == 0 else GRID, 1.6 if tick == 0 else 1), text(x, 594, f"{tick:+d}", "small", "middle")]
for i, (domain, value_head) in enumerate((d, v) for d in ("delivery", "tpp", "zenotravel") for v in ("off", "on")):
    y = top + 30 + i * 69
    rows = [r for r in preserve if r["domain"] == domain and r["value_head"] == value_head]
    values = [float(r["change"]) for r in rows]
    mean = sum(values) / len(values)
    parts.append(text(left - 15, y + 4, f"{LABEL[domain]} / VH-{value_head} (n=10)", "label", "end"))
    value_counts = defaultdict(int)
    for row in rows:
        value = float(row["change"])
        stack = value_counts[value]
        value_counts[value] += 1
        x = scale(value)
        color = RED if value < 0 else (GREEN if value > 0 else "#8793a0")
        parts.append(circle(x, y - min(stack, 5) * 8, 5, color))
        if value <= -5:
            parts.append(text(x + 8, y - min(stack, 5) * 8 - 8, f"seed {row['seed']}: {row['stage1_selected']}→{row['stage2_selected']}", "small"))
    mean_x = scale(mean)
    parts += [line(mean_x, y - 20, mean_x, y + 20, INK, 3), text(mean_x + 6, y + 25, f"mean {mean:+.1f}", "small")]
parts += [
    text(left, 635, "Change in test instances solved (Stage 2 − Stage 1)", "label"),
    circle(760, 632, 5, RED), text(773, 636, "loss", "small"), circle(825, 632, 5, "#8793a0"), text(838, 636, "unchanged", "small"), circle(925, 632, 5, GREEN), text(938, 636, "gain", "small"),
    text(left, 674, "Conclusion: Delivery and Zenotravel remain near ceiling. TPP is usually unchanged, but two runs lose 11 and 5 plans; rare regressions are the robustness risk.", "sub"),
]
save("04_preserve3_validation_seed_robustness", parts)


# 5. Explain checkpoint-selection regret in the graphic itself.
mprime = read(BEFORE / "05_mprime_validation_problem.csv")
parts = start(
    1470, 690, "5 — MPrime checkpoint selection is limited by saturated validation scores",
    "For each lineage, validation chooses a checkpoint without test access. Blue shows the retrospectively best saved checkpoint on test: a diagnostic upper bound, not an allowed selector.",
)
left, right, top = 370, 1000, 145
scale = lambda value: left + (value - 8) / 12 * (right - left)
for tick in range(8, 21, 2):
    x = scale(tick)
    parts += [line(x, top, x, 515, GRID), text(x, 539, tick, "small", "middle")]
for i, row in enumerate(mprime):
    y = top + 28 + i * 58
    selected = float(row["selected_test_mean"])
    best = float(row["observed_best_test_mean"])
    tied = float(row["fraction_checkpoints_tied_at_validation_max"])
    sx, bx = scale(selected), scale(best)
    parts += [
        text(left - 15, y + 4, f"{row['group']} (n={row['n']} lineages)", "label", "end"),
        line(sx, y, bx, y, "#8b98a5", 4), circle(sx, y, 7, ORANGE), circle(bx, y, 7, BLUE),
        text(sx - 4, y - 12, f"{selected:.1f}", "small", "middle"), text(bx + 4, y + 18, f"{best:.1f}", "small", "middle"),
        text(1040, y + 4, f"regret {best-selected:.1f}; {tied:.0%} of saved checkpoints tie at validation maximum", "small"),
    ]
parts += [
    text(left, 576, "Mean MPrime test instances solved (of 20)", "label"),
    circle(375, 618, 7, ORANGE), text(390, 622, "test score of validation-selected checkpoint", "small"),
    circle(670, 618, 7, BLUE), text(685, 622, "best observed test score (hindsight only)", "small"),
    text(1035, 622, "Stage 2: every saved checkpoint tied at 30/30 validation.", "sub"),
    text(370, 659, "Interpretation: the current Stage-2 validation set cannot rank checkpoints; Phase B tests two harder frozen replacements.", "sub"),
]
save("05_mprime_validation_problem", parts)


(OUT / "README.md").write_text(
    """# Advisor figures — revised after independent review

The reviewer saw only the five original PNGs and a short project description.
This revision fixes clipped annotations, labels every scientific axis and unit,
states aggregation and seed counts, separates the dense MCTS result by stage,
shows the complete Stage-1 and Stage-2 curves, limits PRESERVE-3 to the
validation-led branch, and explains MPrime selection regret inside the figure.

Every figure is backed by the immutable colocated CSV in `before_review/`;
those CSVs retain the row-level source/provenance paths.
""",
    encoding="utf-8",
)
print(OUT)
