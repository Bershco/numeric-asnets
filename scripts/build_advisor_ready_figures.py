#!/usr/bin/env python3
"""Build static, auditable SVG figures from authoritative CSV ledgers."""

from __future__ import annotations

import csv
import html
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT = TRACK / "advisor_figures"
PW = TRACK / "mcts_progressive_widening_sensitivity"
DOMAIN_ORDER = ["block_grouping", "drone", "fo_counters", "rover", "counters"]
LABELS = {"block_grouping": "Block Grouping", "drone": "Drone",
          "fo_counters": "FO Counters", "rover": "Rover", "counters": "Counters"}
ARMS = ["policy", "fixed_top20", "pw_kmin3"]
ARM_LABELS = {"policy": "Policy", "fixed_top20": "Fixed top-20", "pw_kmin3": "PW Kmin=3"}
COLORS = {"policy": "#4c78a8", "fixed_top20": "#f58518", "pw_kmin3": "#54a24b"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def esc(value) -> str:
    return html.escape(str(value))


def svg_text(x, y, value, size=12, anchor="start", weight="normal", fill="#222", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{transform}>{esc(value)}</text>')


def save_svg(name: str, width: int, height: int, body: list[str]) -> None:
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>'] + body + ['</svg>']
    (OUT / name).write_text("\n".join(content) + "\n", encoding="utf-8")


def mainstream_forest() -> None:
    rows = [r for r in read_csv(TRACK / "experiment_statistics.csv")
            if r["record_type"] == "paired" and r["experiment_id"] in {"MAIN-VAL", "MAIN-TERM"}
            and "S2 validation-selected" in r["comparison"]]
    rows.sort(key=lambda r: (r["experiment_id"], DOMAIN_ORDER.index(r["domain"]), r["value_head"]))
    fields = ["experiment_id", "domain", "value_head", "n", "mean_before", "mean_after",
              "mean_difference", "ci95_low", "ci95_high", "raw_p", "holm_p", "source_results_file"]
    write_csv(OUT / "mainstream_stage1_stage2_forest_data.csv", rows, fields)

    width, height = 1500, 720
    panel_w, plot_left, plot_right = 720, 205, 490
    x_min, x_max = -25.0, 32.0
    body = [svg_text(width / 2, 31, "Stage-1 → Stage-2 policy change across ten matched seeds", 20, "middle", "bold")]
    for panel, (exp, title) in enumerate([
            ("MAIN-VAL", "Validation-led: S1 selected → S2 selected"),
            ("MAIN-TERM", "Terminal-led: S1 final → S2 selected")]):
        ox = 20 + panel * 745
        subset = [r for r in rows if r["experiment_id"] == exp]
        body.append(svg_text(ox + panel_w / 2, 67, title, 15, "middle", "bold"))
        for tick in [-20, -10, 0, 10, 20, 30]:
            tx = ox + plot_left + (tick - x_min) / (x_max - x_min) * (plot_right - plot_left)
            body.append(f'<line x1="{tx:.1f}" y1="88" x2="{tx:.1f}" y2="625" stroke="#e3e3e3"/>')
            body.append(svg_text(tx, 647, tick, 10, "middle"))
        zero_x = ox + plot_left + (0 - x_min) / (x_max - x_min) * (plot_right - plot_left)
        body.append(f'<line x1="{zero_x:.1f}" y1="88" x2="{zero_x:.1f}" y2="625" stroke="#111" stroke-width="1.2"/>')
        for idx, row in enumerate(subset):
            y = 112 + idx * 51
            mean, lo, hi = map(float, [row["mean_difference"], row["ci95_low"], row["ci95_high"]])
            scale = lambda value: ox + plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)
            color = "#2ca02c" if lo > 0 else "#d62728" if hi < 0 else "#4c78a8"
            body.append(svg_text(ox + plot_left - 10, y + 4,
                                 f"{LABELS[row['domain']]} · VH {row['value_head']}", 11, "end"))
            body.append(f'<line x1="{scale(lo):.1f}" y1="{y}" x2="{scale(hi):.1f}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            body.append(f'<circle cx="{scale(mean):.1f}" cy="{y}" r="5" fill="{color}"/>')
            body.append(svg_text(ox + plot_right + 14, y + 4,
                                 f"{float(row['mean_before']):.1f}→{float(row['mean_after']):.1f}; "
                                 f"p={float(row['raw_p']):.3f}, pH={float(row['holm_p']):.3f}", 10))
        body.append(svg_text(ox + (plot_left + plot_right) / 2, 676,
                             "Mean paired coverage change (plans)", 12, "middle"))
    body.append(svg_text(width / 2, 707,
                         "95% paired t-intervals; p = exact sign-flip; pH = Holm-adjusted.", 11, "middle"))
    save_svg("mainstream_stage1_stage2_forest.svg", width, height, body)


def elapsed_seconds(token: str) -> float:
    h, m, s = token.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def pw_runtime_and_cutoff() -> None:
    jobs = read_csv(PW / "kmin3_runtime_jobs.csv")
    instances = read_csv(PW / "kmin3_success_runtime_instances.csv")
    threshold = 1800
    scores, totals, job_times, over = defaultdict(int), defaultdict(int), defaultdict(list), defaultdict(int)
    for row in jobs:
        key = row["arm"], row["value_head"]
        scores[key] += int(row["score"]); totals[key] += int(row["total"])
        job_times[key].append(elapsed_seconds(row["elapsed"]))
    for row in instances:
        if float(row["runtime_seconds"]) > threshold:
            over[(row["arm"], row["value_head"])] += 1

    above_rows = [dict(row, runtime_minutes=f"{float(row['runtime_seconds']) / 60:.3f}")
                  for row in instances if float(row["runtime_seconds"]) > threshold]
    write_csv(OUT / "pw_kmin3_over_30m_instances.csv", above_rows,
              list(instances[0]) + ["runtime_minutes"])

    output = []
    for vh in ["off", "on", "overall"]:
        for arm in ARMS:
            keys = [(arm, vh)] if vh != "overall" else [(arm, "off"), (arm, "on")]
            success, total = sum(scores[k] for k in keys), sum(totals[k] for k in keys)
            above = sum(over[k] for k in keys)
            within = success if arm == "policy" else success - above
            times = [v for k in keys for v in job_times[k]]
            output.append({"arm": arm, "value_head": vh, "seeds": len(times), "total_instances": total,
                           "original_successes": success, "successful_instances_over_30m": above,
                           "successes_under_30m_counterfactual": within,
                           "original_mean_coverage": f"{20 * success / total:.3f}",
                           "mean_coverage_under_30m": f"{20 * within / total:.3f}",
                           "mean_whole_job_runtime_seconds": f"{statistics.mean(times):.3f}"})
    write_csv(OUT / "pw_kmin3_30min_sensitivity.csv", output, list(output[0]))

    width, height = 1260, 500
    body = [svg_text(width / 2, 31, "Drone coverage under a 30-minute per-instance cutoff", 20, "middle", "bold")]
    for panel, (vh, title) in enumerate([("off", "VH off"), ("on", "VH on"), ("overall", "Combined")]):
        ox, left, right, top, bottom = panel * 420, 72, 395, 70, 410
        subset = [r for r in output if r["value_head"] == vh]
        body.append(svg_text(ox + 210, 59, title, 15, "middle", "bold"))
        for tick in range(0, 21, 5):
            y = top + (20 - tick) / 20 * (bottom - top)
            body.append(f'<line x1="{ox+left}" y1="{y:.1f}" x2="{ox+right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            if panel == 0: body.append(svg_text(ox + left - 8, y + 4, tick, 10, "end"))
        for idx, row in enumerate(subset):
            center, bw = ox + 122 + idx * 101, 30
            original, capped = float(row["original_mean_coverage"]), float(row["mean_coverage_under_30m"])
            y1 = top + (20 - original) / 20 * (bottom - top)
            y2 = top + (20 - capped) / 20 * (bottom - top)
            body.append(f'<rect x="{center-bw-2}" y="{y1:.1f}" width="{bw}" height="{bottom-y1:.1f}" fill="{COLORS[row["arm"]]}" opacity="0.38"/>')
            body.append(f'<rect x="{center+2}" y="{y2:.1f}" width="{bw}" height="{bottom-y2:.1f}" fill="{COLORS[row["arm"]]}"/>')
            body.append(svg_text(center + 17, y2 - 6, f"{capped:.2f}", 9, "middle"))
            body.append(svg_text(center, 435, ARM_LABELS[row["arm"]], 10, "middle", rotate=-18))
    body.append(svg_text(18, 245, "Mean coverage /20", 12, "middle", rotate=-90))
    body.append(svg_text(width / 2, 487,
                         "Pale = recorded 6h cap; solid = remove successes taking >30m. Policy jobs finished in minutes.",
                         11, "middle"))
    save_svg("pw_kmin3_30min_coverage.svg", width, height, body)

    width, height = 820, 500
    left, right, top, bottom = 82, 780, 60, 425
    body = [svg_text(width / 2, 30, "Drone successful-instance runtime distribution", 19, "middle", "bold")]
    min_x, max_x = math.log10(0.5), math.log10(120)
    xmap = lambda minutes: left + (math.log10(minutes) - min_x) / (max_x - min_x) * (right - left)
    ymap = lambda p: bottom - p * (bottom - top)
    for minute in [0.5, 1, 2, 5, 10, 30, 60, 120]:
        x = xmap(minute)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#e5e5e5"/>')
        body.append(svg_text(x, 448, minute, 10, "middle"))
    for p in [0, .25, .5, .75, 1]:
        y = ymap(p)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        body.append(svg_text(left - 9, y + 4, f"{p:.2f}", 10, "end"))
    for arm in ["fixed_top20", "pw_kmin3"]:
        values = sorted(float(r["runtime_seconds"]) / 60 for r in instances if r["arm"] == arm)
        points = [(xmap(max(0.5, value)), ymap((i + 1) / len(values))) for i, value in enumerate(values)]
        path = " ".join(("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}" for i, (x, y) in enumerate(points))
        body.append(f'<path d="{path}" fill="none" stroke="{COLORS[arm]}" stroke-width="3"/>')
    cutoff_x = xmap(30)
    body.append(f'<line x1="{cutoff_x:.1f}" y1="{top}" x2="{cutoff_x:.1f}" y2="{bottom}" stroke="#d62728" stroke-width="2" stroke-dasharray="7 5"/>')
    body.append(svg_text(590, 79, "Fixed top-20 (84 successes; 2 >30m)", 11, fill=COLORS["fixed_top20"]))
    body.append(svg_text(590, 98, "PW Kmin=3 (76 successes; 0 >30m)", 11, fill=COLORS["pw_kmin3"]))
    body.append(svg_text((left + right) / 2, 477, "Successful-instance runtime (minutes, log scale)", 12, "middle"))
    body.append(svg_text(20, (top + bottom) / 2, "Empirical cumulative proportion", 12, "middle", rotate=-90))
    save_svg("pw_kmin3_success_runtime_ecdf.svg", width, height, body)


def drone_endpoint_plot() -> None:
    rows = [r for r in read_csv(TRACK / "drone_endpoint_mcts_paired_results.csv") if r["experiment_id"] == "MAIN-TERM"]
    width, height = 900, 490
    body = [svg_text(width / 2, 30, "Drone MAIN-TERM Stage-2 selected: paired policy vs MCTS", 19, "middle", "bold")]
    for panel, vh in enumerate(["off", "on"]):
        subset = sorted((r for r in rows if r["value_head"] == vh), key=lambda r: int(r["seed"]))
        ox, x1, x2, top, bottom = panel * 440, 165, 340, 65, 420
        body.append(svg_text(ox + 250, 57, f"VH {vh} · n={len(subset)}", 14, "middle", "bold"))
        for tick in range(0, 21, 5):
            y = top + (20 - tick) / 20 * (bottom - top)
            body.append(f'<line x1="{ox+95}" y1="{y:.1f}" x2="{ox+405}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            if panel == 0: body.append(svg_text(ox + 86, y + 4, tick, 10, "end"))
        for row in subset:
            p, m = float(row["policy_score"]), float(row["mcts_score"])
            yp, ym = top + (20 - p) / 20 * (bottom - top), top + (20 - m) / 20 * (bottom - top)
            color = "#2ca02c" if m > p else "#d62728" if m < p else "#7f7f7f"
            body.append(f'<line x1="{ox+x1}" y1="{yp:.1f}" x2="{ox+x2}" y2="{ym:.1f}" stroke="{color}" opacity="0.65"/>')
            body.append(f'<circle cx="{ox+x1}" cy="{yp:.1f}" r="4" fill="{color}"/><circle cx="{ox+x2}" cy="{ym:.1f}" r="4" fill="{color}"/>')
        body.append(svg_text(ox + x1, 446, "Policy", 12, "middle")); body.append(svg_text(ox + x2, 446, "MCTS", 12, "middle"))
    body.append(svg_text(18, 245, "Solved instances /20", 12, "middle", rotate=-90))
    save_svg("drone_mainterm_policy_mcts.svg", width, height, body)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mainstream_forest(); pw_runtime_and_cutoff(); drone_endpoint_plot()
    print(f"Wrote advisor figures to {OUT}")


if __name__ == "__main__":
    main()
