#!/usr/bin/env python3
"""Build a locally cached, provenance-bearing RQ1/RQ3 curve package."""

from __future__ import annotations

import csv
import html
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parents[1]
SOURCE_ROOT = MAIN / "codex_local" / "outputs"
OUT = ROOT / "experiment_tracking" / "learning_curves" / "latest"
AGG_SOURCE = SOURCE_ROOT / "20260819" / "learning_curves" / "rq1_rq3_five_domain_learning_curve_aggregates.csv"
RAW_SOURCE = SOURCE_ROOT / "20260811" / "completed_policy_evaluations.csv"
DOMAIN_LABELS = {
    "block_grouping": "Block Grouping",
    "drone": "Drone",
    "fo_counters": "FO Counters",
    "rover": "Rover",
    "counters": "Counters",
}


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def plot_domain(domain: str, rows: list[dict[str, str]]) -> None:
    width, height = 1120, 430
    panel_width, left, top, chart_height = 530, 62, 62, 285
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#20242a}.title{font-size:22px;font-weight:600}.panel{font-size:15px;font-weight:600}.tick,.note{font-size:10px;fill:#59636e}.axis{font-size:11px}.frame{fill:white;stroke:#9aa3ad}.grid{stroke:#d9dee4}.band{fill:#4b7fb8;opacity:.15}.line{fill:none;stroke-width:2.2}.median{stroke:#2455a4;stroke-width:3}.mean{stroke:#f28e2b;stroke-dasharray:5 3}.selected{stroke:#d1495b}.s1{stroke:#333;stroke-dasharray:7 5}.paper{stroke:#7b2cbf;stroke-dasharray:2 4}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" class="title">{html.escape(DOMAIN_LABELS[domain])} Stage-2 learning curves</text>',
    ]
    for column, rq in enumerate(("RQ1", "RQ3")):
        data = sorted((r for r in rows if r["domain"] == domain and r["research_question"] == rq), key=lambda r: int(r["epoch"]))
        x0 = column * 560 + left
        chart_width = panel_width - 85
        parts.append(f'<text x="{x0 + chart_width/2}" y="49" text-anchor="middle" class="panel">{rq}</text>')
        parts.append(f'<rect x="{x0}" y="{top}" width="{chart_width}" height="{chart_height}" class="frame"/>')
        for tick in (0, 25, 50, 75, 100):
            y = top + chart_height * (1 - tick / 100)
            parts.append(f'<line x1="{x0}" x2="{x0+chart_width}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick}</text>')
        for tick in (0, 20, 40, 60, 80, 99):
            x = x0 + chart_width * tick / 99
            parts.append(f'<text x="{x:.1f}" y="{top+chart_height+18}" text-anchor="middle" class="tick">{tick}</text>')
        def point(row: dict[str, str], field: str) -> tuple[float, float]:
            return x0 + chart_width * int(row["epoch"]) / 99, top + chart_height * (1 - float(row[field]) / 100)
        lower = [point(row, "minimum_percent") for row in data]
        upper = [point(row, "maximum_percent") for row in reversed(data)]
        parts.append('<polygon class="band" points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in lower + upper) + '"/>')
        for field, css in (("median_percent", "median"), ("mean_percent", "mean"), ("best_configuration_mean_percent", "selected")):
            parts.append('<polyline class="line ' + css + '" points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in (point(row, field) for row in data)) + '"/>')
        for field, css in (("stage1_baseline_percent", "s1"), ("paper_baseline_percent", "paper")):
            y = point(data[0], field)[1]
            parts.append(f'<line x1="{x0}" x2="{x0+chart_width}" y1="{y:.1f}" y2="{y:.1f}" class="{css}"/>')
        parts.append(f'<text x="{x0+chart_width/2}" y="{top+chart_height+38}" text-anchor="middle" class="axis">Stage-2 epoch</text>')
    parts.append('<text x="560" y="414" text-anchor="middle" class="note">Blue median · orange all-run mean · red selected-configuration mean · shadow min-max · dashed Stage-1 · dotted paper</text>')
    parts.append('</svg>')
    (OUT / f"by_domain_{domain}.svg").write_text("".join(parts), encoding="utf-8")


def write_index(rows: list[dict[str, str]]) -> None:
    raw = load(OUT / "rq1_rq3_policy_curve_source.csv")
    sources = defaultdict(lambda: {"jobs": set(), "training_logs": set(), "evaluation_logs": set()})
    for row in raw:
        key = (row["research_question"], row["domain"])
        sources[key]["jobs"].add(row["stage2_training_job_id"])
        sources[key]["training_logs"].add(row["source_training_log"])
        sources[key]["evaluation_logs"].add(row["evaluation_log"])
    records = []
    for rq, domain in sorted(sources):
        item = sources[(rq, domain)]
        records.append({
            "research_question": rq,
            "domain": domain,
            "source_rows": sum(1 for row in raw if row["research_question"] == rq and row["domain"] == domain),
            "stage2_training_job_ids": ";".join(sorted(item["jobs"])),
            "source_training_logs": ";".join(sorted(item["training_logs"])),
            "source_evaluation_logs": ";".join(sorted(item["evaluation_logs"])),
            "aggregate_csv": "experiment_tracking/learning_curves/latest/rq1_rq3_five_domain_learning_curve_aggregates.csv",
            "combined_figure": "experiment_tracking/learning_curves/latest/by_rq_rq1_rq3.png",
            "domain_figure": f"experiment_tracking/learning_curves/latest/by_domain_{domain}.svg",
        })
    with (OUT / "learning_curve_provenance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RAW_SOURCE, OUT / "rq1_rq3_policy_curve_source.csv")
    shutil.copy2(AGG_SOURCE, OUT / "rq1_rq3_five_domain_learning_curve_aggregates.csv")
    source_figures = SOURCE_ROOT / "20260819" / "learning_curves"
    copies = {
        "rq1_rq3_five_domain_learning_curves.png": "by_rq_rq1_rq3.png",
        "rq1_rq3_five_domain_learning_curves.svg": "by_rq_rq1_rq3.svg",
        "rq1_five_domain_learning_curves.png": "by_rq_rq1.png",
        "rq1_five_domain_learning_curves.svg": "by_rq_rq1.svg",
        "rq3_five_domain_learning_curves.png": "by_rq_rq3.png",
        "rq3_five_domain_learning_curves.svg": "by_rq_rq3.svg",
    }
    for source, destination in copies.items():
        shutil.copy2(source_figures / source, OUT / destination)
    rows = load(OUT / "rq1_rq3_five_domain_learning_curve_aggregates.csv")
    for domain in DOMAIN_LABELS:
        plot_domain(domain, rows)
    write_index(rows)
    print(f"learning-curve package: {OUT}")


if __name__ == "__main__":
    main()
