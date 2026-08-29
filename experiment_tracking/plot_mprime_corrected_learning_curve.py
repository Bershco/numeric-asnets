#!/usr/bin/env python3
"""Plot corrected MPrime validation/test learning curves from frozen evidence."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "mprime_validation_ipc_scale_v1" / "validation_test_checkpoint_audit.csv"
OUT = ROOT / "mprime_validation_ipc_scale_v1"


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # Use the regular every-five curve only. Selected/final off-grid endpoints
    # remain in the source audit but are not silently mixed into aggregate epochs.
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        epoch = int(row["epoch"])
        if epoch % 5 == 0:
            grouped[(row["value_head"], epoch)].append(row)

    aggregate: list[dict[str, object]] = []
    for (vh, epoch), group in sorted(grouped.items()):
        test = [100.0 * float(row["test_success"]) / float(row["test_total"]) for row in group]
        validation = [100.0 * float(row["validation_success"]) for row in group]
        aggregate.append({
            "value_head": vh,
            "epoch": epoch,
            "n_runs": len(group),
            "total_runs": 10,
            "test_min_pct": min(test),
            "test_max_pct": max(test),
            "test_mean_pct": statistics.fmean(test),
            "test_median_pct": statistics.median(test),
            "test_q25_pct": percentile(test, 0.25),
            "test_q75_pct": percentile(test, 0.75),
            "validation_min_pct": min(validation),
            "validation_max_pct": max(validation),
            "validation_mean_pct": statistics.fmean(validation),
            "validation_median_pct": statistics.median(validation),
        })

    aggregate_path = OUT / "corrected_learning_curve_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    width, height = 1500, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    svg: list[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']

    def panel(vh: str, left: int) -> None:
        top, plot_w, plot_h = 90, 620, 470
        data = [row for row in aggregate if row["value_head"] == vh]
        max_epoch = max(int(row["epoch"]) for row in data)
        x = lambda epoch: left + 65 + epoch / max_epoch * plot_w
        y = lambda value: top + plot_h - value / 100.0 * plot_h
        draw.rectangle((left + 65, top, left + 65 + plot_w, top + plot_h), outline="#444444")
        svg.append(f'<rect x="{left+65}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#444"/>')
        for tick in range(0, 101, 20):
            yy = y(tick)
            draw.line((left + 65, yy, left + 65 + plot_w, yy), fill="#dddddd")
            draw.text((left + 25, yy - 6), str(tick), fill="#333333", font=font)
            svg.append(f'<line x1="{left+65}" y1="{yy:.1f}" x2="{left+65+plot_w}" y2="{yy:.1f}" stroke="#ddd"/>')
            svg.append(f'<text x="{left+50}" y="{yy+4:.1f}" text-anchor="end" font-size="12">{tick}</text>')
        for epoch_tick in range(0, max_epoch + 1, 20):
            xx = x(epoch_tick)
            draw.line((xx, top + plot_h, xx, top + plot_h + 5), fill="#444444")
            draw.text((xx - 7, top + plot_h + 9), str(epoch_tick), fill="#333333", font=font)
            svg.append(f'<line x1="{xx:.1f}" y1="{top+plot_h}" x2="{xx:.1f}" y2="{top+plot_h+5}" stroke="#444"/><text x="{xx:.1f}" y="{top+plot_h+18}" text-anchor="middle" font-size="11">{epoch_tick}</text>')
        upper = [(x(int(row["epoch"])), y(float(row["test_max_pct"]))) for row in data]
        lower = [(x(int(row["epoch"])), y(float(row["test_min_pct"]))) for row in reversed(data)]
        envelope = upper + lower
        draw.polygon(envelope, fill=(76, 120, 168, 38))
        svg.append('<polygon points="' + ' '.join(f'{xx:.1f},{yy:.1f}' for xx, yy in envelope) + '" fill="#4C78A8" opacity="0.15"/>')
        series = [
            ("test_median_pct", "#F58518", "Test median"),
            ("test_mean_pct", "#54A24B", "Test mean"),
            ("validation_mean_pct", "#B279A2", "Validation mean"),
        ]
        for key, color, _ in series:
            points = [(x(int(row["epoch"])), y(float(row[key]))) for row in data]
            draw.line(points, fill=color, width=3)
            svg.append(f'<polyline points="' + ' '.join(f'{xx:.1f},{yy:.1f}' for xx, yy in points) + f'" fill="none" stroke="{color}" stroke-width="3"/>')
        draw.text((left + 250, 55), f"MPrime - value head {vh}", fill="#111111", font=font)
        draw.text((left + 300, top + plot_h + 25), "Stage-1 epoch", fill="#222222", font=font)
        svg.append(f'<text x="{left+375}" y="65" text-anchor="middle" font-size="18">MPrime - value head {vh}</text>')
        svg.append(f'<text x="{left+375}" y="{top+plot_h+35}" text-anchor="middle" font-size="14">Stage-1 epoch</text>')
        for idx, (_, color, label) in enumerate(series):
            yy = top + 25 + idx * 18
            draw.line((left + 480, yy, left + 515, yy), fill=color, width=3)
            draw.text((left + 522, yy - 6), label, fill="#222222", font=font)
            svg.append(f'<line x1="{left+480}" y1="{yy}" x2="{left+515}" y2="{yy}" stroke="{color}" stroke-width="3"/><text x="{left+522}" y="{yy+4}" font-size="12">{label}</text>')

    panel("off", 10)
    panel("on", 760)
    draw.text((540, 20), "Corrected MPrime validation and held-out test learning curves", fill="#111111", font=font)
    draw.text((4, 300), "Coverage (%)", fill="#222222", font=font)
    svg.append('<text x="750" y="28" text-anchor="middle" font-size="20">Corrected MPrime validation and held-out test learning curves</text>')
    svg.append('</svg>')
    image.save(OUT / "corrected_learning_curves.png")
    (OUT / "corrected_learning_curves.svg").write_text("\n".join(svg), encoding="utf-8")
    print(f"rows={len(aggregate)} aggregate={aggregate_path}")


if __name__ == "__main__":
    main()
