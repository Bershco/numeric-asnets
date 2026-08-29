#!/usr/bin/env python3
"""Join corrected MPrime validation metrics with all checkpoint test scores."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path


VALID_RE = re.compile(
    r"\[VALIDATION\] Current network validation success rate: ([0-9.]+) "
    r"with an average plan length of ([0-9.]+)"
)
FINAL_RE = re.compile(r"\[EVAL FINAL\] success=([0-9.]+)/([0-9.]+)=")


def read(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def accounting(ids: list[str]) -> dict[str, tuple[str, Path]]:
    answer: dict[str, tuple[str, Path]] = {}
    for start in range(0, len(ids), 250):
        text = subprocess.check_output([
            "sacct", "-X", "-n", "-P", "-j", ",".join(ids[start:start + 250]),
            "-o", "JobIDRaw,State,StdOut%1000",
        ], text=True)
        for line in text.splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3 and parts[0].isdigit():
                answer[parts[0]] = (parts[1].split()[0], Path(parts[2].replace("%j", parts[0])))
    return answer


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for pos in order[i:j]:
            result[pos] = rank
        i = j
    return result


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return math.nan
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    num = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    ld = sum((a - lm) ** 2 for a in left)
    rd = sum((b - rm) ** 2 for b in right)
    return num / math.sqrt(ld * rd) if ld and rd else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("output_rows", type=Path)
    parser.add_argument("output_summary", type=Path)
    args = parser.parse_args()
    manifest = {row["manifest_id"]: row for row in read(args.manifest, ",")}
    ledger = read(args.ledger, "\t")
    training_ids = sorted({row["training_job"] for row in manifest.values()}, key=int)
    eval_ids = [row["slurm_job_id"] for row in ledger]
    info = accounting(training_ids + eval_ids)
    validation: dict[str, dict[int, tuple[float, float]]] = {}
    training_logs: dict[str, Path] = {}
    for job in training_ids:
        state, log = info[job]
        if not log.is_file():
            raise RuntimeError(f"missing training log {job}: {log}")
        metrics = [(float(rate), float(length)) for rate, length in VALID_RE.findall(
            log.read_text(encoding="utf-8", errors="replace"))]
        validation[job] = {index: item for index, item in enumerate(metrics, 1)}
        training_logs[job] = log
    rows = []
    for submitted in ledger:
        template = manifest[submitted["manifest_id"]]
        job = submitted["slurm_job_id"]
        state, log = info[job]
        if state != "COMPLETED" or not log.is_file():
            raise RuntimeError(f"evaluation not complete {job}: {state} {log}")
        matches = FINAL_RE.findall(log.read_text(encoding="utf-8", errors="replace"))
        if not matches:
            raise RuntimeError(f"missing final score {job}: {log}")
        test_success, test_total = map(float, matches[-1])
        epoch = int(template["epoch"])
        rate, length = validation[template["training_job"]][epoch]
        rows.append({
            "manifest_id": template["manifest_id"], "value_head": template["value_head"],
            "seed": template["seed"], "training_job": template["training_job"],
            "epoch": str(epoch), "roles": template["roles"],
            "validation_success": f"{rate:.9f}", "validation_plan_length": f"{length:.9f}",
            "test_success": f"{test_success:g}", "test_total": f"{test_total:g}",
            "checkpoint": template["checkpoint"], "training_log": str(training_logs[template["training_job"]]),
            "evaluation_job": job, "evaluation_log": str(log),
        })
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["value_head"], row["seed"])].append(row)
    summary = []
    for (vh, seed), points in sorted(groups.items()):
        val = [float(row["validation_success"]) for row in points]
        test = [float(row["test_success"]) for row in points]
        selected = [row for row in points if "selected" in row["roles"]]
        test_best = max(test)
        summary.append({
            "scope": "lineage", "value_head": vh, "seed": seed, "n": str(len(points)),
            "pearson_validation_test": f"{pearson(val, test):.9f}",
            "spearman_validation_test": f"{pearson(ranks(val), ranks(test)):.9f}",
            "selected_epoch": selected[0]["epoch"] if len(selected) == 1 else "",
            "selected_test_success": selected[0]["test_success"] if len(selected) == 1 else "",
            "best_observed_test_success": f"{test_best:g}",
            "selected_test_regret": f"{test_best - float(selected[0]['test_success']):g}" if len(selected) == 1 else "",
        })
    for vh in ("off", "on"):
        points = [row for row in rows if row["value_head"] == vh]
        val = [float(row["validation_success"]) for row in points]
        test = [float(row["test_success"]) for row in points]
        summary.append({
            "scope": "pooled_vh", "value_head": vh, "seed": "ALL", "n": str(len(points)),
            "pearson_validation_test": f"{pearson(val, test):.9f}",
            "spearman_validation_test": f"{pearson(ranks(val), ranks(test)):.9f}",
            "selected_epoch": "", "selected_test_success": "",
            "best_observed_test_success": "", "selected_test_regret": "",
        })
    for path, data in ((args.output_rows, rows), (args.output_summary, summary)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(data)
    print(f"rows={len(rows)} lineages={len(groups)} summaries={len(summary)}")


if __name__ == "__main__":
    main()
