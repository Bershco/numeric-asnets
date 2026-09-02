#!/usr/bin/env python3
"""Extract compact, reproducible TPP VH-off Stage-2 training diagnostics."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import fmean


JOB_SEED_RE = re.compile(r"^(\d+)_.*?_s(\d+)_")
SCALAR_RE = re.compile(r"/train(?P<metric>-loss|/policy_loss|/policy_anchor_kl_loss)\s+:\s+(?P<value>[0-9.eE+-]+)")
VAL_RE = re.compile(r"Current network validation success rate:\s*([0-9.]+)")
BEST_RE = re.compile(r"\[VALIDATION\] New best! succ=([0-9.]+).*?iter_num=(\d+).*?snapshot_name=(\S+)")


def summary(values: list[float]) -> tuple[str, str, str, str]:
    if not values:
        return "", "", "", ""
    return tuple(f"{value:.6f}" for value in (values[0], fmean(values), values[-1], max(values)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", type=Path)
    parser.add_argument("accounting", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.accounting.open(newline="", encoding="utf-8") as stream:
        accounting = {row["job_id"]: row for row in csv.DictReader(stream)}

    rows = []
    for path in sorted(args.logs_dir.glob("206848*_Re-Tr_*.txt")):
        match = JOB_SEED_RE.match(path.name)
        if not match:
            continue
        job_id, seed = match.groups()
        text = path.read_text(encoding="utf-8", errors="replace")
        metrics = {"-loss": [], "/policy_loss": [], "/policy_anchor_kl_loss": []}
        for scalar in SCALAR_RE.finditer(text):
            metrics[scalar.group("metric")].append(float(scalar.group("value")))
        validations = [float(value) for value in VAL_RE.findall(text)]
        best = list(BEST_RE.finditer(text))
        selected = best[-1] if best else None
        acc = accounting[job_id]
        train = summary(metrics["-loss"])
        policy = summary(metrics["/policy_loss"])
        anchor = summary(metrics["/policy_anchor_kl_loss"])
        rows.append({
            "job_id": job_id,
            "seed": seed,
            "state": acc["state"],
            "elapsed_seconds": acc["elapsed_seconds"],
            "elapsed_hours": f"{int(acc['elapsed_seconds']) / 3600:.3f}",
            "epochs_logged": len(metrics["-loss"]),
            "selected_epoch": selected.group(2) if selected else "",
            "selected_validation_coverage": selected.group(1) if selected else "",
            "validation_first": f"{validations[0]:.6f}" if validations else "",
            "validation_mean": f"{fmean(validations):.6f}" if validations else "",
            "validation_final": f"{validations[-1]:.6f}" if validations else "",
            "validation_peak": f"{max(validations):.6f}" if validations else "",
            "train_loss_first": train[0], "train_loss_mean": train[1],
            "train_loss_final": train[2], "train_loss_max": train[3],
            "policy_loss_first": policy[0], "policy_loss_mean": policy[1],
            "policy_loss_final": policy[2], "policy_loss_max": policy[3],
            "anchor_kl_first": anchor[0], "anchor_kl_mean": anchor[1],
            "anchor_kl_final": anchor[2], "anchor_kl_max": anchor[3],
            "worker_timeout_warnings": text.count("[TRAINER WARNING]"),
            "source_training_log": acc["source_training_log"],
        })

    if len(rows) != 8:
        raise RuntimeError(f"Expected eight held-out logs, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
