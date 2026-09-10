#!/usr/bin/env python3
"""Run one approved external-generator policy-evaluation manifest row."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--completion-root", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    row = rows[args.index]
    data_root = Path(os.environ["YARIN_EXTERNAL_DATA_ROOT"])
    domain = row["domain"]
    module = "fo_counters_yarin_external" if domain == "fo_counters" else "rover_yarin_external"
    architecture = "fo_counters" if domain == "fo_counters" else "rover"
    env = os.environ.copy()
    env["YARIN_FO_COUNTERS_PDDL_DIR" if domain == "fo_counters" else "YARIN_ROVER_PDDL_DIR"] = str(
        data_root / ("fo-counters" if domain == "fo_counters" else "rover")
    )
    args.completion_root.mkdir(parents=True, exist_ok=True)
    completion = args.completion_root / f"{row['domain']}_{row['value_head']}_{row['seed']}.jsonl"
    command = [
        "./run_experiment", f"experiments_numeric.architecture_2.{architecture}",
        f"experiments_numeric.domain.{module}", "--resume-from", row["source_checkpoint"],
        "--eval-scheduling", "rolling", "--eval-completion-file", str(completion),
        "--eval-instance-timeout", "21600", "--num-workers", row["workers"],
        "--jpddl-max-heap", "4g", "--worker-logs", "--random-seed", row["seed"],
    ]
    if row["value_head"] == "off":
        command.append("--disable-value-head")
    print("[EXTERNAL SCREEN]", {key: row[key] for key in ("domain", "value_head", "seed", "source_training_job_id")}, flush=True)
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()

