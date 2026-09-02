#!/usr/bin/env python3
"""Match every Stage-2 selected policy endpoint to historical cluster MCTS logs.

This is a read-only cluster audit.  It inventories candidate log paths and
writes a local provenance ledger; it does not submit or modify cluster work.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment_tracking/experiment_results.csv"
OUTPUT = ROOT / "experiment_tracking/stage2_mcts_historical_log_audit_20260902.csv"
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
CONFIG = r"C:\Users\roeeh\.ssh\config"


def endpoints() -> list[dict[str, str]]:
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [
        row for row in rows
        if row["experiment_id"] in {"MAIN-VAL", "MAIN-TERM"}
        and row["task_type"] == "policy_eval"
        and row["stage"] == "stage2"
        and row["endpoint"] == "validation_selected"
    ]
    if len(selected) != 200:
        raise RuntimeError(f"Expected 200 selected Stage-2 endpoints, got {len(selected)}")
    return selected


def inventory() -> list[str]:
    command = (
        "find /home/hersco/training_new_domains "
        "/home/hersco/bershco-nu-asnets/numeric-asnets "
        "/home/hersco/thesis-reproducibility-bundle "
        "-type f \\( -name '*.txt' -o -name '*.out' \\) "
        "-path '*mcts*' -print 2>/dev/null"
    )
    result = subprocess.run(
        [SSH, "-F", CONFIG, "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
         "uni-cluster", command],
        check=True, text=True, capture_output=True, timeout=300,
    )
    return [line for line in result.stdout.splitlines() if line.startswith("/home/")]


def is_mcts_eval(path: str) -> bool:
    lower = path.lower()
    name = Path(path).name.lower()
    return (
        "mcts" in lower
        and ("_mcts_orig_" in name or "mcts_eval" in lower)
        and "policy_eval" not in lower
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    paths = [path for path in inventory() if is_mcts_eval(path)]
    rows = []
    for target in endpoints():
        source = target["source_training_job_id"]
        epoch = int(target["epoch"])
        seed = target["seed"]
        vh_token = "_vh_" if target["value_head"] == "on" else "_novh_"
        source_epoch = re.compile(rf"_src{re.escape(source)}_e0*{epoch}(?:\D|$)")
        seed_token = f"_s{seed}_"
        candidates = [
            path for path in paths
            if seed_token in Path(path).name
            and vh_token in Path(path).name
            and source_epoch.search(Path(path).name)
        ]
        rows.append({
            "experiment_id": target["experiment_id"],
            "domain": target["domain"],
            "value_head": target["value_head"],
            "seed": seed,
            "source_training_job_id": source,
            "selected_epoch": str(epoch),
            "policy_score": target["score"],
            "policy_log": target["source_evaluation_log"],
            "candidate_mcts_log_count": str(len(candidates)),
            "candidate_mcts_logs": ";".join(sorted(candidates)),
            "audit_state": "candidate_logs_found" if candidates else "no_candidate_log_found",
        })
    fields = list(rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((r["experiment_id"], r["domain"], r["value_head"], r["audit_state"]) for r in rows)
    print(f"Inventoried {len(paths)} MCTS-like files; wrote {len(rows)} endpoint rows")
    for key, count in sorted(counts.items()):
        print("|".join((*key, str(count))))


if __name__ == "__main__":
    main()
