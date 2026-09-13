#!/usr/bin/env python3
"""Reconstruct exact FO Counters validation-led Stage-2 MCTS seed results.

Run on the cluster, where the historical logs and durable completion ledgers
exist.  Multiple operational retries are merged by evaluator instance number;
a success wins over a failure and, among successes, the shortest elapsed time
is retained for deterministic cutoff reporting.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path


INSTANCE = re.compile(
    r"\[EVAL INSTANCE\] completed number=(\d+) path=(\S+) status=(\S+) "
    r"elapsed=([0-9.]+)s success=([0-9.]+) steps=(\d+)"
)
JOB = re.compile(r"^(\d+)")
EPOCH = re.compile(r"_e(\d{4})\.txt$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def candidate(old: tuple[bool, float] | None, new: tuple[bool, float]) -> tuple[bool, float]:
    if old is None or (new[0], -new[1]) > (old[0], -old[1]):
        return new
    return old


def ledger_records(path: Path, records: dict[int, tuple[bool, float]]) -> None:
    if not path.is_file():
        return
    for line in path.open(errors="replace"):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        number = item.get("instance_number", item.get("number"))
        if number is None:
            continue
        success = bool(item.get("hit_goal", item.get("success", False)))
        elapsed = float(item.get("elapsed_seconds", item.get("elapsed", 0)) or 0)
        records[int(number)] = candidate(records.get(int(number)), (success, elapsed))


def log_records(path: Path, records: dict[int, tuple[bool, float]]) -> None:
    if not path.is_file():
        return
    for line in path.open(errors="replace"):
        match = INSTANCE.search(line)
        if not match:
            continue
        number = int(match.group(1))
        item = (float(match.group(5)) > 0, float(match.group(4)))
        records[number] = candidate(records.get(number), item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = [
        row for row in read_csv(args.repo / "experiment_tracking/policy_paired_seed_results.csv")
        if row["experiment_id"] == "MAIN-VAL"
        and row["domain"] == "fo_counters"
        and row["value_head"] == "off"
    ]
    output = []
    for row in sorted(policy, key=lambda item: int(item["seed"])):
        epoch_match = EPOCH.search(row["after_log"])
        if not epoch_match:
            raise RuntimeError(f"cannot parse selected epoch: {row['after_log']}")
        source = row["after_training_job"]
        seed = row["seed"]
        epoch = epoch_match.group(1)
        patterns = [
            f"/home/hersco/training_new_domains/*/*/*_fo_counters_*_s{seed}_K0_SR10M_src{source}_e{epoch}.txt",
            f"/home/hersco/training_new_domains/2026-09-10/fo_stage2_validation_recovery/*_off_{seed}.txt",
            f"/home/hersco/training_new_domains/2026-09-12/fo_stage2_validation_exact9/*_off_{seed}_instance9.txt",
        ]
        logs = sorted({Path(item) for pattern in patterns for item in glob.glob(pattern)})
        records: dict[int, tuple[bool, float]] = {}
        ledgers: set[Path] = set()
        for log in logs:
            log_records(log, records)
            match = JOB.match(log.name)
            if not match:
                continue
            job = match.group(1)
            for pattern in (
                f"/home/hersco/training_new_domains/*/*/.resume_state/{job}.eval_completed.jsonl",
                f"/home/hersco/training_new_domains/*/*/completion/{job}.jsonl",
            ):
                ledgers.update(Path(item) for item in glob.glob(pattern))
        for ledger in sorted(ledgers):
            ledger_records(ledger, records)
        successes = [elapsed for success, elapsed in records.values() if success]
        # Eight identities were already complete historical evaluations.  The
        # two interrupted VH-off identities are now closed by the exact
        # recovery logs listed below.  Older successful jobs recorded only
        # successes in their JSONL files, so len(records) is not a completeness
        # test for those historical runs.
        historically_complete = {
            "534933607", "1073581256", "1239739722", "1472491096",
            "1510771779", "1963100312", "1972442430", "2011206605",
        }
        recovery_closed = {
            "923500475": any("21178379_" in str(path) for path in logs),
            "2082152039": any("21219947_" in str(path) for path in logs),
        }
        exact = seed in historically_complete or recovery_closed.get(seed, False)
        output.append({
            "seed": seed,
            "policy_score": row["after_score"],
            "mcts_30m": sum(elapsed <= 1800 for elapsed in successes),
            "mcts_2h": sum(elapsed <= 7200 for elapsed in successes),
            "mcts_6h": sum(elapsed <= 21600 for elapsed in successes),
            "classified_instances": 20 if exact else len(records),
            "unclassified_instances": "" if exact else "unknown_historical_failure_records",
            "source_training_job_id": source,
            "source_policy_job_id": row["after_evaluation_job"],
            "source_policy_log": row["after_log"],
            "source_mcts_logs": ";".join(map(str, logs)),
            "source_completion_ledgers": ";".join(map(str, sorted(ledgers))),
            "evidence_status": "complete_exact_declared_budget" if exact else "partial_lower_bound",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    print(json.dumps({
        "rows": len(output),
        "complete": sum(row["evidence_status"].startswith("complete_exact") for row in output),
        "means": {
            cutoff: sum(int(row[f"mcts_{cutoff}"]) for row in output) / len(output)
            for cutoff in ("30m", "2h", "6h")
        },
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
