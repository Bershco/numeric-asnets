#!/usr/bin/env python3
"""Audit terminal Counters Stage-2 policy versus narrow-MCTS pairs by instance."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


PAIRS = [
    ("1963100312", "20451601", "20649844"),
    ("1073581256", "20451602", "20649846"),
    ("1239739722", "20465054", "20649847"),
    ("2082152039", "20465166", "20649850"),
    ("1510771779", "20465208", "20649851"),
    ("923500475", "20465209", "20649852"),
]
PLAN_RE = re.compile(r"^\[EVAL\]\[PLAN\].*?\|\s*([^| ]+\.pddl)\s*\|", re.M)
DONE_RE = re.compile(
    r"^\[EVAL INSTANCE\] completed .*?path=.*?/([^/ ]+\.pddl) "
    r"status=([^ ]+) elapsed=([0-9.]+)s success=([0-9.]+)", re.M)
START_RE = re.compile(r"^\[EVAL INSTANCE\] started .*?path=.*?/([^/ ]+\.pddl)", re.M)


def accounting(ids: list[str]) -> dict[str, tuple[str, Path]]:
    text = subprocess.check_output([
        "sacct", "-X", "-n", "-P", "-j", ",".join(ids),
        "-o", "JobIDRaw,State,StdOut%1000",
    ], text=True)
    result = {}
    for line in text.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3 and parts[0].isdigit():
            result[parts[0]] = (parts[1].split()[0], Path(parts[2].replace("%j", parts[0])))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    info = accounting([job for _, p, m in PAIRS for job in (p, m)])
    rows = []
    for seed, policy_job, mcts_job in PAIRS:
        policy_state, policy_log = info[policy_job]
        mcts_state, mcts_log = info[mcts_job]
        policy_text = policy_log.read_text(encoding="utf-8", errors="replace")
        mcts_text = mcts_log.read_text(encoding="utf-8", errors="replace")
        policy_success = set(PLAN_RE.findall(policy_text))
        completed = {name: (status, float(elapsed), float(success))
                     for name, status, elapsed, success in DONE_RE.findall(mcts_text)}
        started = set(START_RE.findall(mcts_text))
        mcts_success = {name for name, (_, _, success) in completed.items() if success > 0}
        all_instances = {f"fz_instance_{number}.pddl" for number in range(2, 61)}
        for name in sorted(all_instances, key=lambda value: int(re.search(r"(\d+)", value).group(1))):
            status, elapsed, success = completed.get(name, ("not_completed", None, 0.0))
            if name not in started:
                status = "not_started"
            policy_ok, mcts_ok = name in policy_success, name in mcts_success
            classification = (
                "both_success" if policy_ok and mcts_ok else
                "policy_only_success" if policy_ok else
                "mcts_only_success" if mcts_ok else "both_failure"
            )
            rows.append({
                "seed": seed, "instance": name, "policy_success": int(policy_ok),
                "mcts_success": int(mcts_ok), "classification": classification,
                "mcts_status": status, "mcts_elapsed_seconds": "" if elapsed is None else f"{elapsed:.3f}",
                "policy_job": policy_job, "mcts_job": mcts_job,
                "policy_job_state": policy_state, "mcts_job_state": mcts_state,
                "policy_log": str(policy_log), "mcts_log": str(mcts_log),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for seed, _, _ in PAIRS:
        selected = [row for row in rows if row["seed"] == seed]
        counts = {label: sum(row["classification"] == label for row in selected)
                  for label in ("both_success", "policy_only_success", "mcts_only_success", "both_failure")}
        causes = {}
        for row in selected:
            if row["classification"] == "policy_only_success":
                causes[row["mcts_status"]] = causes.get(row["mcts_status"], 0) + 1
        print(f"seed={seed} counts={counts} policy_only_causes={causes}")
    print(f"rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
