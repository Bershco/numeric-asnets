#!/usr/bin/env python3
"""Materialize policy-curve evaluations for terminal Stage-2 confirmations.

The input ledger is written by ``finalize_anchor_domain.py``.  Running and
pending lineages are deliberately skipped; repeated invocations therefore add
newly terminal lineages without changing the identity of existing work.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


BEST_RE = re.compile(
    r"\[VALIDATION\] New best reached! .*?iteration (\d+) .*?snapshot name: (snapshot_\d+_[^\]]+)"
)
LAST_RE = re.compile(r"Last valid checkpoint is (.+/snapshot_\d+_[^\s]+)")
TEACHERS = {"delivery": "hadd-astar", "tpp": "hadd-astar", "zenotravel": "hadd-gbfs"}
FIELDS = [
    "manifest_id", "task_type", "domain", "value_head", "seed", "stage",
    "status", "teacher", "source_checkpoint_ref", "source_training_job_id",
    "snapshot_epoch", "analysis_roles", "training_state", "training_log",
]


def read(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def accounting(job_ids: list[str]) -> dict[str, tuple[str, Path]]:
    result: dict[str, tuple[str, Path]] = {}
    if not job_ids:
        return result
    text = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
         "-o", "JobIDRaw,State,StdOut%1000"], text=True
    )
    for line in text.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3 and parts[0].isdigit():
            result[parts[0]] = (parts[1].split()[0], Path(parts[2].replace("%j", parts[0])))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    submissions = read(args.ledger)
    info = accounting([row["slurm_job_id"] for row in submissions])
    output: list[dict[str, str]] = []
    terminal_lineages = 0
    for submitted in submissions:
        job_id = submitted["slurm_job_id"]
        state, log = info.get(job_id, ("", Path()))
        if state in {"RUNNING", "PENDING", "REQUEUED", "COMPLETING"} or not log.is_file():
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        last = LAST_RE.findall(text)
        if not last:
            print(f"[SKIP] job={job_id} state={state} no valid checkpoint")
            continue
        snapshot_dir = Path(last[-1]).parent
        candidates: dict[int, list[Path]] = {}
        for entry in snapshot_dir.iterdir():
            match = re.fullmatch(r"snapshot_(\d+)_.+", entry.name)
            if match:
                candidates.setdefault(int(match.group(1)), []).append(entry)
        if not candidates:
            continue
        final_epoch = max(candidates)
        best = BEST_RE.findall(text)
        selected_epoch = int(best[-1][0]) if best else None
        epochs = {epoch for epoch in candidates if epoch % 5 == 0}
        epochs.add(final_epoch)
        if selected_epoch is not None:
            epochs.add(selected_epoch)
        terminal_lineages += 1
        domain = submitted["domain"]
        for epoch in sorted(epochs):
            paths = candidates.get(epoch, [])
            if len(paths) != 1:
                raise RuntimeError(f"job {job_id} epoch {epoch}: expected one snapshot, got {paths}")
            roles = ["preserve4_stage2_policy_curve"]
            if epoch == selected_epoch:
                roles.append("stage2_validation_selected_policy")
            if epoch == final_epoch:
                roles.append("stage2_final_policy")
            output.append({
                "manifest_id": f"{submitted['manifest_id']}-policy-e{epoch:04d}",
                "task_type": "policy_eval", "domain": domain,
                "value_head": submitted["value_head"], "seed": submitted["seed"],
                "stage": "stage2", "status": "ready", "teacher": TEACHERS[domain],
                "source_checkpoint_ref": str(paths[0]),
                "source_training_job_id": job_id, "snapshot_epoch": str(epoch),
                "analysis_roles": ";".join(roles), "training_state": state,
                "training_log": str(log),
            })
    if len({row["manifest_id"] for row in output}) != len(output):
        raise RuntimeError("duplicate manifest IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    print(f"terminal_lineages={terminal_lineages} policy_rows={len(output)} output={args.output}")


if __name__ == "__main__":
    main()
