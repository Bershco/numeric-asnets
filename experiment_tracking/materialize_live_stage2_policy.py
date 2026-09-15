#!/usr/bin/env python3
"""Materialize policy-curve work from immutable checkpoints of live Stage-2 jobs.

Unlike ``materialize_stage2_policy_from_ledger.py``, this scanner is allowed to
observe RUNNING training jobs.  It emits only every-five learning-curve points
while a job is live.  Validation-selected and final roles are added only after
the training job is terminal, so a partial curve can never select an endpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
import time
from pathlib import Path


BEST_RE = re.compile(
    r"\[VALIDATION\] New best(?: reached!)?.*?"
    r"(?:iteration\s+|iter_num=)(\d+).*?"
    r"(?:snapshot name:\s*|snapshot_name=)(snapshot_\d+_[^\s\]]+)"
)
SNAPSHOT_DIR_RE = re.compile(r"^Snapshot directory:\s*(.+)$", re.MULTILINE)
SAVED_RE = re.compile(r"^\[CHECKPOINT SAVED\]\s+(snapshot_(\d+)_[^\s|]+)", re.MULTILINE)
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED",
    "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED",
}
FIELDS = [
    "manifest_id", "task_type", "domain", "value_head", "seed", "stage",
    "status", "teacher", "source_checkpoint_ref", "source_checkpoint_sha256",
    "source_training_job_id", "snapshot_epoch", "analysis_roles",
    "training_state", "training_log",
]


def read(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def accounting(job_ids: list[str]) -> dict[str, tuple[str, Path]]:
    text = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
         "-o", "JobIDRaw,State,StdOut%1000"], text=True
    )
    result: dict[str, tuple[str, Path]] = {}
    for line in text.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        state = parts[1].split()[0].split("+")[0]
        result[parts[0]] = (state, Path(parts[2].replace("%j", parts[0])))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role-prefix", required=True)
    parser.add_argument("--minimum-stability-seconds", type=int, default=60)
    parser.add_argument("--expected-final-epoch", type=int, default=99)
    args = parser.parse_args()

    manifest = read(args.training_manifest, ",")
    submissions = read(args.training_ledger, "\t")
    by_key = {(row["value_head"], row["seed"]): row for row in manifest}
    if len(by_key) != len(manifest):
        raise RuntimeError("duplicate value_head/seed rows in training manifest")
    if len(submissions) != len(manifest):
        raise RuntimeError("training ledger does not cover the complete manifest")
    job_ids = [row["slurm_job_id"] for row in submissions]
    info = accounting(job_ids)

    now = time.time()
    output: list[dict[str, str]] = []
    terminal_count = 0
    complete_count = 0
    continuation_required = 0
    discovered_counts: dict[str, int] = {}
    for submission in submissions:
        key = (submission["value_head"], submission["seed"])
        source = by_key.get(key)
        if source is None:
            raise RuntimeError(f"submission absent from manifest: {key}")
        job_id = submission["slurm_job_id"]
        state, log = info.get(job_id, ("", Path()))
        terminal = state in TERMINAL_STATES
        terminal_count += int(terminal)
        if not log.is_file():
            if terminal:
                raise RuntimeError(f"terminal job {job_id} has no readable log: {log}")
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        directories = SNAPSHOT_DIR_RE.findall(text)
        if not directories:
            if terminal:
                raise RuntimeError(f"terminal job {job_id} has no snapshot directory")
            continue
        snapshot_dir = Path(directories[-1].strip())
        saved = [(name, int(epoch)) for name, epoch in SAVED_RE.findall(text)]
        if not saved:
            if terminal:
                raise RuntimeError(f"terminal job {job_id} has no saved checkpoints")
            continue
        latest_epoch = max(epoch for _, epoch in saved)
        scientifically_complete = terminal and latest_epoch >= args.expected_final_epoch
        complete_count += int(scientifically_complete)
        continuation_required += int(terminal and not scientifically_complete)
        best = BEST_RE.findall(text)
        if scientifically_complete and not best:
            raise RuntimeError(f"terminal job {job_id} has no validation-best checkpoint")
        selected_epoch = int(best[-1][0]) if scientifically_complete and best else None
        final_epoch = latest_epoch if scientifically_complete else None
        emitted = 0
        for name, epoch in sorted(saved, key=lambda item: item[1]):
            if epoch % 5 != 0 and epoch not in {selected_epoch, final_epoch}:
                continue
            checkpoint = snapshot_dir / name
            weights = checkpoint / "weights.joblib"
            if not checkpoint.is_dir() or not weights.is_file():
                if terminal:
                    raise RuntimeError(f"logged checkpoint is incomplete: {checkpoint}")
                continue
            if now - weights.stat().st_mtime < args.minimum_stability_seconds:
                continue
            roles = [f"{args.role_prefix}_learning_curve"]
            if epoch == selected_epoch:
                roles.append(f"{args.role_prefix}_validation_selected_policy")
            if epoch == final_epoch:
                roles.append(f"{args.role_prefix}_final_policy")
            base_id = f"mprime-final-s2-{key[0]}-{key[1]}"
            output.append({
                "manifest_id": f"{base_id}-policy-e{epoch:04d}",
                "task_type": "policy_eval", "domain": "mprime",
                "value_head": key[0], "seed": key[1], "stage": "stage2",
                "status": "ready", "teacher": source["teacher"],
                "source_checkpoint_ref": str(checkpoint),
                "source_checkpoint_sha256": sha256(weights),
                "source_training_job_id": job_id, "snapshot_epoch": str(epoch),
                "analysis_roles": ";".join(roles), "training_state": state,
                "training_log": str(log),
            })
            emitted += 1
        discovered_counts[job_id] = emitted

    ids = [row["manifest_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate policy manifest IDs")
    output.sort(key=lambda row: (row["value_head"], int(row["seed"]), int(row["snapshot_epoch"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    temporary.replace(args.output)
    print(
        f"lineages={len(manifest)} terminal={terminal_count} complete={complete_count} "
        f"continuation_required={continuation_required} "
        f"policy_rows={len(output)} min_rows={min(discovered_counts.values(), default=0)} "
        f"max_rows={max(discovered_counts.values(), default=0)}"
    )


if __name__ == "__main__":
    main()
