#!/usr/bin/env python3
"""Evaluate one final MPrime Stage-2 lineage on frozen Phase-B replicate A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


WORKER_FAILURE = re.compile(r"Worker (?:died without result|crashed)|CRASH_EXIT|AttributeError.*(?:send|put)")
FINAL_SCORE = re.compile(r"\[EVAL FINAL\].*?success=\d+(?:\.\d+)?/30(?:\.0)?")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--validator-manifest", type=Path, required=True)
    parser.add_argument("--validator-candidates", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    manifest = read(args.manifest)
    ready = read(args.ready_manifest)
    validator = read(args.validator_manifest)
    if len(manifest) != 20 or len(ready) != 420:
        raise RuntimeError("expected 20 lineage rows and 420 ready checkpoints")
    if not 0 <= args.task < 20:
        raise RuntimeError("task outside 0..19")
    row = manifest[args.task]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.code_commit:
        raise RuntimeError(f"checkout commit {actual_commit} != declared {args.code_commit}")
    if sha(args.ready_manifest) != row["ready_manifest_sha256"]:
        raise RuntimeError("ready manifest checksum mismatch")
    if sha(args.validator_manifest) != row["validation_manifest_sha256"]:
        raise RuntimeError("validator manifest checksum mismatch")
    module = args.repo / "asnets/experiments_numeric/domain" / f"{row['validation_module']}.py"
    if sha(module) != row["validation_module_sha256"]:
        raise RuntimeError("validator module checksum mismatch")
    rep_a = [item for item in validator if item["replicate"] == "0"]
    if len(rep_a) != 30:
        raise RuntimeError("expected thirty Phase-B-A instances")
    for item in rep_a:
        problem = args.validator_candidates / item["file"]
        if not problem.is_file() or sha(problem) != item["sha256"]:
            raise RuntimeError(f"missing/changed validator problem: {problem}")

    checkpoints = [
        item for item in ready
        if item["value_head"] == row["value_head"] and item["seed"] == row["seed"]
    ]
    checkpoints.sort(key=lambda item: int(item["snapshot_epoch"]))
    if len(checkpoints) != 21:
        raise RuntimeError("lineage does not contain twenty-one checkpoints")
    epochs = [int(item["snapshot_epoch"]) for item in checkpoints]
    if epochs != list(range(0, 100, 5)) + [99]:
        raise RuntimeError(f"unexpected checkpoint epochs: {epochs}")
    if args.preflight:
        checkpoints = checkpoints[:1]

    output = args.work / "rescore" / row["manifest_id"]
    output.mkdir(parents=True, exist_ok=True)
    for item in checkpoints:
        epoch = int(item["snapshot_epoch"])
        stem = f"epoch_{epoch}_phase_b_a"
        log = output / f"{stem}.log"
        summary = output / f"{stem}.val.csv"
        done = output / f"{stem}.done.json"
        identity = {
            "checkpoint": item["source_checkpoint_ref"],
            "checkpoint_sha256": item["source_checkpoint_sha256"],
            "training_job_id": item["source_training_job_id"],
            "validation_manifest_sha256": row["validation_manifest_sha256"],
            "validation_module_sha256": row["validation_module_sha256"],
            "code_commit": actual_commit,
        }
        if done.is_file() and summary.is_file() and json.loads(done.read_text()) == identity:
            print(f"SKIP {row['manifest_id']} epoch={epoch}", flush=True)
            continue
        checkpoint = Path(item["source_checkpoint_ref"])
        weights = checkpoint / "weights.joblib"
        if not weights.is_file() or sha(weights) != item["source_checkpoint_sha256"]:
            raise RuntimeError(f"missing/changed checkpoint: {checkpoint}")
        command = [
            "./run_experiment", "experiments_numeric.architecture_2.mprime",
            f"experiments_numeric.domain.{row['validation_module']}",
            "--resume-from", str(checkpoint), "--num-workers", "3",
            "--jpddl-max-heap", "4g", "--random-seed", row["seed"], "--worker-logs",
        ]
        if row["value_head"] == "off":
            command.append("--disable-value-head")
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=args.repo / "asnets", stdout=stream, stderr=subprocess.STDOUT, text=True)
        content = log.read_text(errors="replace")
        if result.returncode or WORKER_FAILURE.search(content) or not FINAL_SCORE.search(content):
            raise RuntimeError(f"incomplete Phase-B-A result {row['manifest_id']} epoch={epoch} rc={result.returncode}")
        subprocess.run([
            sys.executable, str(args.repo / "asnets/tools/validate_eval_log_with_summary.py"),
            "--log", str(log), "--domain", row["validation_module"],
            "--validator", "/home/hersco/tools/VAL/build/bin/Validate",
            "--summary-csv", str(summary),
        ], cwd=args.repo / "asnets", check=True)
        if not summary.is_file() or summary.stat().st_size == 0:
            raise RuntimeError(f"missing validation summary: {summary}")
        temporary = done.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
        os.replace(temporary, done)
        print(f"VALIDATED {row['manifest_id']} epoch={epoch}", flush=True)


if __name__ == "__main__":
    main()
