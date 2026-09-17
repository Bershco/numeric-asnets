#!/usr/bin/env python3
"""Evaluate exactly one frozen MPrime Stage-2 checkpoint on Phase-B-A."""

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
FINAL_SCORE = re.compile(r"\[EVAL FINAL\].*?success=(\d+)(?:\.0+)?/30(?:\.0+)?")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(done: Path, summary: Path, identity: dict[str, str]) -> bool:
    if not done.is_file() or not summary.is_file() or summary.stat().st_size == 0:
        return False
    try:
        rows = read(summary)
        log = summary.with_name(summary.name.replace(".val.csv", ".log"))
        scores = FINAL_SCORE.findall(log.read_text(errors="replace"))
        return (
            json.loads(done.read_text(encoding="utf-8")) == identity
            and len(scores) == 1
            and len(rows) == int(scores[0])
            and all(row.get("val_valid") == "1" for row in rows)
        )
    except (OSError, json.JSONDecodeError, csv.Error):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--key-manifest", type=Path, required=True)
    parser.add_argument("--key-manifest-sha256", required=True)
    parser.add_argument("--validator-manifest", type=Path, required=True)
    parser.add_argument("--validator-candidates", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()

    if sha(args.key_manifest) != args.key_manifest_sha256:
        raise RuntimeError("missing-key manifest checksum mismatch")
    rows = read(args.key_manifest)
    if not 0 <= args.task < len(rows):
        raise RuntimeError("task outside missing-key manifest")
    row = rows[args.task]
    if int(row["array_index"]) != args.task:
        raise RuntimeError("array-index mismatch")
    actual_commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != row["code_commit"]:
        raise RuntimeError(f"checkout commit {actual_commit} != declared {row['code_commit']}")
    if sha(args.validator_manifest) != row["validation_manifest_sha256"]:
        raise RuntimeError("validator manifest checksum mismatch")
    module = args.repo / "asnets/experiments_numeric/domain" / f"{row['validation_module']}.py"
    if sha(module) != row["validation_module_sha256"]:
        raise RuntimeError("validator module checksum mismatch")
    validator = [item for item in read(args.validator_manifest) if item["replicate"] == "0"]
    if len(validator) != 30:
        raise RuntimeError("expected thirty Phase-B-A instances")
    for item in validator:
        problem = args.validator_candidates / item["file"]
        if not problem.is_file() or sha(problem) != item["sha256"]:
            raise RuntimeError(f"missing/changed validator problem: {problem}")

    checkpoint = Path(row["source_checkpoint_ref"])
    weights = checkpoint / "weights.joblib"
    if not weights.is_file() or sha(weights) != row["source_checkpoint_sha256"]:
        raise RuntimeError(f"missing/changed checkpoint: {checkpoint}")
    epoch = int(row["snapshot_epoch"])
    output = args.work / "rescore" / row["manifest_id"]
    output.mkdir(parents=True, exist_ok=True)
    stem = f"epoch_{epoch}_phase_b_a"
    log, summary, done = output / f"{stem}.log", output / f"{stem}.val.csv", output / f"{stem}.done.json"
    identity = {
        "checkpoint": row["source_checkpoint_ref"],
        "checkpoint_sha256": row["source_checkpoint_sha256"],
        "training_job_id": row["source_training_job_id"],
        "validation_manifest_sha256": row["validation_manifest_sha256"],
        "validation_module_sha256": row["validation_module_sha256"],
        "code_commit": actual_commit,
    }
    if complete(done, summary, identity):
        print(f"SKIP {row['manifest_id']} epoch={epoch}")
        return
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
    scores = FINAL_SCORE.findall(content)
    if result.returncode or WORKER_FAILURE.search(content) or len(scores) != 1:
        raise RuntimeError(f"incomplete Phase-B-A result {row['manifest_id']} epoch={epoch} rc={result.returncode}")
    subprocess.run([
        sys.executable, str(args.repo / "asnets/tools/validate_eval_log_with_summary.py"),
        "--log", str(log), "--domain", row["validation_module"],
        "--validator", "/home/hersco/tools/VAL/build/bin/Validate", "--summary-csv", str(summary),
    ], cwd=args.repo / "asnets", check=True)
    if not summary.is_file() or summary.stat().st_size == 0:
        raise RuntimeError(f"missing validation summary: {summary}")
    summary_rows = read(summary)
    if len(summary_rows) != int(scores[0]) or any(row.get("val_valid") != "1" for row in summary_rows):
        raise RuntimeError(f"VAL summary does not match terminal score: {summary}")
    temporary = done.with_suffix(".tmp")
    temporary.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    os.replace(temporary, done)
    print(f"VALIDATED {row['manifest_id']} epoch={epoch}")


if __name__ == "__main__":
    main()
