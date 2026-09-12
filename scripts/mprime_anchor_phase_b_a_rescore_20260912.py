#!/usr/bin/env python3
"""Idempotently evaluate saved MPrime anchor checkpoints on Phase-B set A."""

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


WORKER_FAILURE = re.compile(
    r"Worker (?:died without result|crashed)|CRASH_EXIT|"
    r"AttributeError.*(?:send|put)"
)
FINAL_SCORE = re.compile(r"\[EVAL FINAL\].*?success=\d+(?:\.\d+)?/30(?:\.0)?")
SNAPSHOT_EPOCH = re.compile(r"^snapshot_(\d+)_")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_snapshot_directory(source_log: Path) -> Path:
    for line in source_log.read_text(errors="replace").splitlines():
        if line.startswith("Snapshot directory: "):
            return Path(line.removeprefix("Snapshot directory: ").strip())
    raise RuntimeError(f"missing snapshot-directory declaration: {source_log}")


def select_checkpoints(snapshots_dir: Path) -> list[tuple[int, Path]]:
    candidates = []
    for path in snapshots_dir.glob("snapshot_*"):
        if not path.is_dir():
            continue
        match = SNAPSHOT_EPOCH.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    candidates.sort()
    if not candidates:
        raise RuntimeError(f"no checkpoints in {snapshots_dir}")
    last = candidates[-1]
    selected = [item for item in candidates if item[0] % 5 == 0 or item == last]
    if len(selected) != 21:
        raise RuntimeError(
            f"expected 21 every-five/final checkpoints in {snapshots_dir}, got {len(selected)}"
        )
    return selected


def validate_static_contract(
    manifest: Path,
    validator_manifest: Path,
    validator_candidates: Path,
) -> tuple[list[dict[str, str]], str]:
    rows = read_csv(manifest)
    if len(rows) != 28 or [int(row["array_index"]) for row in rows] != list(range(28)):
        raise RuntimeError("rescore manifest must contain contiguous array indices 0..27")
    if len({row["manifest_id"] for row in rows}) != 28:
        raise RuntimeError("rescore manifest identities are not unique")
    frozen = read_csv(validator_manifest)
    replicate_a = [row for row in frozen if row["replicate"] == "0"]
    if len(frozen) != 60 or len(replicate_a) != 30:
        raise RuntimeError("validator must contain 60 rows, exactly 30 in replicate A")
    validator_sha = sha256(validator_manifest)
    if any(row["validation_manifest_sha256"] != validator_sha for row in rows):
        raise RuntimeError("rescore manifest does not match frozen validator checksum")
    for row in replicate_a:
        problem = validator_candidates / row["file"]
        if not problem.is_file() or sha256(problem) != row["sha256"]:
            raise RuntimeError(f"missing or changed validator PDDL: {problem}")
    return rows, validator_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validator-manifest", type=Path, required=True)
    parser.add_argument("--validator-candidates", type=Path, required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    rows, validator_sha = validate_static_contract(
        args.manifest, args.validator_manifest, args.validator_candidates
    )
    module_path = (
        args.repo / "asnets/experiments_numeric/domain"
        / f"{rows[0]['validation_module']}.py"
    )
    if not module_path.is_file() or sha256(module_path) != rows[0]["validation_module_sha256"]:
        raise RuntimeError(f"missing or changed frozen validator module: {module_path}")
    if args.check_only:
        print(f"STATIC CONTRACT OK rows={len(rows)} validator_sha256={validator_sha}")
        return
    if args.task < 0 or args.task >= len(rows):
        raise RuntimeError(f"task index outside 0..{len(rows)-1}: {args.task}")

    row = rows[args.task]
    source_log = Path(row["source_training_log"])
    if not source_log.is_file():
        raise RuntimeError(f"missing source training log: {source_log}")
    checkpoints = select_checkpoints(parse_snapshot_directory(source_log))
    if args.preflight:
        checkpoints = checkpoints[:1]

    output = args.work / "rescore" / row["manifest_id"]
    output.mkdir(parents=True, exist_ok=True)
    module = row["validation_module"]
    for epoch, checkpoint in checkpoints:
        stem = f"epoch_{epoch}_phase_b_a"
        log = output / f"{stem}.log"
        summary = output / f"{stem}.val.csv"
        done = output / f"{stem}.done.json"
        identity = {
            "checkpoint": str(checkpoint),
            "training_job_id": row["training_job_id"],
            "validation_module": module,
            "validation_manifest_sha256": validator_sha,
        }
        if done.is_file() and summary.is_file():
            if json.loads(done.read_text()) == identity:
                print(f"SKIP {row['manifest_id']} epoch={epoch}", flush=True)
                continue
            raise RuntimeError(f"identity conflict in {done}")

        command = [
            "./run_experiment",
            "experiments_numeric.architecture_2.mprime",
            f"experiments_numeric.domain.{module}",
            "--resume-from", str(checkpoint),
            "--num-workers", "3",
            "--jpddl-max-heap", "4g",
            "--random-seed", row["seed"],
            "--worker-logs",
        ]
        if row["value_head"] == "off":
            command.append("--disable-value-head")
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=args.repo / "asnets",
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        content = log.read_text(errors="replace")
        if result.returncode or WORKER_FAILURE.search(content) or not FINAL_SCORE.search(content):
            raise RuntimeError(
                f"incomplete evaluator result {row['manifest_id']} epoch={epoch} "
                f"returncode={result.returncode}"
            )
        validate = [
            sys.executable,
            str(args.repo / "asnets/tools/validate_eval_log_with_summary.py"),
            "--log", str(log),
            "--domain", module,
            "--validator", "/home/hersco/tools/VAL/build/bin/Validate",
            "--summary-csv", str(summary),
        ]
        subprocess.run(validate, cwd=args.repo / "asnets", check=True)
        if not summary.is_file() or summary.stat().st_size == 0:
            raise RuntimeError(f"missing validated summary: {summary}")
        done.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
        print(f"VALIDATED {row['manifest_id']} epoch={epoch}", flush=True)


if __name__ == "__main__":
    main()
