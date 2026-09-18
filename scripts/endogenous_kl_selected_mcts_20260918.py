#!/usr/bin/env python3
"""Run one selected-endpoint fixed20/70 or PW70 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--production", type=Path, default=Path("/home/hersco/bershco-nu-asnets/numeric-asnets"))
    parser.add_argument("--container", type=Path, default=Path("/home/hersco/Docker/image.sif"))
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8-sig")))
    matches = [row for row in rows if int(row["array_index"]) == args.index]
    if len(rows) != 16 or len(matches) != 1:
        raise RuntimeError(f"expected 16 rows and one index {args.index}")
    row = matches[0]
    if row["method"] not in {"fixed20x70", "pw70"}:
        raise RuntimeError(f"invalid method: {row['method']}")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.code_commit or row["code_commit"] != args.code_commit:
        raise RuntimeError("code commit mismatch")
    weights = Path(row["checkpoint"]) / "weights.joblib"
    if not weights.is_file() or sha256(weights) != row["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint hash mismatch")

    output = args.campaign / "selected_mcts" / row["method"] / f"arm_{row['arm_index']}"
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "completion.jsonl"
    attempt = output / f"attempt_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
    command = [
        "./run_experiment", row["evaluation_module"], f"experiments_numeric.domain.{row['domain']}",
        "--resume-from", row["checkpoint"], "--disable-value-head", "--eval-with-mcts",
        "--mcts-expansion-size", row["width"], "--mcts-iterations", row["iterations"],
        "--mcts-exploration-weight", row["puct"], "--use-estimator", row["estimator"],
        "--eval-scheduling", "rolling", "--eval-completion-file", str(completion),
        "--eval-instance-timeout", row["instance_timeout_seconds"],
        "--eval-max-actions", row["max_external_actions"], "--num-workers", row["workers"],
        "--jpddl-max-heap", "4g", "--worker-logs", "--random-seed", row["seed"],
    ]
    if row["method"] == "pw70":
        command += [
            "--mcts-progressive-widening", "--mcts-pw-min-width", row["pw_min_width"],
            "--mcts-pw-c", row["pw_c"], "--mcts-pw-alpha", row["pw_alpha"],
        ]
    inner = (
        f"set -euo pipefail; cd {shlex.quote(str(args.checkout / 'asnets'))}; "
        f"source {shlex.quote(str(args.production / 'venv-asnets/bin/activate'))}; "
        f"export PYTHONPATH={shlex.quote(str(args.checkout))}:{shlex.quote(str(args.checkout / 'asnets'))}:${{PYTHONPATH:-}}; "
        f"export ENHSP_CONFIG_OVERRIDE={shlex.quote(row['teacher'])}; {shlex.join(command)}"
    )
    outer = [
        "apptainer", "exec", "--bind", "/home/hersco:/home/hersco",
        "--bind", "/home/hersco/apptainer_fake_passwd:/etc/passwd",
        str(args.container), "/bin/bash", "-lc", inner,
    ]
    with attempt.open("w", encoding="utf-8") as stream:
        process = subprocess.run(outer, stdout=stream, stderr=subprocess.STDOUT)
    metadata = {
        **row, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "attempt_log": str(attempt), "returncode": process.returncode,
    }
    (output / f"attempt_{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
