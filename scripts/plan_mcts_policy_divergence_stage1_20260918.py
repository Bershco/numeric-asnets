#!/usr/bin/env python3
"""Validate and print, but never execute, the Stage-1 Slurm release plan."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

from run_mcts_policy_divergence_stage1_20260918 import load_frozen_tasks


def array_spec(indices: list[int]) -> str:
    if not indices:
        raise RuntimeError("cannot render an empty Slurm array")
    spans = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value == previous + 1:
            previous = value
            continue
        spans.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    spans.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(spans)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--smoke-job-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--primary-concurrency", type=int, default=10)
    parser.add_argument("--optional-concurrency", type=int, default=2)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{7,40}", args.code_commit):
        raise RuntimeError("code commit must be an explicit hexadecimal Git revision")
    if not args.smoke_job_id.isdigit():
        raise RuntimeError("smoke job id must be numeric")
    if args.primary_concurrency < 1 or args.optional_concurrency < 1:
        raise RuntimeError("array concurrency must be positive")
    rows = load_frozen_tasks(args.manifest, args.freeze)
    primary = [index for index, row in enumerate(rows)
               if row["release_class"] == "primary_fixed"]
    optional = [index for index, row in enumerate(rows)
                if row["release_class"] == "optional_fo_pw"]
    gate = f"{args.output_root.rstrip('/')}/compute_smoke/gate.json"
    batch = args.repo.rstrip("/") + "/scripts/mcts_policy_divergence_stage1_grouped_20260918.sbatch"
    common = [
        "--parsable",
        f"--dependency=afterok:{args.smoke_job_id}",
        "--export=ALL," + ",".join([
            f"CODE_COMMIT={args.code_commit}",
            f"CHECKOUT={args.repo}",
            f"STAGE1_OUTPUT_ROOT={args.output_root}",
            f"SMOKE_GATE_JSON={gate}",
        ]),
    ]
    plans = [
        ("primary_fixed", primary, args.primary_concurrency),
        ("optional_fo_pw", optional, args.optional_concurrency),
    ]
    print("# VALIDATED DRY RELEASE PLAN: commands below were not executed")
    for release_class, indices, concurrency in plans:
        command = [
            "sbatch", *common,
            f"--array={array_spec(indices)}%{min(concurrency, len(indices))}",
            batch,
        ]
        print(f"# {release_class}: indices={indices}")
        print(shlex.join(command))
    print("# The batch payload independently rejects a missing/stale smoke gate.")


if __name__ == "__main__":
    main()
