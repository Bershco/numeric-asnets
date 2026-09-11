"""Evaluate frozen Phase-C checkpoint candidates on the frozen Phase-C set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    frozen = list(csv.DictReader((args.root / "frozen_validation_manifest.csv").open()))
    assert len(frozen) == 30
    for row in frozen:
        assert hashlib.sha256((args.root / "candidates" / row["file"]).read_bytes()).hexdigest() == row["sha256"]
    inventory = list(csv.DictReader((args.root / "checkpoint_candidates.csv").open()))
    lineages = sorted({row["lineage"] for row in inventory})
    assert len(lineages) == 40
    group = [row for row in inventory if row["lineage"] == lineages[args.task]]
    assert 1 <= len(group) <= 12
    if args.preflight:
        group = group[:1]
    manifest_digest = hashlib.sha256((args.root / "frozen_validation_manifest.csv").read_bytes()).hexdigest()
    candidate_digest = hashlib.sha256((args.root / "checkpoint_candidates.csv").read_bytes()).hexdigest()
    out = args.root / "rescore" / lineages[args.task]
    out.mkdir(parents=True, exist_ok=True)
    for row in group:
        checkpoint = Path(row["checkpoint"])
        assert checkpoint.exists(), checkpoint
        log = out / f"epoch_{row['epoch']}.log"
        summary = log.with_suffix(".val.csv")
        done = log.with_suffix(".done.json")
        identity = {
            "checkpoint": row["checkpoint"],
            "validation_manifest_sha256": manifest_digest,
            "checkpoint_candidates_sha256": candidate_digest,
        }
        if done.exists() and json.loads(done.read_text()) == identity and summary.exists():
            continue
        complete = log.exists() and re.search(
            r"\[EVAL FINAL\].*?success=\d+(?:\.\d+)?/30(?:\.0)?",
            log.read_text(errors="replace"),
        )
        if not complete:
            command = [
                "./run_experiment",
                "experiments_numeric.architecture_2.mprime",
                "experiments_numeric.domain.mprime_phase_c_20260911",
                "--resume-from", row["checkpoint"],
                "--num-workers", "3",
                "--jpddl-max-heap", "4g",
                "--random-seed", row["seed"],
                "--worker-logs",
            ]
            if row["value_head"] == "off":
                command.append("--disable-value-head")
            with log.open("w") as stream:
                subprocess.run(command, cwd=args.repo / "asnets", stdout=stream, stderr=subprocess.STDOUT, check=True)
        content = log.read_text(errors="replace")
        assert re.search(r"\[EVAL FINAL\].*?success=\d+(?:\.\d+)?/30(?:\.0)?", content)
        assert not re.search(r"Worker (?:died without result|crashed)|CRASH_EXIT", content)
        subprocess.run(
            [
                "python", str(args.repo / "asnets/tools/validate_eval_log_with_summary.py"),
                "--log", str(log), "--domain", "mprime_phase_c_20260911",
                "--validator", "/home/hersco/tools/VAL/build/bin/Validate",
                "--summary-csv", str(summary),
            ],
            cwd=args.repo / "asnets", check=True,
        )
        done.write_text(json.dumps(identity, sort_keys=True))
        print(f"VALIDATED lineage={lineages[args.task]} epoch={row['epoch']}", flush=True)


if __name__ == "__main__":
    main()
