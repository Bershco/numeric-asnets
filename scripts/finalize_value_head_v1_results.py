#!/usr/bin/env python3
"""Verify all sixteen V1 outputs without overstating scientific conclusions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


LABELS = {"replay_target", "deterministic_continuation", "enhsp_raw_h", "enhsp_search_v"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-root", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--resource-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    gate = json.loads(args.resource_gate.read_text(encoding="utf-8"))
    if gate.get("status") != "pass" or gate.get("candidate_manifest_sha256") != sha256(args.candidates):
        raise RuntimeError("resource gate/candidate hash mismatch")
    with args.candidates.open(newline="", encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream))
    if len(candidates) != 16:
        raise RuntimeError("expected sixteen candidate rows")
    outputs = []
    errors = []
    for candidate in candidates:
        directory = args.science_root / candidate["task_id"]
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            errors.append(f"missing summary {candidate['task_id']}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        output_path = Path(summary["output_path"])
        if summary.get("mode") != "full" or summary.get("states") != 60:
            errors.append(f"invalid full summary {candidate['task_id']}")
        if summary.get("task_id") != candidate["task_id"]:
            errors.append(f"task identity mismatch {candidate['task_id']}")
        if not output_path.is_file() or sha256(output_path) != summary.get("output_sha256"):
            errors.append(f"output hash mismatch {candidate['task_id']}")
            continue
        counts = summary.get("label_counts", {})
        if set(counts) != LABELS:
            errors.append(f"label factorial mismatch {candidate['task_id']}")
        for source in ("deterministic_continuation", "enhsp_raw_h", "enhsp_search_v"):
            if counts.get(source, {}).get("error", 0):
                errors.append(f"provider error {candidate['task_id']}/{source}")
        if counts.get("enhsp_raw_h") != counts.get("enhsp_search_v"):
            errors.append(f"raw/transformed mismatch {candidate['task_id']}")
        expected_rows = int(summary.get("successors", 0)) * 4
        actual_rows = sum(1 for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if actual_rows != expected_rows:
            errors.append(f"row-count mismatch {candidate['task_id']}")
        outputs.append(summary)
    result = {
        "schema": "value-head-audit-completion-v1",
        "status": "complete" if not errors and len(outputs) == 16 else "blocked",
        "candidate_manifest": str(args.candidates),
        "candidate_manifest_sha256": sha256(args.candidates),
        "resource_gate": str(args.resource_gate),
        "resource_gate_sha256": sha256(args.resource_gate),
        "tasks": outputs,
        "errors": errors,
        "interpretation": "Operational completion only; RQ claims require lineage/domain aggregation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(("V1_OUTPUTS_COMPLETE" if result["status"] == "complete" else "V1_OUTPUTS_BLOCKED") + f"|{args.output}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
