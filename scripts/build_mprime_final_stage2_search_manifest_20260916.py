#!/usr/bin/env python3
"""Build exact fixed/PW70 MPrime searches from final Stage-2 endpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


SCORE_RE = re.compile(
    r"^(?P<log>.+?/(?P<job>\d+)_.*?_src(?P<src>\d+)_e(?P<epoch>\d{4})\.txt):"
    r"Inference success rate:\s*(?P<rate>[0-9.]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--selected-policy-lines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ready = read_csv(args.ready_manifest)
    selected = [
        row for row in ready
        if "mprime_final_validation_stage2_validation_selected_policy"
        in row["analysis_roles"].split(";")
    ]
    if len(selected) != 20:
        raise RuntimeError(f"expected 20 selected endpoints, found {len(selected)}")
    if {row["snapshot_epoch"] for row in selected} != {"0"}:
        raise RuntimeError("final selector did not choose epoch 0 for every lineage")
    if len({(row["value_head"], row["seed"]) for row in selected}) != 20:
        raise RuntimeError("duplicate or missing value-head/seed endpoint")

    scores: dict[tuple[str, int], dict[str, str]] = {}
    for line in args.selected_policy_lines.read_text(encoding="utf-8-sig").splitlines():
        match = SCORE_RE.search(line)
        if not match:
            continue
        key = (match["src"], int(match["epoch"]))
        scores[key] = {
            "source_policy_job_id": match["job"],
            "source_policy_log": match["log"],
            "selected_test_policy_score": str(round(float(match["rate"]) * 20)),
        }
    if len(scores) != 20:
        raise RuntimeError(f"expected 20 selected policy results, found {len(scores)}")

    rows: list[dict[str, str]] = []
    for endpoint in sorted(selected, key=lambda row: (row["value_head"], int(row["seed"]))):
        key = (endpoint["source_training_job_id"], int(endpoint["snapshot_epoch"]))
        if key not in scores:
            raise RuntimeError(f"missing selected policy result for {key}")
        for method in ("fixed", "pw70"):
            rows.append({
                "array_index": str(len(rows)),
                "manifest_id": (
                    f"mprime-final-s2-{endpoint['value_head']}-{endpoint['seed']}-"
                    f"e{int(endpoint['snapshot_epoch']):04d}-{method}20x70"
                ),
                "experiment_id": "MPRIME-FINAL-S2-SEARCH",
                "branch": "validation_led",
                "stage": "stage2",
                "checkpoint_selection": "phase_b_replicate_a",
                "search_method": method,
                "value_head": endpoint["value_head"],
                "seed": endpoint["seed"],
                "selected_epoch": endpoint["snapshot_epoch"],
                "selected_validation_score": "30",
                **scores[key],
                "source_training_job_id": endpoint["source_training_job_id"],
                "source_training_log": endpoint["training_log"],
                "checkpoint": endpoint["source_checkpoint_ref"],
                "checkpoint_sha256": endpoint["source_checkpoint_sha256"],
                "domain_module": "experiments_numeric.domain.mprime",
                "architecture_module": "experiments_numeric.architecture_2.mprime_mcts",
                "teacher": "hmrp-ha-gbfs",
                "width": "20",
                "iterations": "70",
                "puct": "0.1",
                "estimator": "0.5",
                "pw_min_width": "3" if method == "pw70" else "",
                "pw_c": "0.6" if method == "pw70" else "",
                "pw_alpha": "0.5" if method == "pw70" else "",
                "terminal_safe": "false",
                "workers": "3",
                "cpus": "6",
                "memory": "120G",
                "walltime": "3-00:00:00",
                "instance_timeout_seconds": "21600",
                "max_external_actions": "10000",
                "evaluation_scheduling": "rolling",
                "completion_mode": "identity_jsonl+attempt_VAL",
                "expected_test_instances": "20",
                "status": "ready_not_submitted",
                "ready_manifest": str(args.ready_manifest).replace("\\", "/"),
                "ready_manifest_sha256": sha256(args.ready_manifest),
                "selected_policy_source": str(args.selected_policy_lines).replace("\\", "/"),
                "selected_policy_source_sha256": sha256(args.selected_policy_lines),
            })

    if len(rows) != 40:
        raise RuntimeError(f"expected 40 searches, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
