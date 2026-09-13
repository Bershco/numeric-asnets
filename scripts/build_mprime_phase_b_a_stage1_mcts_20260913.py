#!/usr/bin/env python3
"""Build the exact Phase-B-replicate-A MPrime Stage-1 MCTS manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTORS = (
    ROOT
    / "experiment_tracking/mprime_validation_phase_b_20260906/"
    "phase_b_lineage_selector_comparison_latest.csv"
)
CHECKPOINT_SCORES = (
    ROOT
    / "experiment_tracking/mprime_validation_phase_b_20260906/"
    "phase_b_checkpoint_scores_latest.csv"
)
OUTPUT_DIR = (
    ROOT / "experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913"
)

FIELDS = [
    "array_index",
    "manifest_id",
    "experiment_id",
    "branch",
    "stage",
    "checkpoint_selection",
    "value_head",
    "seed",
    "selected_epoch",
    "selected_validation_score",
    "selected_test_policy_score",
    "source_training_job_id",
    "source_training_log",
    "source_policy_job_id",
    "source_policy_log",
    "checkpoint",
    "domain_module",
    "architecture_module",
    "teacher",
    "width",
    "iterations",
    "puct",
    "estimator",
    "workers",
    "cpus",
    "memory",
    "walltime",
    "instance_timeout_seconds",
    "max_external_actions",
    "evaluation_scheduling",
    "completion_mode",
    "expected_test_instances",
    "status",
    "selector_source",
    "selector_sha256",
    "checkpoint_score_source",
    "checkpoint_score_sha256",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _epoch_from_checkpoint(checkpoint: str) -> int:
    match = re.search(r"/snapshots/snapshot_(\d+)_", checkpoint.replace("\\", "/"))
    if not match:
        raise ValueError(f"cannot parse checkpoint epoch: {checkpoint}")
    return int(match.group(1))


def build_rows(
    selector_path: Path = SELECTORS,
    checkpoint_score_path: Path = CHECKPOINT_SCORES,
) -> dict[str, list[dict[str, str]]]:
    selectors = [
        row
        for row in read_csv(selector_path)
        if row.get("branch") == "stage1" and row.get("selector") == "replicate_a"
    ]
    if len(selectors) != 20:
        raise ValueError(f"expected 20 Stage-1 replicate-A selectors, found {len(selectors)}")
    selector_keys = [(row["value_head"], row["seed"]) for row in selectors]
    if len(selector_keys) != len(set(selector_keys)):
        raise ValueError("duplicate Stage-1 replicate-A value-head/seed selector")

    score_rows = read_csv(checkpoint_score_path)
    by_lineage_epoch: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in score_rows:
        by_lineage_epoch.setdefault((row.get("lineage", ""), row.get("epoch", "")), []).append(row)

    def provenance_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()

    source_selector = provenance_path(selector_path)
    source_scores = provenance_path(checkpoint_score_path)
    selector_digest = sha256(selector_path)
    score_digest = sha256(checkpoint_score_path)
    grouped: dict[str, list[dict[str, str]]] = {"off": [], "on": []}
    for selector in sorted(selectors, key=lambda row: (row["value_head"], int(row["seed"]))):
        vh = selector["value_head"]
        if vh not in grouped:
            raise ValueError(f"invalid value-head mode: {vh}")
        epoch = selector["selected_epoch"]
        matches = by_lineage_epoch.get((selector["lineage"], epoch), [])
        if len(matches) != 1:
            raise ValueError(
                f"{selector['lineage']} epoch {epoch}: expected one checkpoint-score row, "
                f"found {len(matches)}"
            )
        score = matches[0]
        checkpoint = score.get("checkpoint", "").strip()
        if not checkpoint or _epoch_from_checkpoint(checkpoint) != int(epoch):
            raise ValueError(f"{selector['lineage']}: checkpoint/selected-epoch mismatch")
        if score.get("value_head") != vh or score.get("seed") != selector["seed"]:
            raise ValueError(f"{selector['lineage']}: checkpoint identity mismatch")
        if score.get("test_score") != selector.get("selected_test_score"):
            raise ValueError(f"{selector['lineage']}: selected policy score mismatch")
        required_provenance = ("training_job", "training_log", "policy_job", "policy_log")
        missing = [field for field in required_provenance if not score.get(field, "").strip()]
        if missing:
            raise ValueError(f"{selector['lineage']}: missing provenance fields {missing}")

        grouped[vh].append({
            "array_index": "",  # assigned independently within each VH manifest
            "manifest_id": f"mprime-s1-phase-b-a-{vh}-{selector['seed']}-e{int(epoch):04d}-fixed20x70",
            "experiment_id": "MPRIME-VAL",
            "branch": "stage1",
            "stage": "stage1",
            "checkpoint_selection": "phase_b_replicate_a",
            "value_head": vh,
            "seed": selector["seed"],
            "selected_epoch": epoch,
            "selected_validation_score": selector["selected_validation_score"],
            "selected_test_policy_score": selector["selected_test_score"],
            "source_training_job_id": score["training_job"],
            "source_training_log": score["training_log"],
            "source_policy_job_id": score["policy_job"],
            "source_policy_log": score["policy_log"],
            "checkpoint": checkpoint,
            "domain_module": "experiments_numeric.domain.mprime",
            "architecture_module": "experiments_numeric.architecture_2.mprime_mcts",
            "teacher": "hmrp-ha-gbfs",
            "width": "20",
            "iterations": "70",
            "puct": "0.1",
            "estimator": "0.5",
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
            "selector_source": source_selector,
            "selector_sha256": selector_digest,
            "checkpoint_score_source": source_scores,
            "checkpoint_score_sha256": score_digest,
        })

    expected_seeds = {row["seed"] for row in grouped["off"]}
    if len(grouped["off"]) != 10 or len(grouped["on"]) != 10:
        raise ValueError("expected ten selectors in each value-head mode")
    if expected_seeds != {row["seed"] for row in grouped["on"]}:
        raise ValueError("VH-off and VH-on seed sets differ")
    checkpoints = [row["checkpoint"] for rows in grouped.values() for row in rows]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("selected checkpoint paths are not unique")
    for rows in grouped.values():
        for index, row in enumerate(rows):
            row["array_index"] = str(index)
    return grouped


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", type=Path, default=SELECTORS)
    parser.add_argument("--checkpoint-scores", type=Path, default=CHECKPOINT_SCORES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    grouped = build_rows(args.selector, args.checkpoint_scores)
    for vh, rows in grouped.items():
        path = args.output_dir / f"manifest_{vh}.csv"
        write_manifest(path, rows)
        print(f"wrote {len(rows)} rows to {path}")
    print("total=20 reusable_existing=0 fixed_search=20/70 selection=phase_b_replicate_a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
