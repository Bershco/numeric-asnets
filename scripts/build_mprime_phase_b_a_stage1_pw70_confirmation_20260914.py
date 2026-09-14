#!/usr/bin/env python3
"""Build the remaining 16-row Phase-B-A MPrime Stage-1 PW70 confirmation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913"
OUTPUT_DIR = ROOT / "experiment_tracking/mprime_phase_b_a_stage1_pw70_20260913"
SCREEN_SEEDS = {"1963100312", "2011206605"}

FIELDS = [
    "array_index", "manifest_id", "experiment_id", "branch", "stage",
    "checkpoint_selection", "value_head", "seed", "selected_epoch",
    "selected_validation_score", "selected_test_policy_score",
    "source_training_job_id", "source_training_log", "source_policy_job_id",
    "source_policy_log", "checkpoint", "domain_module", "architecture_module",
    "teacher", "width", "iterations", "puct", "estimator", "pw_min_width",
    "pw_c", "pw_alpha", "terminal_safe", "workers", "cpus", "memory",
    "walltime", "instance_timeout_seconds", "max_external_actions",
    "evaluation_scheduling", "completion_mode", "expected_test_instances",
    "status", "source_fixed_manifest", "source_fixed_manifest_sha256",
    "selector_source", "selector_sha256", "checkpoint_score_source",
    "checkpoint_score_sha256",
]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seeds_by_mode: dict[str, set[str]] = {}
    for vh in ("off", "on"):
        source = SOURCE_DIR / f"manifest_{vh}.csv"
        source_rows = _read(source)
        if len(source_rows) != 10:
            raise ValueError(f"{source}: expected ten canonical fixed-MCTS rows")
        selected = [row for row in source_rows if row.get("seed") not in SCREEN_SEEDS]
        if len(selected) != 8:
            raise ValueError(f"{source}: expected eight unused seeds, found {len(selected)}")
        seeds_by_mode[vh] = {row["seed"] for row in selected}
        source_rel = source.resolve().relative_to(ROOT.resolve()).as_posix()
        source_hash = _sha256(source)
        for row in sorted(selected, key=lambda item: int(item["seed"])):
            if (
                row.get("value_head") != vh
                or row.get("checkpoint_selection") != "phase_b_replicate_a"
                or row.get("width") != "20"
                or row.get("iterations") != "70"
                or row.get("puct") != "0.1"
                or row.get("estimator") != "0.5"
                or row.get("instance_timeout_seconds") != "21600"
                or row.get("max_external_actions") != "10000"
                or row.get("workers") != "3"
                or row.get("cpus") != "6"
                or row.get("memory") != "120G"
                or row.get("walltime") != "3-00:00:00"
            ):
                raise ValueError(f"unexpected fixed-comparator identity: {row.get('manifest_id')}")
            epoch = int(row["selected_epoch"])
            rows.append({
                **{field: row.get(field, "") for field in FIELDS},
                "array_index": str(len(rows)),
                "manifest_id": f"mprime-s1-phase-b-a-{vh}-{row['seed']}-e{epoch:04d}-pw70-k3",
                # This is the confirmatory extension of the already-smoked
                # MPRIME-PBA-S1-PW-SCREEN runtime identity. Keeping the same
                # experiment_id lets the unchanged, compute-smoked runner
                # enforce every scientific parameter for these new seeds.
                "experiment_id": "MPRIME-PBA-S1-PW-SCREEN",
                "pw_min_width": "3",
                "pw_c": "0.6",
                "pw_alpha": "0.5",
                "terminal_safe": "false",
                "status": "ready_not_submitted",
                "source_fixed_manifest": source_rel,
                "source_fixed_manifest_sha256": source_hash,
            })
    if seeds_by_mode["off"] != seeds_by_mode["on"]:
        raise ValueError("VH modes do not contain the same eight matched seeds")
    if len(rows) != 16 or len({row["manifest_id"] for row in rows}) != 16:
        raise ValueError("confirmation must contain sixteen unique identities")
    if any(row["seed"] in SCREEN_SEEDS for row in rows):
        raise ValueError("two completed screen seeds leaked into confirmation")
    return rows


def main() -> int:
    rows = build_rows()
    _write(OUTPUT_DIR / "manifest_confirmation_remaining.csv", rows)
    # The already compute-smoked runner validates four rows. Four independent
    # four-task arrays preserve that exact runtime path and add no concurrency cap.
    for part, start in enumerate(range(0, 16, 4), start=1):
        chunk = [{**row, "array_index": str(index)} for index, row in enumerate(rows[start:start + 4])]
        _write(OUTPUT_DIR / f"manifest_confirmation_part{part}.csv", chunk)
    print("wrote 16 matched confirmation rows in four unthrottled four-task arrays")
    print("excluded screen seeds=" + ",".join(sorted(SCREEN_SEEDS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
