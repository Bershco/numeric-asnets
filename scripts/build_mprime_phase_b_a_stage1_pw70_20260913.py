#!/usr/bin/env python3
"""Build the four-row Phase-B-A MPrime Stage-1 PW70 screen manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913"
OUTPUT_DIR = ROOT / "experiment_tracking/mprime_phase_b_a_stage1_pw70_20260913"
SEEDS = ("1963100312", "2011206605")

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


def build_rows(source_dir: Path = SOURCE_DIR) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for vh in ("off", "on"):
        source = source_dir / f"manifest_{vh}.csv"
        source_rows = _read(source)
        selected = [row for row in source_rows if row.get("seed") in SEEDS]
        if {row.get("seed") for row in selected} != set(SEEDS) or len(selected) != 2:
            raise ValueError(f"{source}: expected exactly the two predeclared seeds")
        source_rel = source.resolve().relative_to(ROOT.resolve()).as_posix()
        source_hash = _sha256(source)
        for row in sorted(selected, key=lambda item: SEEDS.index(item["seed"])):
            if (
                row.get("checkpoint_selection") != "phase_b_replicate_a"
                or row.get("width") != "20"
                or row.get("iterations") != "70"
                or row.get("puct") != "0.1"
                or row.get("estimator") != "0.5"
                or row.get("instance_timeout_seconds") != "21600"
                or row.get("max_external_actions") != "10000"
            ):
                raise ValueError(f"unexpected fixed-comparator identity: {row.get('manifest_id')}")
            epoch = int(row["selected_epoch"])
            rows.append({
                **{field: row.get(field, "") for field in FIELDS},
                "array_index": str(len(rows)),
                "manifest_id": f"mprime-s1-phase-b-a-{vh}-{row['seed']}-e{epoch:04d}-pw70-k3",
                "experiment_id": "MPRIME-PBA-S1-PW-SCREEN",
                "pw_min_width": "3",
                "pw_c": "0.6",
                "pw_alpha": "0.5",
                # The scientific comparator has terminal-safe selection disabled.
                # PW is the only intended algorithmic difference.
                "terminal_safe": "false",
                "status": "ready_not_submitted",
                "source_fixed_manifest": source_rel,
                "source_fixed_manifest_sha256": source_hash,
            })
    if len(rows) != 4:
        raise ValueError(f"expected four PW rows, found {len(rows)}")
    identities = {(row["value_head"], row["seed"]) for row in rows}
    expected = {(vh, seed) for vh in ("off", "on") for seed in SEEDS}
    if identities != expected:
        raise ValueError("PW manifest does not form the declared 2x2 matched screen")
    if len({row["checkpoint"] for row in rows}) != 4:
        raise ValueError("PW manifest checkpoint paths are not unique")
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = build_rows()
    output = OUTPUT_DIR / "manifest.csv"
    write_manifest(output, rows)
    print(f"wrote {len(rows)} rows to {output}")
    print("seeds=1963100312,2011206605 modes=off,on algorithm=PW70 Kmin3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
