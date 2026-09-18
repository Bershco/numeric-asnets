#!/usr/bin/env python3
"""Freeze value-head V1 checkpoint candidates from canonical local ledgers.

This creates a candidate manifest only.  It deliberately leaves unavailable
checkpoint/state hashes blank so the strict preflight cannot mistake endpoint
identity evidence for a submission-ready payload.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiment_tracking" / "value_head_quality_audit"
SEEDS = ("534933607", "923500475")
MAIN_DOMAINS = ("drone", "fo_counters", "rover")
RESOURCES = {
    "drone": (4, 48, 8),
    "fo_counters": (4, 64, 8),
    "rover": (4, 96, 12),
    "mprime": (4, 120, 12),
}
FIELDS = (
    "task_id", "domain", "seed", "stage", "value_head", "checkpoint_role",
    "checkpoint_path", "checkpoint_sha256", "source_training_job_id",
    "snapshot_epoch", "source_manifest", "source_manifest_sha256",
    "selection_rule", "state_manifest_path", "state_manifest_sha256",
    "state_sources", "label_sources", "cpus", "memory_gib",
    "time_limit_hours", "status",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_row(*, domain: str, seed: str, stage: str) -> dict[str, str]:
    cpus, memory, hours = RESOURCES[domain]
    task_id = f"vhv1-{domain}-{seed}-{stage}"
    return {
        "task_id": task_id,
        "domain": domain,
        "seed": seed,
        "stage": stage,
        "value_head": "on",
        "checkpoint_role": (
            "validation_selected_stage1" if stage == "stage1"
            else "validation_led_stage2"
        ),
        "state_manifest_path": f"v1_states/vhv1-{domain}-{seed}-paired.jsonl",
        "state_manifest_sha256": "",
        "state_sources": "common_planner+common_random_legal+stage1_on_policy+stage2_on_policy",
        "label_sources": "replay_target+deterministic_continuation+enhsp_raw_h",
        "cpus": str(cpus),
        "memory_gib": str(memory),
        "time_limit_hours": str(hours),
        "status": "frozen_candidate_hashes_and_states_pending",
    }


def main_rows() -> list[dict[str, str]]:
    source = ROOT / "experiment_tracking" / "policy_endpoint_results.csv"
    chosen = [
        row for row in read_csv(source)
        if row["experiment_id"] == "MAIN-VAL"
        and row["domain"] in MAIN_DOMAINS
        and row["value_head"] == "on"
        and row["seed"] in SEEDS
        and row["stage"] in {"stage1", "stage2"}
        and row["endpoint"] == "validation_selected"
    ]
    expected = {(d, s, t) for d in MAIN_DOMAINS for s in SEEDS for t in ("stage1", "stage2")}
    actual = {(r["domain"], r["seed"], r["stage"]) for r in chosen}
    if actual != expected or len(chosen) != len(expected):
        raise RuntimeError(f"MAIN-VAL endpoint mismatch: missing={expected - actual}")
    rows: list[dict[str, str]] = []
    for item in chosen:
        row = base_row(domain=item["domain"], seed=item["seed"], stage=item["stage"])
        row.update({
            "checkpoint_path": item["checkpoint"],
            "checkpoint_sha256": "",
            "source_training_job_id": item["source_training_job_id"],
            "snapshot_epoch": item["snapshot_epoch"],
            "source_manifest": str(source.relative_to(ROOT)).replace("\\", "/"),
            "source_manifest_sha256": sha256(source),
            "selection_rule": "canonical_MAIN-VAL_validation_selected;seeds_first_two_replication_ids_not_outcome_selected",
        })
        rows.append(row)
    return rows


def mprime_rows() -> list[dict[str, str]]:
    stage1_source = ROOT / "experiment_tracking" / "mprime_phase_b_a_stage1_mcts_20260913" / "manifest_on.csv"
    stage2_source = ROOT / "experiment_tracking" / "mprime_final_stage2_search_20260916" / "manifest.csv"
    stage1 = [r for r in read_csv(stage1_source) if r["seed"] in SEEDS]
    stage2 = [
        r for r in read_csv(stage2_source)
        if r["value_head"] == "on" and r["seed"] in SEEDS
        and r["search_method"] == "fixed" and r["branch"] == "validation_led"
    ]
    if len(stage1) != 2 or len(stage2) != 2:
        raise RuntimeError("MPrime exact Stage-1/validation-led Stage-2 endpoints are not unique")
    rows: list[dict[str, str]] = []
    for stage, source, selected in (("stage1", stage1_source, stage1), ("stage2", stage2_source, stage2)):
        for item in selected:
            row = base_row(domain="mprime", seed=item["seed"], stage=stage)
            row.update({
                "checkpoint_path": item["checkpoint"],
                "checkpoint_sha256": item.get("checkpoint_sha256", ""),
                "source_training_job_id": item["source_training_job_id"],
                "snapshot_epoch": item["selected_epoch"],
                "source_manifest": str(source.relative_to(ROOT)).replace("\\", "/"),
                "source_manifest_sha256": sha256(source),
                "selection_rule": "phase_b_replicate_a;seeds_first_two_replication_ids_not_outcome_selected",
            })
            if row["checkpoint_sha256"]:
                row["status"] = "frozen_candidate_state_manifest_pending"
            rows.append(row)
    return rows


def main() -> None:
    rows = sorted(main_rows() + mprime_rows(), key=lambda r: (r["domain"], int(r["seed"]), r["stage"]))
    output = OUT_DIR / "v1_checkpoint_candidates.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} frozen candidates to {output}")


if __name__ == "__main__":
    main()
