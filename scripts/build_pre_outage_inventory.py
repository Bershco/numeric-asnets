#!/usr/bin/env python3
"""Build the frozen job inventory used after the September 2026 outage.

This script performs no cluster operations.  It joins the authoritative local
submission ledgers that existed at the last successful scheduler snapshot.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "2026-09-01T10:37:46+03:00"
OUTPUT = ROOT / "experiment_tracking" / "pre_outage_job_inventory_20260901_1037.csv"

FIELDS = [
    "snapshot_time",
    "experiment_id",
    "component",
    "job_id",
    "manifest_id",
    "domain",
    "value_head",
    "seed",
    "expected_pre_outage_state",
    "cpus",
    "memory_gib",
    "time_limit",
    "artifact_kind",
    "evidence_source",
    "recovery_policy",
    "notes",
]


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def add(rows: list[dict[str, str]], **values: object) -> None:
    row = {field: "" for field in FIELDS}
    row.update({key: str(value) for key, value in values.items()})
    row["snapshot_time"] = SNAPSHOT
    rows.append(row)


def main() -> None:
    rows: list[dict[str, str]] = []
    tracking = ROOT / "experiment_tracking"

    # The last snapshot recorded this campaign as 21 terminal and 39 running.
    # It did not freeze the exact per-job split, so all 60 IDs must be queried.
    preserve_path = tracking / "four_domain_preservation" / "terminal_stage2_submissions.tsv"
    for item in read_rows(preserve_path, "\t"):
        add(
            rows,
            experiment_id="PRESERVE-3-TERM",
            component="stage2_training",
            job_id=item["slurm_job_id"],
            manifest_id=item["manifest_id"],
            domain=item["domain"],
            value_head=item["value_head"],
            seed=item["seed"],
            expected_pre_outage_state="campaign_mixed:21_terminal/39_running",
            cpus=6,
            memory_gib=48,
            time_limit="3-00:00:00",
            artifact_kind="training",
            evidence_source=preserve_path.relative_to(ROOT),
            recovery_policy="preserve_terminal_or_resume_from_latest_checkpoint",
            notes="Query every ID; do not infer its individual state from submission order.",
        )

    pw_path = tracking / "mcts_progressive_widening_cross_domain" / "pw70_followup_submissions.tsv"
    pw_manifest = {
        row["manifest_id"]: row
        for row in read_rows(
            tracking / "mcts_progressive_widening_cross_domain" / "pw70_followup_manifest.csv"
        )
    }
    for item in read_rows(pw_path, "\t"):
        meta = pw_manifest[item["manifest_id"]]
        add(
            rows,
            experiment_id="MCTS-PW70-CROSS-DOMAIN",
            component="mcts_evaluation",
            job_id=item["slurm_job_id"],
            manifest_id=item["manifest_id"],
            domain=meta["domain"],
            value_head=meta["value_head"],
            seed=meta["seed"],
            expected_pre_outage_state="running",
            cpus=meta["cpus"],
            memory_gib=str(meta["memory"]).rstrip("G"),
            time_limit=meta["time_limit"],
            artifact_kind="mcts_evaluation",
            evidence_source=pw_path.relative_to(ROOT),
            recovery_policy="resume_from_completion_jsonl_and_validate_printed_plans",
            notes="PW70 only; never pool with the completed mixed-budget/PW20 screen.",
        )

    horizon_path = tracking / "mcts_horizon_binding" / "counters_submissions.tsv"
    horizon_manifest = {
        row["manifest_id"]: row
        for row in read_rows(tracking / "mcts_horizon_binding" / "counters_manifest.csv")
    }
    for item in read_rows(horizon_path, "\t"):
        meta = horizon_manifest[item["manifest_id"]]
        add(
            rows,
            experiment_id="MCTS-HORIZON-COUNTERS",
            component="mcts_evaluation",
            job_id=item["slurm_job_id"],
            manifest_id=item["manifest_id"],
            domain=meta["domain"],
            value_head=meta["value_head"],
            seed=meta["seed"],
            expected_pre_outage_state="running",
            cpus=6,
            memory_gib=120,
            time_limit="3-00:00:00",
            artifact_kind="mcts_evaluation",
            evidence_source=horizon_path.relative_to(ROOT),
            recovery_policy="resume_from_completion_jsonl_and_validate_printed_plans",
            notes=f"Arm={meta['arm']}; preserve aware/unaware pairing.",
        )

    drone_path = tracking / "main_val_stage2_drone_mcts_submissions.tsv"
    for item in read_rows(drone_path, "\t"):
        running = item["slurm_job_id"] == "20726041"
        add(
            rows,
            experiment_id="MAIN-VAL-S2-MCTS",
            component="drone_stage2_mcts",
            job_id=item["slurm_job_id"],
            manifest_id=item["manifest_id"],
            domain=item["domain"],
            value_head=item["value_head"],
            seed=item["seed"],
            expected_pre_outage_state="running" if running else "terminal",
            cpus=6,
            memory_gib=120,
            time_limit="3-00:00:00",
            artifact_kind="mcts_evaluation",
            evidence_source=drone_path.relative_to(ROOT),
            recovery_policy=(
                "resume_from_completion_jsonl_and_validate_printed_plans"
                if running else "preserve_static_result"
            ),
            notes="Fifteen submitted rows were terminal; four additional historical rows are outside this ledger.",
        )

    add(
        rows,
        experiment_id="MCTS-LEGACY-ROVER",
        component="terminal_stage2_mcts",
        job_id="20559649",
        manifest_id="rover-on-2082152039-stage2-mcts",
        domain="rover",
        value_head="on",
        seed="2082152039",
        expected_pre_outage_state="running",
        cpus=6,
        memory_gib=120,
        time_limit="3-00:00:00",
        artifact_kind="mcts_evaluation",
        evidence_source="experiment_tracking/live_jobs.csv",
        recovery_policy="resume_from_completion_jsonl_and_validate_printed_plans",
        notes="The final live Rover row at the 10:37 snapshot.",
    )
    add(
        rows,
        experiment_id="MCTS-LEGACY-ROVER",
        component="posthoc_validation_needed",
        job_id="20559583",
        manifest_id="rover-on-534933607-stage2-mcts",
        domain="rover",
        value_head="on",
        seed="534933607",
        expected_pre_outage_state="terminal_out_of_memory",
        cpus=6,
        memory_gib=120,
        time_limit="3-00:00:00",
        artifact_kind="mcts_evaluation",
        evidence_source="experiment_tracking/approved_recovery_actions_20260901.csv",
        recovery_policy="deduplicate_retry_plans_and_posthoc_val_only",
        notes="Do not rerun inference merely to repair retry-duplicated VAL provenance.",
    )

    dependency_path = tracking / "dependency_controllers_20260831.csv"
    for item in read_rows(dependency_path):
        if item["state"] != "pending":
            continue
        add(
            rows,
            experiment_id=item["experiment_id"],
            component="policy_controller",
            job_id=item["job_id"],
            domain=item["domain"],
            value_head=item["value_head"],
            expected_pre_outage_state="dependency_pending",
            cpus=item["cpus"],
            memory_gib=item["ram_gib"],
            time_limit=item["time_limit"],
            artifact_kind="controller",
            evidence_source=dependency_path.relative_to(ROOT),
            recovery_policy="rerun_idempotently_only_if_dependency_chain_was_lost",
            notes=item["dependency_scope"],
        )

    add(
        rows,
        experiment_id="PRESERVE-3-TERM",
        component="policy_evaluation",
        job_id="20812836",
        manifest_id="zenotravel-off-operational-tail",
        domain="zenotravel",
        value_head="off",
        expected_pre_outage_state="operationally_held_requeued",
        cpus=10,
        memory_gib=20,
        time_limit="04:00:00",
        artifact_kind="policy_evaluation",
        evidence_source="experiment_tracking/approved_recovery_actions_20260901.csv",
        recovery_policy="inspect_then_release_or_idempotently_resubmit",
        notes="Not a scientific hold; environment-retrieval failure caused the requeue/hold.",
    )

    mprime_path = tracking / "mprime_validation_ipc_scale_v1" / "stage2_submissions.tsv"
    for item in read_rows(mprime_path, "\t"):
        job_id = item["slurm_job_id"]
        # Four validation-led tuning winners are already-completed training jobs.
        if int(job_id) < 20760292:
            continue
        add(
            rows,
            experiment_id=(
                "MAIN-EXT6-MPRIME" if item["branch"] == "validation_led"
                else "MAIN-TERM-EXT6-MPRIME"
            ),
            component="stage2_training",
            job_id=job_id,
            manifest_id=item["manifest_id"],
            domain="mprime",
            value_head=item["value_head"],
            seed=item["seed"],
            expected_pre_outage_state="deliberately_held",
            cpus=6,
            memory_gib=48,
            time_limit="3-00:00:00",
            artifact_kind="training",
            evidence_source=mprime_path.relative_to(ROOT),
            recovery_policy="remain_held_until_corrected_anchor_freeze",
            notes="Do not release during outage recovery.",
        )
    for experiment_id, job_id, count in (
        ("MAIN-EXT6-MPRIME", "20760692", 16),
        ("MAIN-TERM-EXT6-MPRIME", "20760693", 20),
    ):
        add(
            rows,
            experiment_id=experiment_id,
            component="policy_controller",
            job_id=job_id,
            expected_pre_outage_state="deliberately_held",
            cpus=2,
            memory_gib=1,
            time_limit="06:00:00",
            artifact_kind="controller",
            evidence_source="experiment_tracking/stage2_policy_controllers_20260831.csv",
            recovery_policy="remain_held_until_corrected_anchor_freeze",
            notes=f"Controller for {count} new Stage-2 training jobs.",
        )

    live_jobs_path = tracking / "live_jobs.csv"
    for item in read_rows(live_jobs_path):
        if item["experiment_id"] != "MCTS-RESOURCE-FO-HELD":
            continue
        add(
            rows,
            experiment_id="MCTS-LEGACY-FO-COUNTERS",
            component="stage2_mcts",
            job_id=item["job_id"],
            expected_pre_outage_state="deliberately_held",
            cpus=item["cpus"],
            memory_gib=str(item["memory"]).rstrip("G"),
            time_limit=item["time_limit"],
            artifact_kind="mcts_evaluation",
            evidence_source=live_jobs_path.relative_to(ROOT),
            recovery_policy="remain_held_until_explicit_reprioritization",
            notes=item["job_name"],
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        if row["job_id"] in seen:
            duplicates.append(row["job_id"])
        seen.add(row["job_id"])
    if duplicates:
        raise RuntimeError(f"duplicate job IDs in recovery inventory: {duplicates}")

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} frozen job records to {OUTPUT}")


if __name__ == "__main__":
    main()
