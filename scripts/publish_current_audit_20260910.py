#!/usr/bin/env python3
"""Refresh current experiment/catalog documentation without rewriting history."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
STAMP = datetime.now().astimezone().isoformat(timespec="seconds")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


registry_path = TRACK / "experiment_registry.csv"
registry = read(registry_path)
fields = list(registry[0])
by_id = {row["experiment_id"]: row for row in registry}


def upsert(experiment_id: str, **values: str) -> None:
    row = by_id.get(experiment_id)
    if row is None:
        row = {field: "" for field in fields}
        row["experiment_id"] = experiment_id
        registry.append(row)
        by_id[experiment_id] = row
    row.update(values)


upsert(
    "MPRIME-VAL-ADEQUACY",
    status="live-tail-2243-of2260-replicates-58-of60-lineages-complete",
    scope="All 1,130 saved MPrime checkpoints x two independently frozen harder validation sets",
    results_file="experiment_tracking/mprime_validation_phase_b_20260906/phase_b_cell_summary_latest.csv",
    manifest_path="experiment_tracking/mprime_validation_phase_b_20260906/checkpoints.csv;experiment_tracking/mprime_validation_phase_b_20260906/protocol.json;problems/numeric/mprime/validation_ipc_scale_v1/manifest.csv",
    next_action="Exact two-lineage tail array 21178598 is running after the import-path repair; rebuild summaries at 2260/2260, inspect replicate agreement, then freeze validation-led checkpoints only.",
)
upsert(
    "MCTS-PW",
    manifest_path="experiment_tracking/mcts_progressive_widening_pilot/results.csv",
)
upsert(
    "PRESERVE-3-VAL",
    manifest_path="experiment_tracking/four_domain_preservation/stable_domain_stage2_seed_pairs_20260902.csv",
)
upsert(
    "ANCHOR-KL-CONTROL",
    status="live-outlier-98-of100-updates-control-complete",
    next_action="Outlier adaptive job 21144388 has 98/100 logged updates; coefficient is still 3 because post-update KL stays below the target. Evaluate comparable test checkpoints before any method claim.",
)
upsert(
    "MCTS-PW70-CROSS-DOMAIN",
    status="completed",
    next_action="Corrected PW70 screening is terminal; use the ten-seed FO Counters/Rover extension for confirmatory conclusions.",
)
upsert(
    "MCTS-PW70-TEN-SEED",
    status="completed-declared-budget-with-one-starred-partial",
    next_action="Retain all ten declared-budget seeds. FO/off is 8.4/20 with one 7/20 OOM-partial seed; report the complete-allocation n=9 sensitivity separately.",
)
upsert(
    "MCTS-LEGACY-FO",
    status="completed-terminal-led-archived",
    next_action="No live terminal-led tail remains. Preserve the declared-budget result for sensitivity only; primary analysis is validation-led.",
)
upsert(
    "MCTS-STAGE2-BRANCH-COMPLETION",
    status="completed-terminal-led-archive",
    scope="Archived terminal-led Stage-2 MCTS completion evidence; no live primary work",
    next_action="All historical branch-completion allocations have left Slurm. Current primary FO validation-led partial completion is tracked separately as FO-S2-VAL-RECOVERY.",
)
upsert(
    "YARIN-EXTERNAL-GENERATORS",
    display_name="External generator distribution-shift screen",
    role="validation-robustness",
    status="completed-three-seed-screen",
    scope="FO Counters and Rover x two VH modes x three validation-selected Stage-1 checkpoints",
    primary_question="Do conclusions survive frozen instances from an independently implemented generator?",
    configuration_summary="Policy-only evaluation; same checkpoints; 20 frozen instances per domain; Yarin generator revision b64e5d086117ebd5c1d53fbe9a9d93ae609a59fb",
    results_file="experiment_tracking/advisor_followup_20260910/yarin_external_results_latest.csv",
    manifest_path="experiment_tracking/advisor_followup_20260910/yarin_external_screen_manifest.csv",
    next_action="FO improves descriptively while Rover collapses on the smaller external graphs. Treat as n=3 distribution-shift evidence, not a confirmatory test; topology alone is not established as the cause.",
)
upsert(
    "COUNTERS-VISIT-AUDIT",
    display_name="Counters root visit-distribution audit",
    role="search-diagnostic",
    status="live-two-jobs",
    scope="Three exact validation-led Stage-2 Counters instances under VH-off failure and matched VH-on control",
    primary_question="Does 20-visit root selection override a good policy because visit evidence is too coarse?",
    configuration_summary="Narrow 5 children/20 simulations; one worker; 6h per instance; action, prior, visit, Q and U traces",
    results_file="experiment_tracking/advisor_followup_20260910/counters_visit_audit_manifest.csv",
    manifest_path="experiment_tracking/advisor_followup_20260910/counters_visit_audit_manifest.csv",
    next_action="Jobs 21178320 and 21178321 are live; summarize exact 30m/2h/6h root-decision distributions after completion.",
)
upsert(
    "FO-S2-VAL-RECOVERY",
    display_name="FO Counters validation-led Stage-2 exact completion",
    role="operational-completion",
    status="live-three-minimal-recovery-jobs",
    scope="Only 42 unclassified instances from three genuinely partial evaluation identities",
    primary_question="Complete the primary validation-led FO Counters Stage-2 MCTS cells without duplicating classified instances.",
    configuration_summary="Normal 20 children/70 simulations; 3 workers; 120 GiB; 6h per instance; 36h allocation",
    results_file="experiment_tracking/advisor_followup_20260910/fo_stage2_validation_partial_recovery_manifest.csv",
    manifest_path="experiment_tracking/advisor_followup_20260910/fo_stage2_validation_partial_recovery_manifest.csv",
    next_action="Jobs 21178377, 21178379 and 21178380 are live after three 90-second signature-check failures were corrected.",
)
upsert(
    "MAIN-VAL-S2-MCTS",
    status="live-fo-exact-recovery",
    scope="Validation-led Stage-2 policy versus MCTS across the five primary domains; FO Counters has three censored identities under exact recovery",
    next_action="Use jobs 21178377/21178379/21178380 to complete only the 42 unclassified FO instances, then rebuild final five-domain RQ2/RQ4 inference.",
)

registry.sort(key=lambda row: row["experiment_id"])
write(registry_path, registry, fields)
write(TRACK / "experiments.csv", registry, fields)

# Close lifecycle drift in submitted manifests and keep legacy IDs explicitly
# joinable to the canonical registry.
external_manifest = TRACK / "advisor_followup_20260910" / "yarin_external_screen_manifest.csv"
external_rows = read(external_manifest)
for row in external_rows:
    row["experiment_id"] = "YARIN-EXTERNAL-GENERATORS"
    row["submission_state"] = "completed"
    row["source_result_row"] = "experiment_tracking/advisor_followup_20260910/yarin_external_results_latest.csv"
write(external_manifest, external_rows)

publication = TRACK / "advisor_followup_20260910" / "reproducibility_publication_manifest.csv"
publication_rows = read(publication)
for row in publication_rows:
    if row.get("artifact") == "External-generator screen results":
        row["publication_state"] = "completed_ready"
write(publication, publication_rows)

write(TRACK / "experiment_id_aliases.csv", [
    {"historical_id": "EXTERNAL-GENERATOR-SCREEN", "canonical_id": "YARIN-EXTERNAL-GENERATORS", "reason": "pre-registry screen name"},
    {"historical_id": "MCTS-HORIZON-BINDING", "canonical_id": "MCTS-HORIZON", "reason": "binding-horizon manifest name"},
    {"historical_id": "MCTS-SAFE-CONTEXT-DIAG", "canonical_id": "MCTS-SAFE-CONTEXT", "reason": "diagnostic subcampaign name"},
])

# This historically named *_latest file is retained for its mixed two-/five-/
# ten-seed screen schema, but no row may continue to claim that work is live.
pw_live = TRACK / "mcts_progressive_widening_cross_domain" / "pw70_live_summary_latest.csv"
pw_rows = read(pw_live)
for row in pw_rows:
    if "live" in row.get("status", "") or "lower_bound" in row.get("status", ""):
        row["status"] = "historical_screen_superseded"
        row["conclusion"] = "Superseded by the terminal result tables; do not use this row for current inference"
        row["row_level_provenance"] = "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_ten_seed_statistics_latest.csv"
write(pw_live, pw_rows)

# Build a current catalog by joining the master registry to the scheduler snapshot.
workload = read(TRACK / "cluster_workload_latest.csv")
catalog = []
for row in registry:
    label = row["display_name"].lower()
    matches = [job for job in workload if row["experiment_id"].lower() in job["experiment"].lower()
               or any(token in job["experiment"].lower() for token in label.split()[:2])]
    catalog.append({
        "experiment_id": row["experiment_id"],
        "display_name": row["display_name"],
        "role": row["role"],
        "status": row["status"],
        "primary_question": row["primary_question"],
        "configuration_summary": row["configuration_summary"],
        "results_file": row["results_file"],
        "manifest_path": row["manifest_path"],
        "live_jobs": len(matches),
        "live_cpus": sum(int(job["cpus"]) for job in matches),
        "live_memory_gib": sum(float(job["memory_gib"]) for job in matches),
        "next_action": row["next_action"],
    })
write(TRACK / "experiment_catalog_latest.csv", catalog)

live_rows = []
for experiment in sorted({row["experiment"] for row in workload}):
    jobs = [row for row in workload if row["experiment"] == experiment]
    live_rows.append({
        "snapshot_time_idt": jobs[0]["snapshot_time_idt"],
        "experiment": experiment,
        "running": sum(row["state"] == "RUNNING" for row in jobs),
        "pending": sum(row["state"] == "PENDING" for row in jobs),
        "jobs": len(jobs),
        "requested_cpus": sum(int(row["cpus"]) for row in jobs),
        "requested_memory_gib": sum(float(row["memory_gib"]) for row in jobs),
        "job_ids": ";".join(row["job_id"] for row in jobs),
        "row_level_provenance": "experiment_tracking/cluster_workload_latest.csv",
    })
write(TRACK / "live_experiment_status_latest.csv", live_rows)

# Add explicit drill-down pointers to the two canonical aggregate tables whose
# rows are intentionally not job-level.  Their referenced detail files contain
# the original job/checkpoint/log identities.
for aggregate, provenance in (
    (TRACK / "advisor_followup_20260910" / "generator_distribution_summary.csv",
     "experiment_tracking/advisor_followup_20260910/generator_distribution_instances.csv"),
    (TRACK / "mprime_validation_phase_b_20260906" / "phase_b_cell_summary_latest.csv",
     "experiment_tracking/mprime_validation_phase_b_20260906/phase_b_lineage_summary_latest.csv;experiment_tracking/mprime_validation_phase_b_20260906/phase_b_checkpoint_scores_latest.csv"),
):
    values = read(aggregate)
    for value in values:
        value["row_level_provenance"] = provenance
    write(aggregate, values)

# Canonical result/provenance inventory. Historical dated snapshots remain immutable.
csv_rows = []
duplicate_groups: dict[str, list[str]] = defaultdict(list)
provenance_tokens = ("job", "log", "path", "checkpoint", "manifest", "source", "provenance", "ledger")
result_tokens = ("score", "success", "coverage", "effect", "mean", "p_value", "raw_p", "holm_p", "runtime")
supplemental_manifests = [
    ROOT / "problem_generator/frozen_external_yarin/frozen_manifest.csv",
    ROOT / "problems/numeric/mprime/validation_ipc_scale_v1/manifest.csv",
    ROOT / "problem_generator/generated_validation_instances/final_validation_manifest.csv",
]
csv_paths = sorted(set(TRACK.rglob("*.csv")) | {path for path in supplemental_manifests if path.exists()})
for path in csv_paths:
    rel = path.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    duplicate_groups[digest].append(rel)
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            dict_reader = csv.DictReader(stream)
            header = dict_reader.fieldnames or []
            parsed_rows = list(dict_reader)
            row_count = len(parsed_rows)
        parse_state = "ok"
    except Exception as exc:  # pragma: no cover - audit path
        header, row_count, parse_state = [], 0, f"parse_error:{type(exc).__name__}"
    lower = [field.lower() for field in header]
    result_like = any(any(token in field for token in result_tokens) for field in lower)
    provenance_fields = [field for field in header if any(token in field.lower() for token in provenance_tokens)]
    populated_provenance_rows = sum(
        any(str(row.get(field, "")).strip() for field in provenance_fields)
        for row in parsed_rows
    ) if parse_state == "ok" else 0
    current = ("latest" in path.name or "advisor_followup_20260910" in rel
               or path.name in {"experiments.csv", "experiment_registry.csv"})
    csv_rows.append({
        "csv_path": rel,
        "rows": row_count,
        "classification": "current_canonical" if current else "historical_or_supporting",
        "result_like": int(result_like),
        "provenance_fields": ";".join(provenance_fields),
        "populated_provenance_rows": populated_provenance_rows,
        "provenance_status": "inline_or_pointer_present" if provenance_fields and populated_provenance_rows == row_count else (
            "partial_or_empty_provenance" if provenance_fields else (
            "not_a_result_table" if not result_like else "legacy_gap_not_primary"),
        ),
        "parse_state": parse_state,
        "sha256": digest,
    })
write(TRACK / "result_csv_provenance_index_latest.csv", csv_rows)

reference_rows = []
for row in registry:
    for field in ("results_file", "manifest_path"):
        tokens = [token.strip() for token in row.get(field, "").split(";") if token.strip()]
        if not tokens:
            reference_rows.append({
                "experiment_id": row["experiment_id"], "field": field,
                "reference": "", "state": "not_declared",
            })
        for token in tokens:
            if token.startswith(("http://", "https://", "/home/")):
                state = "remote_reference"
            else:
                state = "exists_local" if (ROOT / token).exists() else "missing_local"
            reference_rows.append({
                "experiment_id": row["experiment_id"], "field": field,
                "reference": token, "state": state,
            })
write(TRACK / "registry_reference_audit_latest.csv", reference_rows)

duplicate_rows = []
for digest, paths in duplicate_groups.items():
    if len(paths) > 1:
        registry_mirror = set(paths) == {
            "experiment_tracking/experiment_registry.csv",
            "experiment_tracking/experiments.csv",
        }
        duplicate_rows.append({
            "sha256": digest,
            "copies": len(paths),
            "paths": ";".join(paths),
            "disposition": (
                "experiments.csv is a generated compatibility mirror; edit only experiment_registry.csv"
                if registry_mirror else
                "retain if dated snapshot; canonical readers must use the *_latest or master path"
            ),
        })
write(TRACK / "documentation_redundancy_audit_latest.csv", duplicate_rows,
      ["sha256", "copies", "paths", "disposition"])

index = f"""# Canonical experiment-documentation index

Updated: {STAMP}

This index resolves the apparent duplication created by dated audit snapshots.
Dated files are immutable historical evidence, not current status.  New analysis
must read the canonical files below; it must not infer liveness from an old date.

| Purpose | Canonical file |
|---|---|
| Current Slurm jobs | `experiment_tracking/cluster_workload_latest.csv` |
| Current workload totals | `experiment_tracking/cluster_workload_summary_latest.csv` |
| Master experiment registry | `experiment_tracking/experiment_registry.csv` |
| Joined current catalog | `experiment_tracking/experiment_catalog_latest.csv` |
| RQ-separated primary statistics | `experiment_tracking/advisor_followup_20260910/rq_primary_validation_led.csv` |
| Advisor narrative and tables | `experiment_tracking/advisor_followup_20260910/README.md` |
| Dynamic job evidence | `experiment_tracking/dynamic_experiment_jobs_latest.csv` |
| MPrime Phase-B checkpoint evidence | `experiment_tracking/mprime_validation_phase_b_20260906/phase_b_checkpoint_scores_latest.csv` |
| CSV provenance audit | `experiment_tracking/result_csv_provenance_index_latest.csv` |
| Registry reference integrity | `experiment_tracking/registry_reference_audit_latest.csv` |
| Historical experiment-ID aliases | `experiment_tracking/experiment_id_aliases.csv` |
| Byte-identical duplicate audit | `experiment_tracking/documentation_redundancy_audit_latest.csv` |

## Retention rule

- Keep dated snapshots because they prove what was known at a particular time.
- Keep one current alias for each changing concept (`*_latest` or the master path).
- Do not cite an old `status_YYYYMMDD` file as current state.
- Current statistical tables must contain direct job/log fields or point to a
  seed-level ledger that contains them.
"""
(TRACK / "documentation_index_latest.md").write_text(index, encoding="utf-8")

workload_lines = [
    f"| {row['experiment']} | "
    f"{'running' if row['running'] else 'pending'} | {row['jobs']} | "
    f"{row['requested_cpus']} | {row['requested_memory_gib']} GiB |"
    for row in live_rows
]
status = f"""# Current experiment status

Updated: {workload[0]['snapshot_time_idt'] if workload else STAMP}

This is the only canonical changing Markdown status page. Dated status files are
historical snapshots. Full RQ tables, methods and conclusions are in
`experiment_tracking/advisor_followup_20260910/README.md`.

## Live workload

| Experiment | State | Jobs | CPU | RAM |
|---|---|---:|---:|---:|
{chr(10).join(workload_lines)}

## Current scientific endpoints

- MPrime Phase B: 2,243/2,260 checkpoint-replicates and 58/60 complete
  lineages; repaired exact two-lineage tail `21178598[15,42]` is running.
- Adaptive KL: stable control complete; outlier has 98/100 logged updates and
  has not changed its coefficient from 3.
- Counters visit audit: two jobs are running on the exact three-instance failure
  set and matched VH-on control.
- FO Counters validation-led Stage-2 MCTS: three minimal jobs are running only
  the 42 instances left unclassified by three historical partial allocations.

## Canonical sources

- Scheduler rows: `experiment_tracking/cluster_workload_latest.csv`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
- RQ statistics: `experiment_tracking/advisor_followup_20260910/rq_primary_validation_led.csv`
- Provenance audit: `experiment_tracking/result_csv_provenance_index_latest.csv`
"""
(TRACK / "status_latest.md").write_text(status, encoding="utf-8")

print(f"registry={len(registry)} csv_audit={len(csv_rows)} duplicates={len(duplicate_rows)}")
