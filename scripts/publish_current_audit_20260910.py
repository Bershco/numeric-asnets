#!/usr/bin/env python3
"""Refresh current experiment/catalog documentation without rewriting history."""

from __future__ import annotations

import csv
import hashlib
import re
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


def task_count(job_id: str) -> int:
    """Count concrete Slurm tasks represented by a compact job-id cell."""
    total = 0
    for token in job_id.split(";"):
        match = re.search(r"_\[([^]]+)\]$", token.strip())
        if not match:
            total += 1
            continue
        for part in match.group(1).split(","):
            bounds = part.split("-")
            total += int(bounds[-1]) - int(bounds[0]) + 1
    return total


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
    status="completed-2260-of2260-60-of60-lineages",
    scope="All 1,130 saved MPrime checkpoints x two independently frozen harder validation sets",
    primary_question="Does validation rank saved checkpoints reliably without saturating?",
    results_file="experiment_tracking/mprime_validation_phase_b_20260906/phase_b_cell_summary_latest.csv",
    manifest_path="experiment_tracking/mprime_validation_phase_b_20260906/checkpoints.csv;experiment_tracking/mprime_validation_phase_b_20260906/protocol.json;problems/numeric/mprime/validation_ipc_scale_v1/manifest.csv",
    next_action="Phase B removed saturation but Stage2 validation-test rank agreement remains weak; do not repeat the same full design. If MPrime continues, prepare a structurally redesigned candidate-limited Phase C.",
)
upsert(
    "MPRIME-VAL-ADEQUACY-C",
    status="completed-347-of347",
    results_file="experiment_tracking/mprime_validation_phase_c_20260911/validator_decision_latest.csv",
    manifest_path="experiment_tracking/mprime_validation_phase_c_20260911/checkpoint_candidates.csv",
    next_action="Phase-B replicate A had the best available rank/stability trade-off and is frozen as final MPrime validator; Phase C remains methodological evidence.",
)
upsert(
    "MAIN-EXT6-MPRIME",
    status="completed-policy-with-validation-audit",
    primary_question="Do MAIN-VAL policy conclusions extend from the original five imperfect domains to MPrime as a sixth?",
    held_reason="Phase B shows that both the old and harder validation distributions rank Stage-2 checkpoints weakly",
    next_action="Retain completed policy results as provisional extension. If MPrime continues, use a structurally redesigned candidate-limited validation audit before selecting further checkpoints.",
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
    status="completed-no-controller-activation",
    results_file="experiment_tracking/anchor_kl_control_summary_latest.csv",
    next_action="Both adaptive jobs completed 100 updates but coefficient remained 3 throughout; the adaptive treatment never activated.",
)
upsert(
    "MCTS-PW70-CROSS-DOMAIN",
    status="completed",
    next_action="Corrected PW70 screening is terminal; use the ten-seed FO Counters/Rover extension for confirmatory conclusions.",
)
upsert(
    "MCTS-PW70-TEN-SEED",
    status="completed-declared-budget",
    next_action="Freeze all ten declared-budget seeds. FO/off is final at 8.4/20; one 7/20 seed is an OOM-terminal declared-budget endpoint whose unclassified instance counts unsuccessful.",
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
    status="completed-both-vh-arms",
    scope="Three exact validation-led Stage-2 Counters instances under VH-off failure plus a VH-on behavior arm",
    primary_question="Does 20-visit root selection override a good policy because visit evidence is too coarse?",
    configuration_summary="Narrow 5 children/20 simulations; one worker; 6h per instance; action, prior, visit, Q and U traces",
    results_file="experiment_tracking/advisor_followup_20260910/counters_visit_audit_latest_milestones.csv",
    manifest_path="experiment_tracking/advisor_followup_20260910/counters_visit_audit_manifest.csv",
    next_action="Both arms complete; VH-off first divergences at steps 881/993/1105 have tied9-visit maxima and equal Q; the VH-on arm is not a positive control because its policy solved none of the targets.",
)
upsert(
    "MCTS-COUNTERS-TIEBREAK-3WAY",
    status="completed-targeted-causal-screen",
    scope="Validation-led Stage2 Counters VH-off selected failures plus a VH-on behavior arm x three tie-break rules",
    results_file="experiment_tracking/counters_tie_break_3way_20260911/results_latest.csv",
    manifest_path="experiment_tracking/advisor_followup_20260910/counters_tie_break_3way_manifest.csv",
    next_action="VH-off actionID0/3 Q0/3 policy3/3 by2h with all plans VAL-valid; VH-on0/3 is not a positive control because its policy solved none of the targets.",
)
upsert(
    "MCTS-COUNTERS-TIEBREAK-STRICT",
    status="live-16-running-4-complete",
    results_file="experiment_tracking/advisor_followup_20260910/live_submission_update_20260913.md",
    manifest_path="experiment_tracking/counters_tie_break_strict_stage1_20260913/manifest.csv",
    next_action="Two completed matched pairs are neutral:59/59 and20/59 under both rules; wait for all ten pairs before changing any RQ score.",
)
upsert(
    "MCTS-BG-TIEBREAK-SCREEN",
    status="live-two-running-three-of-four-classified",
    results_file="experiment_tracking/block_grouping_tie_break_screen_20260913/README.md",
    manifest_path="experiment_tracking/block_grouping_tie_break_screen_20260913/manifest.csv",
    next_action="Both same-build rules are0/3 ordinary-unsolved on the first three targets with the fourth active; expand only if a selected failure is rescued through an observed visit tie.",
)
upsert(
    "MPRIME-ANCHOR-PBA",
    status="live-466-of588",
    results_file="experiment_tracking/mprime_anchor_phase_b_a_20260912/README.md",
    manifest_path="experiment_tracking/mprime_anchor_phase_b_a_20260912/manifest.csv",
    next_action="Six recovery tasks running and original attempts out of queue; controller21233927 skips completed points and finalizes only after manual curve review.",
)
upsert(
    "MPRIME-PBA-S1-MCTS",
    status="live-20-running-no-terminal-seeds",
    results_file="experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913/README.md",
    manifest_path="experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913/manifest_off.csv;experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913/manifest_on.csv",
    next_action="All20 canonical checkpoints are running; current success lower bounds are at least6.8/20 off and7.2/20 on with no CI or inference until seeds terminate.",
)
upsert(
    "MPRIME-PBA-S1-PW-SCREEN",
    display_name="MPrime final-validator Stage1 PW70 screen",
    role="search-screen",
    status="immediate-next-not-submitted",
    scope="Two predeclared Phase-B-A Stage1 seeds x two VH modes",
    primary_question="Does PW70 retain the MPrime fixed-search benefit with lower search cost?",
    configuration_summary="PW70 Kmin3; two matched seeds per VH; normal fixed comparator; 6h per instance; planned4 tasks",
    results_file="experiment_tracking/advisor_followup_20260910/live_submission_update_20260913.md",
    manifest_path="",
    next_action="Prepare and smoke-test four-task matched screen after the running fixed comparator yields interpretable reference scores; not a Stage2 dependency.",
)
upsert(
    "FO-S2-VAL-RECOVERY",
    display_name="FO Counters validation-led Stage-2 exact completion",
    role="operational-completion",
    status="completed-exact",
    scope="Only the unclassified instances from three censored evaluation identities",
    primary_question="Complete the primary validation-led FO Counters Stage-2 MCTS cells without duplicating classified instances.",
    configuration_summary="Normal 20 children/70 simulations; 3 workers; 120 GiB; 6h per instance; minimal exact-instance recovery",
    results_file="experiment_tracking/advisor_followup_20260910/fo_stage2_validation_vh_off_exact_seed_results_20260913.csv",
    manifest_path="experiment_tracking/advisor_followup_20260910/fo_stage2_validation_partial_recovery_manifest.csv",
    next_action="Final VH-off mean is6.1/20 at30m2h6h versus policy2.9; delta+3.2 CI[1.74 4.66] rawp.00195 Holmp.00977.",
)
upsert(
    "MAIN-VAL-S2-MCTS",
    status="completed-five-domain",
    scope="Validation-led Stage-2 policy versus MCTS across the five primary domains",
    next_action="All five-domain rows are exact and final; MPrime enters only after corrected Stage2 endpoints.",
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
live_experiment_map = {
    "COUNTERS-VISIT-AUDIT": ["Counters root-visit distribution audit"],
    "FO-S2-VAL-RECOVERY": ["FO Counters validation-led Stage-2 exact recovery"],
    "MAIN-VAL-S2-MCTS": ["FO Counters validation-led Stage-2 exact recovery"],
    "MCTS-COUNTERS-TIEBREAK-STRICT": ["Counters Stage1 strict tie-break confirmation"],
    "MCTS-BG-TIEBREAK-SCREEN": ["Block Grouping selected-failure tie-break screen"],
    "MPRIME-ANCHOR-PBA": ["MPrime Phase-B-A anchor rescore original tasks", "MPrime Phase-B-A anchor rescore recovery tasks", "MPrime Phase-B-A rescore controller"],
    "MPRIME-PBA-S1-MCTS": ["MPrime Phase-B-A Stage1 fixed MCTS"],
}
catalog = []
for row in registry:
    live_names = live_experiment_map.get(row["experiment_id"], [])
    matches = [job for job in workload if job["experiment"] in live_names]
    catalog.append({
        "experiment_id": row["experiment_id"],
        "display_name": row["display_name"],
        "role": row["role"],
        "status": row["status"],
        "primary_question": row["primary_question"],
        "configuration_summary": row["configuration_summary"],
        "results_file": row["results_file"],
        "manifest_path": row["manifest_path"],
        "live_jobs": sum(task_count(job["job_id"]) for job in matches),
        "live_cpus": sum(int(job["cpus"]) for job in matches),
        "live_memory_gib": sum(float(job["memory_gib"]) for job in matches),
        "next_action": row["next_action"],
    })
write(TRACK / "experiment_catalog_latest.csv", catalog)

live_rows = []
for experiment in sorted({row["experiment"] for row in workload}):
    jobs = [row for row in workload if row["experiment"] == experiment]
    running_tasks = sum(task_count(row["job_id"]) for row in jobs if row["state"] == "RUNNING")
    pending_tasks = sum(task_count(row["job_id"]) for row in jobs if row["state"] == "PENDING")
    live_rows.append({
        "snapshot_time_idt": jobs[0]["snapshot_time_idt"],
        "experiment": experiment,
        "running": running_tasks,
        "pending": pending_tasks,
        "jobs": running_tasks + pending_tasks,
        "requested_cpus": sum(int(row["cpus"]) for row in jobs),
        "requested_memory_gib": sum(float(row["memory_gib"]) for row in jobs),
        "job_ids": ";".join(row["job_id"] for row in jobs),
        "row_level_provenance": "experiment_tracking/cluster_workload_latest.csv",
    })
write(TRACK / "live_experiment_status_latest.csv", live_rows)
write(TRACK / "cluster_workload_summary_latest.csv", [
    {
        "snapshot_time_idt": row["snapshot_time_idt"],
        "experiment": row["experiment"],
        "state": "RUNNING" if row["running"] and not row["pending"] else (
            "PENDING" if row["pending"] and not row["running"] else "RUNNING_PLUS_PENDING"
        ),
        "jobs": row["jobs"],
        "requested_cpus": row["requested_cpus"],
        "requested_memory_gib": row["requested_memory_gib"],
        "scheduler_source": "experiment_tracking/cluster_workload_latest.csv",
    }
    for row in live_rows
])

# These two legacy *_latest ledgers are retained for row-level provenance, but
# must not masquerade as the current queue/branch status.
dynamic_jobs = read(TRACK / "dynamic_experiment_jobs_latest.csv")
for row in dynamic_jobs:
    row["record_scope"] = "historical observation; state is not current"
    row["current_live_state_source"] = "experiment_tracking/cluster_workload_latest.csv"
write(TRACK / "dynamic_experiment_jobs_latest.csv", dynamic_jobs)

branch_coverage = read(TRACK / "stage2_mcts_branch_coverage_latest.csv")
for row in branch_coverage:
    row["snapshot_time_idt"] = STAMP
    row["validation_scheduler_terminal"] = "10"
    row["validation_live"] = "0"
    row["validation_unsubmitted"] = "0"
    row["terminal_scheduler_terminal"] = ""
    row["terminal_live"] = ""
    row["terminal_unsubmitted"] = ""
    row["scientific_status"] = "validation-led primary complete; terminal-led columns archived and not maintained"
    row["row_level_provenance"] = "experiment_tracking/advisor_followup_20260910/rq_primary_validation_led.csv;experiment_tracking/result_csv_provenance_index_latest.csv"
write(TRACK / "stage2_mcts_branch_coverage_latest.csv", branch_coverage)

rq_results = read(TRACK / "rq_results_latest.csv")
for row in rq_results:
    if row["research_question"] == "RQ2":
        row["status"] = "complete_original_five_domain_family"
        row["headline"] = "FO Counters gains significantly at both stages; Drone/Rover improve modestly; Block Grouping is budget-sensitive and Counters can be harmed."
    elif row["research_question"] == "RQ4":
        row["status"] = "complete_original_five_domain_family"
        row["headline"] = "Drone has the clear positive value-head interaction; FO benefits from search in both modes without a significant interaction."
write(TRACK / "rq_results_latest.csv", rq_results)

for legacy_latest in (
    TRACK / "best_configuration_by_domain_latest.csv",
    TRACK / "stage2_policy_mcts_all_cutoff_statistics_latest.csv",
):
    rows = read(legacy_latest)
    for row in rows:
        row["record_scope"] = "historical mixed-branch snapshot; superseded for primary inference"
        row["canonical_replacement"] = "experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md"
    write(legacy_latest, rows)

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
| RQ2 raw levels | `experiment_tracking/advisor_followup_20260910/rq2_raw_means_validation_led.csv` |
| RQ3 raw levels and interaction | `experiment_tracking/advisor_followup_20260910/rq3_raw_means_validation_led.csv` |
| RQ4 raw levels | `experiment_tracking/advisor_followup_20260910/rq4_raw_means_validation_led.csv` |
| Advisor narrative and tables | `experiment_tracking/advisor_followup_20260910/README.md` |
| Historical job-level evidence (not live state) | `experiment_tracking/dynamic_experiment_jobs_latest.csv` |
| MPrime Phase-B checkpoint evidence | `experiment_tracking/mprime_validation_phase_b_20260906/phase_b_checkpoint_scores_latest.csv` |
| MPrime Phase-B selector comparison | `experiment_tracking/mprime_validation_phase_b_20260906/phase_b_cell_selector_comparison_latest.csv` |
| Counters visit milestones | `experiment_tracking/advisor_followup_20260910/counters_visit_audit_latest_milestones.csv` |
| FO recovery progress | `experiment_tracking/advisor_followup_20260910/fo_stage2_validation_recovery_progress_latest.csv` |
| Adaptive-KL summary | `experiment_tracking/anchor_kl_control_summary_latest.csv` |
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

- MPrime Phase B: complete at 2,260/2,260 checkpoint-replicates and 60/60
  lineages. Harder validation removed saturation but Stage-2 rank agreement with
  test remains weak.
- Adaptive KL: both arms completed 100 updates. Neither changed coefficient 3,
  so the adaptive treatment never activated.
- Counters visit audit: both arms are complete. The first VH-off divergences
  occur after 881–1,105 actions under tied visit maxima and equal Q values, not
  at the first action. The VH-on behavior arm is not a positive control because
  its policy solved none of the three targets.
- FO Counters validation-led Stage-2 MCTS is complete. The exact VH-off mean is
  6.1/20 at every cutoff versus policy 2.9/20; the final recovery instance used
  its full six-hour allowance and did not add a success.

## Canonical sources

- Scheduler rows: `experiment_tracking/cluster_workload_latest.csv`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
- RQ statistics: `experiment_tracking/advisor_followup_20260910/rq_primary_validation_led.csv`
- Provenance audit: `experiment_tracking/result_csv_provenance_index_latest.csv`
"""
(TRACK / "status_latest.md").write_text(status, encoding="utf-8")

print(f"registry={len(registry)} csv_audit={len(csv_rows)} duplicates={len(duplicate_rows)}")
