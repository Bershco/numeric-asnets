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
    status="live-tail-2241-of2260-replicates-58-of60-lineages-complete",
    scope="All 1,130 saved MPrime checkpoints x two independently frozen harder validation sets",
    results_file="experiment_tracking/mprime_validation_phase_b_20260906/phase_b_cell_summary_latest.csv",
    manifest_path="experiment_tracking/mprime_validation_phase_b_20260906/checkpoints.csv;experiment_tracking/mprime_validation_phase_b_20260906/frozen_validation_manifest.csv",
    next_action="Exact two-lineage tail array 21178405 is running; rebuild summaries at 2260/2260, inspect replicate agreement, then freeze validation-led checkpoints only.",
)
upsert(
    "ANCHOR-KL-CONTROL",
    status="live-adaptive-outlier-control-complete",
    next_action="Outlier adaptive job 21144388 remains live; coefficient is still 3 because post-update KL stays below the target. Evaluate comparable test checkpoints before any method claim.",
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
    "YARIN-EXTERNAL-GENERATORS",
    display_name="External generator distribution-shift screen",
    role="validation-robustness",
    status="completed-three-seed-screen",
    scope="FO Counters and Rover x two VH modes x three validation-selected Stage-1 checkpoints",
    primary_question="Do conclusions survive frozen instances from an independently implemented generator?",
    configuration_summary="Policy-only evaluation; same checkpoints; 20 frozen instances per domain; Yarin generator revision b64e5d086117ebd5c1d53fbe9a9d93ae609a59fb",
    results_file="experiment_tracking/advisor_followup_20260910/yarin_external_results_latest.csv",
    manifest_path="experiment_tracking/advisor_followup_20260910/yarin_external_screen_manifest.csv",
    next_action="FO improves descriptively while Rover collapses under the sparser external topology. Treat as n=3 distribution-shift evidence, not a confirmatory test.",
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

registry.sort(key=lambda row: row["experiment_id"])
write(registry_path, registry, fields)
write(TRACK / "experiments.csv", registry, fields)

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
for path in sorted(TRACK.rglob("*.csv")):
    rel = path.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    duplicate_groups[digest].append(rel)
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            row_count = sum(1 for _ in reader)
        parse_state = "ok"
    except Exception as exc:  # pragma: no cover - audit path
        header, row_count, parse_state = [], 0, f"parse_error:{type(exc).__name__}"
    lower = [field.lower() for field in header]
    result_like = any(any(token in field for token in result_tokens) for field in lower)
    provenance_fields = [field for field in header if any(token in field.lower() for token in provenance_tokens)]
    current = ("latest" in path.name or "advisor_followup_20260910" in rel
               or path.name in {"experiments.csv", "experiment_registry.csv"})
    csv_rows.append({
        "csv_path": rel,
        "rows": row_count,
        "classification": "current_canonical" if current else "historical_or_supporting",
        "result_like": int(result_like),
        "provenance_fields": ";".join(provenance_fields),
        "provenance_status": "inline_or_pointer_present" if provenance_fields else (
            "not_a_result_table" if not result_like else "legacy_gap_not_primary"),
        "parse_state": parse_state,
        "sha256": digest,
    })
write(TRACK / "result_csv_provenance_index_latest.csv", csv_rows)

duplicate_rows = []
for digest, paths in duplicate_groups.items():
    if len(paths) > 1:
        duplicate_rows.append({
            "sha256": digest,
            "copies": len(paths),
            "paths": ";".join(paths),
            "disposition": "retain if dated snapshot; canonical readers must use the *_latest or master path",
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
| Byte-identical duplicate audit | `experiment_tracking/documentation_redundancy_audit_latest.csv` |

## Retention rule

- Keep dated snapshots because they prove what was known at a particular time.
- Keep one current alias for each changing concept (`*_latest` or the master path).
- Do not cite an old `status_YYYYMMDD` file as current state.
- Current statistical tables must contain direct job/log fields or point to a
  seed-level ledger that contains them.
"""
(TRACK / "documentation_index_latest.md").write_text(index, encoding="utf-8")

print(f"registry={len(registry)} csv_audit={len(csv_rows)} duplicates={len(duplicate_rows)}")
