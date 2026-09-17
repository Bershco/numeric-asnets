#!/usr/bin/env python3
"""Freeze the local Stage-0 policy/MCTS divergence evidence join.

This script never reads the cluster.  It normalizes the locally available
instance-level fixed/PW outcomes, overlays the existing first-divergence
diagnostics, and emits an exact missing-strata manifest for a later compact
recorder campaign.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
SHARED_ROOT = WORKTREE.parents[1]


def locate(relative: str) -> Path:
    candidates = (WORKTREE / relative, SHARED_ROOT / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing Stage-0 input {relative!r}; tried {candidates}")


def read_csv(relative: str) -> tuple[Path, list[dict[str, str]]]:
    path = locate(relative)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return path, list(csv.DictReader(handle))


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path = HERE / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def basename(value: str) -> str:
    return os.path.basename(value.replace("\\", "/"))


def checkpoint_identity(log_path: str, fallback_epoch: str = "") -> str:
    match = re.search(r"src(\d+)_e(\d+)", log_path or "")
    if match:
        return f"src{match.group(1)}_e{int(match.group(2)):04d}"
    if fallback_epoch:
        return f"seed-selected-e{int(fallback_epoch):04d}"
    return "unknown"


def outcome(policy_success: bool, search_success: bool) -> str:
    if policy_success and not search_success:
        return "policy_success_search_failure"
    if not policy_success and not search_success:
        return "both_fail_outcome_only"
    if not policy_success and search_success:
        return "policy_failure_search_success"
    return "both_success"


def boolish(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "success"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SOURCE_SPECS = [
    ("experiment_tracking/policy_mcts_instance_audit.csv", "primary Stage-1 fixed policy/search instance universe"),
    ("experiment_tracking/mcts_progressive_widening_cross_domain/pw70_followup_instances_20260904.csv", "Stage-1 PW70 Block Grouping/Counters classified instances"),
    ("experiment_tracking/mcts_progressive_widening_cross_domain/pw70_confirmatory_instances_20260904.csv", "Stage-1 PW70 FO Counters/Rover classified instances"),
    ("experiment_tracking/counters_seed_divergence_audit_20260914/per_instance_pair_join.csv", "Counters strict first-divergence timing and outcomes"),
    ("experiment_tracking/counters_seed_divergence_audit_20260914/per_instance_divergence.csv", "Counters strict failing-arm actions and terminal causes"),
    ("experiment_tracking/advisor_followup_20260910/counters_visit_audit_manifest.csv", "Counters Stage-2 selected-case provenance"),
    ("experiment_tracking/advisor_followup_20260910/counters_visit_audit_latest_steps.csv", "Counters Stage-2 aggregate root statistics"),
    ("experiment_tracking/advisor_followup_20260910/counters_tie_break_counterfactual_latest.csv", "Counters Stage-2 tie counterfactuals"),
    ("experiment_tracking/counters_tie_break_3way_20260911/results_latest.csv", "Counters three-way causal treatment outcomes"),
    ("experiment_tracking/block_grouping_tie_break_screen_20260913/trace_audit_summary_20260914.csv", "Block Grouping fixed tie-screen traces"),
    ("experiment_tracking/block_grouping_tie_break_screen_20260913/policy_alignment_20260914.csv", "Block Grouping fixed pure-policy alignment"),
    ("experiment_tracking/block_grouping_pw70_tie_trace_20260914/results_20260915.csv", "Block Grouping PW70 selected-case outcomes"),
    ("experiment_tracking/block_grouping_pw70_tie_trace_20260914/first_divergence_q_u_20260915.csv", "Block Grouping PW70 first-divergence N/Q/U/prior comparison"),
    ("experiment_tracking/mcts_safe_drone/targeted_results_20260828.csv", "Drone SAFE selected-case intervention outcomes"),
]


source_rows: list[dict[str, object]] = []
sources: dict[str, tuple[Path, list[dict[str, str]]]] = {}
for relative, purpose in SOURCE_SPECS:
    path, rows = read_csv(relative)
    sources[relative] = (path, rows)
    source_rows.append(
        {
            "relative_artifact": relative,
            "resolved_path": str(path),
            "sha256": sha256(path),
            "data_rows": len(rows),
            "purpose": purpose,
        }
    )


# Normalize the complete fixed-search outcome universe.
broad_rel = "experiment_tracking/policy_mcts_instance_audit.csv"
broad_path, broad = sources[broad_rel]
instance_rows: list[dict[str, object]] = []
broad_index: dict[tuple[str, str, str, str], dict[str, str]] = {}
relation_map = {
    "policy_only_success": "policy_success_search_failure",
    "both_failure": "both_fail_outcome_only",
    "mcts_only_success": "policy_failure_search_success",
    "both_success": "both_success",
}
for source_row, row in enumerate(broad, start=2):
    instance = basename(row["instance"])
    key = (row["domain"], row["value_head"], row["seed"], instance)
    if key in broad_index:
        raise ValueError(f"Duplicate primary fixed identity: {key}")
    broad_index[key] = row
    instance_rows.append(
        {
            "domain": row["domain"],
            "stage": "stage1_validation_selected",
            "value_head": row["value_head"],
            "seed": row["seed"],
            "checkpoint_identity": checkpoint_identity(row["policy_log"]),
            "search_family": "fixed",
            "search_variant": "narrow_5x20" if row["domain"] in {"block_grouping", "counters"} else "normal_20x70",
            "instance": instance,
            "policy_status": row["policy_status"],
            "search_status": row["mcts_status"],
            "outcome_stratum": relation_map[row["relation"]],
            "policy_steps": row["policy_steps"],
            "search_steps": row["mcts_steps"],
            "search_elapsed_seconds": row["mcts_elapsed_seconds"],
            "policy_log": row["policy_log"],
            "search_log": row["mcts_log"],
            "source_artifact": broad_rel,
            "source_row": source_row,
        }
    )


# Normalize every locally classified Stage-1 PW70 row and join its exact policy.
pw_inputs = [
    "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_followup_instances_20260904.csv",
    "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_confirmatory_instances_20260904.csv",
]
pw_seen: set[tuple[str, str, str, str]] = set()
for relative in pw_inputs:
    _, rows = sources[relative]
    for source_row, row in enumerate(rows, start=2):
        if row["stage"] != "stage1" or row["iterations"] != "70":
            continue
        instance = basename(row["instance_path"])
        key = (row["domain"], row["value_head"], row["seed"], instance)
        if key in pw_seen:
            raise ValueError(f"Duplicate Stage-1 PW70 identity: {key}")
        pw_seen.add(key)
        policy = broad_index.get(key)
        if policy is None:
            raise ValueError(f"PW70 row has no exact primary-policy join: {key}")
        policy_success = policy["policy_status"] == "success"
        search_success = boolish(row["success"])
        instance_rows.append(
            {
                "domain": row["domain"],
                "stage": "stage1_validation_selected",
                "value_head": row["value_head"],
                "seed": row["seed"],
                "checkpoint_identity": checkpoint_identity(policy["policy_log"]),
                "search_family": "pw",
                "search_variant": "pw70_kmin3",
                "instance": instance,
                "policy_status": policy["policy_status"],
                "search_status": row["status"],
                "outcome_stratum": outcome(policy_success, search_success),
                "policy_steps": policy["policy_steps"],
                "search_steps": row["steps"],
                "search_elapsed_seconds": row["elapsed_seconds"],
                "policy_log": policy["policy_log"],
                "search_log": row["source_log"],
                "source_artifact": relative,
                "source_row": source_row,
            }
        )


instance_fields = [
    "domain", "stage", "value_head", "seed", "checkpoint_identity",
    "search_family", "search_variant", "instance", "policy_status",
    "search_status", "outcome_stratum", "policy_steps", "search_steps",
    "search_elapsed_seconds", "policy_log", "search_log", "source_artifact",
    "source_row",
]
instance_rows.sort(key=lambda r: tuple(str(r[k]) for k in ("domain", "stage", "value_head", "seed", "search_family", "instance")))
write_csv("stage0_instance_outcomes.csv", instance_fields, instance_rows)


# Freeze first-divergence evidence.  Blank N/Q/U fields mean the local compact
# artifact never recorded them; they are not imputed from aggregate scores.
divergence_fields = [
    "domain", "stage", "value_head", "seed", "checkpoint_identity",
    "search_family", "search_variant", "tie_break", "instance",
    "outcome_stratum", "first_divergence_decision", "policy_action",
    "search_action", "network_argmax_action", "root_visits",
    "search_action_visits", "policy_action_visits", "search_action_q",
    "policy_action_q", "search_action_u", "policy_action_u",
    "search_action_prior", "policy_action_prior", "max_visit_tie_count",
    "visit_entropy", "policy_visit_js_divergence", "terminal_cause",
    "mechanism_label", "local_root_evidence_level", "policy_log",
    "search_log", "source_artifact", "source_row",
]
divergence_rows: list[dict[str, object]] = []


# Counters strict Stage-1 action-ID and policy-prior trajectories.
pair_rel = "experiment_tracking/counters_seed_divergence_audit_20260914/per_instance_pair_join.csv"
_, pair_rows = sources[pair_rel]
div_rel = "experiment_tracking/counters_seed_divergence_audit_20260914/per_instance_divergence.csv"
_, failing_rows = sources[div_rel]
failing_index = {
    (row["seed"], row["tie_break"], row["instance"]): row for row in failing_rows
}
for source_row, row in enumerate(pair_rows, start=2):
    for arm, state_key, divergence_key, cause_key, log_key in (
        ("action_id", "action_id_state", "action_id_first_divergence", "action_id_terminal_cause", "action_id_completion_log"),
        ("policy_prior", "policy_prior_state", "policy_prior_first_divergence", "policy_prior_terminal_cause", "policy_prior_completion_log"),
    ):
        if not row[divergence_key]:
            continue
        state = row[state_key]
        if state not in {"success", "failure"}:
            continue
        detail = failing_index.get((row["seed"], "policy" if arm == "policy_prior" else "action_id", row["instance"]), {})
        divergence_rows.append(
            {
                "domain": "counters",
                "stage": "stage1_validation_selected",
                "value_head": "off",
                "seed": row["seed"],
                "checkpoint_identity": checkpoint_identity(row["policy_log"], row["selected_epoch"]),
                "search_family": "fixed" if arm == "action_id" else "fixed_policy_prior_treatment",
                "search_variant": "narrow_5x20",
                "tie_break": arm,
                "instance": row["instance"],
                "outcome_stratum": "both_success" if state == "success" else "policy_success_search_failure",
                "first_divergence_decision": row[divergence_key],
                "policy_action": detail.get("policy_action_at_divergence", ""),
                "search_action": detail.get("search_action_at_divergence", ""),
                "terminal_cause": row[cause_key],
                "mechanism_label": detail.get("selection_mechanism", row["pair_mechanism"]),
                "local_root_evidence_level": "trajectory_first_divergence_no_root_vector",
                "policy_log": row["policy_log"],
                "search_log": row[log_key],
                "source_artifact": pair_rel,
                "source_row": source_row,
            }
        )


# Counters Stage-2 selected causal pilot: first tied root per exact instance.
counterfactual_rel = "experiment_tracking/advisor_followup_20260910/counters_tie_break_counterfactual_latest.csv"
_, counterfactual = sources[counterfactual_rel]
steps_rel = "experiment_tracking/advisor_followup_20260910/counters_visit_audit_latest_steps.csv"
_, visit_steps = sources[steps_rel]
visit_index = {(row["instance"], row["step"]): row for row in visit_steps if row["arm"] == "failure"}
first_counterfactual: dict[str, tuple[int, int, dict[str, str]]] = {}
for source_row, row in enumerate(counterfactual, start=2):
    step = int(row["step"])
    current = first_counterfactual.get(row["instance"])
    if current is None or step < current[0]:
        first_counterfactual[row["instance"]] = (step, source_row, row)
for raw_instance, (step, source_row, row) in sorted(first_counterfactual.items()):
    visit = visit_index.get((raw_instance, str(step)), {})
    instance = re.search(r"(fz_instance_\d+\.pddl)", raw_instance).group(1)
    divergence_rows.append(
        {
            "domain": "counters",
            "stage": "stage2_validation_selected",
            "value_head": "off",
            "seed": "1963100312",
            "checkpoint_identity": "src20430416_e0033",
            "search_family": "fixed",
            "search_variant": "narrow_5x20",
            "tie_break": "action_id",
            "instance": instance,
            "outcome_stratum": "policy_success_search_failure",
            "first_divergence_decision": step,
            "policy_action": row["network_argmax_action"],
            "search_action": row["current_action"],
            "network_argmax_action": row["network_argmax_action"],
            "root_visits": visit.get("total_edge_visits", ""),
            "search_action_visits": row["max_visit_count"],
            "max_visit_tie_count": row["number_tied_at_max_visits"],
            "visit_entropy": visit.get("visit_entropy", ""),
            "policy_visit_js_divergence": visit.get("policy_visit_js_divergence_expanded_children", ""),
            "terminal_cause": "ordinary_10000_action_cap",
            "mechanism_label": "equal_max_visits_action_id_overrode_policy_argmax",
            "local_root_evidence_level": "aggregate_root_tie_and_entropy_no_child_q_u",
            "policy_log": "/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage2_policy_eval/20451601_Ev_counters_counters_orig_novh_c.1_s1963100312_SR10P_src20430416_e0033.txt",
            "search_log": "/home/hersco/training_new_domains/2026-09-10/counters_visit_audit/21178320_failure_off.txt",
            "source_artifact": counterfactual_rel,
            "source_row": source_row,
        }
    )


# Block Grouping fixed tie screen.
bg_trace_rel = "experiment_tracking/block_grouping_tie_break_screen_20260913/trace_audit_summary_20260914.csv"
_, bg_trace = sources[bg_trace_rel]
bg_align_rel = "experiment_tracking/block_grouping_tie_break_screen_20260913/policy_alignment_20260914.csv"
_, bg_align = sources[bg_align_rel]
bg_align_index = {row["instance"]: row for row in bg_align}
for source_row, row in enumerate(bg_trace, start=2):
    aligned = bg_align_index.get(row["instance"], {})
    bg_policy_log = aligned.get(
        "source_policy_log",
        "/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage1_policy_eval/20424828_Ev_block_grouping_block_grouping_orig_novh_c.1_s1963100312_SR10P_src20401187_e0044.txt",
    )
    divergence_rows.append(
        {
            "domain": "block_grouping",
            "stage": "stage1_validation_selected",
            "value_head": "off",
            "seed": "1963100312",
            "checkpoint_identity": checkpoint_identity(bg_policy_log, "44"),
            "search_family": "fixed",
            "search_variant": "narrow_5x20",
            "tie_break": "action_id",
            "instance": row["instance"],
            "outcome_stratum": "policy_success_search_failure",
            "first_divergence_decision": aligned.get("action_id_first_divergence_decision", row["first_same_physical_state_action_difference"]),
            "policy_action": aligned.get("policy_selected", ""),
            "search_action": aligned.get("action_id_selected", row["action_id_selected"]),
            "network_argmax_action": row["network_argmax"],
            "max_visit_tie_count": aligned.get("action_id_max_visit_tie_count", ""),
            "terminal_cause": "ordinary_10000_action_cap",
            "mechanism_label": "clean_equal_max_tie" if row["clean_equal_max_visit_tie"].lower() == "true" else "tree_or_history_statistics_already_differed",
            "local_root_evidence_level": "tie_summary_no_child_q_u",
            "policy_log": bg_policy_log,
            "search_log": row["action_id_log"],
            "source_artifact": bg_trace_rel,
            "source_row": source_row,
        }
    )


# Block Grouping PW70 selected roots, including locally cached selected-vs-policy N/Q/U/prior.
bg_pw_rel = "experiment_tracking/block_grouping_pw70_tie_trace_20260914/results_20260915.csv"
_, bg_pw = sources[bg_pw_rel]
bg_qu_rel = "experiment_tracking/block_grouping_pw70_tie_trace_20260914/first_divergence_q_u_20260915.csv"
_, bg_qu = sources[bg_qu_rel]
bg_qu_index = {
    (row["value_head"], row["seed"], row["target_instance"]): (source_row, row)
    for source_row, row in enumerate(bg_qu, start=2)
}
for source_row, row in enumerate(bg_pw, start=2):
    q_source_row, q = bg_qu_index[(row["value_head"], row["seed"], row["target_instance"])]
    divergence_rows.append(
        {
            "domain": "block_grouping",
            "stage": "stage1_validation_selected",
            "value_head": row["value_head"],
            "seed": row["seed"],
            "checkpoint_identity": checkpoint_identity(row["source_policy_log"]),
            "search_family": "pw",
            "search_variant": "pw70_kmin3",
            "tie_break": "action_id",
            "instance": row["target_instance"],
            "outcome_stratum": "policy_success_search_failure",
            "first_divergence_decision": row["first_policy_divergence_step"],
            "policy_action": q["policy_action"],
            "search_action": q["selected_action"],
            "root_visits": q["root_visits"],
            "search_action_visits": q["selected_N"],
            "policy_action_visits": q["policy_N"],
            "search_action_q": q["selected_Q"],
            "policy_action_q": q["policy_Q"],
            "search_action_u": q["selected_U"],
            "policy_action_u": q["policy_U"],
            "search_action_prior": q["selected_prior"],
            "policy_action_prior": q["policy_prior"],
            "terminal_cause": "six_hour_timeout",
            "mechanism_label": q["dominant_evidence"],
            "local_root_evidence_level": "selected_vs_policy_n_q_u_prior_full_children_in_remote_trace",
            "policy_log": row["source_policy_log"],
            "search_log": row["remote_trace_log"],
            "source_artifact": bg_qu_rel,
            "source_row": q_source_row,
        }
    )


divergence_rows.sort(key=lambda r: tuple(str(r.get(k, "")) for k in ("domain", "stage", "value_head", "seed", "search_family", "tie_break", "instance")))
write_csv("stage0_first_divergence_evidence.csv", divergence_fields, divergence_rows)


# Normalize the causal/intervention evidence that cannot be represented as a
# complete first-divergence root row.
intervention_fields = [
    "domain", "stage", "value_head", "seed", "method_change", "instances",
    "baseline_successes", "treatment_successes", "conclusion",
    "policy_log", "treatment_log", "source_artifact", "source_row",
]
interventions: list[dict[str, object]] = []
drone_rel = "experiment_tracking/mcts_safe_drone/targeted_results_20260828.csv"
_, drone_rows = sources[drone_rel]
for source_row, row in enumerate(drone_rows, start=2):
    interventions.append(
        {
            "domain": "drone",
            "stage": "stage1_validation_selected",
            "value_head": row["value_head"],
            "seed": row["seed"],
            "method_change": "SAFE_terminal_child_handling",
            "instances": row["instance"],
            "baseline_successes": 0,
            "treatment_successes": 1 if row["status"] == "success" else 0,
            "conclusion": "selected original policy-success/search-failure repaired" if boolish(row["safe_repaired_original_policy_only_failure"]) else "selected failure not repaired",
            "treatment_log": row["source_log"],
            "source_artifact": drone_rel,
            "source_row": source_row,
        }
    )
threeway_rel = "experiment_tracking/counters_tie_break_3way_20260911/results_latest.csv"
_, threeway = sources[threeway_rel]
for source_row, row in enumerate(threeway, start=2):
    if row["value_head"] != "off":
        continue
    interventions.append(
        {
            "domain": "counters",
            "stage": "stage2_validation_selected",
            "value_head": "off",
            "seed": "1963100312",
            "method_change": f"tie_break_{row['tie_break']}",
            "instances": "fz_instance_51.pddl;fz_instance_55.pddl;fz_instance_59.pddl",
            "baseline_successes": 0,
            "treatment_successes": row["coverage_6h"],
            "conclusion": row["interpretation"],
            "policy_log": "/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage2_policy_eval/20451601_Ev_counters_counters_orig_novh_c.1_s1963100312_SR10P_src20430416_e0033.txt",
            "treatment_log": row["remote_log"],
            "source_artifact": threeway_rel,
            "source_row": source_row,
        }
    )
write_csv("stage0_intervention_evidence.csv", intervention_fields, interventions)


# Coverage matrix: every planned domain/VH/family cell is present even when no
# local per-instance universe exists yet (Drone/MPrime PW, MPrime fixed).
primary_families = [("fixed", "primary"), ("pw", "diagnostic_only")]
domains = ["block_grouping", "counters", "drone", "fo_counters", "rover", "mprime"]
value_heads = ["off", "on"]
strata = [
    "policy_success_search_failure",
    "both_fail_outcome_only",
    "policy_failure_search_success",
    "both_success",
]
outcome_counts = Counter(
    (str(row["domain"]), str(row["value_head"]), str(row["search_family"]), str(row["outcome_stratum"]))
    for row in instance_rows
    if row["stage"] == "stage1_validation_selected"
)
divergence_counts = Counter(
    (str(row["domain"]), str(row["value_head"]), str(row["search_family"]), str(row["outcome_stratum"]))
    for row in divergence_rows
    if row["stage"] == "stage1_validation_selected" and row["search_family"] in {"fixed", "pw"}
)
root_counts = Counter(
    (str(row["domain"]), str(row["value_head"]), str(row["search_family"]), str(row["outcome_stratum"]))
    for row in divergence_rows
    if row["stage"] == "stage1_validation_selected"
    and row["search_family"] in {"fixed", "pw"}
    and "n_q_u_prior" in str(row["local_root_evidence_level"])
)
coverage_rows: list[dict[str, object]] = []
missing_rows: list[dict[str, object]] = []
instance_index_by_cell_stratum: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
for row in instance_rows:
    instance_index_by_cell_stratum[(str(row["domain"]), str(row["value_head"]), str(row["search_family"]), str(row["outcome_stratum"]))].append(row)

for domain in domains:
    for vh in value_heads:
        for family, policy in primary_families:
            counts = {stratum: outcome_counts[(domain, vh, family, stratum)] for stratum in strata}
            divergences = {stratum: divergence_counts[(domain, vh, family, stratum)] for stratum in strata}
            roots = {stratum: root_counts[(domain, vh, family, stratum)] for stratum in strata}
            missing = [stratum for stratum in strata if counts[stratum] > 0 and divergences[stratum] == 0]
            unavailable = [stratum for stratum in strata if counts[stratum] == 0]
            total = sum(counts.values())
            if family == "fixed":
                recommendation = "one compact-recorder task (max one frozen instance per missing observed stratum)" if missing else "no task: observed strata have first-divergence evidence"
            elif domain in {"fo_counters", "mprime"}:
                recommendation = "optional PW compact-recorder task after fixed-search Stage 2; max four instances"
            else:
                recommendation = "no new PW task at Stage 0; do not duplicate fixed-search campaign"
            if total == 0:
                recommendation = "wait for exact per-instance outcome join before selecting targets"
            coverage_rows.append(
                {
                    "domain": domain,
                    "stage": "stage1_validation_selected",
                    "value_head": vh,
                    "search_family": family,
                    "outcome_rows": total,
                    "policy_success_search_failure": counts["policy_success_search_failure"],
                    "both_fail_outcome_only": counts["both_fail_outcome_only"],
                    "policy_failure_search_success": counts["policy_failure_search_success"],
                    "both_success": counts["both_success"],
                    "first_divergence_rows": sum(divergences.values()),
                    "local_n_q_u_prior_rows": sum(roots.values()),
                    "observed_strata_missing_first_divergence": ";".join(missing),
                    "strata_without_local_outcomes": ";".join(unavailable),
                    "stage2_submission_policy": policy,
                    "proposed_minimal_job": recommendation,
                }
            )
            if family == "fixed" or domain in {"fo_counters", "mprime"}:
                for stratum in missing:
                    candidates = sorted(
                        instance_index_by_cell_stratum[(domain, vh, family, stratum)],
                        key=lambda r: (str(r["seed"]), str(r["instance"])),
                    )
                    candidate = candidates[0]
                    missing_rows.append(
                        {
                            "domain": domain,
                            "stage": "stage1_validation_selected",
                            "value_head": vh,
                            "search_family": family,
                            "missing_stratum": stratum,
                            "candidate_seed": candidate["seed"],
                            "candidate_checkpoint_identity": candidate["checkpoint_identity"],
                            "candidate_instance": candidate["instance"],
                            "candidate_policy_log": candidate["policy_log"],
                            "candidate_search_log": candidate["search_log"],
                            "candidate_source_artifact": candidate["source_artifact"],
                            "reason": "outcome is locally classified but no compact first-divergence record exists for this cell/stratum",
                            "submission_group": f"{domain}:{vh}:{family}",
                        }
                    )


coverage_fields = [
    "domain", "stage", "value_head", "search_family", "outcome_rows",
    "policy_success_search_failure", "both_fail_outcome_only",
    "policy_failure_search_success", "both_success", "first_divergence_rows",
    "local_n_q_u_prior_rows", "observed_strata_missing_first_divergence",
    "strata_without_local_outcomes", "stage2_submission_policy",
    "proposed_minimal_job",
]
write_csv("stage0_coverage_missing_strata.csv", coverage_fields, coverage_rows)
missing_fields = [
    "domain", "stage", "value_head", "search_family", "missing_stratum",
    "candidate_seed", "candidate_checkpoint_identity", "candidate_instance",
    "candidate_policy_log", "candidate_search_log", "candidate_source_artifact",
    "reason", "submission_group",
]
write_csv("stage0_missing_strata_manifest.csv", missing_fields, missing_rows)
write_csv(
    "stage0_source_artifacts.csv",
    ["relative_artifact", "resolved_path", "sha256", "data_rows", "purpose"],
    source_rows,
)


print(f"fixed instance outcomes: {sum(1 for row in instance_rows if row['search_family'] == 'fixed')}")
print(f"PW70 instance outcomes: {sum(1 for row in instance_rows if row['search_family'] == 'pw')}")
print(f"first-divergence records: {len(divergence_rows)}")
print(f"intervention records: {len(interventions)}")
print(f"coverage cells: {len(coverage_rows)}")
print(f"missing-stratum candidates: {len(missing_rows)}")
