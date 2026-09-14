#!/usr/bin/env python3
"""Join pure-policy and strict-search trajectories for two Counters seeds."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"

SEEDS = {
    534933607: {
        "epoch": 43,
        "policy_score": 59,
        "policy_log": "20429101_policy.txt",
        "action_id": "21233925_0.completed.jsonl",
        "policy": "21233925_10.completed.jsonl",
        "action_id_log": "21233925_0.txt",
        "policy_log_search": "21233925_10.txt",
    },
    2082152039: {
        "epoch": 12,
        "policy_score": 35,
        "policy_log": "20429130_policy.txt",
        "action_id": "21233925_9.completed.jsonl",
        "policy": "21233925_19.completed.jsonl",
        "action_id_log": "21233925_9.txt",
        "policy_log_search": "21233925_19.txt",
    },
}

PLAN_RE = re.compile(
    r"^\[EVAL\]\[PLAN\]\s+EASY\s+\|\s+(?P<instance>[^|]+?)\s+\|\s+"
    r"steps=(?P<steps>\d+)\s+\|\s+plan=(?P<plan>\[.*\])$"
)


def read_policy_plans(path: Path) -> dict[str, list[str]]:
    plans: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            match = PLAN_RE.match(raw.rstrip("\r\n"))
            if not match:
                continue
            plan = ast.literal_eval(match.group("plan"))
            assert len(plan) == int(match.group("steps"))
            plans[match.group("instance").strip()] = plan
    return plans


def read_completion(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            name = Path(item["instance_path"]).name
            rows[name] = item
    return rows


def first_divergence(policy: list[str], search: list[str]) -> tuple[int | None, str, str]:
    for index, (policy_action, search_action) in enumerate(zip(policy, search), start=1):
        if policy_action != search_action:
            return index, policy_action, search_action
    if len(policy) == len(search):
        return None, "", ""
    index = min(len(policy), len(search)) + 1
    return index, policy[index - 1] if index <= len(policy) else "<policy-ended>", search[index - 1] if index <= len(search) else "<search-ended>"


def outcome_cause(item: dict) -> str:
    status = str(item.get("status", "")).lower()
    steps = int(item.get("steps", 0) or 0)
    elapsed = float(item.get("elapsed_seconds", 0.0) or 0.0)
    if status == "timeout" or elapsed >= 21599.0:
        return "timeout_6h"
    if not item.get("hit_goal", False) and steps >= 10000:
        return "ordinary_10000_action_cap"
    if status in {"oom", "unclassified"}:
        return "oom_or_unclassified"
    if item.get("hit_goal", False):
        return "success"
    return "other_terminal_failure"


def main() -> None:
    detail_rows: list[dict] = []
    pair_rows: list[dict] = []
    for seed, spec in SEEDS.items():
        pure = read_policy_plans(RAW / spec["policy_log"])
        by_rule = {rule: read_completion(RAW / spec[rule]) for rule in ("action_id", "policy")}
        for rule in ("action_id", "policy"):
            completions = by_rule[rule]
            for instance, item in sorted(completions.items(), key=lambda pair: item_number(pair[0])):
                pure_plan = pure.get(instance)
                search_plan = item.get("plan") or []
                policy_success = pure_plan is not None
                search_success = bool(item.get("hit_goal", False))
                if not (policy_success and not search_success):
                    continue
                divergence, policy_action, search_action = first_divergence(pure_plan, search_plan)
                terminal_cause = outcome_cause(item)
                detail_rows.append(
                    {
                        "seed": seed,
                        "selected_epoch": spec["epoch"],
                        "policy_score": spec["policy_score"],
                        "tie_break": rule,
                        "instance": instance,
                        "instance_number": item.get("instance_number", ""),
                        "policy_steps": len(pure_plan),
                        "search_steps": len(search_plan),
                        "first_divergence_decision": divergence or "",
                        "policy_action_at_divergence": policy_action,
                        "search_action_at_divergence": search_action,
                        "terminal_cause": terminal_cause,
                        "selection_mechanism": "insufficient_root_trace",
                        "search_status": item.get("status", ""),
                        "elapsed_seconds": item.get("elapsed_seconds", ""),
                        "policy_log": remote_policy_path(seed),
                        "search_completion_log": remote_completion_path(seed, rule),
                        "search_text_log": remote_text_path(seed, rule),
                    }
                )

        for instance, pure_plan in sorted(pure.items(), key=lambda pair: item_number(pair[0])):
            arms = {}
            for rule in ("action_id", "policy"):
                item = by_rule[rule].get(instance)
                if item is None:
                    arms[rule] = {"state": "unclassified", "div": None, "action": "", "cause": "unclassified"}
                    continue
                search_plan = item.get("plan") or []
                div, _, search_action = first_divergence(pure_plan, search_plan)
                arms[rule] = {
                    "state": "success" if item.get("hit_goal", False) else "failure",
                    "div": div,
                    "action": search_action,
                    "cause": outcome_cause(item),
                }
            aid, prior = arms["action_id"], arms["policy"]
            mechanism = classify_pair_mechanism(aid, prior)
            pair_rows.append(
                {
                    "seed": seed,
                    "selected_epoch": spec["epoch"],
                    "policy_score": spec["policy_score"],
                    "instance": instance,
                    "policy_steps": len(pure_plan),
                    "action_id_state": aid["state"],
                    "action_id_first_divergence": aid["div"] or "",
                    "action_id_terminal_cause": aid["cause"],
                    "policy_prior_state": prior["state"],
                    "policy_prior_first_divergence": prior["div"] or "",
                    "policy_prior_terminal_cause": prior["cause"],
                    "pair_mechanism": mechanism,
                    "policy_log": remote_policy_path(seed),
                    "action_id_completion_log": remote_completion_path(seed, "action_id"),
                    "policy_prior_completion_log": remote_completion_path(seed, "policy"),
                }
            )

    fieldnames = list(detail_rows[0]) if detail_rows else []
    with (ROOT / "per_instance_divergence.csv").open("w", newline="", encoding="utf-8") as handle:
        pair_index = {(row["seed"], row["instance"]): row for row in pair_rows}
        for row in detail_rows:
            pair = pair_index[(row["seed"], row["instance"])]
            if row["tie_break"] == "policy":
                # At this root the trajectory was still identical to pure
                # policy. The source policy action is the highest-prior
                # retained child. Had it shared the maximum visit count, the
                # policy tie-break would have selected it. Its rejection
                # therefore proves that it was not visit-maximal; the logs do
                # not distinguish a unique winner from a tie among other
                # non-policy actions.
                row["selection_mechanism"] = "search_override_consistent_with_policy_action_not_visit_max_root_trace_absent"
            elif pair["pair_mechanism"] == "initial_tie_rescued_but_later_search_override":
                row["selection_mechanism"] = "max_visit_tie_action_id_overrode_policy"
            elif pair["pair_mechanism"] == "same_first_divergence_action_both_rules_root_cause_unresolved":
                row["selection_mechanism"] = "same_first_divergence_action_both_rules_root_cause_unresolved"
            else:
                row["selection_mechanism"] = "insufficient_root_trace"
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    group_fields = ["seed", "tie_break", "terminal_cause", "selection_mechanism"]
    counts = Counter(tuple(str(row[field]) for field in group_fields) for row in detail_rows)
    with (ROOT / "aggregate_causes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*group_fields, "count"])
        for key, count in sorted(counts.items()):
            writer.writerow([*key, count])

    with (ROOT / "per_instance_pair_join.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    pair_counts = Counter((str(row["seed"]), row["pair_mechanism"]) for row in pair_rows)
    with (ROOT / "aggregate_pair_mechanisms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "pair_mechanism", "count"])
        for key, count in sorted(pair_counts.items()):
            writer.writerow([*key, count])

    with (ROOT / "raw_file_checksums.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_file", "bytes", "sha256"])
        for path in sorted(RAW.glob("*")):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            writer.writerow([str(path.relative_to(ROOT)), path.stat().st_size, file_sha256(path)])


def item_number(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 10**9


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_pair_mechanism(aid: dict, prior: dict) -> str:
    """Classify treatment behavior without inventing unlogged root statistics."""
    if aid["state"] == "unclassified" or prior["state"] == "unclassified":
        return "at_least_one_arm_unclassified"
    if aid["state"] == "success" and prior["state"] == "success":
        return "both_search_arms_succeed"
    if aid["state"] == "failure" and prior["state"] == "success":
        return "policy_prior_rescues_action_id_failure"
    if aid["state"] == "success" and prior["state"] == "failure":
        return "policy_prior_introduces_failure"
    # Both fail. The rules differ only at maximum-visit ties. If the policy arm
    # stays on the pure-policy trajectory longer, it resolved the earlier tie
    # correctly, but later search evidence overrode the policy.
    if aid["div"] and prior["div"] and prior["div"] > aid["div"]:
        return "initial_tie_rescued_but_later_search_override"
    if aid["div"] == prior["div"] and aid["action"] == prior["action"]:
        return "same_first_divergence_action_both_rules_root_cause_unresolved"
    if aid["div"] == prior["div"]:
        return "same_step_different_nonpolicy_actions_insufficient_root_trace"
    return "both_fail_changed_trajectory_insufficient_root_trace"


def remote_policy_path(seed: int) -> str:
    job = 20429101 if seed == 534933607 else 20429130
    suffix = (
        "20429101_Ev_counters_counters_orig_novh_c.1_s534933607_SR10P_src20401263_e0043.txt"
        if seed == 534933607
        else "20429130_Ev_counters_counters_orig_novh_c.1_s2082152039_SR10P_src20401264_e0012.txt"
    )
    return f"/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage1_policy_eval/{suffix}"


def remote_completion_path(seed: int, rule: str) -> str:
    index = {("action_id", 534933607): 0, ("policy", 534933607): 10, ("action_id", 2082152039): 9, ("policy", 2082152039): 19}[(rule, seed)]
    directory = "action_id" if rule == "action_id" else "policy"
    return f"/home/hersco/training_new_domains/2026-09-13/counters_tie_break_strict_stage1/{directory}/{seed}/21233925_{index}.completed.jsonl"


def remote_text_path(seed: int, rule: str) -> str:
    index = {("action_id", 534933607): 0, ("policy", 534933607): 10, ("action_id", 2082152039): 9, ("policy", 2082152039): 19}[(rule, seed)]
    directory = "action_id" if rule == "action_id" else "policy"
    return f"/home/hersco/training_new_domains/2026-09-13/counters_tie_break_strict_stage1/{directory}/{seed}/21233925_{index}_{directory}_{seed}.txt"


if __name__ == "__main__":
    main()
