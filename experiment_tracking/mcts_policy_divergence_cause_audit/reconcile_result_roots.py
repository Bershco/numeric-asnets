#!/usr/bin/env python3
"""Join divergence-audit result artifacts across output roots and schemas.

The first campaign wrote terminal outcomes in the top-level
``completion_record`` field (schema v1).  Later recoveries write the normalized
``outcome.classification`` field (schema v2).  This reader deliberately accepts
both layouts, prefers explicit terminal evidence, and deduplicates by frozen
``(task_id, candidate_index)`` identity.
"""

from __future__ import annotations

import argparse
import csv
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any


FIELDS = [
    "task_id",
    "candidate_index",
    "instance",
    "missing_stratum",
    "outcome",
    "outcome_evidence",
    "first_divergence_observed",
    "selector",
    "selection_mechanism",
    "selector_input_changed_from_raw_visits",
    "duplicate_control_applied",
    "step",
    "policy_action",
    "selected_action",
    "policy_action_expanded",
    "policy_entropy",
    "visit_entropy",
    "policy_visit_js_divergence",
    "selected_action_policy_prior",
    "selected_action_policy_rank",
    "winner_visit_share",
    "top1_top2_visit_margin",
    "max_visit_tie_count",
    "selected_is_signed_q_argmax",
    "selected_is_u_argmax",
    "selected_is_signed_q_plus_u_argmax",
    "root_visits",
    "policy_action_visits",
    "selected_action_visits",
    "raw_visit_argmax_action",
    "selected_is_raw_visit_argmax",
    "policy_action_prior",
    "policy_action_q",
    "selected_action_q",
    "selected_minus_policy_q",
    "policy_action_u",
    "selected_action_u",
    "selected_minus_policy_u",
    "policy_action_q_plus_u",
    "selected_action_q_plus_u",
    "selected_minus_policy_q_plus_u",
    "policy_status",
    "current_outcome_stratum",
    "source_result",
]


def terminal_outcome(record: dict[str, Any]) -> tuple[str, str]:
    outcome = record.get("outcome")
    if isinstance(outcome, dict) and outcome.get("classification"):
        return str(outcome["classification"]), str(
            outcome.get("evidence", "outcome")
        )

    completion = record.get("completion_record")
    if not isinstance(completion, dict):
        return "unknown", "missing"
    if completion.get("status") == "success" and completion.get("hit_goal") is True:
        return "success", "completion_record"
    if completion.get("status") == "finished_unsolved":
        steps = int(completion.get("steps", -1))
        return (
            "action_limit" if steps >= 10_000 else "finished_unsolved",
            "completion_record",
        )
    return "unknown", "unsupported_completion_record"


def normalize(record: dict[str, Any], source: Path) -> dict[str, Any]:
    outcome, evidence = terminal_outcome(record)
    divergence = record.get("first_divergence")
    summary = divergence.get("summary", {}) if isinstance(divergence, dict) else {}
    selection = (
        divergence.get("selection", {}) if isinstance(divergence, dict) else {}
    )
    selector = selection.get("final_stage")
    if not selector and isinstance(divergence, dict):
        # Schema v1 predates the normalized ``selection`` object.  The actual
        # terminal selector is still frozen in the ordered override path.
        selected = divergence.get("selected_action")
        for stage in reversed(divergence.get("override_path", [])):
            if stage.get("selected_action") == selected:
                selector = stage.get("stage")
                break
    if not selector:
        selector = "no_divergence"
    override_path = (
        divergence.get("override_path", []) if isinstance(divergence, dict) else []
    )
    duplicate_control_applied = any(
        stage.get("stage") == "duplicate_penalty" and stage.get("applied") is True
        for stage in override_path
    )
    root = divergence.get("root", {}) if isinstance(divergence, dict) else {}
    policy_action = (
        divergence.get("policy_action") if isinstance(divergence, dict) else None
    )
    selected_action = (
        divergence.get("selected_action") if isinstance(divergence, dict) else None
    )

    def at(values: Any, index: Any) -> Any:
        if not isinstance(values, list) or not isinstance(index, int):
            return None
        if not 0 <= index < len(values):
            return None
        return values[index]

    def difference(left: Any, right: Any) -> Any:
        if left is None or right is None:
            return None
        return float(left) - float(right)

    policy_status = (
        "success"
        if record.get("missing_stratum")
        in {"both_success", "policy_success_search_failure"}
        else "failure"
    )
    search_status = "success" if outcome == "success" else "failure"
    current_outcome_stratum = {
        ("success", "success"): "both_success",
        ("success", "failure"): "policy_success_search_failure",
        ("failure", "success"): "policy_failure_search_success",
        ("failure", "failure"): "both_fail",
    }[(policy_status, search_status)]

    policy_q = at(root.get("q_values"), policy_action)
    selected_q = at(root.get("q_values"), selected_action)
    policy_u = at(root.get("u_values"), policy_action)
    selected_u = at(root.get("u_values"), selected_action)
    policy_score = at(root.get("signed_q_plus_u"), policy_action)
    selected_score = at(root.get("signed_q_plus_u"), selected_action)
    visit_distribution = root.get("visit_distribution")
    selector_distribution = selection.get("selector_input_distribution")
    selector_input_changed = None
    if isinstance(visit_distribution, list) and isinstance(selector_distribution, list):
        selector_input_changed = (
            len(visit_distribution) != len(selector_distribution)
            or any(
                abs(float(left) - float(right))
                > 1e-8 + 1e-6 * max(abs(float(left)), abs(float(right)))
                for left, right in zip(visit_distribution, selector_distribution)
            )
        )
    raw_visit_argmax = None
    if isinstance(visit_distribution, list) and visit_distribution:
        raw_visit_argmax = max(
            range(len(visit_distribution)),
            key=lambda action: (float(visit_distribution[action]), -action),
        )
    if not isinstance(divergence, dict):
        selection_mechanism = "no_divergence"
    elif selector == "goal_chase":
        selection_mechanism = "goal_chase"
    elif duplicate_control_applied or selector_input_changed is True:
        selection_mechanism = "transformed_then_root_argmax"
    else:
        selection_mechanism = "raw_root_visit_argmax"
    return {
        "task_id": record["task_id"],
        "candidate_index": int(record["candidate_index"]),
        "instance": record.get("instance"),
        "missing_stratum": record.get("missing_stratum"),
        "outcome": outcome,
        "outcome_evidence": evidence,
        "first_divergence_observed": bool(record.get("first_divergence_observed")),
        "selector": selector,
        "selection_mechanism": selection_mechanism,
        "selector_input_changed_from_raw_visits": selector_input_changed,
        "duplicate_control_applied": duplicate_control_applied,
        "step": divergence.get("step") if isinstance(divergence, dict) else None,
        "policy_action": policy_action,
        "selected_action": selected_action,
        "policy_action_expanded": summary.get("policy_action_expanded"),
        "policy_entropy": summary.get("policy_entropy"),
        "visit_entropy": summary.get("visit_entropy"),
        "policy_visit_js_divergence": summary.get("policy_visit_js_divergence"),
        "selected_action_policy_prior": summary.get("selected_action_policy_prior"),
        "selected_action_policy_rank": summary.get("selected_action_policy_rank"),
        "winner_visit_share": summary.get("winner_visit_share"),
        "top1_top2_visit_margin": summary.get("top1_top2_visit_margin"),
        "max_visit_tie_count": summary.get("max_visit_tie_count"),
        "selected_is_signed_q_argmax": summary.get("selected_is_signed_q_argmax"),
        "selected_is_u_argmax": summary.get("selected_is_u_argmax"),
        "selected_is_signed_q_plus_u_argmax": summary.get(
            "selected_is_signed_q_plus_u_argmax"
        ),
        "root_visits": root.get("root_visits"),
        "policy_action_visits": at(root.get("edge_visit_counts"), policy_action),
        "selected_action_visits": at(root.get("edge_visit_counts"), selected_action),
        "raw_visit_argmax_action": raw_visit_argmax,
        "selected_is_raw_visit_argmax": selected_action == raw_visit_argmax,
        "policy_action_prior": at(root.get("edge_priors"), policy_action),
        "policy_action_q": policy_q,
        "selected_action_q": selected_q,
        "selected_minus_policy_q": difference(selected_q, policy_q),
        "policy_action_u": policy_u,
        "selected_action_u": selected_u,
        "selected_minus_policy_u": difference(selected_u, policy_u),
        "policy_action_q_plus_u": policy_score,
        "selected_action_q_plus_u": selected_score,
        "selected_minus_policy_q_plus_u": difference(selected_score, policy_score),
        "policy_status": policy_status,
        "current_outcome_stratum": current_outcome_stratum,
        "source_result": str(source),
    }


def join_roots(roots: list[Path]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, int], tuple[tuple[bool, int], dict[str, Any]]] = {}
    for root_rank, root in enumerate(roots):
        for path in root.rglob("*.result.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if "task_id" not in raw or "candidate_index" not in raw:
                continue
            row = normalize(raw, path)
            key = (str(row["task_id"]), int(row["candidate_index"]))
            score = (row["outcome"] != "unknown", root_rank)
            if key not in rows or score > rows[key][0]:
                rows[key] = (score, row)
    return [
        value[1]
        for _key, value in sorted(rows.items(), key=lambda item: item[0])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument(
        "--format", choices=("csv", "json", "summary"), default="csv"
    )
    args = parser.parse_args()
    rows = join_roots(args.roots)
    if args.format == "json":
        print(json.dumps({"count": len(rows), "rows": rows}, indent=2))
        return
    if args.format == "summary":
        def counts(field: str, subset: list[dict[str, Any]] = rows) -> dict[str, int]:
            return dict(sorted(collections.Counter(
                str(row[field]) for row in subset
            ).items()))

        divergences = [row for row in rows if row["first_divergence_observed"]]
        ordinary = [
            row for row in divergences
            if row["selection_mechanism"] == "raw_root_visit_argmax"
        ]
        transformed = [
            row for row in divergences
            if row["selection_mechanism"] == "transformed_then_root_argmax"
        ]
        no_divergence = [row for row in rows if not row["first_divergence_observed"]]

        def medians(subset: list[dict[str, Any]]) -> dict[str, Any]:
            fields = (
                "policy_entropy", "visit_entropy", "policy_visit_js_divergence",
                "selected_action_policy_prior", "selected_action_policy_rank",
                "winner_visit_share", "selected_minus_policy_q",
                "selected_minus_policy_u", "selected_minus_policy_q_plus_u",
            )
            answer = {}
            for field in fields:
                values = [
                    float(row[field]) for row in subset if row[field] is not None
                ]
                answer[field] = statistics.median(values) if values else None
            return answer

        ordinary_by_stratum = {}
        for stratum in (
            "policy_success_search_failure", "policy_failure_search_success",
            "both_fail", "both_success",
        ):
            subset = [
                row for row in ordinary if row["current_outcome_stratum"] == stratum
            ]
            ordinary_by_stratum[stratum] = {
                "count": len(subset),
                "medians": medians(subset),
            }
        print(json.dumps({
            "count": len(rows),
            "outcomes": counts("outcome"),
            "first_divergence_observed": len(divergences),
            "no_divergence": len(no_divergence),
            "no_divergence_outcomes": counts("outcome", no_divergence),
            "selectors": counts("selector", divergences),
            "selection_mechanisms": counts("selection_mechanism", divergences),
            "transformed_root_outcomes": counts("outcome", transformed),
            "transformed_root_duplicate_control": sum(
                row["duplicate_control_applied"] is True for row in transformed
            ),
            "ordinary_outcomes": counts("outcome", ordinary),
            "ordinary_q_argmax": sum(
                row["selected_is_signed_q_argmax"] is True for row in ordinary
            ),
            "ordinary_u_argmax": sum(
                row["selected_is_u_argmax"] is True for row in ordinary
            ),
            "ordinary_q_plus_u_argmax": sum(
                row["selected_is_signed_q_plus_u_argmax"] is True
                for row in ordinary
            ),
            "ordinary_visit_ties": sum(
                int(row["max_visit_tie_count"] or 0) > 1 for row in ordinary
            ),
            "ordinary_selected_q_greater_than_policy": sum(
                row["selected_minus_policy_q"] is not None
                and row["selected_minus_policy_q"] > 1e-12
                for row in ordinary
            ),
            "ordinary_selected_q_equal_policy": sum(
                row["selected_minus_policy_q"] is not None
                and abs(row["selected_minus_policy_q"]) <= 1e-12
                for row in ordinary
            ),
            "ordinary_selected_q_less_than_policy": sum(
                row["selected_minus_policy_q"] is not None
                and row["selected_minus_policy_q"] < -1e-12
                for row in ordinary
            ),
            "ordinary_policy_action_unexpanded": sum(
                row["policy_action_expanded"] is False for row in ordinary
            ),
            "ordinary_by_current_outcome_stratum": ordinary_by_stratum,
        }, indent=2, sort_keys=True))
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
