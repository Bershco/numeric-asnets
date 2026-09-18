"""Compact, opt-in evidence for the first policy/MCTS divergence.

The recorder is deliberately a pure observer.  It receives the completed root
and the action already chosen by the ordinary selector; it never chooses or
modifies an action.  Full-length vectors retain unexpanded progressive-
widening actions as ``None`` rather than silently treating them as zero-valued
children.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from post_training.monte_carlo_tree_search import action_history_digest


SCHEMA_VERSION = "mcts-first-divergence-v1"


def _normalise(vector: Sequence[float]) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("probability vector must be one-dimensional")
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("probability vector must be finite and nonnegative")
    total = float(result.sum())
    if total <= 0.0:
        return np.zeros_like(result)
    return result / total


def entropy(vector: Sequence[float]) -> float:
    """Natural-log entropy after robust normalization."""
    probabilities = _normalise(vector)
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def jensen_shannon_divergence(
        left: Sequence[float], right: Sequence[float]) -> float:
    """Natural-log Jensen-Shannon divergence for aligned vectors."""
    p = _normalise(left)
    q = _normalise(right)
    if p.shape != q.shape:
        raise ValueError("Jensen-Shannon vectors must have equal shape")
    midpoint = 0.5 * (p + q)

    def kl_term(source: np.ndarray) -> float:
        positive = source > 0.0
        return float(np.sum(
            source[positive]
            * (np.log(source[positive]) - np.log(midpoint[positive]))
        ))

    return 0.5 * kl_term(p) + 0.5 * kl_term(q)


def _digest_bytes(value: bytes, *, person: bytes) -> str:
    return hashlib.blake2b(
        value, digest_size=16, person=person).hexdigest()


def _digest_array(value, *, dtype, person: bytes) -> str:
    canonical = np.asarray(value, dtype=dtype).tobytes(order="C")
    return _digest_bytes(canonical, person=person)


def _action_names(root, act_dim: int) -> list[str | None]:
    names: list[str | None] = [None] * act_dim
    enabled = getattr(root.state, "acts_enabled", None)
    if enabled is None:
        return names
    for action_id in range(min(act_dim, len(enabled))):
        entry = enabled[action_id]
        bound_action = entry[0] if isinstance(entry, tuple) else entry
        names[action_id] = getattr(bound_action, "unique_ident", str(bound_action))
    return names


def _child_state_digest(child) -> str | None:
    if child is None:
        return None
    state_key = getattr(child, "state_key", None)
    if state_key is None:
        return None
    return _digest_bytes(state_key, person=b"asnet-child-key")


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def build_first_divergence_record(
        mcts,
        *,
        visit_policy: Sequence[float],
        selected_action: int,
        step: int,
        instance_name: str,
        elapsed_seconds: float | None,
        override_path: Sequence[Mapping[str, Any]],
        provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a complete first-divergence root record, or ``None``.

    The policy comparator is exactly the policy-only evaluator's convention:
    raw root network probabilities masked to applicable actions and then
    normalized.  The visit vector is the already-computed search distribution
    supplied to the external selector.
    """
    root = mcts.curr_tree_root
    raw_policy = np.asarray(root.act_dist, dtype=np.float64)
    applicable = np.asarray(root.applicable_action_mask, dtype=bool)
    if raw_policy.ndim != 1 or applicable.shape != raw_policy.shape:
        raise ValueError("root policy and applicable-action mask must align")
    act_dim = len(raw_policy)
    visits_distribution = _normalise(visit_policy)
    if visits_distribution.shape != raw_policy.shape:
        raise ValueError("visit policy and root policy must align")
    masked_policy = _normalise(raw_policy * applicable)
    if float(masked_policy.sum()) <= 0.0:
        return None
    policy_action = int(np.argmax(masked_policy))
    selected_action = int(selected_action)
    if selected_action == policy_action:
        return None

    action_ids = list(range(act_dim))
    action_names = _action_names(root, act_dim)
    expanded = [False] * act_dim
    edge_visits = [0] * act_dim
    edge_priors: list[float | None] = [None] * act_dim
    q_values: list[float | None] = [None] * act_dim
    u_values: list[float | None] = [None] * act_dim
    scores: list[float | None] = [None] * act_dim
    terminal: list[bool | None] = [None] * act_dim
    goal: list[bool | None] = [None] * act_dim
    child_state_digests: list[str | None] = [None] * act_dim

    children = root.children
    sqrt_root_n = math.sqrt(max(1.0, float(root.visit_count)))
    if children is not None:
        for child_index, (action, child) in enumerate(children.items()):
            action = int(action)
            expanded[action] = True
            edge_n = int(children.visits[child_index])
            prior = float(children.priors[child_index])
            q_value = float(child.Q_value)
            u_value = float(
                mcts.exploration_weight * prior
                * sqrt_root_n / (1.0 + edge_n))
            edge_visits[action] = edge_n
            edge_priors[action] = prior
            q_values[action] = _json_number(q_value)
            u_values[action] = _json_number(u_value)
            scores[action] = _json_number(mcts.sign * q_value + u_value)
            terminal[action] = bool(child.terminal_state)
            goal[action] = bool(child.goal_state)
            child_state_digests[action] = _child_state_digest(child)

    ranked_edge_visits = sorted(edge_visits, reverse=True)
    top_visit = ranked_edge_visits[0] if ranked_edge_visits else 0
    runner_up = ranked_edge_visits[1] if len(ranked_edge_visits) > 1 else 0
    total_edge_visits = int(sum(edge_visits))
    tied_at_max = [
        action for action, count in enumerate(edge_visits)
        if count == top_visit and expanded[action]
    ]
    other_selected_visits = [
        count for action, count in enumerate(edge_visits)
        if action != selected_action and expanded[action]
    ]
    best_other = max(other_selected_visits, default=0)

    expanded_actions = [action for action in action_ids if expanded[action]]

    def argmax_nullable(values: Sequence[float | None]) -> int | None:
        eligible = [action for action in expanded_actions
                    if values[action] is not None]
        if not eligible:
            return None
        return min(eligible, key=lambda action: (-float(values[action]), action))

    signed_q = [
        None if value is None else float(mcts.sign * value)
        for value in q_values
    ]
    q_argmax = argmax_nullable(signed_q)
    u_argmax = argmax_nullable(u_values)
    score_argmax = argmax_nullable(scores)
    act_history = action_history_digest(root.state)
    record = {
        "schema_version": SCHEMA_VERSION,
        "provenance": dict(provenance),
        "instance": instance_name,
        "step": int(step),
        "elapsed_seconds": (
            None if elapsed_seconds is None else float(elapsed_seconds)),
        "policy_action": policy_action,
        "policy_action_name": action_names[policy_action],
        "selected_action": selected_action,
        "selected_action_name": action_names[selected_action],
        "override_path": [dict(stage) for stage in override_path],
        "state": {
            "physical_state_digest": _digest_bytes(
                root.state_key, person=b"asnet-state-key"),
            "action_history_digest": (
                None if act_history is None else act_history.hex()),
            "applicable_action_digest": _digest_array(
                applicable, dtype=np.uint8, person=b"asnet-app-mask"),
        },
        "root": {
            "root_visits": int(root.visit_count),
            "total_edge_visits": total_edge_visits,
            "expanded_width": len(expanded_actions),
            "act_dim": act_dim,
            "action_ids": action_ids,
            "action_names": action_names,
            "applicable": applicable.tolist(),
            "expanded": expanded,
            "raw_network_policy": raw_policy.tolist(),
            "masked_network_policy": masked_policy.tolist(),
            "visit_distribution": visits_distribution.tolist(),
            "edge_visit_counts": edge_visits,
            "edge_priors": edge_priors,
            "q_values": q_values,
            "u_values": u_values,
            "signed_q_plus_u": scores,
            "child_terminal": terminal,
            "child_goal": goal,
            "child_state_digests": child_state_digests,
        },
        "summary": {
            "policy_entropy": entropy(masked_policy),
            "visit_entropy": entropy(visits_distribution),
            "policy_visit_js_divergence": jensen_shannon_divergence(
                masked_policy, visits_distribution),
            "policy_action_expanded": bool(expanded[policy_action]),
            "selected_action_policy_prior": float(
                masked_policy[selected_action]),
            "selected_action_policy_rank": int(
                1 + np.count_nonzero(
                    masked_policy > masked_policy[selected_action])),
            "max_visit_tie_count": len(tied_at_max),
            "max_visit_tied_actions": tied_at_max,
            "top1_top2_visit_margin": int(top_visit - runner_up),
            "winner_visit_share": (
                float(top_visit / total_edge_visits)
                if total_edge_visits else 0.0),
            "selected_leave_one_out_margin": int(
                edge_visits[selected_action] - best_other),
            "selected_leave_one_out_ratio": (
                float(edge_visits[selected_action] / best_other)
                if best_other else None),
            "signed_q_argmax": q_argmax,
            "u_argmax": u_argmax,
            "signed_q_plus_u_argmax": score_argmax,
            "selected_is_signed_q_argmax": selected_action == q_argmax,
            "selected_is_u_argmax": selected_action == u_argmax,
            "selected_is_signed_q_plus_u_argmax": (
                selected_action == score_argmax),
        },
        "timing_cumulative": {
            "selection_seconds": float(mcts.selection_seconds),
            "successor_generation_seconds": float(
                getattr(mcts, "successor_generation_seconds", 0.0)),
            "network_inference_seconds": float(
                getattr(mcts, "network_inference_seconds", 0.0)),
            "evaluation_seconds": float(mcts.evaluation_seconds),
            "backpropagation_seconds": float(mcts.backpropagation_seconds),
        },
        "selection_depth_histogram": {
            str(depth): int(count)
            for depth, count in sorted(mcts.selection_depth_hist.items())
        },
    }
    return record
