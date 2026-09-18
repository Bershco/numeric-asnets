"""Small, opt-in primitives for the value-head quality audit.

The primary training and inference paths do not import this module.  It keeps
successor enumeration, batched raw-value evaluation, and row construction
separate from label generation so MCTS, continuation, and planner labels are
never silently pooled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .value_head_audit_manifest import validate_task_manifest_rows


@dataclass(frozen=True)
class SuccessorValueRow:
    state_id: str
    action_id: int
    successor_index: int
    transition_probability: float
    raw_network_value: float
    successor_terminal: bool
    successor_goal: bool


@dataclass(frozen=True)
class LabelResult:
    """One label-family result for a single, stable successor identity.

    A non-``valid`` result never carries a numeric value.  In particular,
    timeout and unsolved are observations, not substitute heuristic values.
    """

    label_source: str
    label_status: str
    label_value: float | None
    label_higher_is_better: bool
    label_scale_comparable: bool
    label_log_path: str


class LabelProvider(Protocol):
    """Resolve one label family without falling back to another family."""

    def label(self, row: SuccessorValueRow) -> LabelResult:
        ...


def successor_key(row: SuccessorValueRow) -> tuple[str, int, int]:
    """Return the stable join key shared by all independent label caches."""

    return row.state_id, row.action_id, row.successor_index


class MappingLabelProvider:
    """Label provider for a checksumed, already-materialized numeric cache."""

    def __init__(
        self,
        *,
        label_source: str,
        values: Mapping[tuple[str, int, int], float],
        higher_is_better: bool,
        scale_comparable: bool,
        label_log_path: str,
    ):
        if not label_source:
            raise ValueError("label_source must be non-empty")
        if not label_log_path:
            raise ValueError("label_log_path must be non-empty")
        self._label_source = label_source
        self._values = values
        self._higher_is_better = bool(higher_is_better)
        self._scale_comparable = bool(scale_comparable)
        self._label_log_path = label_log_path

    def label(self, row: SuccessorValueRow) -> LabelResult:
        value = self._values.get(successor_key(row))
        return LabelResult(
            label_source=self._label_source,
            label_status="valid" if value is not None else "missing",
            label_value=None if value is None else float(value),
            label_higher_is_better=self._higher_is_better,
            label_scale_comparable=self._scale_comparable,
            label_log_path=self._label_log_path,
        )


class CallableLabelProvider:
    """Adapter for continuation/planner runners that preserve status.

    The injected callable is responsible for resolving the persisted
    successor identity and returning a :class:`LabelResult`.  This keeps the
    audit module independent of ENHSP and simulator startup while still
    enforcing the no-fallback/no-timeout-as-number contract.
    """

    def __init__(self, fn: Callable[[SuccessorValueRow], LabelResult]):
        self._fn = fn

    def label(self, row: SuccessorValueRow) -> LabelResult:
        return validate_label_result(self._fn(row))


def validate_label_result(result: LabelResult) -> LabelResult:
    """Validate status/value semantics shared by every V1 label family."""

    allowed = {"valid", "missing", "timeout", "unsolved", "error", "not_applicable"}
    if result.label_status not in allowed:
        raise ValueError(f"unknown label status: {result.label_status}")
    if not result.label_source:
        raise ValueError("label_source must be non-empty")
    if not result.label_log_path:
        raise ValueError("label_log_path must be non-empty")
    if result.label_status == "valid":
        if result.label_value is None or not np.isfinite(result.label_value):
            raise ValueError("valid label requires a finite numeric value")
    elif result.label_value is not None:
        raise ValueError(
            f"{result.label_status} label must not carry a numeric value"
        )
    return result


def apply_label_provider(
    rows: Sequence[SuccessorValueRow], provider: LabelProvider
) -> list[dict[str, object]]:
    """Join exactly one label family to raw values without pooling families."""

    labelled: list[dict[str, object]] = []
    for row in rows:
        result = validate_label_result(provider.label(row))
        labelled.append({**asdict(row), **asdict(result)})
    return labelled


def evaluate_successor_values(
    *,
    state,
    state_id: str,
    network,
    successor_fn: Callable[[object, int], Iterable[tuple[float, object]]],
) -> list[SuccessorValueRow]:
    """Enumerate applicable successors and evaluate raw values in one batch.

    ``successor_fn`` is injected deliberately: the production worker can use
    its numeric-domain simulator while unit tests use a small deterministic
    stand-in.  Stochastic actions produce one row per probabilistic successor.
    """

    mask = np.asarray(state.get_applicable_action_mask(), dtype=bool)
    pending: list[tuple[int, int, float, object]] = []
    for action_id in np.flatnonzero(mask):
        successors = list(successor_fn(state, int(action_id)))
        if not successors:
            raise ValueError(f"applicable action {action_id} has no successor")
        probability_sum = sum(float(probability) for probability, _ in successors)
        if not np.isclose(probability_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"action {action_id} successor probabilities sum to {probability_sum}"
            )
        for successor_index, (probability, successor) in enumerate(successors):
            pending.append((int(action_id), successor_index, float(probability), successor))

    if not pending:
        return []
    observations = np.asarray(
        [successor.to_network_input() for _, _, _, successor in pending],
        dtype=np.float32,
    )
    output = network(observations, training=False)
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise ValueError("value-head audit requires a VH-on network output")
    raw_values = np.asarray(output[1]).reshape(-1)
    if len(raw_values) != len(pending):
        raise ValueError(
            f"network returned {len(raw_values)} values for {len(pending)} successors"
        )

    return [
        SuccessorValueRow(
            state_id=state_id,
            action_id=action_id,
            successor_index=successor_index,
            transition_probability=probability,
            raw_network_value=float(raw_value),
            successor_terminal=bool(successor.is_terminal),
            successor_goal=bool(successor.is_goal),
        )
        for (action_id, successor_index, probability, successor), raw_value
        in zip(pending, raw_values)
    ]


def rows_as_dicts(rows: Sequence[SuccessorValueRow]) -> list[dict[str, object]]:
    """Return stable serializable rows for CSV/JSONL writers."""

    return [asdict(row) for row in rows]
