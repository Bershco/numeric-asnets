"""Small, opt-in primitives for the value-head quality audit.

The primary training and inference paths do not import this module.  It keeps
successor enumeration, batched raw-value evaluation, and row construction
separate from label generation so MCTS, continuation, and planner labels are
never silently pooled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .value_head_audit_manifest import validate_task_manifest_rows


STATE_RECORD_SCHEMA = "value-head-audit-canonical-state-v1"
STATE_HASH_FIELDS = (
    "schema", "atoms", "fluents", "aux_data", "aux_data_interp",
    "network_input", "is_goal", "is_terminal",
)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _state_hash_payload(record: Mapping[str, object]) -> dict[str, object]:
    return {field: record[field] for field in STATE_HASH_FIELDS}


def canonical_state_record(state, **metadata: object) -> dict[str, object]:
    """Serialize an exact audit state, including history-dependent features.

    Physical PDDL state alone is insufficient when action-history data
    generators are enabled.  The frozen payload therefore includes the exact
    auxiliary vector and its interpretation as well as a network-input witness.
    """

    # ``to_tup_state`` deliberately drops MDPSim special fluents because it is
    # intended for temporary PDDL initial-state replacement.  Exact round-trip
    # restoration needs the complete fluent vector used by ``to_mdpsim``.
    atoms = [
        proposition.unique_ident
        for proposition, truth in state.props_true if truth
    ]
    fluents = [
        (fluent.unique_ident, float(value))
        for fluent, value in state.flnt_values
    ]
    payload: dict[str, object] = {
        "schema": STATE_RECORD_SCHEMA,
        "atoms": atoms,
        "fluents": [[name, float(value)] for name, value in fluents],
        "aux_data": np.asarray(state.aux_data, dtype=np.float32).tolist(),
        "aux_data_interp": list(state._aux_data_interp or ()),
        "network_input": np.asarray(
            state.to_network_input(), dtype=np.float32
        ).tolist(),
        "is_goal": bool(state.is_goal),
        "is_terminal": bool(state.is_terminal),
    }
    payload.update(metadata)
    payload["state_sha256"] = hashlib.sha256(
        _canonical_json_bytes(_state_hash_payload(payload))
    ).hexdigest()
    return payload


def validate_canonical_state_record(record: Mapping[str, object]) -> None:
    """Reject a mutated or incomplete serialized canonical state."""

    if record.get("schema") != STATE_RECORD_SCHEMA:
        raise ValueError("unknown canonical-state record schema")
    recorded_hash = record.get("state_sha256")
    if not isinstance(recorded_hash, str):
        raise ValueError("state_sha256 is missing")
    try:
        hash_payload = _state_hash_payload(record)
    except KeyError as exc:
        raise ValueError(
            f"canonical-state record is missing {exc.args[0]}"
        ) from exc
    actual_hash = hashlib.sha256(_canonical_json_bytes(hash_payload)).hexdigest()
    if actual_hash != recorded_hash:
        raise ValueError("canonical-state record hash mismatch")
    for field in ("atoms", "fluents", "aux_data", "aux_data_interp", "network_input"):
        if field not in record:
            raise ValueError(f"canonical-state record is missing {field}")


def restore_canonical_state(record: Mapping[str, object], planner_exts):
    """Restore and verify a manifest state against one PlannerExtensions."""

    from .state_reprs import CanonicalState

    validate_canonical_state_record(record)
    prop_string = ", ".join(str(atom) for atom in record["atoms"])
    flnt_string = ", ".join(
        f"{name}: {float(value)}" for name, value in record["fluents"]
    )
    mdpsim_state = planner_exts.mdpsim_problem.intermediate_state(
        prop_string, flnt_string
    )
    state = CanonicalState.from_mdpsim(
        mdpsim_state, planner_exts, is_init_cstate=False
    )
    state._aux_data = np.asarray(record["aux_data"], dtype=np.float32)
    state._aux_data_interp = list(record["aux_data_interp"])
    state._aux_data_interp_to_id = {
        name: index for index, name in enumerate(state._aux_data_interp)
    }
    expected = np.asarray(record["network_input"], dtype=np.float32)
    actual = np.asarray(state.to_network_input(), dtype=np.float32)
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise ValueError("restored state does not reproduce exact network input")
    if bool(record["is_goal"]) != bool(state.is_goal):
        raise ValueError("restored state goal flag mismatch")
    if bool(record["is_terminal"]) != bool(state.is_terminal):
        raise ValueError("restored state terminal flag mismatch")
    return state


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


class ResultCacheLabelProvider:
    """Serve already-computed per-successor statuses without semantic fallback."""

    def __init__(
        self,
        *,
        label_source: str,
        results: Mapping[tuple[str, int, int], tuple[str, float | None]],
        higher_is_better: bool,
        scale_comparable: bool,
        label_log_path: str,
        missing_status: str = "missing",
    ):
        self._label_source = label_source
        self._results = results
        self._higher_is_better = bool(higher_is_better)
        self._scale_comparable = bool(scale_comparable)
        self._label_log_path = label_log_path
        self._missing_status = missing_status

    def label(self, row: SuccessorValueRow) -> LabelResult:
        status, value = self._results.get(
            successor_key(row), (self._missing_status, None)
        )
        return validate_label_result(LabelResult(
            label_source=self._label_source,
            label_status=status,
            label_value=value,
            label_higher_is_better=self._higher_is_better,
            label_scale_comparable=self._scale_comparable,
            label_log_path=self._label_log_path,
        ))


class ReplayTargetProvider(ResultCacheLabelProvider):
    """Exact persisted replay ``z``; absent identities remain explicitly missing."""

    def __init__(self, results, *, label_log_path: str):
        super().__init__(
            label_source="replay_target", results=results,
            higher_is_better=True, scale_comparable=True,
            label_log_path=label_log_path, missing_status="missing",
        )


class DeterministicContinuationProvider(ResultCacheLabelProvider):
    """Checkpoint-specific remaining action count with explicit failures."""

    def __init__(self, results, *, label_log_path: str):
        super().__init__(
            label_source="deterministic_continuation", results=results,
            higher_is_better=False, scale_comparable=False,
            label_log_path=label_log_path, missing_status="error",
        )


class ENHSPRawProvider(ResultCacheLabelProvider):
    """Independent raw ENHSP heuristic values, never silently transformed."""

    def __init__(self, results, *, label_log_path: str):
        super().__init__(
            label_source="enhsp_raw_h", results=results,
            higher_is_better=False, scale_comparable=False,
            label_log_path=label_log_path, missing_status="error",
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


def enhsp_search_value(
    raw_h: float, *, coefficient: float = 1.0, minimization: bool = False
) -> float:
    """Apply the exact ENHSP transform used by current search.

    This mirrors ``post_training.training_mcts.get_est_v`` after the raw
    estimator result is available.  V1 freezes ``coefficient=1.0`` and
    ``minimization=False`` in a checksumed configuration artifact.
    """

    raw_h = float(raw_h)
    coefficient = float(coefficient)
    if not np.isfinite(raw_h) or raw_h < 0:
        raise ValueError("raw ENHSP h must be finite and nonnegative")
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("ENHSP coefficient must be finite and positive")
    if minimization:
        return raw_h
    return float(np.exp(-coefficient * raw_h))


class ENHSPSearchValueProvider:
    """Transform an independent raw-ENHSP provider into search-scale value."""

    def __init__(
        self,
        raw_provider: LabelProvider,
        *,
        coefficient: float = 1.0,
        minimization: bool = False,
    ):
        self._raw_provider = raw_provider
        self._coefficient = coefficient
        self._minimization = minimization

    def label(self, row: SuccessorValueRow) -> LabelResult:
        raw = validate_label_result(self._raw_provider.label(row))
        if raw.label_source != "enhsp_raw_h":
            raise ValueError("ENHSPSearchValueProvider requires enhsp_raw_h")
        value = None
        if raw.label_status == "valid":
            value = enhsp_search_value(
                raw.label_value,
                coefficient=self._coefficient,
                minimization=self._minimization,
            )
        return LabelResult(
            label_source="enhsp_search_v",
            label_status=raw.label_status,
            label_value=value,
            label_higher_is_better=not self._minimization,
            label_scale_comparable=not self._minimization,
            label_log_path=raw.label_log_path,
        )


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
