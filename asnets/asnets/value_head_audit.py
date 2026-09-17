"""Small, opt-in primitives for the value-head quality audit.

The primary training and inference paths do not import this module.  It keeps
successor enumeration, batched raw-value evaluation, and row construction
separate from label generation so MCTS, continuation, and planner labels are
never silently pooled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SuccessorValueRow:
    state_id: str
    action_id: int
    successor_index: int
    transition_probability: float
    raw_network_value: float
    successor_terminal: bool
    successor_goal: bool


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
