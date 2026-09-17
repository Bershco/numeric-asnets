from __future__ import annotations

import unittest

import numpy as np

from asnets.value_head_audit import evaluate_successor_values, rows_as_dicts


class _State:
    def __init__(self, obs, *, mask=None, terminal=False, goal=False):
        self._obs = np.asarray(obs, dtype=np.float32)
        self._mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.is_terminal = terminal
        self.is_goal = goal

    def get_applicable_action_mask(self):
        return self._mask

    def to_network_input(self):
        return self._obs


class _ValueNetwork:
    def __call__(self, observations, *, training):
        assert training is False
        values = np.sum(observations, axis=1, keepdims=True)
        policy = np.zeros((len(observations), 3), dtype=np.float32)
        return policy, values


class ValueHeadAuditTest(unittest.TestCase):
    def test_batches_all_applicable_successors_and_preserves_identity(self):
        root = _State([0], mask=[True, False, True])
        successors = {
            0: [(1.0, _State([1, 2], terminal=False, goal=False))],
            2: [
                (0.25, _State([3, 4], terminal=True, goal=False)),
                (0.75, _State([5, 6], terminal=True, goal=True)),
            ],
        }
        rows = evaluate_successor_values(
            state=root,
            state_id="state-7",
            network=_ValueNetwork(),
            successor_fn=lambda _state, action: successors[action],
        )
        self.assertEqual([row.action_id for row in rows], [0, 2, 2])
        self.assertEqual([row.successor_index for row in rows], [0, 0, 1])
        self.assertEqual([row.raw_network_value for row in rows], [3.0, 7.0, 11.0])
        self.assertAlmostEqual(sum(row.transition_probability for row in rows[1:]), 1.0)
        self.assertTrue(rows[-1].successor_goal)
        self.assertEqual(rows_as_dicts(rows)[0]["state_id"], "state-7")

    def test_rejects_invalid_transition_distribution(self):
        root = _State([0], mask=[True])
        with self.assertRaisesRegex(ValueError, "probabilities sum"):
            evaluate_successor_values(
                state=root,
                state_id="bad",
                network=_ValueNetwork(),
                successor_fn=lambda _state, _action: [(0.5, _State([1]))],
            )

    def test_requires_value_head_output(self):
        root = _State([0], mask=[True])
        with self.assertRaisesRegex(ValueError, "VH-on"):
            evaluate_successor_values(
                state=root,
                state_id="novh",
                network=lambda observations, training: np.zeros((len(observations), 1)),
                successor_fn=lambda _state, _action: [(1.0, _State([1]))],
            )


if __name__ == "__main__":
    unittest.main()
