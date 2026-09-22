from dataclasses import dataclass
import pickle
import unittest

import numpy as np

from asnets.spawn_train_worker import (
    DataSource,
    GoalConditionedState,
    WorkerCollector,
    her_monte_carlo_value_target,
    positive_dynamic_goal_mask,
    satisfies_positive_goal,
    validate_her_options,
)
from asnets.explorer import WeightedReplayBuffer


@dataclass(frozen=True)
class FakeState:
    props_true: tuple
    raw_input: tuple

    def to_network_input(self):
        return np.asarray(self.raw_input, dtype=np.float32)


def state(*truths):
    return FakeState(
        tuple((f"p{index}", bool(value)) for index, value in enumerate(truths)),
        (1.0, 2.0, 3.0),
    )


class HerFutureTests(unittest.TestCase):
    def test_future_goal_uses_only_newly_true_dynamic_facts(self):
        current = state(True, False, False, True)
        future = state(True, True, False, True)
        mask = positive_dynamic_goal_mask(current, future)
        self.assertEqual(mask, (0.0, 1.0, 0.0, 0.0))
        self.assertFalse(satisfies_positive_goal(current, mask))
        self.assertTrue(satisfies_positive_goal(future, mask))

    def test_goal_conditioned_input_changes_only_by_explicit_tail(self):
        base = state(False, False)
        relabelled = GoalConditionedState(base, (0.0, 1.0)).to_network_input()
        np.testing.assert_array_equal(relabelled[:3], base.to_network_input())
        np.testing.assert_array_equal(relabelled[3:], np.asarray([0.0, 1.0]))

    def test_value_target_is_declared_monte_carlo_return(self):
        mask = (0.0, 1.0)
        achieved = state(False, True)
        self.assertEqual(
            her_monte_carlo_value_target(achieved, mask, 2, gamma=1.0),
            1.0,
        )
        self.assertAlmostEqual(
            her_monte_carlo_value_target(achieved, mask, 2, gamma=0.9),
            0.81,
        )
        with self.assertRaises(ValueError):
            her_monte_carlo_value_target(
                state(False, False), mask, 2, gamma=1.0)

    def test_goal_conditioned_state_retains_provenance(self):
        wrapped = GoalConditionedState(
            state(False, False), (0.0, 1.0), provenance="HER_FUTURE")
        self.assertEqual(wrapped.provenance, "HER_FUTURE")
        restored = pickle.loads(pickle.dumps(wrapped))
        self.assertEqual(restored, wrapped)
        self.assertEqual(hash(restored), hash(wrapped))

    def test_replay_provenance_survives_insertion_and_eviction(self):
        replay = WeightedReplayBuffer()
        replay.update(["ordinary", "ordinary"], provenance="TRAJECTORY")
        replay.update(["relabelled"], provenance="HER_FUTURE")
        self.assertEqual(
            replay.provenance_counter,
            {"TRAJECTORY": 2, "HER_FUTURE": 1},
        )
        replay.remove_oldest()
        self.assertEqual(replay.provenance_counter, {"HER_FUTURE": 1})

    def test_her_off_collector_is_byte_equivalent(self):
        base = state(False, True)
        collector = WorkerCollector()
        collector.add_sample(
            cstate=base,
            children=None,
            action=0,
            pi=np.asarray([1.0], dtype=np.float32),
            z=0.0,
            source=DataSource.TRAJECTORY,
        )
        obs, _pi, _z = collector.as_batches()
        self.assertEqual(
            obs[0].tobytes(), base.to_network_input().tobytes())

    def test_invalid_or_confounded_her_options_fail_closed(self):
        for strategy, her_k, tree_k in (
                ("off", 1, 0), ("future", 0, 0),
                ("future", 1, 10), ("bogus", 1, 0)):
            with self.subTest(strategy=strategy, her_k=her_k, tree_k=tree_k):
                with self.assertRaises(ValueError):
                    validate_her_options(strategy, her_k, tree_k)

    def test_valid_her_options_are_explicit(self):
        validate_her_options("off", 0, 0)
        validate_her_options("future", 1, 0)


if __name__ == "__main__":
    unittest.main()
