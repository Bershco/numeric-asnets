import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from asnets.validation_state import (
    TRAINER_STATE_FILE,
    ValidationState,
    cumulative_epoch_offset,
    load_trainer_state,
    validation_set_fingerprint,
)


def _spec(domain, problem, difficulty="EASY"):
    return SimpleNamespace(
        pddls=(str(domain), str(problem)),
        difficulty=SimpleNamespace(name=difficulty),
    )


class ValidationStateTests(unittest.TestCase):
    def test_fresh_run_cannot_load_unrelated_state(self):
        self.assertIsNone(load_trainer_state(None))
        state, legacy = ValidationState.restore(
            expected_fingerprint="current",
            expected_trainer_kind="stage1_imitation",
            trainer_state=None,
            is_resume=False)
        self.assertFalse(legacy)
        self.assertEqual(state.fingerprint, "current")
        self.assertIsNone(state.best_rate)

    def test_round_trip_restores_selection_and_counters(self):
        state = ValidationState(
            fingerprint="same", trainer_kind="stage1_imitation")
        self.assertTrue(state.observe(0.7, 12.0, "snapshot_10", 10))
        self.assertFalse(state.observe(0.6, 9.0, "snapshot_11", 11))
        trainer_state = {
            "epoch_num": 11,
            "cumulative_epoch": 111,
            "validation_state": state.to_dict(),
        }
        restored, legacy = ValidationState.restore(
            expected_fingerprint="same",
            expected_trainer_kind="stage1_imitation",
            trainer_state=trainer_state,
            is_resume=True)
        self.assertFalse(legacy)
        self.assertEqual(restored.best_rate, 0.7)
        self.assertEqual(restored.best_checkpoint, "snapshot_10")
        self.assertEqual(restored.best_cumulative_epoch, 10)
        self.assertEqual(restored.non_improving_count, 1)
        self.assertEqual(cumulative_epoch_offset(trainer_state), 112)

    def test_changed_validation_set_fails_clearly(self):
        state = ValidationState(
            fingerprint="old", trainer_kind="stage1_imitation")
        with self.assertRaisesRegex(ValueError, "does not match"):
            ValidationState.restore(
                expected_fingerprint="new",
                expected_trainer_kind="stage1_imitation",
                trainer_state={"validation_state": state.to_dict()},
                is_resume=True,
            )

    def test_legacy_checkpoint_requires_baseline(self):
        state, legacy = ValidationState.restore(
            expected_fingerprint="current",
            expected_trainer_kind="stage1_imitation",
            trainer_state={
                "epoch_num": 63, "iter_num": 64, "best_rate": 1.0},
            is_resume=True,
        )
        self.assertTrue(legacy)
        self.assertIsNone(state.best_rate)
        self.assertEqual(cumulative_epoch_offset({"epoch_num": 63}), 64)

    def test_legacy_direct_weight_file_requires_baseline(self):
        state, legacy = ValidationState.restore(
            expected_fingerprint="current",
            expected_trainer_kind="stage1_imitation",
            trainer_state=None,
            is_resume=True,
        )
        self.assertTrue(legacy)
        self.assertIsNone(state.best_rate)

    def test_stage_transition_does_not_restore_prior_validation_selection(self):
        old = ValidationState(
            fingerprint="same", trainer_kind="stage1_imitation")
        old.observe(1.0, 5.0, "stage1_best", 10)
        state, legacy = ValidationState.restore(
            expected_fingerprint="same",
            expected_trainer_kind="stage2_mcts",
            trainer_state={"validation_state": old.to_dict()},
            is_resume=True,
        )
        self.assertFalse(legacy)
        self.assertIsNone(state.best_rate)
        self.assertEqual(state.trainer_kind, "stage2_mcts")

    def test_legacy_stage_transition_is_inferred_from_trainer_state(self):
        state, legacy = ValidationState.restore(
            expected_fingerprint="same",
            expected_trainer_kind="stage2_mcts",
            trainer_state={"epoch_num": 63, "iter_num": 64},
            is_resume=True,
        )
        self.assertFalse(legacy)
        self.assertIsNone(state.best_rate)

    def test_checkpoint_file_loading_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "snapshot"
            checkpoint.mkdir()
            expected = {"epoch_num": 4}
            state_path = checkpoint / TRAINER_STATE_FILE
            with state_path.open("wb") as stream:
                pickle.dump(expected, stream)
            fake_joblib = SimpleNamespace(
                load=lambda path: pickle.loads(Path(path).read_bytes()))
            with mock.patch.dict(sys.modules, {"joblib": fake_joblib}):
                self.assertEqual(load_trainer_state(str(checkpoint)), expected)
                self.assertIsNone(
                    load_trainer_state(str(checkpoint / "weights.pkl")))

    def test_fingerprint_uses_pddl_contents_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            domain = Path(directory) / "domain.pddl"
            first = Path(directory) / "first.pddl"
            second = Path(directory) / "second.pddl"
            domain.write_text("domain")
            first.write_text("first")
            second.write_text("second")
            one = validation_set_fingerprint([
                _spec(domain, first), _spec(domain, second)])
            same = validation_set_fingerprint([
                _spec(domain, first), _spec(domain, second)])
            reordered = validation_set_fingerprint([
                _spec(domain, second), _spec(domain, first)])
            self.assertEqual(one, same)
            self.assertNotEqual(one, reordered)


if __name__ == "__main__":
    unittest.main()
