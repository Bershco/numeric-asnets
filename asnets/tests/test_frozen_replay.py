import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from asnets.frozen_replay import FrozenReplaySchedule


class _Problem:
    def __init__(self, obs_dim=3, act_dim=2):
        self.obs_dim = obs_dim
        self.act_dim = act_dim


def _write_step(path: Path, *, signature="sig", offset=0.0):
    np.savez_compressed(
        path,
        obs_0=np.asarray([[1.0 + offset, 2.0, 3.0]], dtype=np.float32),
        pi_tgt_0=np.asarray([[0.25, 0.75]], dtype=np.float32),
        policy_weights_0=np.asarray([1.0], dtype=np.float32),
        problem_signature_0=np.asarray(signature),
    )


class FrozenReplayScheduleTests(unittest.TestCase):
    def test_loads_exact_numbered_batches_and_reports_file_digest(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            _write_step(directory / "optimizer_step_000.npz")
            _write_step(
                directory / "optimizer_step_001.npz", offset=1.0)
            schedule = FrozenReplaySchedule(directory, expected_steps=2)
            step = schedule.load_step(1, {"sig": _Problem()})
            self.assertEqual(step.index, 1)
            self.assertEqual(step.batches[0][0], "sig")
            np.testing.assert_array_equal(
                step.batches[0][1][0],
                np.asarray([[2.0, 2.0, 3.0]], dtype=np.float32),
            )
            self.assertEqual(
                step.sha256,
                hashlib.sha256(step.path.read_bytes()).hexdigest(),
            )

    def test_rejects_incomplete_schedule_before_training(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            _write_step(directory / "optimizer_step_000.npz")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                FrozenReplaySchedule(directory, expected_steps=2)

    def test_rejects_unknown_problem_signature(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            _write_step(directory / "optimizer_step_000.npz")
            schedule = FrozenReplaySchedule(directory, expected_steps=1)
            with self.assertRaisesRegex(ValueError, "no initialized bucket"):
                schedule.load_step(0, {"different": _Problem()})

    def test_legacy_blank_signature_requires_unique_shape_match(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            _write_step(
                directory / "optimizer_step_000.npz", signature="")
            schedule = FrozenReplaySchedule(directory, expected_steps=1)
            step = schedule.load_step(0, {
                "right": _Problem(obs_dim=3, act_dim=2),
                "other": _Problem(obs_dim=4, act_dim=3),
            })
            self.assertEqual(step.batches[0][0], "right")
            with self.assertRaisesRegex(ValueError, "no initialized bucket"):
                schedule.load_step(0, {
                    "ambiguous-a": _Problem(),
                    "ambiguous-b": _Problem(),
                })

    def test_rejects_shape_or_nonfinite_corruption(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            np.savez_compressed(
                directory / "optimizer_step_000.npz",
                obs_0=np.asarray([[1.0, np.nan, 3.0]], dtype=np.float32),
                pi_tgt_0=np.asarray([[0.25, 0.75]], dtype=np.float32),
                policy_weights_0=np.asarray([1.0], dtype=np.float32),
                problem_signature_0=np.asarray("sig"),
            )
            schedule = FrozenReplaySchedule(directory, expected_steps=1)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                schedule.load_step(0, {"sig": _Problem()})


if __name__ == "__main__":
    unittest.main()
