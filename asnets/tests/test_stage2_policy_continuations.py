"""Regression tests for continuation-aware Stage-2 policy materialization."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "experiment_tracking" / "materialize_stage2_policy_from_ledger.py"
SPEC = spec_from_file_location("materialize_stage2_policy_from_ledger", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ContinuationMaterializationTests(unittest.TestCase):
    def test_continuation_segments_share_one_cumulative_axis(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            original = root_path / "original"
            continuation = root_path / "continuation"
            original.mkdir()
            continuation.mkdir()
            (original / "snapshot_75_1.0000").mkdir()
            (original / "snapshot_84_0.9000").mkdir()
            (continuation / "snapshot_0_0.9000").mkdir()
            (continuation / "snapshot_15_0.8000").mkdir()

            candidates = MODULE.override_candidates({
                "snapshot_segments": f"{original}@0;{continuation}@85",
            })

            self.assertEqual(sorted(candidates), [75, 84, 85, 100])
            self.assertEqual(candidates[75][0].name, "snapshot_75_1.0000")
            self.assertEqual(candidates[100][0].name, "snapshot_15_0.8000")

    def test_continuation_segments_reject_overlapping_epochs(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            first = root_path / "first"
            second = root_path / "second"
            first.mkdir()
            second.mkdir()
            (first / "snapshot_85_1.0000").mkdir()
            (second / "snapshot_0_1.0000").mkdir()

            with self.assertRaisesRegex(RuntimeError, "overlapping/ambiguous"):
                MODULE.override_candidates({
                    "snapshot_segments": f"{first}@0;{second}@85",
                })


if __name__ == "__main__":
    unittest.main()
