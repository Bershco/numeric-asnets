from pathlib import Path
import unittest

from scripts.build_mprime_anchor_phase_b_a_manifest_20260912 import build_rows
from scripts.mprime_anchor_phase_b_a_rescore_20260912 import select_checkpoints


ROOT = Path(__file__).resolve().parents[2]


class MPrimeAnchorPhaseBARescoreTests(unittest.TestCase):
    def test_manifest_freezes_complete_anchor_grid(self):
        rows = build_rows(
            ROOT / "experiment_tracking/mprime_validation_ipc_scale_v1/anchor_selection_invalid.csv",
            ROOT / "experiment_tracking/mprime_validation_ipc_scale_v1/anchor_tuning_submissions.tsv",
            ROOT / "experiment_tracking/mprime_validation_phase_b_20260906/frozen_validation_manifest.csv",
        )
        self.assertEqual(len(rows), 28)
        self.assertEqual(sum(int(row["expected_checkpoint_count"]) for row in rows), 588)
        self.assertEqual({row["validation_replicate"] for row in rows}, {"phase_b_a"})

    def test_checkpoint_selection_is_every_five_plus_final(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch in range(100):
                (root / f"snapshot_{epoch}_1.0000").mkdir()
            selected = select_checkpoints(root)
            self.assertEqual(
                [epoch for epoch, _ in selected], list(range(0, 100, 5)) + [99]
            )

    def test_checkpoint_selection_rejects_incomplete_lineage(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch in range(15):
                (root / f"snapshot_{epoch}_1.0000").mkdir()
            with self.assertRaisesRegex(RuntimeError, "expected 21"):
                select_checkpoints(root)


if __name__ == "__main__":
    unittest.main()
