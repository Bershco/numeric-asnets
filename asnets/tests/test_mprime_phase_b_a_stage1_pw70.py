import tempfile
import unittest
from pathlib import Path

from scripts.build_mprime_phase_b_a_stage1_pw70_20260913 import build_rows, write_manifest
from scripts.run_mprime_phase_b_a_stage1_pw70_20260913 import experiment_arguments, load_row


class MPrimePhaseBAStage1PW70Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_rows()

    def test_exact_predeclared_2x2_screen(self):
        self.assertEqual(len(self.rows), 4)
        self.assertEqual(
            {(row["value_head"], row["seed"]) for row in self.rows},
            {(vh, seed) for vh in ("off", "on") for seed in ("1963100312", "2011206605")},
        )

    def test_only_pw_differs_from_fixed_identity(self):
        for row in self.rows:
            self.assertEqual(row["checkpoint_selection"], "phase_b_replicate_a")
            self.assertEqual(
                (row["width"], row["iterations"], row["puct"], row["estimator"]),
                ("20", "70", "0.1", "0.5"),
            )
            self.assertEqual(
                (row["pw_min_width"], row["pw_c"], row["pw_alpha"], row["terminal_safe"]),
                ("3", "0.6", "0.5", "false"),
            )
            self.assertEqual(
                (row["workers"], row["cpus"], row["memory"], row["walltime"]),
                ("3", "6", "120G", "3-00:00:00"),
            )
            self.assertTrue(row["source_fixed_manifest_sha256"])

    def test_runtime_command_is_pw70_without_terminal_safe(self):
        args = experiment_arguments(self.rows[0], Path("completion.jsonl"))
        self.assertIn("--mcts-progressive-widening", args)
        self.assertEqual(args[args.index("--mcts-pw-min-width") + 1], "3")
        self.assertEqual(args[args.index("--mcts-iterations") + 1], "70")
        self.assertNotIn("--eval-mcts-terminal-safe-action-selection", args)

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            write_manifest(path, self.rows)
            for index in range(4):
                self.assertEqual(load_row(path, index)["array_index"], str(index))


if __name__ == "__main__":
    unittest.main()
