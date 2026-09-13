import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_mprime_phase_b_a_stage1_mcts_20260913 import (
    CHECKPOINT_SCORES,
    SELECTORS,
    build_rows,
    write_manifest,
)
from scripts.run_mprime_phase_b_a_stage1_mcts_20260913 import (
    ensure_val_summary,
    experiment_arguments,
    load_row,
)


class MPrimePhaseBAStage1MCTSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grouped = build_rows(SELECTORS, CHECKPOINT_SCORES)

    def test_exact_twenty_campaign_split_by_value_head(self):
        self.assertEqual(set(self.grouped), {"off", "on"})
        self.assertEqual(len(self.grouped["off"]), 10)
        self.assertEqual(len(self.grouped["on"]), 10)
        self.assertEqual(
            {row["seed"] for row in self.grouped["off"]},
            {row["seed"] for row in self.grouped["on"]},
        )
        self.assertEqual(
            len({row["checkpoint"] for rows in self.grouped.values() for row in rows}),
            20,
        )

    def test_identity_and_configuration_are_frozen(self):
        for vh, rows in self.grouped.items():
            for index, row in enumerate(rows):
                self.assertEqual(row["array_index"], str(index))
                self.assertEqual(row["value_head"], vh)
                self.assertIn(f"-{vh}-{row['seed']}-", row["manifest_id"])
                self.assertEqual(row["checkpoint_selection"], "phase_b_replicate_a")
                self.assertEqual(row["domain_module"], "experiments_numeric.domain.mprime")
                self.assertEqual(row["architecture_module"], "experiments_numeric.architecture_2.mprime_mcts")
                self.assertEqual(
                    (
                        row["width"], row["iterations"], row["puct"], row["estimator"],
                        row["workers"], row["cpus"], row["memory"], row["walltime"],
                        row["instance_timeout_seconds"], row["max_external_actions"],
                    ),
                    ("20", "70", "0.1", "0.5", "3", "6", "120G", "3-00:00:00", "21600", "10000"),
                )
                self.assertTrue(row["source_training_log"])
                self.assertTrue(row["source_policy_log"])
                self.assertTrue(row["selector_source"].startswith("experiment_tracking/"))
                self.assertTrue(row["checkpoint_score_source"].startswith("experiment_tracking/"))
                self.assertIn(
                    f"/snapshot_{int(row['selected_epoch'])}_",
                    row["checkpoint"].replace("\\", "/"),
                )

    def test_runtime_loader_rejects_a_mixed_value_head_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.csv"
            rows = [dict(self.grouped["off"][0]), dict(self.grouped["on"][0])]
            rows.extend(dict(row) for row in self.grouped["off"][2:])
            for index, row in enumerate(rows):
                row["array_index"] = str(index)
            write_manifest(path, rows)
            with self.assertRaisesRegex(ValueError, "only one value-head"):
                load_row(path, 0)

    def test_full_and_smoke_commands_preserve_search_identity(self):
        row = self.grouped["off"][0]
        full = experiment_arguments(row, Path("completion.jsonl"))
        self.assertIn("--disable-value-head", full)
        self.assertEqual(full[full.index("--num-workers") + 1], "3")
        self.assertEqual(full[full.index("--eval-instance-timeout") + 1], "21600")
        self.assertNotIn("--skip-instance-numbers", full)

        smoke = experiment_arguments(
            row, Path("smoke.jsonl"), smoke_instance=1, smoke_timeout=600
        )
        self.assertEqual(smoke[smoke.index("--num-workers") + 1], "1")
        self.assertEqual(smoke[smoke.index("--eval-instance-timeout") + 1], "600")
        skipped = smoke[smoke.index("--skip-instance-numbers") + 1]
        self.assertEqual(skipped, ",".join(str(number) for number in range(2, 21)))

    def test_manifests_round_trip_through_runtime_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            for vh, rows in self.grouped.items():
                path = Path(directory) / f"manifest_{vh}.csv"
                write_manifest(path, rows)
                with path.open(newline="", encoding="utf-8") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 10)
                for index in range(10):
                    self.assertEqual(load_row(path, index)["value_head"], vh)

    def test_zero_success_attempt_still_has_a_val_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.val.csv"
            ensure_val_summary(path)
            ensure_val_summary(path)
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ["instance,candidate_plans,selected_steps,val_valid,reason"],
            )


if __name__ == "__main__":
    unittest.main()
