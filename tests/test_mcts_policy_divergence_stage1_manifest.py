import csv
import hashlib
import json
import unittest
from pathlib import Path


AUDIT_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiment_tracking"
    / "mcts_policy_divergence_cause_audit"
)


class Stage1GroupedManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (AUDIT_DIR / "stage1_grouped_tasks.csv").open(
                newline="", encoding="utf-8-sig") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.freeze = json.loads(
            (AUDIT_DIR / "stage1_grouped_tasks.freeze.json").read_text(
                encoding="utf-8"))

    def test_expected_task_and_candidate_counts_are_frozen(self):
        self.assertEqual(len(self.rows), 12)
        self.assertEqual(
            sum(row["release_class"] == "primary_fixed"
                for row in self.rows), 10)
        self.assertEqual(
            sum(row["release_class"] == "optional_fo_pw"
                for row in self.rows), 2)
        self.assertEqual(
            sum(int(row["candidate_count"]) for row in self.rows), 40)
        self.assertTrue(all(row["submitted"] == "false" for row in self.rows))

    def test_fixed_search_config_matches_source_experiment_by_domain(self):
        expected = {
            "block_grouping": (5, 20),
            "counters": (5, 20),
            "drone": (20, 70),
            "fo_counters": (20, 70),
            "rover": (20, 70),
        }
        for row in self.rows:
            if row["search_family"] != "fixed":
                continue
            config = json.loads(row["search_config"])
            with self.subTest(task=row["task_id"]):
                self.assertEqual(
                    (config["mcts_expansion_k"], config["mcts_iterations"]),
                    expected[row["domain"]],
                )
                self.assertEqual(config["mcts_exploration_weight"], 0.1)
                self.assertEqual(config["estimator_coeff"], 0.5)
                self.assertFalse(config["progressive_widening"])

    def test_optional_fo_pw_and_freeze_checksums(self):
        optional = [
            row for row in self.rows
            if row["release_class"] == "optional_fo_pw"
        ]
        self.assertEqual({row["domain"] for row in optional}, {"fo_counters"})
        self.assertEqual({row["value_head"] for row in optional}, {"off", "on"})
        for row in optional:
            config = json.loads(row["search_config"])
            self.assertEqual(config["mcts_expansion_k"], 20)
            self.assertEqual(config["mcts_iterations"], 70)
            self.assertEqual(config["mcts_exploration_weight"], 0.1)
            self.assertEqual(config["estimator_coeff"], 0.5)
            self.assertTrue(config["progressive_widening"])
            self.assertEqual(config["pw_min_width"], 3)
        grouped_hash = hashlib.sha256(
            (AUDIT_DIR / "stage1_grouped_tasks.csv").read_bytes()).hexdigest()
        source_hash = hashlib.sha256(
            (AUDIT_DIR / "stage0_missing_strata_manifest.csv")
            .read_bytes()).hexdigest()
        self.assertEqual(
            grouped_hash, self.freeze["grouped_manifest_sha256"])
        self.assertEqual(
            source_hash, self.freeze["source_manifest_sha256"])
        self.assertFalse(self.freeze["submitted"])


if __name__ == "__main__":
    unittest.main()
