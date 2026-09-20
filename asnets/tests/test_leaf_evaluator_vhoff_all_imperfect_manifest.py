import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_mcts_leaf_evaluator_vhoff_all_imperfect.py"
MANIFEST = (
    ROOT
    / "experiment_tracking"
    / "mcts_leaf_evaluator_vhoff_all_imperfect_20260920"
    / "manifest.csv"
)
SPEC = importlib.util.spec_from_file_location("vhoff_leaf_manifest_verify", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class VHOffLeafEvaluatorManifestTests(unittest.TestCase):
    def setUp(self):
        self.rows = VERIFY.read_rows(MANIFEST)

    def test_frozen_two_arm_design_is_complete(self):
        summary = VERIFY.verify_manifest(self.rows)
        self.assertEqual(summary["science_tasks"], 24)
        self.assertEqual(summary["matched_cells"], 12)
        self.assertEqual({row["value_head"] for row in self.rows}, {"off"})
        self.assertEqual({row["arm"] for row in self.rows}, {"rollout", "current"})

    def test_value_head_or_budget_mutation_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["value_head"] = "on"
        with self.assertRaisesRegex(ValueError, "Stage-1 VH-off"):
            VERIFY.verify_manifest(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["width"] = "20"
        with self.assertRaisesRegex(ValueError, "width must be 5"):
            VERIFY.verify_manifest(rows)

    def test_runner_and_smoke_indices_freeze_vh_off(self):
        self.assertEqual(
            {self.rows[index]["manifest_id"] for index in (0, 4, 8, 12, 16, 20)},
            VERIFY.SMOKE_IDS,
        )
        runner = (
            ROOT / "scripts" / "mcts_leaf_evaluator_vhoff_all_imperfect.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("--disable-value-head --eval-with-mcts", runner)
        self.assertIn("ise-cpu-intl-[01,05,08-15,18,24-28]", runner)

    def test_unsolved_smoke_is_compatibility_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            done = Path(tmp)
            by_id = {row["manifest_id"]: row for row in self.rows}
            for manifest_id in VERIFY.SMOKE_IDS:
                row = by_id[manifest_id]
                payload = {
                    "manifest_id": manifest_id,
                    "mode": "smoke",
                    "value_head": "off",
                    "checkpoint_sha256": "not-empty",
                    "leaf_evaluator": "policy_rollout",
                    "width": int(row["width"]),
                    "iterations": int(row["iterations"]),
                    "code_commit": row["code_commit"],
                    "evaluation_outcome": "unsolved",
                    "coverage": 0,
                }
                (done / f"{manifest_id}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            summary = VERIFY.verify_smoke(self.rows, done)
            self.assertEqual(summary["compatibility_records"], 6)
            self.assertEqual(
                summary["release_basis"], "compatibility_only_not_performance"
            )


if __name__ == "__main__":
    unittest.main()
