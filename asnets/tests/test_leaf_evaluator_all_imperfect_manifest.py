import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_mcts_leaf_evaluator_all_imperfect.py"
MANIFEST = (
    ROOT
    / "experiment_tracking"
    / "mcts_leaf_evaluator_all_imperfect_20260920"
    / "manifest.csv"
)
SPEC = importlib.util.spec_from_file_location("leaf_manifest_verify", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class LeafEvaluatorAllImperfectManifestTests(unittest.TestCase):
    def setUp(self):
        self.rows = VERIFY.read_rows(MANIFEST)

    def test_frozen_extension_is_complete_and_nonduplicative(self):
        summary = VERIFY.verify_manifest(self.rows)
        self.assertEqual(summary["new_science_tasks"], 16)
        self.assertEqual(summary["complete_design_tasks"], 32)
        self.assertEqual(
            {row["domain"] for row in self.rows},
            {"block_grouping", "rover", "counters", "mprime"},
        )

    def test_domain_budget_mutation_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["width"] = "20"
        with self.assertRaisesRegex(ValueError, "width must be 5"):
            VERIFY.verify_manifest(rows)

    def test_unsolved_smoke_is_compatibility_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            done = Path(tmp)
            by_id = {row["manifest_id"]: row for row in self.rows}
            for manifest_id in VERIFY.SMOKE_IDS:
                row = by_id[manifest_id]
                payload = {
                    "manifest_id": manifest_id,
                    "mode": "smoke",
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
            self.assertEqual(
                summary["release_basis"],
                "compatibility_only_not_performance",
            )


if __name__ == "__main__":
    unittest.main()
