import importlib.util
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "endogenous_kl_incremental_controller_20260918.py"
SPEC = importlib.util.spec_from_file_location("controller", SCRIPT)
controller = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(controller)


class IncrementalControllerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.campaign = self.root / "campaign"
        self.campaign.mkdir()
        self.arm = {
            "arm_index": "0", "domain": "drone", "seed": "7",
            "semantics": "legacy_dropout_current", "total": "20",
            "evaluation_module": "experiments_numeric.architecture_2.drone",
            "training_module": "experiments_numeric.architecture_2.drone_mcts",
            "teacher": "hadd-astar", "source_checkpoint": "/source",
            "source_checkpoint_sha256": "source-hash", "source_training_job_id": "1",
        }
        self.output = self.campaign / "outputs" / "arm_0_drone_7_legacy_dropout_current"
        self.snapshots = self.root / "snapshots"
        self.output.mkdir(parents=True)
        self.snapshots.mkdir()
        (self.output / "snapshot_root.txt").write_text(str(self.snapshots))
        (self.output / "training.stdout").write_text(
            "[VALIDATION] Current network validation success rate: 0.8\n"
        )

    def tearDown(self):
        self.temp.cleanup()

    def checkpoint(self, epoch, score, payload=b"weights"):
        path = self.snapshots / f"snapshot_{epoch}_{score}"
        path.mkdir()
        weights = path / "weights.joblib"
        weights.write_bytes(payload)
        os.utime(weights, (1, 1))
        return path

    def test_discovery_is_append_only_and_hash_frozen(self):
        checkpoint = self.checkpoint(0, "0.7500")
        rows = controller.discover_policy_rows(
            self.campaign, [self.arm], [], code_commit="abc",
            min_age_seconds=0, now=datetime.now(timezone.utc),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["identity"], "arm00-epoch0000")
        self.assertEqual(rows[0]["validation_score"], "0.8")
        (checkpoint / "weights.joblib").write_bytes(b"changed")
        os.utime(checkpoint / "weights.joblib", (1, 1))
        with self.assertRaisesRegex(RuntimeError, "identity mutated"):
            controller.discover_policy_rows(
                self.campaign, [self.arm], rows, code_commit="abc",
                min_age_seconds=0, now=datetime.now(timezone.utc),
            )

    def test_policy_result_requires_exact_identity_and_hash(self):
        self.checkpoint(0, "0.5000")
        row = controller.discover_policy_rows(
            self.campaign, [self.arm], [], code_commit="abc",
            min_age_seconds=0, now=datetime.now(timezone.utc),
        )[0]
        target = self.output / "policy_epoch_0000" / "result.json"
        target.parent.mkdir()
        target.write_text(json.dumps({
            "identity": row["identity"], "checkpoint_sha256": row["checkpoint_sha256"],
            "domain": "drone", "seed": 7, "semantics": "legacy_dropout_current",
            "epoch": 0, "total": 20, "seen": 20,
        }))
        self.assertTrue(controller.valid_policy_result(self.campaign, row))
        data = json.loads(target.read_text()); data["checkpoint_sha256"] = "wrong"
        target.write_text(json.dumps(data))
        self.assertFalse(controller.valid_policy_result(self.campaign, row))

    def test_continuation_segment_maps_local_to_canonical_epoch(self):
        self.checkpoint(0, "0.5000", b"base")
        segment = self.output / "continuations" / "segment_001"
        segment.mkdir(parents=True)
        continued = self.root / "continued"
        continued.mkdir()
        (segment / "plan.json").write_text(json.dumps({"start_epoch": 26}))
        (segment / "snapshot_root.txt").write_text(str(continued))
        (segment / "training.stdout").write_text(
            "[VALIDATION] Current network validation success rate: 0.9\n"
        )
        checkpoint = continued / "snapshot_0_1.0000"
        checkpoint.mkdir()
        (checkpoint / "weights.joblib").write_bytes(b"continued")
        os.utime(checkpoint / "weights.joblib", (1, 1))
        rows = controller.discover_policy_rows(
            self.campaign, [self.arm], [], code_commit="abc",
            min_age_seconds=0, now=datetime.now(timezone.utc),
        )
        self.assertEqual([row["epoch"] for row in rows], [0])
        # Epoch 26 is not a predeclared curve point, but epoch 30 is local 4.
        for local in range(1, 5):
            path = continued / f"snapshot_{local}_1.0000"
            path.mkdir(); (path / "weights.joblib").write_bytes(bytes([local]))
            os.utime(path / "weights.joblib", (1, 1))
        (segment / "training.stdout").write_text("".join(
            f"[VALIDATION] Current network validation success rate: {0.9-local/100}\n"
            for local in range(5)
        ))
        rows = controller.discover_policy_rows(
            self.campaign, [self.arm], rows, code_commit="abc",
            min_age_seconds=0, now=datetime.now(timezone.utc),
        )
        self.assertEqual([row["epoch"] for row in rows], [0, 30])
        self.assertEqual(rows[1]["validation_score"], "0.86")

    def test_validation_best_uses_earliest_epoch_and_builds_both_searches(self):
        policy_rows, results = [], []
        for arm in range(8):
            for position, epoch in enumerate(controller.EPOCHS):
                identity = f"arm{arm:02d}-epoch{epoch:04d}"
                score = 0.9 if epoch in {10, 15} else 0.5
                row = {
                    field: "" for field in controller.POLICY_FIELDS
                }
                row.update({
                    "identity": identity, "curve_index": str(arm * 21 + position),
                    "arm_index": str(arm), "domain": "drone", "seed": str(arm),
                    "semantics": "legacy_dropout_current", "epoch": str(epoch),
                    "validation_score": str(score), "total": "20",
                    "evaluation_module": "eval", "teacher": "hadd-astar",
                    "mcts_module": "eval_mcts",
                    "checkpoint": f"/c/{arm}/{epoch}",
                    "checkpoint_sha256": f"hash-{arm}-{epoch}",
                })
                policy_rows.append(row)
                results.append({
                    "identity": identity, "score": epoch % 20,
                    "evaluation_log": f"/log/{identity}",
                })
        mcts = controller.freeze_selected_and_mcts(
            self.campaign, policy_rows, [], code_commit="abc",
            now=datetime.now(timezone.utc),
        )
        selected = controller.read_csv(self.campaign / "selected_endpoints.csv")
        self.assertEqual({row["selected_epoch"] for row in selected}, {"10"})
        self.assertEqual(mcts, [])
        mcts = controller.freeze_selected_and_mcts(
            self.campaign, policy_rows, results, code_commit="abc",
            now=datetime.now(timezone.utc),
        )
        self.assertEqual(len(mcts), 16)
        self.assertEqual({row["method"] for row in mcts}, {"fixed20x70", "pw70"})


if __name__ == "__main__":
    unittest.main()
