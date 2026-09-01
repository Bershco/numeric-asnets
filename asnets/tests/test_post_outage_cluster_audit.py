import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = ROOT / "scripts" / "post_outage_cluster_audit.py"
    spec = importlib.util.spec_from_file_location("post_outage_cluster_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PostOutageDurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_slurm_durations(self):
        self.assertEqual(self.module.parse_duration("01:02:03"), 3723)
        self.assertEqual(self.module.parse_duration("2-01:02:03"), 176523)
        self.assertEqual(self.module.parse_duration("09:30"), 570)

    def test_duration_round_trip(self):
        self.assertEqual(self.module.format_duration(176523), "2-01:02:03")


class PostOutageClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def classify(self, **updates):
        values = {
            "expected": "running",
            "artifact_kind": "training",
            "state": "NODE_FAIL",
            "reason": "Node failure",
            "has_checkpoint": False,
            "eval_final": False,
            "partial_eval": False,
        }
        values.update(updates)
        return self.module.classify(**values)

    def test_interrupted_training_with_checkpoint_is_resumable(self):
        classification, action = self.classify(has_checkpoint=True)
        self.assertEqual(classification, "interrupted_resumable_training")
        self.assertIn("latest_checkpoint", action)

    def test_interrupted_mcts_with_partial_evidence_is_resumable(self):
        classification, action = self.classify(
            artifact_kind="mcts_evaluation", partial_eval=True)
        self.assertEqual(classification, "interrupted_resumable_mcts")
        self.assertIn("completion_jsonl", action)

    def test_resource_limited_mcts_is_a_terminal_fixed_budget_result(self):
        classification, _ = self.classify(
            artifact_kind="mcts_evaluation", state="OUT_OF_MEMORY",
            partial_eval=True)
        self.assertEqual(classification, "fixed_budget_terminal_result")

    def test_scientific_holds_stay_held(self):
        classification, action = self.classify(
            expected="deliberately_held", artifact_kind="mcts_evaluation",
            state="PENDING", reason="JobHeldUser")
        self.assertEqual(classification, "held_intact")
        self.assertEqual(action, "keep_held_and_review_state")

    def test_missing_scientific_hold_is_not_reported_intact(self):
        classification, action = self.classify(
            expected="deliberately_held", artifact_kind="mcts_evaluation",
            state="", reason="")
        self.assertEqual(classification, "held_missing_from_scheduler")
        self.assertIn("recreate_as_held", action)

    def test_known_terminal_row_is_not_blindly_rerun_if_accounting_is_lost(self):
        classification, action = self.classify(
            expected="terminal", artifact_kind="mcts_evaluation",
            state="", reason="")
        self.assertEqual(classification, "preoutage_terminal_accounting_missing")
        self.assertIn("preserve_local_static_result", action)

    def test_completed_controller_is_not_resubmitted_blindly(self):
        classification, action = self.classify(
            artifact_kind="controller", state="COMPLETED", reason="")
        self.assertEqual(classification, "controller_completed")
        self.assertEqual(action, "verify_idempotent_outputs")


class PostOutageArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_bounded_log_parser_finds_training_and_eval_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            (snapshots / "snapshot_41_0.750").touch()
            log = root / "job.out"
            log.write_text(
                "\n".join((
                    f"Snapshot directory: {snapshots}",
                    "epoch: validation 41/100",
                    "[EVAL][PLAN] instance=p01.pddl steps=2",
                    "[EVAL FINAL] success=1/20",
                )),
                encoding="utf-8",
            )
            parsed = self.module.parse_artifacts(log)
            self.assertEqual(parsed["latest_epoch"], 41)
            self.assertEqual(parsed["target_epoch"], 100)
            self.assertTrue(str(parsed["latest_checkpoint"]).endswith("snapshot_41_0.750"))
            self.assertEqual(parsed["eval_final_success"], 1)
            self.assertEqual(parsed["eval_final_total"], 20)
            self.assertEqual(parsed["printed_plan_instances"], 1)


class FrozenInventoryTests(unittest.TestCase):
    def test_inventory_is_unique_and_reconciles_held_resources(self):
        path = (
            ROOT / "experiment_tracking" /
            "pre_outage_job_inventory_20260901_1037.csv"
        )
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 162)
        self.assertEqual(len({row["job_id"] for row in rows}), 162)
        held = [
            row for row in rows
            if row["expected_pre_outage_state"] == "deliberately_held"
        ]
        self.assertEqual(len(held), 58)
        self.assertEqual(sum(int(row["cpus"]) for row in held), 340)
        self.assertEqual(sum(int(row["memory_gib"]) for row in held), 4130)

    def test_known_running_tail_plus_preserve_group_matches_snapshot(self):
        path = (
            ROOT / "experiment_tracking" /
            "pre_outage_job_inventory_20260901_1037.csv"
        )
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        exact = [row for row in rows if row["expected_pre_outage_state"] == "running"]
        self.assertEqual(len(exact) + 39, 61)
        self.assertEqual(sum(int(row["cpus"]) for row in exact) + 234, 366)
        self.assertEqual(sum(int(row["memory_gib"]) for row in exact) + 1872, 4512)


class SlurmQueryParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_squeue_queries_user_once_and_reports_untracked_jobs(self):
        output = "\n".join((
            "11|RUNNING|None|01:00:00|03:00:00|6|120G|node-a",
            "99|PENDING|Resources|00:00|01:00:00|2|8G|",
        ))
        with mock.patch.object(self.module, "run", return_value=output) as runner:
            tracked, all_rows = self.module.collect_squeue(["11"])
        self.assertEqual(set(tracked), {"11"})
        self.assertEqual([row["JobIDRaw"] for row in all_rows], ["11", "99"])
        command = runner.call_args.args[0]
        self.assertIn("-u", command)
        self.assertNotIn("-j", command)

    def test_sacct_discards_steps_and_retains_root_record(self):
        values = [
            "11", "job", "NODE_FAIL", "0:1", "01:00:00", "03:00:00",
            "6", "120G", "start", "end", "node-a", "Node failure",
            "/tmp/11.out", "/tmp",
        ]
        with mock.patch.object(self.module, "run", return_value="|".join(values)):
            parsed = self.module.collect_sacct(["11"])
        self.assertEqual(parsed["11"]["State"], "NODE_FAIL")
        self.assertEqual(parsed["11"]["StdOut"], "/tmp/11.out")


if __name__ == "__main__":
    unittest.main()
