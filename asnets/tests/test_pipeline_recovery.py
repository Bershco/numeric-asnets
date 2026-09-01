import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfirmationCheckpointRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(
            "materialize_confirmation_policy",
            ROOT / "experiment_tracking" / "materialize_confirmation_policy.py",
        )

    def test_normal_shutdown_checkpoint_wins(self):
        text = "\n".join((
            "Snapshot directory: /tmp/run/snapshots",
            "Last valid checkpoint is /tmp/run/snapshots/snapshot_79_0.75",
        ))
        path, source = self.module.resolve_snapshot_dir(text, "TIMEOUT")
        self.assertEqual(path, Path("/tmp/run/snapshots"))
        self.assertEqual(source, "normal_shutdown_marker")

    def test_terminal_job_uses_startup_snapshot_directory(self):
        text = "Snapshot directory: /tmp/run/snapshots\n"
        path, source = self.module.resolve_snapshot_dir(text, "TIMEOUT")
        self.assertEqual(path, Path("/tmp/run/snapshots"))
        self.assertEqual(source, "terminal_snapshot_directory_fallback")

    def test_running_job_never_uses_fallback(self):
        path, source = self.module.resolve_snapshot_dir(
            "Snapshot directory: /tmp/run/snapshots\n", "RUNNING")
        self.assertIsNone(path)
        self.assertEqual(source, "missing")


class RetryPlanDeduplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(
            "validate_eval_log_with_summary",
            ROOT / "asnets" / "tools" / "validate_eval_log_with_summary.py",
        )

    def test_same_plan_retry_is_deduplicated(self):
        plans = [
            {"instance": "p1.pddl", "actions": ["(a)"], "steps": 1},
            {"instance": "p1.pddl", "actions": ["(a)"], "steps": 1},
        ]
        grouped = self.module.group_unique_plans_by_instance(plans)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(len(grouped["p1.pddl"]), 1)

    def test_distinct_retry_plans_remain_candidates_for_one_instance(self):
        plans = [
            {"instance": "p1.pddl", "actions": ["(a)"], "steps": 1},
            {"instance": "p1.pddl", "actions": ["(b)"], "steps": 1},
        ]
        grouped = self.module.group_unique_plans_by_instance(plans)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(len(grouped["p1.pddl"]), 2)


if __name__ == "__main__":
    unittest.main()
