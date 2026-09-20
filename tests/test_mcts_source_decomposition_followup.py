import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_mcts_source_decomposition_followup_20260920.py"
SPEC = importlib.util.spec_from_file_location("source_decomposition_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class SourceDecompositionFollowupTests(unittest.TestCase):
    def setUp(self):
        self.tracking = (
            ROOT / "experiment_tracking/mcts_policy_divergence_cause_audit")
        self.manifest = self.tracking / "source_decomposition_followup_manifest.json"
        self.freeze = self.tracking / "source_decomposition_followup_manifest.freeze.json"
        self.source_manifest = self.tracking / "stage1_grouped_tasks.csv"
        self.reconciliation = self.tracking / "stage1_final_40_reconciled.csv"

    def test_frozen_design_has_three_harmful_descriptive_contrasts(self):
        tasks = runner.load_design(
            self.manifest,
            self.freeze,
            self.source_manifest,
            self.reconciliation,
        )
        self.assertEqual(len(tasks), 6)
        harmful = {task["diagnostic_id"] for task in tasks
                   if task["role"] == "harmful"}
        controls = [task for task in tasks if task["role"] == "control"]
        self.assertEqual(len(harmful), 3)
        self.assertEqual(len(controls), 3)
        self.assertEqual(
            {task["descriptive_control_for"] for task in controls}, harmful)
        self.assertEqual(
            {task["domain"] for task in tasks},
            {"block_grouping", "drone", "fo_counters"},
        )

    def test_diagnostic_result_is_not_a_terminal_classification(self):
        task = json.loads(self.manifest.read_text(encoding="utf-8"))["tasks"][0]
        payload = runner.build_result_payload(
            task=task,
            task_index=0,
            record={"schema_version": runner.RECORD_SCHEMA},
            checkpoint="/frozen/checkpoint",
            log=Path("diagnostic.txt"),
            completion=Path("diagnostic.completed.jsonl"),
            code_commit="abc123",
            manifest_sha256="manifest-hash",
            registry_sha256="registry-hash",
        )
        self.assertTrue(payload["diagnostic_stopped_after_record"])
        self.assertFalse(payload["terminal_outcome_repeated"])
        self.assertTrue(payload["generic_completion_file_disabled"])
        self.assertNotIn("outcome", payload)
        self.assertNotIn("classification", payload)
        self.assertEqual(
            payload["historical_outcome_source"], task["search_log"])

    def test_validator_accepts_agreement_root_with_complete_sources(self):
        task = json.loads(self.manifest.read_text(encoding="utf-8"))["tasks"][3]
        record = {
            "schema_version": runner.RECORD_SCHEMA,
            "instance": task["instance"],
            "step": task["target_step"],
            "diagnostic_stop_after_record": True,
            "actions_diverge": False,
            "policy_action": 0,
            "selected_action": 0,
            "root": {
                "act_dim": 2,
                "expanded": [True, False],
                "edge_visit_counts": [1, 0],
                "q_source_decomposition": [{
                    "scope": "node_global_matches_child_q",
                    "backups": 1,
                    "reconstruction_residual": 0.0,
                    "leaf_source_counts": {"network_only": 1},
                }, None],
            },
        }
        runner.validate_record(record, task)

    def test_dry_run_command_enables_source_and_disables_first_divergence(self):
        tasks = runner.load_design(
            self.manifest,
            self.freeze,
            self.source_manifest,
            self.reconciliation,
        )
        with tempfile.TemporaryDirectory() as temporary:
            command = runner.build_command(
                repo=ROOT,
                task=tasks[0],
                checkpoint="/frozen/checkpoint",
                completion=Path(temporary) / "completion.jsonl",
            )
        self.assertIn("--eval-mcts-source-decomposition-step", command)
        index = command.index("--eval-mcts-source-decomposition-step")
        self.assertEqual(command[index + 1], str(tasks[0]["target_step"]))
        self.assertNotIn("--eval-mcts-first-divergence-record", command)
        self.assertNotIn("--eval-completion-file", command)

    def test_every_checkpoint_resolves_exactly_in_scientific_registry(self):
        tasks = runner.load_design(
            self.manifest,
            self.freeze,
            self.source_manifest,
            self.reconciliation,
        )
        registry = runner.read_csv(
            ROOT / "experiment_tracking/experiment_results.csv")
        resolved = []
        for task in tasks:
            checkpoint, row = runner.source_runner.resolve_checkpoint(
                registry,
                task["checkpoint_identity"],
                policy_log=task["policy_log"],
            )
            self.assertTrue(checkpoint.startswith("/home/hersco/"))
            resolved.append(row["job_id"])
        self.assertEqual(len(set(resolved)), 6)


if __name__ == "__main__":
    unittest.main()
