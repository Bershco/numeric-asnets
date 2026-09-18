from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/run_mcts_policy_divergence_stage1_20260918.py"
SPEC = importlib.util.spec_from_file_location("stage1_repair_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)
RECONCILE_SCRIPT = (
    ROOT / "scripts/reconcile_mcts_policy_divergence_stage1_20260918.py")
RECONCILE_SPEC = importlib.util.spec_from_file_location(
    "stage1_reconciler", RECONCILE_SCRIPT)
reconciler = importlib.util.module_from_spec(RECONCILE_SPEC)
assert RECONCILE_SPEC.loader is not None
RECONCILE_SPEC.loader.exec_module(reconciler)


def legacy_goal_chase_record() -> dict:
    return {
        "schema_version": runner.LEGACY_RECORDER_SCHEMA,
        "instance": "instance_2.pddl",
        "selected_action": 1,
        "policy_action": 0,
        "override_path": [{
            "stage": "goal_chase", "applied": True, "selected_action": 1,
        }],
        "root": {
            "act_dim": 2,
            "action_ids": [0, 1],
            "action_names": ["a", "b"],
            "applicable": [True, True],
            "expanded": [True, True],
            "raw_network_policy": [0.6, 0.4],
            "masked_network_policy": [0.6, 0.4],
            "visit_distribution": [0.7, 0.3],
            "edge_visit_counts": [7, 3],
            "edge_priors": [0.6, 0.4],
            "q_values": [0.1, 0.2],
            "u_values": [0.01, 0.02],
            "signed_q_plus_u": [0.11, 0.22],
            "child_terminal": [False, False],
            "child_goal": [False, True],
            "child_state_digests": ["x", "y"],
        },
    }


class ReconciliationTests(unittest.TestCase):
    def test_legacy_goal_chase_selector_is_exactly_reconstructable(self):
        normalized = runner.normalize_record(
            legacy_goal_chase_record(), instance_name="instance_2.pddl",
            allow_legacy=True)
        self.assertEqual(normalized["schema_version"], runner.RECORDER_SCHEMA)
        self.assertEqual(
            normalized["selection"]["selector_input_distribution"], [0.7, 0.3])
        self.assertEqual(
            normalized["selection"]["selector_distribution_source"],
            "visit_distribution_legacy_goal_chase_input")

    def test_explicit_hard_timeout_is_terminal_without_completion_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "candidate.txt"
            completion = Path(directory) / "candidate.completed.jsonl"
            log.write_text(
                "[EVAL INSTANCE] timeout number=8 "
                "path=../problems/numeric/fo-counters/instances/instance_10.pddl "
                "limit=21600.0s\n"
                "[EVAL FINAL] success=0.0/1=0.000\n",
                encoding="utf-8")
            outcome = runner.terminal_outcome(
                log=log, completion=completion,
                instance_name="instance_10.pddl", evaluation_index=8,
                max_actions=10000)
        self.assertEqual(outcome["classification"], "hard_timeout")
        self.assertEqual(outcome["evidence"], "eval_instance_timeout_log_marker")
        self.assertIsNone(outcome["completion_record"])

    def test_crash_marker_is_not_misclassified_as_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "candidate.txt"
            completion = Path(directory) / "candidate.completed.jsonl"
            log.write_text(
                "[EVAL INSTANCE] crashed number=8 path=instance_10.pddl\n"
                "[EVAL INSTANCE] timeout number=8 path=instance_10.pddl "
                "limit=21600.0s\n",
                encoding="utf-8")
            outcome = runner.terminal_outcome(
                log=log, completion=completion,
                instance_name="instance_10.pddl", evaluation_index=8,
                max_actions=10000)
        self.assertIsNone(outcome)

    def test_explicit_success_log_is_terminal_without_completion_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "candidate.txt"
            completion = Path(directory) / "candidate.completed.jsonl"
            log.write_text(
                "[EVAL INSTANCE] completed number=0 "
                "path=instances/instance_2.pddl status=success "
                "elapsed=12.25s success=True steps=7\n",
                encoding="utf-8")
            outcome = runner.terminal_outcome(
                log=log, completion=completion,
                instance_name="instance_2.pddl", evaluation_index=0,
                max_actions=10000)
        self.assertEqual(outcome["classification"], "success")
        self.assertEqual(outcome["evidence"], "eval_instance_completed_log_marker")

    def test_explicit_action_limit_log_is_terminal_without_completion_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "candidate.txt"
            completion = Path(directory) / "candidate.completed.jsonl"
            log.write_text(
                "[EVAL INSTANCE] completed number=0 "
                "path=instances/instance_2.pddl status=unsolved "
                "elapsed=120.0s success=False steps=10000\n",
                encoding="utf-8")
            outcome = runner.terminal_outcome(
                log=log, completion=completion,
                instance_name="instance_2.pddl", evaluation_index=0,
                max_actions=10000)
        self.assertEqual(outcome["classification"], "action_limit")
        self.assertEqual(outcome["evidence"], "eval_instance_completed_log_marker")

    def test_completion_distinguishes_success_and_action_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "candidate.txt"
            log.write_text("[EVAL FINAL] success=0.0/1=0.000\n", encoding="utf-8")
            completion = root / "candidate.completed.jsonl"
            completion.write_text(json.dumps({
                "instance_number": 0,
                "instance_path": "instances/instance_2.pddl",
                "status": "finished_unsolved",
                "hit_goal": False,
                "steps": 10000,
                "plan": [],
                "elapsed_seconds": 10.0,
            }) + "\n", encoding="utf-8")
            outcome = runner.terminal_outcome(
                log=log, completion=completion,
                instance_name="instance_2.pddl", evaluation_index=0,
                max_actions=10000)
        self.assertEqual(outcome["classification"], "action_limit")

    def test_reconciler_preserves_results_salvages_timeout_and_filters_recovery(self):
        tracking = ROOT / "experiment_tracking/mcts_policy_divergence_cause_audit"
        manifest = tracking / "stage1_grouped_tasks.csv"
        freeze = tracking / "stage1_grouped_tasks.freeze.json"
        rows = runner.load_frozen_tasks(manifest, freeze)
        timeout_key = (1, 0)
        unresolved_key = (1, 1)
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            for task_index, row in enumerate(rows):
                candidates = json.loads(row["candidate_runs_json"])
                for candidate_index, candidate in enumerate(candidates):
                    if (task_index, candidate_index) == unresolved_key:
                        continue
                    stem = reconciler.candidate_stem(candidate_index, candidate)
                    task_dir = artifact_root / row["task_id"]
                    task_dir.mkdir(parents=True, exist_ok=True)
                    if (task_index, candidate_index) == timeout_key:
                        eval_index = runner.instance_index(
                            ROOT, row["domain"], candidate["instance"])
                        (task_dir / f"{stem}.txt").write_text(
                            f"[EVAL INSTANCE] timeout number={eval_index} "
                            f"path=instances/{candidate['instance']} limit=21600.0s\n"
                            "[EVAL FINAL] success=0.0/1=0.000\n",
                            encoding="utf-8")
                        continue
                    (task_dir / f"{stem}.result.json").write_text(json.dumps({
                        "schema_version": (
                            "mcts-divergence-stage1-candidate-result-v1"),
                        "checkpoint_identity": candidate["checkpoint_identity"],
                        "instance": candidate["instance"],
                        "first_divergence": None,
                        "completion_record": {"status": "success"},
                    }), encoding="utf-8")
            report, recovery = reconciler.reconcile(
                repo=ROOT, manifest=manifest, freeze=freeze,
                artifact_root=artifact_root, write=True)
            timeout_candidate = json.loads(rows[1]["candidate_runs_json"])[0]
            timeout_result = (
                artifact_root / rows[1]["task_id"]
                / f"{reconciler.candidate_stem(0, timeout_candidate)}.result.json")
            self.assertTrue(timeout_result.exists())
            saved = json.loads(timeout_result.read_text(encoding="utf-8"))
        self.assertEqual(report["preserved_valid_results"], 38)
        self.assertEqual(report["materialized_terminal_results"], 1)
        self.assertEqual(report["recovery_candidates"], 1)
        self.assertEqual(saved["outcome"]["classification"], "hard_timeout")
        self.assertEqual(recovery["tasks"][0]["candidate_indices"], [1])


if __name__ == "__main__":
    unittest.main()
