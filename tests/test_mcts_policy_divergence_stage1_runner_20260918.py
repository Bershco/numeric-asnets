from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_mcts_policy_divergence_stage1_20260918.py"
SPEC = importlib.util.spec_from_file_location("stage1_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class Stage1RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tracking = ROOT / "experiment_tracking/mcts_policy_divergence_cause_audit"
        cls.manifest = tracking / "stage1_grouped_tasks.csv"
        cls.freeze = tracking / "stage1_grouped_tasks.freeze.json"
        cls.registry = ROOT / "experiment_tracking/experiment_results.csv"
        cls.rows = runner.load_frozen_tasks(cls.manifest, cls.freeze)
        cls.registry_rows = runner.read_csv(cls.registry)

    def test_all_frozen_candidates_resolve_to_exact_checkpoint(self):
        resolved = 0
        for row in self.rows:
            for candidate in json.loads(row["candidate_runs_json"]):
                checkpoint, registry_row = runner.resolve_checkpoint(
                    self.registry_rows,
                    candidate["checkpoint_identity"],
                    policy_log=candidate["policy_log"],
                )
                source_job, epoch = runner._identity(candidate["checkpoint_identity"])
                self.assertEqual(registry_row["source_training_job_id"], source_job)
                self.assertRegex(checkpoint, rf"/snapshot_{epoch}_")
                resolved += 1
        self.assertEqual(resolved, 40)

    def test_fixed_and_pw_commands_preserve_declared_search(self):
        completion = Path("/tmp/completion.jsonl")
        fixed = self.rows[4]
        fixed_candidate = json.loads(fixed["candidate_runs_json"])[0]
        fixed_checkpoint, _ = runner.resolve_checkpoint(
            self.registry_rows, fixed_candidate["checkpoint_identity"],
            policy_log=fixed_candidate["policy_log"])
        fixed_command = runner.build_command(
            repo=ROOT, domain=fixed["domain"], value_head=fixed["value_head"],
            seed=int(fixed_candidate["seed"]), checkpoint=fixed_checkpoint,
            instance_name=fixed_candidate["instance"],
            search_config=json.loads(fixed["search_config"]),
            tie_break=fixed["tie_break"], terminal_safe=False,
            completion=completion, instance_timeout=21600)
        self.assertIn("--eval-mcts-first-divergence-record", fixed_command)
        self.assertEqual(fixed_command[fixed_command.index("--mcts-expansion-size") + 1], "20")
        self.assertEqual(fixed_command[fixed_command.index("--mcts-iterations") + 1], "70")
        self.assertNotIn("--mcts-progressive-widening", fixed_command)

        pw = next(row for row in self.rows if row["release_class"] == "optional_fo_pw")
        pw_candidate = json.loads(pw["candidate_runs_json"])[0]
        pw_checkpoint, _ = runner.resolve_checkpoint(
            self.registry_rows, pw_candidate["checkpoint_identity"],
            policy_log=pw_candidate["policy_log"])
        pw_command = runner.build_command(
            repo=ROOT, domain=pw["domain"], value_head=pw["value_head"],
            seed=int(pw_candidate["seed"]), checkpoint=pw_checkpoint,
            instance_name=pw_candidate["instance"],
            search_config=json.loads(pw["search_config"]),
            tie_break=pw["tie_break"], terminal_safe=True,
            completion=completion, instance_timeout=21600)
        self.assertIn("--mcts-progressive-widening", pw_command)
        self.assertIn("--eval-mcts-terminal-safe-action-selection", pw_command)
        self.assertEqual(pw_command[pw_command.index("--mcts-pw-min-width") + 1], "3")

    def test_compute_smoke_gate_is_commit_and_root_specific(self):
        payload = {
            "schema_version": runner.GATE_SCHEMA,
            "status": "passed",
            "code_commit": "a" * 40,
            "grouped_manifest_sha256": runner.EXPECTED_MANIFEST_SHA256,
            "checkpoint_identity": runner.SMOKE["checkpoint_identity"],
            "instance": runner.SMOKE["instance"],
            "step": runner.SMOKE["step"],
            "policy_action": runner.SMOKE["policy_action"],
            "selected_action": runner.SMOKE["selected_action"],
            "root_visits": runner.SMOKE["root_visits"],
            "total_edge_visits": runner.SMOKE["total_edge_visits"],
            "disabled_mode_invariance": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "gate.json"
            gate.write_text(json.dumps(payload), encoding="utf-8")
            runner.validate_gate(gate, code_commit="a" * 40)
            with self.assertRaisesRegex(RuntimeError, "gate mismatch"):
                runner.validate_gate(gate, code_commit="b" * 40)

    def test_known_smoke_checkpoint_and_instance_resolve(self):
        checkpoint, row = runner.resolve_checkpoint(
            self.registry_rows,
            runner.SMOKE["checkpoint_identity"],
            evaluation_job=runner.SMOKE["evaluation_job"],
        )
        self.assertEqual(row["source_training_job_id"], "20430427")
        self.assertIn("/snapshot_0_", checkpoint)
        self.assertEqual(runner.instance_index(ROOT, "counters", "fz_instance_51.pddl"), 49)


if __name__ == "__main__":
    unittest.main()
