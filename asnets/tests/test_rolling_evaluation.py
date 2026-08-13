import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from asnets.parllel_explore_spawn_grads import (
    _load_completed_evaluations,
    run_rolling_spawn_eval,
)
from asnets.spawn_train_worker import EvalWorkerOutput
from asnets.scripts.run_asnets import select_evaluation_instances


def _spec(number, delay=0.0, outcome="success", timeout=None):
    return SimpleNamespace(
        evaluation_index=number,
        pddls=("domain.pddl", f"instance_{number}.pddl"),
        delay=delay,
        outcome=outcome,
        timeout=timeout,
        trainer_seed=42,
        slot_id=number - 1,
        use_estimator=False,
        use_estimator_decay=False,
    )


def _worker(inp):
    time.sleep(inp.spec.delay)
    if inp.spec.outcome == "crash":
        raise RuntimeError("intentional crash")
    solved = inp.spec.outcome == "success"
    return EvalWorkerOutput(
        hit_goal=solved,
        steps=inp.spec.evaluation_index if solved else -1,
        instance_name=inp.spec.pddls[1],
    )


def _must_not_run(_inp):
    raise AssertionError("persisted instances must not run again")


class RollingEvaluationTests(unittest.TestCase):
    def test_wave_prefix_and_explicit_skips_combine(self):
        instances = [f"instance_{number}.pddl" for number in range(1, 8)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            selected = select_evaluation_instances(
                instances, requested_wave=2, num_workers=3,
                skip_instance_numbers=(5, 99))
        self.assertEqual(selected, [
            (4, "instance_4.pddl"),
            (6, "instance_6.pddl"),
            (7, "instance_7.pddl"),
        ])
        self.assertIn(
            "skip numbers outside test set ignored: [99]", output.getvalue())

    def test_wave_below_one_starts_at_first_instance(self):
        self.assertEqual(select_evaluation_instances(
            ["one.pddl", "two.pddl"], -30, 3), [
                (1, "one.pddl"), (2, "two.pddl")])

    def test_wave_above_available_instances_finishes_cleanly(self):
        self.assertEqual(select_evaluation_instances(
            ["one.pddl"], 10, 3), [])

    def test_rolling_pool_replaces_finished_workers_immediately(self):
        specs = [
            _spec(1, 0.0), _spec(2, 2.0),
            _spec(3, 0.0), _spec(4, 0.0),
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            outputs = run_rolling_spawn_eval(
                specs, {}, max_workers=2, worker_fn=_worker,
                evaluation_signature="sig")
        self.assertTrue(all(output.hit_goal for output in outputs))
        events = output.getvalue()
        self.assertLess(
            events.index("completed number=1"),
            events.index("started number=3"))
        self.assertLess(
            events.index("started number=3"),
            events.index("completed number=2"))

    def test_finished_results_are_printed_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            completion_file = Path(directory) / "completed.jsonl"
            specs = [_spec(1), _spec(2, outcome="unsolved")]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                results = run_rolling_spawn_eval(
                    specs, {}, max_workers=2, worker_fn=_worker,
                    completion_file=str(completion_file),
                    evaluation_signature="sig")
            self.assertEqual(
                [result.hit_goal for result in results], [True, False])
            records = [
                json.loads(line)
                for line in completion_file.read_text().splitlines()
            ]
            records.sort(key=lambda record: record["instance_number"])
            self.assertEqual(
                [record["status"] for record in records],
                ["success", "finished_unsolved"])
            printed = output.getvalue()
            self.assertIn("[EVAL INSTANCE] completed number=1", printed)
            self.assertIn("[EVAL INSTANCE] completed number=2", printed)
            self.assertIn("path=instance_1.pddl", printed)
            self.assertIn("success=True", printed)

    def test_persisted_results_are_skipped_on_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            completion_file = Path(directory) / "completed.jsonl"
            specs = [_spec(1), _spec(2, outcome="unsolved")]
            first = run_rolling_spawn_eval(
                specs, {}, max_workers=2, worker_fn=_worker,
                completion_file=str(completion_file),
                evaluation_signature="sig")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                second = run_rolling_spawn_eval(
                    specs, {}, max_workers=2, worker_fn=_must_not_run,
                    completion_file=str(completion_file),
                    evaluation_signature="sig")
            self.assertEqual(
                [(result.hit_goal, result.steps) for result in second],
                [(result.hit_goal, result.steps) for result in first])
            self.assertIn(
                "[EVAL INSTANCE] skip completed number=1", output.getvalue())

    def test_crash_and_timeout_are_not_persisted(self):
        for outcome, delay in (("crash", 0.0), ("success", 1.0)):
            with self.subTest(outcome=outcome):
                with tempfile.TemporaryDirectory() as directory:
                    completion_file = Path(directory) / "completed.jsonl"
                    outputs = run_rolling_spawn_eval(
                        [_spec(1, delay=delay, outcome=outcome)], {},
                        max_workers=1, worker_fn=_worker,
                        instance_timeout=0.1,
                        completion_file=str(completion_file),
                        evaluation_signature="sig")
                    self.assertIn(outputs[0].steps, (-2, -3))
                    self.assertTrue(
                        not completion_file.exists()
                        or not completion_file.read_text())

    def test_spec_timeout_is_used_without_override(self):
        outputs = run_rolling_spawn_eval(
            [_spec(1, delay=1.0, timeout=0.1)], {}, max_workers=1,
            worker_fn=_worker, evaluation_signature="sig")
        self.assertEqual(outputs[0].steps, -2)

    def test_completion_file_signature_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            completion_file = Path(directory) / "completed.jsonl"
            completion_file.write_text(json.dumps({
                "evaluation_signature": "old",
                "instance_number": 1,
                "instance_path": "instance_1.pddl",
                "status": "success",
                "hit_goal": True,
                "steps": 1,
                "elapsed_seconds": 1.0,
            }) + "\n")
            with self.assertRaisesRegex(
                    RuntimeError, "ordered test-set signature"):
                _load_completed_evaluations(str(completion_file), "new")


if __name__ == "__main__":
    unittest.main()
