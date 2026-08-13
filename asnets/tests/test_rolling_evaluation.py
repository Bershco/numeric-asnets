import json
import time
from types import SimpleNamespace

import pytest

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
        steps=number_or_failure(inp.spec.evaluation_index, solved),
        instance_name=inp.spec.pddls[1],
    )


def number_or_failure(number, solved):
    return number if solved else -1


def _must_not_run(_inp):
    raise AssertionError("persisted instances must not run again")


def test_wave_prefix_and_explicit_skips_combine_without_changing_wave_mode(
        capfd):
    instances = [f"instance_{number}.pddl" for number in range(1, 8)]
    selected = select_evaluation_instances(
        instances, requested_wave=2, num_workers=3,
        skip_instance_numbers=(5, 99))
    assert selected == [
        (4, "instance_4.pddl"),
        (6, "instance_6.pddl"),
        (7, "instance_7.pddl"),
    ]
    assert "skip numbers outside test set ignored: [99]" in capfd.readouterr().out


def test_wave_below_one_starts_at_first_instance():
    assert select_evaluation_instances(
        ["one.pddl", "two.pddl"], -30, 3) == [
            (1, "one.pddl"), (2, "two.pddl")]


def test_wave_above_available_instances_finishes_cleanly():
    assert select_evaluation_instances(["one.pddl"], 10, 3) == []


def test_rolling_pool_replaces_finished_workers_immediately():
    specs = [
        _spec(1, 0.1),
        _spec(2, 1.0),
        _spec(3, 0.1),
        _spec(4, 1.0),
    ]
    start = time.monotonic()
    outputs = run_rolling_spawn_eval(
        specs, {}, max_workers=2, worker_fn=_worker,
        instance_timeout=5, evaluation_signature="sig")
    elapsed = time.monotonic() - start
    assert all(output.hit_goal for output in outputs)
    # Fixed waves take about 2.0s; rolling scheduling takes about 1.2s.
    assert elapsed < 1.7


def test_completed_success_and_unsolved_results_are_printed_and_persisted(
        tmp_path, capfd):
    completion_file = tmp_path / "completed.jsonl"
    specs = [_spec(1, outcome="success"), _spec(2, outcome="unsolved")]
    outputs = run_rolling_spawn_eval(
        specs, {}, max_workers=2, worker_fn=_worker,
        instance_timeout=5, completion_file=str(completion_file),
        evaluation_signature="sig")
    assert [output.hit_goal for output in outputs] == [True, False]
    records = [
        json.loads(line)
        for line in completion_file.read_text().splitlines()
    ]
    records.sort(key=lambda record: record["instance_number"])
    assert [record["status"] for record in records] == [
        "success", "finished_unsolved"]
    printed = capfd.readouterr().out
    assert "[EVAL INSTANCE] completed number=1" in printed
    assert "[EVAL INSTANCE] completed number=2" in printed
    assert "path=instance_1.pddl" in printed
    assert "success=True" in printed


def test_persisted_results_are_skipped_on_retry(tmp_path, capfd):
    completion_file = tmp_path / "completed.jsonl"
    specs = [_spec(1), _spec(2, outcome="unsolved")]
    first = run_rolling_spawn_eval(
        specs, {}, max_workers=2, worker_fn=_worker,
        completion_file=str(completion_file), evaluation_signature="sig")
    second = run_rolling_spawn_eval(
        specs, {}, max_workers=2, worker_fn=_must_not_run,
        completion_file=str(completion_file), evaluation_signature="sig")
    assert [(o.hit_goal, o.steps) for o in second] == [
        (o.hit_goal, o.steps) for o in first]
    assert "[EVAL INSTANCE] skip completed number=1" in capfd.readouterr().out


@pytest.mark.parametrize("outcome,delay", [("crash", 0.0), ("success", 1.0)])
def test_crash_and_timeout_are_not_persisted(tmp_path, outcome, delay):
    completion_file = tmp_path / "completed.jsonl"
    outputs = run_rolling_spawn_eval(
        [_spec(1, delay=delay, outcome=outcome)], {}, max_workers=1,
        worker_fn=_worker, instance_timeout=0.1,
        completion_file=str(completion_file), evaluation_signature="sig")
    assert outputs[0].steps in (-2, -3)
    assert not completion_file.exists() or not completion_file.read_text()


def test_configured_spec_timeout_is_used_when_override_is_absent(tmp_path):
    outputs = run_rolling_spawn_eval(
        [_spec(1, delay=1.0, timeout=0.1)], {}, max_workers=1,
        worker_fn=_worker, evaluation_signature="sig")
    assert outputs[0].steps == -2


def test_completion_file_signature_mismatch_fails(tmp_path):
    completion_file = tmp_path / "completed.jsonl"
    completion_file.write_text(json.dumps({
        "evaluation_signature": "old",
        "instance_number": 1,
        "instance_path": "instance_1.pddl",
        "status": "success",
        "hit_goal": True,
        "steps": 1,
        "elapsed_seconds": 1.0,
    }) + "\n")
    with pytest.raises(RuntimeError, match="ordered test-set signature"):
        _load_completed_evaluations(str(completion_file), "new")
