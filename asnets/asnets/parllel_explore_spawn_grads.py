# asnets/parallel_explore_spawn_grads.py
from __future__ import annotations

import os
import json
import signal
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from queue import Empty
from time import time, monotonic, sleep
from typing import Any, Optional, Tuple
import traceback

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, Future, wait, ALL_COMPLETED, FIRST_COMPLETED
from pathlib import Path

import numpy as np

from asnets.spawn_train_worker import MCTSWorkerInput, WorkerOutput, run_worker_opt_profiled, run_worker_eval_mcts, \
    WorkerInput, EvalWorkerOutput
from asnets.supervised import SupervisedObjective
from asnets.utils.generator_utils import InstanceDifficulty
from asnets.utils.prof_utils import can_profile


def _print_completed_eval_plan(spec, result):
    """Immediately emit a completed successful plan for recovery and VAL."""
    if not result.hit_goal:
        return
    if result.plan is None:
        print(
            f"[EVAL PLAN WARNING] successful instance "
            f"number={spec.evaluation_index} has no serialized plan",
            flush=True,
        )
        return
    print(
        f"[EVAL][PLAN] "
        f"{spec.difficulty.name:<10} | "
        f"{Path(spec.pddls[1]).name} | "
        f"steps={result.steps} | "
        f"plan={result.plan}",
        flush=True,
    )


def run_epoch_spawn_grads(
        specs: list[Any],
        weights_np: dict,
        max_estimator_coeff: float,
        log: bool,
        curr_epoch: Optional[int],
        PROFILE_DIR: Optional[str] = None,
        corrupt_pi: Optional[str] = None,
        corrupt_z: Optional[str] = None,
        minimization: bool = False,
        max_workers: Optional[int] = None,
        epoch_timeout: Optional[float] = None,
) -> list[WorkerOutput]:
    """
    Run one exploration epoch in parallel and return completed worker outputs.

    All workers are submitted together through a forkserver-based process pool.
    The function waits up to `epoch_timeout` seconds for all workers. Workers that
    do not finish in time are treated as timed out; completed workers are kept,
    timed-out workers are skipped, and the executor is shut down early.

    Args:
        specs: Worker specifications for this epoch.
        weights_np: Network weights passed to each worker.
        log: Whether worker logging is enabled.
        curr_epoch: Current epoch index.
        PROFILE_DIR: Optional profiling output directory.
        corrupt_pi: Optional corruption mode for policy targets.
        corrupt_z: Optional corruption mode for value targets.
        max_workers: Maximum number of worker processes.
        epoch_timeout: Maximum wait time in seconds for the whole epoch.

    Returns:
        A list of outputs from workers that finished successfully before timeout.
    """
    ctx = mp.get_context("forkserver")
    outs: list[WorkerOutput] = []
    with ProcessPoolExecutor(
            max_workers=max_workers or len(specs),
            mp_context=ctx,
    ) as ex:
        futs: list[Future[WorkerOutput]] = [
            ex.submit(
                run_worker_opt_profiled,
                MCTSWorkerInput(
                    spec=spec,
                    epoch=curr_epoch,
                    weights_np=weights_np,
                    log=log,
                    PROFILE_DIR=PROFILE_DIR,
                    corrupt_pi=corrupt_pi,
                    corrupt_z=corrupt_z,
                    minimization=minimization,
                    max_estimator_coeff=max_estimator_coeff,
                ),
            )
            for spec in specs
        ]
        done, not_done = wait(
            futs,
            timeout=epoch_timeout,
            return_when=ALL_COMPLETED,
        )
        for fut in done:
            outs.append(fut.result())
        if not_done:
            print(
                f"[TRAINER WARNING] {len(not_done)} worker(s) timed out | "
                f"timeout={epoch_timeout:.1f}s | epoch={curr_epoch}"
            )
            ex.shutdown(wait=False, cancel_futures=True)
        unfinished = [f for f in futs if not f.done()]
        for fut in unfinished:
            fut.cancel()  # failsafe for tasks that never started
        if unfinished:
            plural = "worker has" if len(unfinished) == 1 else "workers have"
            print(
                f"[TRAINER WARNING] {len(unfinished)} {plural} timed out | "
                f"The timeout was {epoch_timeout:.1f} seconds."
            )
    return outs


def _eval_process_entry(idx, worker_fn, inp, result_conn):
    os.setsid()  # worker becomes leader of a new process group/session, this gives us a way to kill all subprocesses of said worker when something crashed inside
    try:
        if inp.PROFILE_DIR:
            result = run_worker_opt_profiled(inp, worker_fn=worker_fn)
        else:
            result = worker_fn(inp)
        result_conn.send((idx, "ok", result, None))
    except BaseException as e:
        result_conn.send(
            (idx, "err", None, f"{repr(e)}\n{traceback.format_exc()}"))
    finally:
        result_conn.close()


def _load_completed_evaluations(completion_file, expected_signature):
    completed = {}
    if not completion_file or not os.path.exists(completion_file):
        return completed
    with open(completion_file, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    signatures = {record["evaluation_signature"] for record in records}
    if signatures and signatures != {expected_signature}:
        raise RuntimeError(
            "Evaluation completion file does not match the current ordered "
            "test-set signature"
        )
    for record in records:
        completed[record["instance_number"]] = record
    return completed


def _append_completed_evaluation(
        completion_file,
        evaluation_signature,
        spec,
        result,
        elapsed,
):
    if not completion_file:
        return
    record = {
        "evaluation_signature": evaluation_signature,
        "instance_number": spec.evaluation_index,
        "instance_path": spec.pddls[1],
        "status": "success" if result.hit_goal else "finished_unsolved",
        "hit_goal": bool(result.hit_goal),
        "steps": int(result.steps),
        "plan": result.plan,
        "elapsed_seconds": float(elapsed),
    }
    os.makedirs(os.path.dirname(os.path.abspath(completion_file)), exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    fd = os.open(
        completion_file,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _terminate_eval_process(process, *, hard=False, reap_timeout=30.0):
    """Terminate and reap one isolated evaluation worker process group.

    Hard evaluation deadlines use SIGKILL immediately.  A JPype-hosted JVM can
    intercept or delay SIGTERM while leaving the Python worker alive but unable
    to restart its JVM.  We must reap the old worker before filling its pool
    slot; otherwise abandoned trees/JVMs can accumulate inside the Slurm cgroup.
    """
    first_signal = signal.SIGKILL if hard else signal.SIGTERM
    try:
        os.killpg(process.pid, first_signal)
    except ProcessLookupError:
        pass
    if hard and process.is_alive():
        process.kill()
    process.join(timeout=reap_timeout)
    if process.is_alive() and not hard:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.is_alive():
            process.kill()
        process.join(timeout=reap_timeout)
    if process.is_alive():
        raise RuntimeError(
            f"Evaluation worker pid={process.pid} was not reaped after "
            f"SIGKILL within {reap_timeout:.1f}s; refusing to overfill the "
            f"worker pool")


def run_rolling_spawn_eval(
        specs,
        weights_np,
        max_workers=None,
        worker_fn=run_worker_eval_mcts,
        minimization: bool = False,
        PROFILE_DIR: Optional[str] = None,
        instance_timeout: Optional[float] = None,
        completion_file: Optional[str] = None,
        evaluation_signature: Optional[str] = None,
) -> list[EvalWorkerOutput]:
    """Evaluate with a rolling worker pool and per-instance hard deadlines."""
    if not specs:
        return []
    if instance_timeout is not None and instance_timeout <= 0:
        raise ValueError("instance_timeout must be positive")
    if completion_file and not evaluation_signature:
        raise ValueError("completion_file requires evaluation_signature")

    max_workers = min(max_workers or len(specs), len(specs))
    completed_records = _load_completed_evaluations(
        completion_file, evaluation_signature)
    outs: list[Optional[EvalWorkerOutput]] = [None] * len(specs)
    pending_specs = []
    for idx, spec in enumerate(specs):
        previous = completed_records.get(spec.evaluation_index)
        if previous is None:
            pending_specs.append((idx, spec))
            continue
        if previous["instance_path"] != spec.pddls[1]:
            raise RuntimeError(
                f"Persisted instance {spec.evaluation_index} path mismatch")
        outs[idx] = EvalWorkerOutput(
            hit_goal=previous["hit_goal"],
            steps=previous["steps"],
            instance_name=previous["instance_path"],
            plan=previous.get("plan"),
        )
        print(
            f"[EVAL INSTANCE] skip completed "
            f"number={spec.evaluation_index} path={spec.pddls[1]} "
            f"status={previous['status']} "
            f"elapsed={previous['elapsed_seconds']:.2f}s "
            f"success={previous['hit_goal']} steps={previous['steps']}",
            flush=True,
        )
        _print_completed_eval_plan(spec, outs[idx])

    ctx = mp.get_context("forkserver")
    active = {}
    pending_iter = iter(pending_specs)

    def launch_available():
        while len(active) < max_workers:
            try:
                idx, spec = next(pending_iter)
            except StopIteration:
                return
            inp = WorkerInput(
                spec=spec,
                epoch=None,
                weights_np=weights_np,
                minimization=minimization,
                PROFILE_DIR=PROFILE_DIR,
            )
            result_conn, worker_conn = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=_eval_process_entry,
                args=(idx, worker_fn, inp, worker_conn),
            )
            process.start()
            worker_conn.close()
            active[idx] = {
                "process": process,
                "result_conn": result_conn,
                "started": monotonic(),
                "spec": spec,
                "dead_since": None,
            }
            print(
                f"[EVAL INSTANCE] started number={spec.evaluation_index} "
                f"path={spec.pddls[1]} pid={process.pid}", flush=True)

    launch_available()
    try:
        while active:
            ready_results = []
            for idx, entry in list(active.items()):
                result_conn = entry["result_conn"]
                if not result_conn.poll():
                    continue
                try:
                    ready_results.append(result_conn.recv())
                except EOFError:
                    # The dead-worker branch below records the exit code.
                    pass
            for idx, status, result, err_text in ready_results:
                entry = active.pop(idx, None)
                if entry is None:
                    continue
                process = entry["process"]
                result_conn = entry["result_conn"]
                spec = entry["spec"]
                elapsed = monotonic() - entry["started"]
                process.join(timeout=1.0)
                if process.is_alive():
                    _terminate_eval_process(process, hard=True)
                if status == "ok":
                    outs[idx] = result
                    _append_completed_evaluation(
                        completion_file,
                        evaluation_signature,
                        spec,
                        result,
                        elapsed,
                    )
                    print(
                        f"[EVAL INSTANCE] completed "
                        f"number={spec.evaluation_index} path={spec.pddls[1]} "
                        f"status={'success' if result.hit_goal else 'unsolved'} "
                        f"elapsed={elapsed:.2f}s success={result.hit_goal} "
                        f"steps={result.steps}", flush=True)
                    _print_completed_eval_plan(spec, result)
                else:
                    print(
                        f"[EVAL INSTANCE] crashed "
                        f"number={spec.evaluation_index} path={spec.pddls[1]} "
                        f"elapsed={elapsed:.2f}s error={err_text}", flush=True)
                    outs[idx] = EvalWorkerOutput(
                        hit_goal=False,
                        steps=-3,
                        instance_name=f"[{spec.pddls[1]}]:[CRASH]",
                    )
                result_conn.close()
                process.close()
                launch_available()

            now = monotonic()
            for idx, entry in list(active.items()):
                process = entry["process"]
                spec = entry["spec"]
                elapsed = now - entry["started"]
                if not process.is_alive() and process.exitcode is not None:
                    if entry["dead_since"] is None:
                        entry["dead_since"] = now
                        continue
                    if now - entry["dead_since"] < 0.5:
                        continue
                    process.join(timeout=0)
                    print(
                        f"[EVAL INSTANCE] died number={spec.evaluation_index} "
                        f"path={spec.pddls[1]} exitcode={process.exitcode}",
                        flush=True,
                    )
                    outs[idx] = EvalWorkerOutput(
                        hit_goal=False,
                        steps=-3,
                        instance_name=(
                            f"[{spec.pddls[1]}]:"
                            f"[CRASH_EXIT_{process.exitcode}]"),
                    )
                    entry["result_conn"].close()
                    process.close()
                    del active[idx]
                    launch_available()
                    continue
                effective_timeout = (
                    instance_timeout
                    if instance_timeout is not None
                    else spec.timeout
                )
                if (effective_timeout is not None
                        and elapsed >= effective_timeout):
                    _terminate_eval_process(process, hard=True)
                    print(
                        f"[EVAL INSTANCE] timeout number={spec.evaluation_index} "
                        f"path={spec.pddls[1]} limit={effective_timeout:.1f}s",
                        flush=True,
                    )
                    outs[idx] = EvalWorkerOutput(
                        hit_goal=False,
                        steps=-2,
                        instance_name=f"[{spec.pddls[1]}]:[HARD_TIMEOUT]",
                    )
                    entry["result_conn"].close()
                    process.close()
                    del active[idx]
                    launch_available()
            if active:
                sleep(0.25)
    finally:
        for entry in active.values():
            _terminate_eval_process(entry["process"], hard=True)
            entry["result_conn"].close()
            entry["process"].close()

    assert all(output is not None for output in outs)
    total_success = float(sum(output.hit_goal for output in outs))
    print(
        f"\n[EVAL FINAL] success={total_success}/{len(specs)}="
        f"{total_success / len(specs):.3f}",
        flush=True,
    )
    return outs

def run_epoch_spawn_eval(
        specs,
        weights_np,
        wave_threshold,
        max_workers=None,
        worker_fn=run_worker_eval_mcts,
        minimization: bool = False,
        PROFILE_DIR: Optional[str] = None,
) -> list[EvalWorkerOutput]:
    if not specs:
        return []
    if max_workers is None:
        max_workers = len(specs)
    max_workers = min(max_workers, len(specs))
    outs: list[Optional[EvalWorkerOutput]] = [None] * len(specs)
    spec_to_idx = {id(spec): i for i, spec in enumerate(specs)}
    grouped_specs = defaultdict(list)
    for spec in specs:
        grouped_specs[spec.difficulty].append(spec)
    total_success = 0
    total_done = 0
    difficulty_gate = {
        InstanceDifficulty.MEDIUM: (InstanceDifficulty.EASY, 0.4),
        InstanceDifficulty.HARD: (InstanceDifficulty.MEDIUM, 0.4),
    }
    ctx = mp.get_context("forkserver")
    for diff, diff_specs in grouped_specs.items():
        # --------------------------------------------------
        # Difficulty gating
        # --------------------------------------------------
        if diff in difficulty_gate:
            required_diff, threshold = difficulty_gate[diff]
            prev_results = [
                outs[spec_to_idx[id(spec)]].hit_goal
                for spec in grouped_specs[required_diff]
                if outs[spec_to_idx[id(spec)]] is not None
            ]
            prev_rate = float(np.mean(prev_results)) if prev_results else 0.0
            if prev_rate < threshold:
                print(
                    f"\n[EVAL] Skipping {diff.name}: "
                    f"{required_diff.name} success "
                    f"{prev_rate:.3f} < {threshold:.3f}"
                )
                for spec in diff_specs:
                    idx = spec_to_idx[id(spec)]
                    outs[idx] = EvalWorkerOutput(
                        hit_goal=False,
                        steps=-1,
                        instance_name=f"[SKIPPED] {spec.pddls[1]}",
                    )
                continue
        print(f"\n[EVAL] === {diff.name} ({len(diff_specs)} instances) ===")
        for wave_start in range(0, len(diff_specs), max_workers):
            wave_specs = diff_specs[wave_start:wave_start + max_workers]
            wave_idx = wave_start // max_workers + 1
            print(
                f"[EVAL] {diff.name} wave {wave_idx}: "
                f"{len(wave_specs)} instances"
            )
            start_wave_time = monotonic()
            wave_success = 0
            wave_done = 0
            spec_timeouts = [
                spec.timeout
                for spec in wave_specs
                if spec.timeout is not None
            ]
            hard_timeout = (
                max(spec_timeouts) * 1.5
                if spec_timeouts
                else None
            )
            wave_deadline = (
                monotonic() + hard_timeout
                if hard_timeout is not None
                else np.inf
            )
            out_q = ctx.Queue()
            processes: dict[int, mp.Process] = {}
            # --------------------------------------------------
            # Spawn wave processes
            # --------------------------------------------------
            for spec in wave_specs:
                idx = spec_to_idx[id(spec)]
                inp = WorkerInput(
                    spec=spec,
                    epoch=None,
                    weights_np=weights_np,
                    minimization=minimization,
                    PROFILE_DIR=PROFILE_DIR,
                )
                p = ctx.Process(
                    target=_eval_process_entry,
                    args=(idx, worker_fn, inp, out_q),
                )
                p.start()
                processes[idx] = p
            pending = set(processes.keys())
            # --------------------------------------------------
            # Collect wave results / enforce hard timeout
            # --------------------------------------------------
            while pending:
                # Drain completed queue messages first.
                while True:
                    try:
                        idx, status, result, err_text = out_q.get_nowait()
                    except Empty:
                        break
                    if idx not in pending:
                        continue
                    p = processes[idx]
                    p.join(timeout=0)
                    if status == "ok":
                        outs[idx] = result
                        _print_completed_eval_plan(specs[idx], result)
                    else:
                        print(
                            f"[EVAL] Worker crashed | "
                            f"{diff.name} wave {wave_idx} | "
                            f"idx={idx} | "
                            f"base_error={err_text}"
                        )
                        outs[idx] = EvalWorkerOutput(
                            hit_goal=False,
                            steps=-3,
                            instance_name=f"[{specs[idx].pddls[1]}]:[CRASH]",
                        )
                    pending.remove(idx)
                    wave_done += 1
                    wave_success += outs[idx].hit_goal
                    total_done += 1
                    total_success += outs[idx].hit_goal
                # Also catch processes that died without putting a result.
                dead_without_result = []
                for idx in list(pending):
                    p = processes[idx]
                    if not p.is_alive() and p.exitcode is not None:
                        dead_without_result.append(idx)
                for idx in dead_without_result:
                    p = processes[idx]
                    p.join(timeout=0)
                    print(
                        f"[EVAL] Worker died without result | "
                        f"{diff.name} wave {wave_idx} | "
                        f"idx={idx} | exitcode={p.exitcode}"
                    )
                    outs[idx] = EvalWorkerOutput(
                        hit_goal=False,
                        steps=-3,
                        instance_name=f"[{specs[idx].pddls[1]}]:[CRASH_EXIT_{p.exitcode}]",
                    )
                    pending.remove(idx)
                    wave_done += 1
                    total_done += 1
                if not pending:
                    break
                remaining = wave_deadline - monotonic()
                # --------------------------------------------------
                # Hard timeout: actually kill processes
                # --------------------------------------------------
                if remaining <= 0:
                    print(
                        f"[EVAL] HARD TIMEOUT | "
                        f"{diff.name} wave {wave_idx}"
                    )
                    for idx in list(pending):
                        p = processes[idx]
                        try:
                            os.killpg(p.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        p.join(timeout=3.0)
                        if p.is_alive():
                            try:
                                os.killpg(p.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            p.join(timeout=3.0)
                        outs[idx] = EvalWorkerOutput(
                            hit_goal=False,
                            steps=-2,
                            instance_name=f"[{specs[idx].pddls[1]}]:[HARD_TIMEOUT]",
                        )
                        pending.remove(idx)
                        wave_done += 1
                        total_done += 1
                    break
                sleep(min(0.25, max(remaining, 0.0)))
            # Clean up all processes.
            for p in processes.values():
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=1.0)
                p.close()
            out_q.close()
            out_q.join_thread()
            wave_rate = wave_success / wave_done if wave_done else 0.0
            print(
                f"[EVAL] {diff.name} wave {wave_idx}: "
                f"{wave_success}/{wave_done} = {wave_rate:.3f} | "
                f"time={monotonic() - start_wave_time:.2f}s"
            )
            if wave_rate < wave_threshold:
                print(
                    f"[EVAL] {diff.name} wave {wave_idx} is the last wave "
                    f"due to not hitting wave threshold: {wave_threshold}"
                )
                remaining_specs = diff_specs[wave_start + max_workers:]
                for spec in remaining_specs:
                    idx = spec_to_idx[id(spec)]
                    outs[idx] = EvalWorkerOutput(
                        hit_goal=False,
                        steps=-1,
                        instance_name=f"[{spec.pddls[1]}]:[WAVE_SKIPPED]",
                    )
                break
        diff_total = len(diff_specs)
        diff_success = sum(
            outs[spec_to_idx[id(spec)]].hit_goal
            for spec in diff_specs
        )
        diff_rate = diff_success / diff_total if diff_total else 0.0
        print(
            f"[EVAL] {diff.name} TOTAL: "
            f"{diff_success}/{diff_total} = {diff_rate:.3f}"
        )
    print(
        f"\n[EVAL FINAL] success={total_success}/{len(specs)}="
        f"{total_success / len(specs):.3f}"
    )
    assert all(o is not None for o in outs)
    return outs


@dataclass(frozen=True)
class SpawnExploreSpec:
    pddls: Tuple[str]
    domain_type: Any  # DomainType enum
    trainer_seed: Optional[int]
    slot_id: Optional[int]
    num_slots: Optional[int]
    # everything else you pass into ProblemServiceConfig that affects behavior:
    ssipp_dg_heuristic: Optional[str]
    use_lm_cuts: bool
    use_numeric_landmarks: bool
    use_contributions: bool
    use_act_history: bool
    fd_heuristic: Optional[str]
    ssipp_teacher_heuristic: Optional[str]
    enhsp_config: Optional[str]
    estimator_h_to_v_coeff: float
    teacher_planner: str
    teacher_timeout_s: int
    only_one_good_action: bool
    use_teacher_envelope: bool
    max_len: int
    mcts_iterations: int
    heuristic_bootstrapping: bool
    mcts_her_strategy: bool
    mcts_expansion_k: int
    use_fluents: bool
    use_comps: bool
    difficulty: InstanceDifficulty
    mcts_progressive_widening: bool = False
    mcts_pw_min_width: int = 2
    mcts_pw_c: float = 0.6
    mcts_pw_alpha: float = 0.5
    fixed_instance_pddl: bool = False
    original_training_set: bool = False
    mcts_exploration_weight: float = 1.0
    sample_k_additional_states: int = 0
    freeze_train_steps: int = 50
    freeze_batch_size: int = 32
    goal_path_reconstruction: Optional[str] = None
    ENHSP_plan_bootstrap: bool = True
    est_plan_z: bool = False
    timeout: Optional[float] = None

    # mcts debugging options
    puct_debug: bool = False
    action_debug: bool = False

    # estimator decay
    use_estimator_decay: bool = False
    estimator_decay_coeff_start: float = 1.0
    estimator_decay_coeff_end: float = 0.2
    estimator_decay_epochs: int = 0

    # estimator without decay
    use_estimator: float = 0.0

    # action policy attributes
    action_policy: str = "argmax"
    action_policy_goal_chase_distance_threshold: int = -1
    action_policy_epsilon: float = None
    action_policy_temperature: float = None
    action_policy_decay_rate: float = None
    action_policy_duplicate_penalty: float = None

    # evaluation only attributes
    evaluation_mode: bool = False
    evaluation_index: Optional[int] = None
    mcts_enforce_remaining_horizon: bool = False
    mcts_terminal_safe_action_selection: bool = False

    def __str__(self) -> str:
        """A stylized and grouped representation of the spec."""
        # Define groupings for visual clarity
        groups = {
            "CORE": ["domain_type", "pddls", "difficulty", "trainer_seed", "slot_id", "num_slots"],
            "PLANNER / TEACHER": ["teacher_planner", "teacher_timeout_s", "use_teacher_envelope", "enhsp_config",
                                  "max_len"],
            "HEURISTICS": ["ssipp_dg_heuristic", "fd_heuristic", "ssipp_teacher_heuristic", "use_lm_cuts",
                           "use_numeric_landmarks", "use_contributions"],
            "MCTS & EXPLORATION": ["mcts_iterations", "mcts_expansion_k", "mcts_exploration_weight",
                                   "mcts_progressive_widening", "mcts_pw_min_width", "mcts_pw_c",
                                   "mcts_pw_alpha", "mcts_her_strategy", "sample_k_additional_states"],
            "ESTIMATOR & DECAY": ["estimator_h_to_v_coeff", "use_estimator", "estimator_decay_coeff_start",
                                  "estimator_decay_coeff_end", "estimator_decay_epochs"],
            "ACTION POLICY": ["action_policy", "action_policy_epsilon", "action_policy_temperature",
                              "action_policy_decay_rate", "action_policy_duplicate_penalty"],
            "MISC / TRAINING": ["use_fluents", "use_comps", "fixed_instance_pddl", "original_training_set",
                                "freeze_train_steps", "freeze_batch_size"]
        }

        header = f"\n=== SpawnExploreSpec ==="
        output = [header]

        # Track which fields we've already categorized
        seen_fields = set()
        for group_name, field_list in groups.items():
            output.append(f"\n  [ {group_name} ]")
            for f_name in field_list:
                if hasattr(self, f_name):
                    val = getattr(self, f_name)
                    output.append(f"    {f_name:<40} : {val}")
                    seen_fields.add(f_name)

        # Catch-all for any fields not explicitly grouped
        remaining = [f.name for f in fields(self) if f.name not in seen_fields]
        if remaining:
            output.append(f"\n  [ OTHER ]")
            for f_name in remaining:
                val = getattr(self, f_name)
                output.append(f"    {f_name:<40} : {val}")

        return "\n".join(output) + "\n"

    def change_diff_to(self, diff: InstanceDifficulty) -> SpawnExploreSpec:
        return self.duplicate({"difficulty": diff})

    def duplicate(self, to_update: dict | None = None):
        """
        Return a copy of this SpawnExploreSpec, optionally overriding fields.

        Example:
            new_spec = spec.duplicate({"max_len": 200})
        """
        if not to_update:
            return replace(self)

        return replace(self, **to_update)


@can_profile
def make_specs(args, specific_instances=None, evaluation_mode=False,
               difficulty: Optional[InstanceDifficulty] = None,
               evaluation_indices=None) -> \
list[SpawnExploreSpec]:
    only_one_good_action = args.sup_objective == SupervisedObjective.THERE_CAN_ONLY_BE_ONE

    num_slots = len(specific_instances) if specific_instances is not None else args.num_workers
    if evaluation_indices is not None:
        if specific_instances is None or len(evaluation_indices) != len(specific_instances):
            raise ValueError(
                "evaluation_indices must align with specific_instances")

    specs = []
    for slot_id in range(num_slots):
        pddls = (
            tuple(args.pddls)
            if specific_instances is None
            else (args.pddls[0], specific_instances[slot_id])
        )

        kwargs = dict(
            pddls=pddls,
            domain_type=args.domain_type,
            trainer_seed=args.seed,
            slot_id=(
                evaluation_indices[slot_id] - 1
                if evaluation_indices is not None else slot_id),
            evaluation_mode=evaluation_mode,
            num_slots=num_slots,
            ssipp_dg_heuristic=args.ssipp_dg_heuristic,
            use_lm_cuts=args.use_lm_cuts,
            use_numeric_landmarks=args.use_numeric_landmarks,
            use_contributions=args.use_contributions,
            use_act_history=args.use_act_history,
            fd_heuristic=args.fd_teacher_heuristic,
            ssipp_teacher_heuristic=args.ssipp_teacher_heuristic,
            enhsp_config=args.enhsp_config,
            estimator_h_to_v_coeff=args.estimator_h_to_v_coeff,
            teacher_planner=args.teacher_planner,
            teacher_timeout_s=args.teacher_timeout_s,
            only_one_good_action=only_one_good_action,
            use_teacher_envelope=args.use_teacher_envelope,
            max_len=args.limit_turns,
            mcts_iterations=args.mcts_iterations,
            heuristic_bootstrapping=args.heuristic_bootstrapping,
            mcts_her_strategy=args.mcts_her_strategy,
            mcts_expansion_k=args.mcts_expansion_size,
            mcts_progressive_widening=args.mcts_progressive_widening,
            mcts_pw_min_width=args.mcts_pw_min_width,
            mcts_pw_c=args.mcts_pw_c,
            mcts_pw_alpha=args.mcts_pw_alpha,
            use_fluents=args.use_fluents,
            use_comps=args.use_comparisons,
            difficulty=difficulty if difficulty is not None else InstanceDifficulty.EASY,
            evaluation_index=(
                evaluation_indices[slot_id]
                if evaluation_indices is not None else None),
            mcts_enforce_remaining_horizon=(
                evaluation_mode
                and args.eval_mcts_enforce_remaining_horizon),
            mcts_terminal_safe_action_selection=(
                evaluation_mode
                and args.eval_mcts_terminal_safe_action_selection),
            fixed_instance_pddl=args.fixed_instance,
            mcts_exploration_weight=args.mcts_exploration_weight,
            action_policy=args.action_policy,
            action_policy_epsilon=args.action_policy_epsilon,
            action_policy_temperature=args.action_policy_temperature,
            action_policy_decay_rate=args.action_policy_decay_rate,
            action_policy_duplicate_penalty=args.action_policy_duplicate_penalty,
            action_debug=args.action_debug,
            puct_debug=args.puct_debug,
            timeout=args.graceful_timeout,
            use_estimator=args.use_estimator,
        )

        if not evaluation_mode:
            kwargs.update(
                sample_k_additional_states=args.sample_k_additional_states,
                goal_path_reconstruction=args.goal_path_reconstruction,
                original_training_set=args.original_training_set,
                use_estimator_decay=args.use_estimator_decay,
                estimator_decay_coeff_start=args.estimator_decay_coeff_start,
                estimator_decay_coeff_end=args.estimator_decay_coeff_end,
                estimator_decay_epochs=(
                    args.estimator_decay_epochs
                    if args.estimator_decay_epochs is not None
                    else int(args.max_opt_epochs / 3)
                ),
                action_policy_goal_chase_distance_threshold=(
                    args.action_policy_goal_chase_distance_threshold
                ),
            )

        specs.append(SpawnExploreSpec(**kwargs))

    return specs
