# asnets/parallel_explore_spawn_grads.py
from __future__ import annotations

from dataclasses import dataclass, fields
from time import time
from typing import Any, Optional, List

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, Future, wait, ALL_COMPLETED

from asnets.spawn_train_worker import WorkerInput, WorkerOutput, run_worker_opt_profile, run_worker_eval, \
    WorkerInputEval


def run_epoch_spawn_grads(
        specs: list[Any],
        weights_np: dict,
        dropout: float,
        debug: bool,
        policy_only: bool,
        mse_coeff: float,
        l2_reg_coeff: float,
        l1_reg_coeff: float,
        l1_l2_reg_coeff: float,
        log: bool,
        curr_epoch: Optional[int],
        PROFILE_DIR: Optional[str] = None,
        corrupt_pi: Optional[str] = None,
        corrupt_z: Optional[str] = None,
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
        dropout: Dropout rate used by workers.
        debug: Whether debug mode is enabled.
        policy_only: Whether to train/evaluate policy only.
        mse_coeff: MSE loss coefficient.
        l2_reg_coeff: L2 regularization coefficient.
        l1_reg_coeff: L1 regularization coefficient.
        l1_l2_reg_coeff: Combined L1/L2 regularization coefficient.
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
    fn_start = time()
    outs: list[WorkerOutput] = []
    with ProcessPoolExecutor(
            max_workers=max_workers or len(specs),
            mp_context=ctx,
    ) as ex:
        submit_start = time()
        futs: list[Future[WorkerOutput]] = [
            ex.submit(
                run_worker_opt_profile,
                WorkerInput(
                    spec=spec,
                    epoch=curr_epoch,
                    weights_np=weights_np,
                    dropout=dropout,
                    debug=debug,
                    policy_only=policy_only,
                    log=log,
                    PROFILE_DIR=PROFILE_DIR,
                    corrupt_pi=corrupt_pi,
                    corrupt_z=corrupt_z,
                    mse_coeff=mse_coeff,
                    l2_reg_coeff=l2_reg_coeff,
                    l1_reg_coeff=l1_reg_coeff,
                    l1_l2_reg_coeff=l1_l2_reg_coeff,
                ),
            )
            for spec in specs
        ]
        wait_start = time()
        done, not_done = wait(
            futs,
            timeout=epoch_timeout,
            return_when=ALL_COMPLETED,
        )
        collect_start = time()
        for fut in done:
            try:
                outs.append(fut.result())
            except Exception as e:
                print(f"[TRAINER WARNING] Worker crashed: {e}")
        shutdown_start = time()
        if not_done:
            print(
                f"[TRAINER WARNING] {len(not_done)} worker(s) timed out | "
                f"timeout={epoch_timeout:.1f}s | epoch={curr_epoch}"
            )
            ex.shutdown(wait=False, cancel_futures=True)
        unfinished_start = time()
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


def run_epoch_spawn_eval(specs, weights_np, max_workers=None):
    ctx = mp.get_context("forkserver")
    outs = []
    with ProcessPoolExecutor(
            max_workers=max_workers or len(specs),
            mp_context=ctx,
    ) as ex:
        futs = []
        for spec in specs:
            inp = WorkerInputEval(
                spec=spec,
                epoch=None,
                weights_np=weights_np,
                dropout=0.0,
                debug=False,
                policy_only=False,
            )
            futs.append(ex.submit(run_worker_eval, inp))
        for f in as_completed(futs):
            outs.append(f.result())
    return outs


@dataclass
class SpawnExploreSpec:
    # minimal picklable config
    pddls: List[str]
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
    difficulty: Any  # InstanceDifficulty enum
    fixed_instance_pddl: bool = False
    original_training_set: bool = False
    mcts_exploration_weight: float = 1.0
    sample_k_additional_states: int = 0
    freeze_train_steps: int = 50
    freeze_batch_size: int = 32
    goal_path_reconstruction: Optional[str] = None

    # estimator decay
    estimator_decay: bool = False
    estimator_decay_coeff_start: float = 1.0
    estimator_decay_coeff_end: float = 0.2
    estimator_decay_epochs: int = 0

    # action policy attributes
    action_policy: str = "argmax"
    action_policy_goal_chase_distance_threshold: int = -1
    action_policy_epsilon: float = None
    action_policy_temperature: float = None
    action_policy_decay_rate: float = None

    # evaluation only attributes
    evaluation_instance_index: Optional[int] = None

    def __str__(self) -> str:
        """A stylized, grouped, and colorized representation of the spec."""
        # Define groupings for visual clarity
        groups = {
            "CORE": ["domain_type", "pddls", "difficulty", "trainer_seed", "slot_id", "num_slots"],
            "PLANNER / TEACHER": ["teacher_planner", "teacher_timeout_s", "use_teacher_envelope", "enhsp_config",
                                  "max_len"],
            "HEURISTICS": ["ssipp_dg_heuristic", "fd_heuristic", "ssipp_teacher_heuristic", "use_lm_cuts",
                           "use_numeric_landmarks", "use_contributions"],
            "MCTS & EXPLORATION": ["training_mcts_iterations", "mcts_expansion_k", "mcts_exploration_weight",
                                   "mcts_her_strategy", "sample_k_additional_states"],
            "ESTIMATOR & DECAY": ["estimator_h_to_v_coeff", "estimator_decay", "estimator_decay_coeff_start",
                                  "estimator_decay_coeff_end", "estimator_decay_epochs"],
            "ACTION POLICY": ["action_policy", "action_policy_epsilon", "action_policy_temperature",
                              "action_policy_decay_rate"],
            "MISC / TRAINING": ["use_fluents", "use_comps", "fixed_instance_pddl", "original_training_set",
                                "freeze_train_steps", "freeze_batch_size"]
        }

        # ANSI Color Codes (Optional - remove if you want plain text)
        CLR = "\033[94m"  # Blue
        VAL = "\033[92m"  # Green
        RST = "\033[0m"  # Reset

        header = f"\n{CLR}=== SpawnExploreSpec ==={RST}"
        output = [header]

        # Track which fields we've already categorized
        seen_fields = set()
        for group_name, field_list in groups.items():
            output.append(f"\n  {CLR}[ {group_name} ]{RST}")
            for f_name in field_list:
                if hasattr(self, f_name):
                    val = getattr(self, f_name)
                    output.append(f"    {f_name:<40} : {VAL}{val}{RST}")
                    seen_fields.add(f_name)

        # Catch-all for any fields not explicitly grouped
        remaining = [f.name for f in fields(self) if f.name not in seen_fields]
        if remaining:
            output.append(f"\n  {CLR}[ OTHER ]{RST}")
            for f_name in remaining:
                val = getattr(self, f_name)
                output.append(f"    {f_name:<40} : {VAL}{val}{RST}")

        return "\n".join(output) + "\n"
