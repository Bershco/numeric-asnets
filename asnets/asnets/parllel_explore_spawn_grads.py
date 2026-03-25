# asnets/parallel_explore_spawn_grads.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from asnets.spawn_train_worker import WorkerInput, WorkerOutput, run_worker_opt_profile


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
    PROFILE_DIR: Optional[str] = None,
    corrupt_pi: Optional[str] = None,
    corrupt_z: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> list[WorkerOutput]:

    ctx = mp.get_context('forkserver')

    outs: list[WorkerOutput] = []
    with ProcessPoolExecutor(max_workers=max_workers or len(specs), mp_context=ctx) as ex:
        futs = []
        for i, spec in enumerate(specs):
            # seed override (optional) per spec/slot
            seed = getattr(spec, "random_seed", None)
            inp = WorkerInput(
                spec=spec,
                weights_np=weights_np,
                seed=seed,
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
            )
            futs.append(ex.submit(run_worker_opt_profile, inp))

        for f in as_completed(futs):
            outs.append(f.result())

    return outs


@dataclass
class SpawnExploreSpec:
    # minimal picklable config
    pddls: List[str]
    domain_type: Any  # DomainType enum
    random_seed: Optional[int]

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
    estimator_decay: bool
    teacher_planner: str
    teacher_timeout_s: int
    only_one_good_action: bool
    use_teacher_envelope: bool
    max_len: int
    training_mcts_iterations: int
    heuristic_bootstrapping: bool
    mcts_her_strategy: bool
    mcts_expansion_k: int
    use_fluents: bool
    use_comps: bool
    difficulty: Any  # InstanceDifficulty enum
    fixed_instance_pddl: bool = False
    mcts_exploration_weight: float = 1.0
    sample_k_additional_states: int = 5
    freeze_train_steps: int = 50
    freeze_batch_size: int = 32
    goal_path_reconstruction: Optional[str] = None

    # action policy attributes
    action_policy: str = "argmax"
    action_policy_goal_chase_distance_threshold: int = -1
    action_policy_epsilon: float = None
    action_policy_temperature: float = None
    action_policy_decay_rate: float = None
