# asnets/spawn_train_worker.py
from __future__ import annotations

import tqdm
from enhsp_wrapper.enhsp import ENHSP, PlanningResult, PlanningStatus
from .interfaces.enhsp_interface import ENHSP_CONFIGS
import cProfile
import os
import pstats
import time
from dataclasses import dataclass, field
from typing import Any, Optional, List

import numpy as np

from .prob_dom_meta import DomainMeta, ProblemMeta
from .utils.tf_utils import configure_tf_gpu_memory_growth
from post_training.action_selection_policy import build_action_policy
from .teacher import ENHSPTeacher, Teacher
from .teacher_cache import TeacherException
from .utils.pddl_utils import hlist_to_sexprs, replace_init_state

import tensorflow as tf

import logging

from asnets.models import PropNetworkWeights, PropNetwork

from asnets.spawn_context import LocalExploreContext
from asnets.state_reprs import CanonicalState, get_init_cstate, sample_next_state
from asnets.supervised import PlannerExtensions, planner_trace
from asnets.utils.generator_utils import extract_domain_name_from_file, Domain, InstanceDifficulty
from asnets.utils.py_utils import set_random_seeds, RandomPopContainer
from post_training.enhspwrapper import ENHSPEstimator, EstimatorMode
from post_training.training_mcts import TrainingMCTS

from enum import Enum, auto

LOGGER = logging.getLogger(__name__)


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True, kw_only=True)
class WorkerInput:
    spec: Any  # SpawnExploreSpec
    weights_np: dict  # PropNetworkWeights.export_numpy() result
    epoch: Optional[int]

    # logging
    log: bool = False
    log_weights: bool = False

    # profiling
    PROFILE_DIR: Optional[str] = None

    @property
    def seed(self):
        epoch_term = 0 if self.epoch is None else self.epoch * 128
        return self.spec.trainer_seed + self.spec.slot_id + epoch_term  # assuming max(workers) >>> 128


@dataclass(frozen=True)
class PolicyDrivenWorkerInput(WorkerInput):
    num_trajectories: int
    dynamic: bool
    min_new_pairs: Optional[int]
    max_new_pairs: Optional[int]
    recent_learning_time: Optional[int]
    expl_learn_ratio: Optional[int]


@dataclass(frozen=True)
class MCTSWorkerInput(WorkerInput):
    # loss cfg
    mse_coeff: float
    l2_reg_coeff: float
    l1_reg_coeff: float
    l1_l2_reg_coeff: float

    # run corruption settings for corruption testing
    corrupt_pi: Optional[str] = None  # "shuffle" | "random" | "zero" | None
    corrupt_z: Optional[str] = None  # "shuffle" | "random" | "zero" | None

    max_estimator_coeff: float = 1.0

    @property
    def estimator_coeff(self):
        if not self.spec.use_estimator:
            return 0.0
        return min(
            self.max_estimator_coeff,
            self.spec.estimator_decay_coeff_start +
            (self.spec.estimator_decay_coeff_end - self.spec.estimator_decay_coeff_start) *
                        min(self.epoch / self.spec.estimator_decay_epochs, 1)
        )


@dataclass
class WorkerOutput:
    hit_goal_mean: float
    n_samples: int
    loss_mean: float
    grads_np: list[np.ndarray]
    root_target_entropy: Optional[np.float64] = None
    root_pred_entropy: Optional[np.float64] = None
    root_kl: Optional[np.float64] = None
    instance_diff: InstanceDifficulty = None


class DataSource(Enum):
    TRAJECTORY = auto()
    TREE_SAMPLE = auto()
    HEURISTIC_BOOTSTRAP = auto()
    GOAL_PATH = auto()
    ENHSP_PLAN = auto()


@dataclass
class WorkerCollector:
    # --- core dataset ---
    cstates: List[Any] = field(default_factory=list)
    children: List[Any] = field(default_factory=list)
    actions: List[Optional[int]] = field(default_factory=list)
    pi_tgt: List[np.ndarray] = field(default_factory=list)
    z_tgt: List[float] = field(default_factory=list)
    sources: List[DataSource] = field(default_factory=list)

    hit_goal: float = 0.0

    # --------- data accumulation ---------

    def add_sample(
            self,
            cstate,
            children,
            action,
            pi,
            z,
            source: DataSource,
    ):
        self.cstates.append(cstate)
        self.children.append(children)
        self.actions.append(action)
        self.pi_tgt.append(pi.astype(np.float32))
        self.z_tgt.append(float(z))
        self.sources.append(source)

    # --------- batching ---------

    def as_batches(self):
        obs_batch = np.asarray([s.to_network_input() for s in self.cstates])
        pi_tgt = np.asarray(self.pi_tgt, dtype=np.float32)
        z_tgt = np.asarray(self.z_tgt, dtype=np.float32)
        return obs_batch, pi_tgt, z_tgt

    def get_trajectory_info_as_list(self):
        return [{'state': self.cstates[i], 'children': self.children[i], 'pi': self.pi_tgt[i], 'z': self.z_tgt[i]} for i
                in range(len(self.sources)) if self.sources[i] == DataSource.TRAJECTORY]

    def __len__(self):
        return len(self.cstates)


@dataclass
class WorkerCollectorWithLogging(WorkerCollector):
    # --- root diagnostics ---
    root_target_entropies: List[float] = field(default_factory=list)
    root_pred_entropies: List[float] = field(default_factory=list)
    root_kls: List[float] = field(default_factory=list)

    # --- prediction tracking ---
    pi_pred: List[np.ndarray] = field(default_factory=list)
    z_pred: List[float] = field(default_factory=list)

    # --- training losses ---
    xent_losses: List[float] = field(default_factory=list)
    mse_losses: List[float] = field(default_factory=list)
    reg_losses: List[float] = field(default_factory=list)

    # --------- logging helpers ---------

    def add_root_stats(self, entropy_t, entropy_p, kl):
        self.root_target_entropies.append(float(entropy_t))
        self.root_pred_entropies.append(float(entropy_p))
        self.root_kls.append(float(kl))

    def add_pred(self, pi_pred, z_pred=None):
        self.pi_pred.append(pi_pred.astype(np.float32))
        if z_pred is not None:
            self.z_pred.append(float(z_pred))

    def add_losses(self, xent, mse, reg):
        self.xent_losses.append(float(xent))
        self.mse_losses.append(float(mse))
        self.reg_losses.append(float(reg))

    # --------- summaries ---------

    def root_summary(self):
        import numpy as np
        return {
            "root_target_entropy": np.mean(self.root_target_entropies) if self.root_target_entropies else None,
            "root_pred_entropy": np.mean(self.root_pred_entropies) if self.root_pred_entropies else None,
            "root_kl": np.mean(self.root_kls) if self.root_kls else None,
        }

    def loss_summary(self):
        import numpy as np
        return {
            "xent_loss": np.mean(self.xent_losses) if self.xent_losses else None,
            "mse_loss": np.mean(self.mse_losses) if self.mse_losses else None,
            "reg_loss": np.mean(self.reg_losses) if self.reg_losses else None,
        }


@dataclass(frozen=True)
class EvalWorkerInput:
    spec: Any  # SpawnExploreSpec
    weights_np: dict
    epoch: Optional[int]

    @property
    def seed(self):
        return self.spec.trainer_seed + self.spec.slot_id


@dataclass
class EvalWorkerOutput:
    hit_goal: float
    steps: int
    instance_name: Optional[str] = None


def _build_planner_exts_from_spec(spec, epoch_num):
    domain_pddl_path = spec.pddls[0]
    domain = Domain.from_pddl_name(extract_domain_name_from_file(domain_pddl_path))
    instance_pddl_paths = spec.pddls[1:]
    if spec.evaluation_mode:
        instance_path = instance_pddl_paths[
            0
        ]
    else:
        if spec.original_training_set:
            num_workers = spec.num_slots
            this_worker_id = spec.slot_id
            dataset_size = len(instance_pddl_paths)
            instance_idx = (epoch_num * num_workers + this_worker_id) % dataset_size
            instance_path = instance_pddl_paths[instance_idx]
        elif spec.fixed_instance_pddl:
            instance_path = instance_pddl_paths[0]
        else:
            instance_path = str(domain.get_realtime_instance(spec.difficulty, spec.trainer_seed, spec.slot_id))
    pddls = [domain_pddl_path, instance_path]
    return PlannerExtensions(
        pddls,
        spec.domain_type,
        dg_ssipp_heuristic_name=spec.ssipp_dg_heuristic,
        dg_use_lm_cuts=spec.use_lm_cuts,
        dg_use_numeric_landmarks=spec.use_numeric_landmarks,
        dg_use_contributions=spec.use_contributions,
        dg_use_act_history=spec.use_act_history,
    )


def _compute_mcts_iterations(branching_factor: int, multiplier: int = 3, constant: int = 10, min_iter: int = 10,
                             max_iter: int = 200):
    return int(np.clip(constant + multiplier * branching_factor, min_iter, max_iter))


def _build_estimator(planner_exts, spec):
    """
    MUST return an object with:
        estimator.get_cstate_h(cstate) -> float
    """
    return ENHSPEstimator(planner_exts, enhsp_config=spec.enhsp_config)


def _rebuild_weight_manager_local(prob_meta, weights_np: dict):
    """
    MUST return PropNetworkWeights instance whose variables are created locally,
    then assigned from weights_np.
    """
    wm = PropNetworkWeights.from_numpy(prob_meta, weights_np)
    return wm


def _build_network_local(weight_manager_local, prob_meta):
    net = PropNetwork(
        weight_manager=weight_manager_local,
        problem_meta=prob_meta,
    )
    return net


# -----------------------------
# Loss (worker-side)
# -----------------------------

def _policy_xent_loss(pi_pred: tf.Tensor, pi_target: tf.Tensor) -> tf.Tensor:
    # pi_pred: probabilities (already masked softmax in your network)
    eps = tf.constant(1e-8, dtype=pi_pred.dtype)
    pi_pred = tf.clip_by_value(pi_pred, eps, 1.0)
    return -tf.reduce_mean(tf.reduce_sum(pi_target * tf.math.log(pi_pred), axis=1))


def _value_mse_loss(v_pred: tf.Tensor, z_target: tf.Tensor) -> tf.Tensor:
    v_pred = tf.squeeze(v_pred, axis=-1) if v_pred.shape.rank == 2 else tf.squeeze(v_pred)
    z_target = tf.cast(z_target, v_pred.dtype)
    return tf.reduce_mean(tf.square(v_pred - z_target))


def _reg_terms(vars_, l2, l1, l1_l2):
    # keep it simple, stupid: optional L1, L2, and combined coefficient
    reg = tf.constant(0.0, dtype=vars_[0].dtype if vars_ else tf.float32)
    if l2:
        reg += tf.add_n([tf.reduce_sum(tf.square(v)) for v in vars_]) * tf.cast(l2, reg.dtype)
    if l1:
        reg += tf.add_n([tf.reduce_sum(tf.abs(v)) for v in vars_]) * tf.cast(l1, reg.dtype)
    if l1_l2:
        reg += tf.add_n([tf.reduce_sum(tf.abs(v)) + tf.reduce_sum(tf.square(v)) for v in vars_]) * tf.cast(l1_l2,
                                                                                                           reg.dtype)
    return reg


def _corrupt_targets(inp, pi_tgt, z_tgt):
    # --------------------------------------------------
    # TARGET CORRUPTION (SANITY TEST MODE)
    # --------------------------------------------------

    rng = np.random.default_rng(inp.seed)

    # ---- corrupt π ----
    if inp.corrupt_pi is not None:
        if inp.corrupt_pi == "shuffle":
            rng.shuffle(pi_tgt)

        elif inp.corrupt_pi == "random":
            # random valid distributions
            pi_tgt = rng.random(pi_tgt.shape)
            pi_tgt = pi_tgt / (pi_tgt.sum(axis=1, keepdims=True) + 1e-8)

        else:
            raise ValueError(f"Unknown corrupt_pi mode: {inp.corrupt_pi}")

    # ---- corrupt z ----
    if inp.corrupt_z is not None:
        if inp.corrupt_z == "shuffle":
            rng.shuffle(z_tgt)

        elif inp.corrupt_z == "random":
            z_tgt = rng.normal(loc=0.0, scale=1.0, size=z_tgt.shape).astype(np.float32)

        elif inp.corrupt_z == "zero":
            z_tgt = np.zeros_like(z_tgt)

        else:
            raise ValueError(f"Unknown corrupt_z mode: {inp.corrupt_z}")

    return pi_tgt, z_tgt


def heuristic_bootstrapping(bootstrap_k: int, trajectory_info: list, ctx: LocalExploreContext,
                            one_best_action=False) -> list:
    result = []
    traj_len = len(trajectory_info)
    if traj_len >= bootstrap_k:
        sampled_traj_indices: List[int] = np.random.choice(traj_len, size=bootstrap_k, replace=False)
    else:
        sampled_traj_indices: List[int] = [i for i in range(traj_len)]
    print('[HEURISTIC_BOOTSTRAPPING] Acquiring sampled states heuristics.')
    for ind in sampled_traj_indices:
        sampled_state = trajectory_info[ind]['state']
        sampled_state_children = trajectory_info[ind]['children']
        if one_best_action:
            sampled_state_v, sampled_state_pi = ctx.get_state_v_pi_one_hot_est(sampled_state)
        else:
            sampled_state_v, _ = ctx.get_state_v_pi_one_hot_est(sampled_state)
            sampled_state_pi = ctx.get_state_pi_est(sampled_state_children)
        result.append(
            {'state': sampled_state, 'children': sampled_state_children, 'pi': sampled_state_pi,
             'z': sampled_state_v})
    return result


def plan_to_trajectory(enhsp_config: str, pddl_files: list[str], act_ident_to_ind, act_dim: int,
                       init_state: CanonicalState, ctx: LocalExploreContext, estimator: ENHSPEstimator,
                       est_plan_z: bool = False, enhsp_timeout: int = 15):
    params = ENHSP_CONFIGS[enhsp_config] + f" -timeout {enhsp_timeout}"
    planner = ENHSP(params)
    domain_path = pddl_files[0]
    instance_path = pddl_files[1]
    plan_res: PlanningResult = planner.plan(domain_path, instance_path)
    if plan_res.status == PlanningStatus.SUCCESS:
        plan_actions_int = [act_ident_to_ind[act_ident] for act_ident in plan_res.plan]
        plan_len = len(plan_actions_int) + 1
        plan_states = [init_state]
        plan_states_pi = []
        plan_states_z = []
        curr_state = init_state
        for i, act_int in enumerate(plan_actions_int):
            prev_state_pi = np.zeros(act_dim, dtype=np.float32)
            prev_state_pi[act_int] = 1.0
            plan_states_pi.append(prev_state_pi)
            prev_state_key = curr_state.state_key
            if est_plan_z:
                cached = estimator.state_key_cache.get(prev_state_key)
                if cached is None:
                    problem_hlist = replace_init_state(
                        estimator._problem_hlist,
                        curr_state.to_tup_state()
                    )
                    oneliner = hlist_to_sexprs(problem_hlist)
                    (h, _) = estimator.get_estimate_batched(
                        [oneliner],
                        EstimatorMode.V_ONLY
                    )[0]
                    coeff = ctx.estimator_h_to_v_coeff
                    est_v = float(np.exp(-coeff * h))
                    estimator.state_key_cache[prev_state_key] = (est_v, None)
                else:
                    est_v, _ = cached
                plan_states_z.append(est_v)
            else:
                dist_from_goal = plan_len - i
                plan_states_z.append(float(1 - (dist_from_goal / plan_len)))
            curr_state = ctx.env_simulate_step(curr_state, act_int)
            plan_states.append(curr_state)
        assert plan_states[-1].is_goal, "Somehow planner found a plan that is successful but does not reach the goal"
        plan_states = plan_states[:-1]
        assert len(plan_states) == len(plan_states_pi) == len(plan_states_z)
        return plan_states, plan_states_pi, plan_states_z


def _dbg_tf_threads(tag=""):
    # TF-configured thread pools (maybe 0/None meaning “default” depending on TF build)
    intra = tf.config.threading.get_intra_op_parallelism_threads()
    inter = tf.config.threading.get_inter_op_parallelism_threads()

    # Env vars that many backends obey
    omp = os.environ.get("OMP_NUM_THREADS")
    mkl = os.environ.get("MKL_NUM_THREADS")
    tfi = os.environ.get("TF_NUM_INTRAOP_THREADS")
    tfe = os.environ.get("TF_NUM_INTEROP_THREADS")

    print(
        f"[TF_THREADS] pid={os.getpid()} {tag} "
        f"intra={intra} inter={inter} "
        f"env(TF_INTRA={tfi}, TF_INTER={tfe}, OMP={omp}, MKL={mkl})",
        flush=True
    )


# -----------------------------
# Worker main
# -----------------------------
def run_worker(inp: MCTSWorkerInput) -> WorkerOutput:
    """
    This runs fully inside a spawned process.
    It returns grads as numpy arrays aligned to local weight_manager_local.all_weights order.
    """
    worker_tag = f"[W{inp.seed}|{os.getpid()}]"
    set_random_seeds(inp.seed, worker_tag=worker_tag)
    configure_tf_gpu_memory_growth()
    # _dbg_tf_threads(tag=f"{worker_tag} worker_start")
    # --- build instance infra ---
    CanonicalState.network_input_config(use_fluents=inp.spec.use_fluents, use_comparisons=inp.spec.use_comps)
    planner_exts = _build_planner_exts_from_spec(inp.spec, inp.epoch)
    estimator = _build_estimator(planner_exts, inp.spec)
    action_policy = build_action_policy(
        base_policy=inp.spec.action_policy,
        worker_tag=worker_tag,
        distance_threshold=np.inf if inp.spec.action_policy_goal_chase_distance_threshold == -1 else inp.spec.action_policy_goal_chase_distance_threshold,
        epsilon=inp.spec.action_policy_epsilon,
        temperature=inp.spec.action_policy_temperature,
        decay_rate=inp.spec.action_policy_decay_rate,
        epoch=inp.epoch,
    )
    act_dim = planner_exts.problem_meta.num_acts
    if hasattr(inp.spec, "mcts_iterations") and inp.spec.mcts_iterations > 0:
        mcts_iter = inp.spec.mcts_iterations
    else:
        branching_f = min(act_dim, inp.spec.mcts_expansion_k)
        mcts_iter = _compute_mcts_iterations(branching_f)
        if inp.log:
            print(f"{worker_tag} mcts_iterations was not set manually, calculated to be:{mcts_iter}")

    # local weight vars
    wm_local = _rebuild_weight_manager_local(planner_exts.problem_meta, inp.weights_np)
    value_head_enabled = wm_local.value_head_enabled
    if inp.log_weights:
        w = wm_local.all_weights[0]
        print(f"{worker_tag} after rebuild:", float(tf.reduce_mean(w)), float(tf.math.reduce_std(w)),
              float(tf.linalg.norm(w)))
    # local network for THIS instance
    net = _build_network_local(wm_local, planner_exts.problem_meta)

    # ctx for TrainingMCTS
    ctx = LocalExploreContext(
        planner_exts=planner_exts,
        estimator=estimator,
        estimator_h_to_v_coeff=inp.spec.estimator_h_to_v_coeff,
    )

    # --- run exploration ---
    # get init state
    cstate = ctx.get_init_state()
    select_logging = False
    mcts = TrainingMCTS(
        network=net,
        ctx=ctx,
        iterations=mcts_iter,
        expansion_k=inp.spec.mcts_expansion_k,
        exploration_weight=inp.spec.mcts_exploration_weight,
        sharpen_pi=0.1,
        log_visitations=False,
        select_logging=select_logging,
        estimator_coeff=inp.estimator_coeff,
    )

    mcts.initialise_tree(cstate)

    max_len = inp.spec.max_len

    collector = (
        WorkerCollectorWithLogging()
        if inp.log
        else WorkerCollector()
    )

    a = None

    # Episode
    for t in range(max_len):
        # terminal?
        if cstate.is_terminal:
            collector.hit_goal = 1.0 if cstate.is_goal else 0.0
            break

        pi, z = mcts.run_search()  # pi: (act_dim,), z: scalar
        collector.add_sample(
            cstate=cstate,
            children=mcts.get_children_of(cstate),
            action=a,
            pi=pi,
            z=z,
            source=DataSource.TRAJECTORY
        )
        # --- ROOT POLICY DIAGNOSTICS ---
        if inp.log:
            obs = cstate.to_network_input()
            obs_tf = tf.expand_dims(tf.convert_to_tensor(obs, tf.float32), 0)

            if value_head_enabled:
                pi_net, _ = net(obs_tf, training=False)
            else:
                pi_net = net(obs_tf, training=False)

            pi_mcts = tf.stop_gradient(tf.convert_to_tensor(pi, tf.float32))
            pi_net = tf.stop_gradient(pi_net[0])

            entropy_t = -tf.reduce_sum(
                pi_mcts * tf.math.log(tf.clip_by_value(pi_mcts, 1e-8, 1.0))
            )

            entropy_p = -tf.reduce_sum(
                pi_net * tf.math.log(tf.clip_by_value(pi_net, 1e-8, 1.0))
            )

            kl_t = tf.reduce_sum(
                pi_mcts * (
                        tf.math.log(tf.clip_by_value(pi_mcts, 1e-8, 1.0))
                        - tf.math.log(tf.clip_by_value(pi_net, 1e-8, 1.0))
                )
            )

            collector.add_root_stats(
                entropy_t.numpy(),
                entropy_p.numpy(),
                kl_t.numpy(),
            )

        if inp.spec.sample_k_additional_states:
            sampled_data = mcts.sample_k_sufficient_nodes(k=inp.spec.sample_k_additional_states)
            for item in sampled_data:
                # item['node'].state is the actual state representation (cstate)
                collector.add_sample(
                    cstate=item['node'].state,
                    children=None,
                    action=None,
                    pi=item['pi'],
                    z=item['z'],
                    source=DataSource.TREE_SAMPLE,
                )

        # sample action from pi masked by available children
        mask = mcts.get_children_mask(act_dim=act_dim)
        masked_pi = pi * mask
        s = masked_pi.sum()
        if s > 0:
            masked_pi = masked_pi / s
        else:
            # fallback uniform over valid actions
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            masked_pi = np.zeros_like(pi)
            masked_pi[valid] = 1.0 / len(valid)
        a = action_policy.select_action(mcts=mcts, pi=masked_pi)
        cstate = mcts.step_forward(a)

    if select_logging:
        mcts.get_select_depth_stats()

    # If ended due to max_len without terminal
    # TODO: make sure I don't miss successes if goal reached after max_len moves
    if not cstate.is_terminal:
        collector.hit_goal = 1.0 if cstate.is_goal else 0.0
    if inp.spec.heuristic_bootstrapping:
        trajectory_info = collector.get_trajectory_info_as_list()
        sampled_data = heuristic_bootstrapping(bootstrap_k=5, trajectory_info=trajectory_info, ctx=ctx)
        for item in sampled_data:
            collector.add_sample(
                cstate=item['state'],
                children=item['children'],
                action=None,
                pi=item['pi'],
                z=item['z'],
                source=DataSource.HEURISTIC_BOOTSTRAP,
            )
    if inp.spec.ENHSP_plan_bootstrap:
        plan_as_traj = plan_to_trajectory(enhsp_config=inp.spec.enhsp_config,
                                          pddl_files=planner_exts.pddl_files,
                                          act_ident_to_ind=planner_exts.act_ident_to_ind,
                                          act_dim=act_dim,
                                          init_state=mcts.original_tree_root.state,
                                          ctx=ctx, estimator=estimator)
        if plan_as_traj:
            plan_states, plan_states_pi, plan_states_z = plan_as_traj
            for state, pi, z in zip(plan_states, plan_states_pi, plan_states_z):
                collector.add_sample(
                    cstate=state,
                    children=None,
                    action=None,
                    pi=pi,
                    z=z,
                    source=DataSource.ENHSP_PLAN,
                )
    reconstruct_goal_path = inp.spec.goal_path_reconstruction
    if reconstruct_goal_path:
        trajectory_info = collector.get_trajectory_info_as_list()
        if reconstruct_goal_path == "closest":
            target_list = mcts.reconstruct_goal_path_closest(trajectory_info)
        elif reconstruct_goal_path == "all":
            target_list = mcts.reconstruct_goal_paths_from_trajectory(trajectory_info)
        else:
            raise NotImplementedError("Only implemented 'all' and 'closest' reconstruction options")
        for item in target_list:
            collector.add_sample(
                cstate=item['state'],
                children=item['children'],
                action=None,
                pi=item['pi'],
                z=item['z'],
                source=DataSource.GOAL_PATH,
            )

    if len(collector) == 0:
        zeros = [np.zeros(v.shape, dtype=np.float32) for v in wm_local.all_weights]
        return WorkerOutput(
            hit_goal_mean=collector.hit_goal,
            n_samples=0,
            loss_mean=0.0,
            grads_np=zeros,
            instance_diff=inp.spec.difficulty,
        )

    obs_batch, pi_tgt, z_tgt = collector.as_batches()

    from collections import Counter

    if inp.log:
        log_lines = []
        log_lines.append("\n=== WORKER DIAGNOSTICS ===")

        # ---- source masks & counts ----
        src_list = collector.sources
        all_sources = list(DataSource)
        masks = {ds: np.asarray([s == ds for s in src_list], dtype=bool) for ds in all_sources}

        counts = Counter(src_list)
        counts_str = ", ".join(f"{ds.name}:{counts[ds]}" for ds in all_sources if counts.get(ds, 0) > 0)

        # Ensure shapes
        pi_tgt_2d = np.atleast_2d(pi_tgt)
        z_tgt_1d = np.asarray(z_tgt, dtype=np.float32)

        applicable = np.count_nonzero(pi_tgt_2d, axis=1)
        log_lines.append(
            f"Samples: {pi_tgt_2d.shape[0]}   [{counts_str}]   Hit:{collector.hit_goal}   AppActsμ:{applicable.mean():.2f}")

        # ---- batched predictions for ALL samples ----
        obs_tf = tf.convert_to_tensor(obs_batch, dtype=tf.float32)

        if value_head_enabled:
            pi_pred_all, v_pred_all = net(obs_tf, training=False)
        else:
            pi_pred_all = net(obs_tf, training=False)
            v_pred_all = None

        # convert to numpy, squeeze if needed
        pi_pred_all = tf.stop_gradient(pi_pred_all).numpy()
        pi_pred_2d = np.atleast_2d(pi_pred_all)

        if v_pred_all is not None:
            v_pred_all = tf.stop_gradient(v_pred_all).numpy()
            v_pred_1d = np.asarray(v_pred_all, dtype=np.float32).reshape(-1)
        else:
            v_pred_1d = None

        # ---- per-source stats helpers ----
        def _mean_std(x: np.ndarray):
            return float(np.mean(x)), float(np.std(x))

        def _tgt_pi_stats(mask):
            if mask.sum() == 0:
                return None
            m = np.max(pi_tgt_2d[mask], axis=1)
            return _mean_std(m)

        def _pred_pi_stats(mask):
            if mask.sum() == 0:
                return None
            m = np.max(pi_pred_2d[mask], axis=1)
            return _mean_std(m)

        def _tgt_v_stats(mask):
            if mask.sum() == 0:
                return None
            return _mean_std(z_tgt_1d[mask])

        def _pred_v_stats(mask):
            if v_pred_1d is None or mask.sum() == 0:
                return None
            return _mean_std(v_pred_1d[mask])

        # ---- compact policy/value table (only sources that exist) ----
        header = (
            f"{'SOURCE':<22} {'n':>5}  "
            f"{'max(pi_tgt) μ±σ':>16}  {'max(pi_pred) μ±σ':>17}  "
            f"{'z_tgt μ±σ':>12}  {'v_pred μ±σ':>12}"
        )
        log_lines.append("\n" + header)
        log_lines.append("-" * len(header))

        for ds in all_sources:
            mask = masks[ds]
            n = int(mask.sum())
            if n == 0:
                continue

            tgt_pi = _tgt_pi_stats(mask)
            pred_pi = _pred_pi_stats(mask)
            tgt_v = _tgt_v_stats(mask)
            pred_v = _pred_v_stats(mask)

            tgt_pi_s = f"{tgt_pi[0]:.4f}±{tgt_pi[1]:.4f}" if tgt_pi else "-"
            pred_pi_s = f"{pred_pi[0]:.4f}±{pred_pi[1]:.4f}" if pred_pi else "-"
            tgt_v_s = f"{tgt_v[0]:.3f}±{tgt_v[1]:.3f}" if tgt_v else "-"
            pred_v_s = f"{pred_v[0]:.3f}±{pred_v[1]:.3f}" if pred_v else "-"

            log_lines.append(f"{ds.name:<22} {n:>5}  {tgt_pi_s:>16}  {pred_pi_s:>17}  {tgt_v_s:>12}  {pred_v_s:>12}")

        # ---- argmax match (trajectory only), overlap-safe ----
        traj_mask = masks[DataSource.TRAJECTORY]
        n_cmp = int(traj_mask.sum())
        if n_cmp > 0:
            arg_tgt = np.argmax(pi_tgt_2d[traj_mask], axis=1)
            arg_pred = np.argmax(pi_pred_2d[traj_mask], axis=1)
            argmax_match = float(np.mean(arg_tgt == arg_pred))
        else:
            argmax_match = float("nan")
        log_lines.append(f"\nPolicy: argmax_match(traj)={argmax_match:.4f} (n={n_cmp})")

        # ---- root diagnostics single-line ----
        root_summary = collector.root_summary()
        rt = root_summary.get("root_target_entropy")
        rp = root_summary.get("root_pred_entropy")
        rk = root_summary.get("root_kl")
        rt_s = f"{rt:.4f}" if rt is not None else "None"
        rp_s = f"{rp:.4f}" if rp is not None else "None"
        rk_s = f"{rk:.4f}" if rk is not None else "None"
        log_lines.append(f"RootDiag: H_tgt={rt_s}  H_pred={rp_s}  KL={rk_s}")
    if inp.corrupt_pi is not None or inp.corrupt_z is not None:
        pi_tgt, z_tgt = _corrupt_targets(inp, pi_tgt, z_tgt)

    # --- compute grads locally ---
    vars_ = wm_local.all_weights

    K_train_steps = 10  # TODO: later put this inside the WorkerInput so it can be changed throughout the run if needed, like LR steps

    accum_grads = [np.zeros(v.shape, dtype=np.float32) for v in vars_]

    for step in range(K_train_steps):

        # optional: sample minibatch
        idx = np.random.choice(obs_batch.shape[0], size=min(32, obs_batch.shape[0]), replace=False)

        obs_mb = obs_batch[idx]
        pi_mb = pi_tgt[idx]
        z_mb = z_tgt[idx] if value_head_enabled else None

        with tf.GradientTape() as tape:
            if value_head_enabled:
                pi_pred, v_pred = net(obs_mb, training=True)
                xent_loss = _policy_xent_loss(pi_pred, tf.convert_to_tensor(pi_mb, dtype=pi_pred.dtype))
                mse_loss = tf.cast(inp.mse_coeff, xent_loss.dtype) * _value_mse_loss(v_pred, tf.convert_to_tensor(z_mb,
                                                                                                                  dtype=v_pred.dtype))
            else:
                pi_pred = net(obs_mb, training=True)
                xent_loss = _policy_xent_loss(pi_pred, tf.convert_to_tensor(pi_mb, dtype=pi_pred.dtype))
                mse_loss = 0.0

            reg_loss = _reg_terms(vars_, inp.l2_reg_coeff, inp.l1_reg_coeff, inp.l1_l2_reg_coeff)
            loss = xent_loss + mse_loss + reg_loss
            if inp.log:
                collector.add_losses(
                    xent_loss.numpy(),
                    float(mse_loss.numpy()) if value_head_enabled else 0.0,
                    reg_loss.numpy(),
                )

        grads = tape.gradient(loss, vars_)

        for i, g in enumerate(grads):
            accum_grads[i] += g.numpy().astype(np.float32)

    # average grads across local steps
    grads_np = [g / K_train_steps for g in accum_grads]
    if inp.log:
        ls = collector.loss_summary()
        x = ls.get("xent_loss")
        m = ls.get("mse_loss")
        r = ls.get("reg_loss")
        x_s = f"{x:.4f}" if x is not None else "None"
        m_s = f"{m:.4f}" if m is not None else "None"
        r_s = f"{r:.4f}" if r is not None else "None"
        log_lines.append(f"Loss: xent={x_s}  mse={m_s}  reg={r_s}")
    root_summary = collector.root_summary() if inp.log else {}
    if inp.log:
        print(f"{worker_tag} " + f"\n{worker_tag} ".join(log_lines), flush=True)

    return WorkerOutput(
        hit_goal_mean=float(collector.hit_goal),
        n_samples=int(obs_batch.shape[0]),
        loss_mean=float(loss.numpy()),
        grads_np=grads_np,
        root_target_entropy=root_summary.get("root_target_entropy"),
        root_pred_entropy=root_summary.get("root_pred_entropy"),
        root_kl=root_summary.get("root_kl"),
        instance_diff=inp.spec.difficulty
    )


def run_worker_opt_profiled(inp: WorkerInput, worker_fn=run_worker) -> WorkerOutput:
    # Make sure this directory exists (spawn safe)
    prof = None
    t0 = time.time()

    if inp.PROFILE_DIR:
        os.makedirs(inp.PROFILE_DIR, exist_ok=True)
        prof = cProfile.Profile()
        prof.enable()

    try:
        return worker_fn(inp)
    finally:
        if prof is not None:
            prof.disable()
            pid = os.getpid()
            seed = getattr(inp, "seed", None)
            spec_name = getattr(getattr(inp, "spec", None), "name", None)
            tag = f"pid{pid}_seed{seed}_spec{spec_name}"
            path = os.path.join(inp.PROFILE_DIR, f"{worker_fn.__name__}_{tag}.prof")
            prof.dump_stats(path)

            # Optional: also write a tiny human-readable "top 30" alongside it
            txt_path = os.path.join(inp.PROFILE_DIR, f"{worker_fn.__name__}_{tag}.top.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                ps = pstats.Stats(prof, stream=f).sort_stats("cumtime")
                ps.print_stats(30)

        # Optional: coarse phase timings even without pstats
        print(f"[WORKER TIMING] pid={os.getpid()} total={time.time() - t0:.2f}s", flush=True)


def init_eval_worker(inp: EvalWorkerInput, worker_tag_addon: Optional[str] = None) -> tuple[str, str]:
    worker_tag_prefix = f"EVAL{'_' + worker_tag_addon if worker_tag_addon else ''}"
    difficulty_str = str(inp.spec.difficulty)
    full_worker_tag = f"[{worker_tag_prefix}|{difficulty_str}|{os.getpid()}]"
    instance_name = f"[{str(inp.spec.slot_id)}] {inp.spec.pddls[1]}"
    set_random_seeds(inp.seed, worker_tag=full_worker_tag)
    configure_tf_gpu_memory_growth()
    CanonicalState.network_input_config(
        use_fluents=inp.spec.use_fluents,
        use_comparisons=inp.spec.use_comps,
    )
    return full_worker_tag, instance_name


def eval_max_len_coeff_by_diff(diff: InstanceDifficulty) -> float:
    coeff_dict = {
        InstanceDifficulty.EASY: 1.0,
        InstanceDifficulty.MEDIUM: 5.0,
        InstanceDifficulty.HARD: 10.0,
    }
    return coeff_dict[diff]

def run_worker_eval_mcts(inp: EvalWorkerInput) -> EvalWorkerOutput:
    worker_tag, instance_name = init_eval_worker(inp)
    planner_exts = _build_planner_exts_from_spec(inp.spec, inp.epoch)
    act_dim = planner_exts.problem_meta.num_acts
    estimator = _build_estimator(planner_exts, inp.spec)
    action_policy = build_action_policy(
        base_policy=inp.spec.action_policy,
        worker_tag=worker_tag,
        distance_threshold=np.inf,  # on evaluation there is always a need to goal chase
        epsilon=inp.spec.action_policy_epsilon,
        temperature=inp.spec.action_policy_temperature,
        decay_rate=inp.spec.action_policy_decay_rate,
    )
    wm_local = _rebuild_weight_manager_local(
        planner_exts.problem_meta,
        inp.weights_np,
    )
    net = _build_network_local(
        wm_local,
        planner_exts.problem_meta,
    )
    ctx = LocalExploreContext(
        planner_exts=planner_exts,
        estimator=estimator,
        estimator_h_to_v_coeff=inp.spec.estimator_h_to_v_coeff,
    )
    if hasattr(inp.spec, "mcts_iterations") and inp.spec.mcts_iterations > 0:
        mcts_iter = inp.spec.mcts_iterations
    else:
        branching_f = min(act_dim, inp.spec.mcts_expansion_k)
        mcts_iter = _compute_mcts_iterations(branching_f)
        print(f"{worker_tag} mcts_iterations was not set manually, calculated to be:{mcts_iter}")
    mcts = TrainingMCTS(
        network=net,
        ctx=ctx,
        iterations=mcts_iter,
        expansion_k=inp.spec.mcts_expansion_k,
        exploration_weight=inp.spec.mcts_exploration_weight,
        sharpen_pi=0.1,
        log_visitations=False,
        select_logging=False,
        estimator_coeff=0.0,  # IMPORTANT difference vs training, estimator must not be used
    )
    cstate = ctx.get_init_state()
    max_len = inp.spec.max_len * eval_max_len_coeff_by_diff(inp.spec.difficulty)
    mcts.initialise_tree(cstate)

    for step in range(max_len):
        if cstate.is_terminal:
            return EvalWorkerOutput(
                hit_goal=float(cstate.is_goal),
                steps=step,
                instance_name=instance_name,
            )
        pi, _ = mcts.run_search()
        mask = mcts.get_children_mask(act_dim=act_dim)
        masked_pi = pi * mask
        s = masked_pi.sum()
        if s > 0:
            masked_pi /= s
        else:
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            masked_pi = np.zeros_like(pi)
            masked_pi[valid] = 1 / len(valid)
        action = action_policy.select_action(
            mcts=mcts,
            pi=masked_pi,
        )
        cstate = mcts.step_forward(action)
    return EvalWorkerOutput(
        hit_goal=float(cstate.is_goal),
        steps=max_len,
        instance_name=instance_name,
    )


def run_worker_eval_policy_only(inp: EvalWorkerInput) -> EvalWorkerOutput:
    worker_tag, instance_name = init_eval_worker(inp, "POLICY")

    planner_exts = _build_planner_exts_from_spec(
        inp.spec,
        inp.epoch,
    )
    action_policy_str = inp.spec.action_policy
    assert action_policy_str in ["argmax",
                                 "sample"], f"Cannot use visit proportional action policy on non-mcts evaluation ({action_policy_str})"
    action_policy = build_action_policy(
        base_policy=action_policy_str,
        worker_tag=worker_tag,
        epsilon=inp.spec.action_policy_epsilon,
        temperature=inp.spec.action_policy_temperature,
    )
    # --------------------------------------------------
    # Rebuild network locally
    # --------------------------------------------------
    wm_local = _rebuild_weight_manager_local(
        planner_exts.problem_meta,
        inp.weights_np,
    )
    net = _build_network_local(
        wm_local,
        planner_exts.problem_meta,
    )
    ctx = LocalExploreContext(
        planner_exts=planner_exts,
        estimator=None,
    )
    cstate = ctx.get_init_state()
    max_len = inp.spec.max_len * eval_max_len_coeff_by_diff(inp.spec.difficulty)
    for step in range(max_len):
        if cstate.is_terminal:
            return EvalWorkerOutput(
                hit_goal=float(cstate.is_goal),
                steps=step,
                instance_name=instance_name,
            )
        obs = cstate.to_network_input()
        if net.value_head_enabled:
            pi, _ = net(obs[None], training=False)
            pi = pi.numpy()[0]
        else:
            pi = net(obs[None], training=False).numpy()[0]
        # mask invalid actions exactly like MCTS worker
        mask = cstate.get_applicable_action_mask()
        masked_pi = pi * mask
        s = masked_pi.sum()
        if s > 0:
            masked_pi /= s
        else:
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            masked_pi = np.zeros_like(pi)
            masked_pi[valid] = 1 / len(valid)
        action_id = action_policy.select_action(
            mcts=None,  # intentionally None
            pi=masked_pi,
        )
        cstate = ctx.env_simulate_step(cstate, action_id)
    return EvalWorkerOutput(
        hit_goal=float(cstate.is_goal),
        steps=max_len,
        instance_name=instance_name,
    )


def make_enhsp_value_target_fn(estimator, h_to_v_coeff: float = 1.0):
    """
    Returns a callable mapping CanonicalState -> scalar value target.

    Uses ENHSP heuristic estimate converted into value signal.
    """

    def value_target_fn(cstate: CanonicalState, distance_to_goal: int):
        h = estimator.evaluate_state(cstate)

        if h is None:
            return 0.0

        # Convert heuristic distance into bounded value
        # smaller h = better → larger value
        return 1.0 / (1.0 + h_to_v_coeff * h)

    return value_target_fn


def distance_to_goal_value_target(cstate: CanonicalState, distance_to_goal: int) -> float:
    return 1.0 / (1.0 + float(distance_to_goal))


def run_multiple_trajectory_collection(inp: PolicyDrivenWorkerInput):
    spec = inp.spec
    epoch_num = inp.epoch
    start_time = time.time()
    # Stage 1 - collect trajectories by current policy
    CanonicalState.network_input_config(
        use_fluents=spec.use_fluents,
        use_comparisons=spec.use_comps
    )
    pe = _build_planner_exts_from_spec(spec, epoch_num)
    wm_local = _rebuild_weight_manager_local(
        pe.problem_meta,
        inp.weights_np,
    )
    net = _build_network_local(
        wm_local,
        pe.problem_meta,
    )
    value_target_fn = None
    if net.value_head_enabled:
        if spec.use_estimator:
            estimator = _build_estimator(pe, spec)
            value_target_fn = make_enhsp_value_target_fn(estimator, spec.estimator_h_to_v_coeff)
        else:
            value_target_fn = distance_to_goal_value_target
    teacher_timeout_s = 15
    teacher = ENHSPTeacher(planner_exts=pe, teacher_timeout_s=teacher_timeout_s, enhsp_config=spec.enhsp_config)
    model_cache = {}
    trajectories = []
    for _ in range(inp.num_trajectories):
        path, hit_goal = collect_single_trajectory(spec, pe, net,
                                                   model_cache)  # model_cache might be useless - in hit rate and in speed of network, both cpu and gpu
        trajectories.append((path, hit_goal))
    # Stage 2 - collect expert trajectories by planner from either dynamic (with _terminate) or static (just grab all of them) explorer
    expert_trajectories = []
    first_explore = epoch_num == 0
    if inp.dynamic:
        t = tqdm.tqdm(desc='dynamic explore', total=inp.max_new_pairs)
        last_progress_time = int(time.time())
        total_new_pairs = 0
        cont = RandomPopContainer()
        for path, _ in trajectories:
            for state, act in path:
                cont.add(state)
        while True:
            terminate, last_progress_time = _terminate(start_time, total_new_pairs, inp.min_new_pairs,
                                                       inp.max_new_pairs,
                                                       last_progress_time, t,
                                                       first_explore, inp.recent_learning_time, inp.expl_learn_ratio)
            if terminate or len(cont) == 0:
                break
            cstate = cont.pop_random()
            tup_output = explore_from_state(spec=spec, epoch_num=epoch_num, cstate=cstate, pe=pe, teacher=teacher,
                                            only_one_good_action=spec.only_one_good_action,
                                            use_teacher_envelope=spec.use_teacher_envelope,
                                            value_target_fn=value_target_fn)
            if tup_output:  # to avoid crashing the exploration process over teacher failure
                expert_trajectories.extend(tup_output)
                total_new_pairs += len(tup_output)

    else:
        total_states = sum(len(path) for path, _ in trajectories)
        pbar = tqdm.tqdm(total=total_states, desc='static explore')
        added_tuples = 0
        for path, _ in trajectories:
            for cstate, act in path:
                tup_output = explore_from_state(spec=spec, epoch_num=epoch_num, cstate=cstate, pe=pe, teacher=teacher,
                                                only_one_good_action=spec.only_one_good_action,
                                                use_teacher_envelope=spec.use_teacher_envelope,
                                                value_target_fn=value_target_fn)
                if tup_output:  # to avoid crashing the exploration process over teacher failure
                    expert_trajectories.extend(tup_output)
                    added_tuples += len(tup_output)
                pbar.set_postfix({"new expert knowledge": added_tuples}, refresh=False)
                pbar.update(1)
    return expert_trajectories, trajectories


def collect_single_trajectory(spec, pe, net, model_cache):
    hit_goal = False
    path = []
    cstate = get_init_cstate(pe)
    for _ in range(spec.max_len):
        obs = cstate.to_network_input()
        obs_bytes = obs.tobytes()
        if obs_bytes not in model_cache:
            if net.value_head_enabled:
                act_dist, _ = net(obs[None], training=False)
            else:
                act_dist = net(obs[None], training=False)
            act_dist = tf.reshape(act_dist, [-1, ], ).numpy()
            s = np.sum(act_dist)
            if s == 0:
                act_dist[:] = 1 / len(act_dist)
            else:
                act_dist /= s
            model_cache[obs_bytes] = act_dist
        else:
            act_dist = model_cache[obs_bytes]
        action = int(np.random.choice(np.arange(act_dist.shape[0]), p=act_dist))

        path.append((cstate, pe.problem_meta.bound_acts_ordered[action]))
        cstate, _ = sample_next_state(cstate=cstate, action_id=action, planner_exts=pe)
        if cstate.is_terminal:
            if cstate.is_goal:
                hit_goal = True
            break
    return path, hit_goal


def explore_from_state(
        spec,
        epoch_num,
        cstate: CanonicalState,
        pe,
        teacher: Teacher,
        only_one_good_action: bool = True,
        use_teacher_envelope: bool = True,
        value_target_fn=None,
):
    """
    Returns planner envelope as:

        [(state, action)]
    OR
        [(state, action, value_target)]

    depending on whether value_target_fn is provided.
    """

    if pe is None:
        pe = _build_planner_exts_from_spec(spec, epoch_num)
    try:
        teacher_experience = planner_trace(
            planner=teacher,
            planner_exts=pe,
            root_cstate=cstate,
            only_one_good_action=only_one_good_action,
            use_teacher_envelope=use_teacher_envelope,
        )
    except TeacherException as ex:
        LOGGER.warning(f"Teacher error on problem {pe.problem_name} ({ex})")
        return None
    filtered_reversed = []
    distance_to_goal = 0
    for env_cstate, act in reversed(teacher_experience):
        nactions = sum(p[1] for p in env_cstate.acts_enabled)
        if nactions > 1:
            if value_target_fn is None:
                filtered_reversed.append((env_cstate, act))
            else:
                z = value_target_fn(env_cstate, distance_to_goal)
                filtered_reversed.append((env_cstate, act, z))
        distance_to_goal += 1
    return list(reversed(filtered_reversed))


def _terminate(start_time: float, total_new_pairs: int, min_new_pairs: int, max_new_pairs: int, last_progress_time: int,
               t: tqdm.tqdm, first_explore: bool, recent_learning_time: int, expl_learn_ratio: int) -> tuple[bool, int]:
    if first_explore:
        t.update(total_new_pairs - t.n)
        last_progress_time = int(time.time())
        return total_new_pairs >= min_new_pairs, last_progress_time

    # Terminating when there seems to be no progress
    if total_new_pairs == t.n:
        if time.time() - last_progress_time > 10:
            LOGGER.warning('No progress in exploration phase for 10s, aborting')
            return True, last_progress_time
    else:
        last_progress_time = int(time.time())
        t.update(total_new_pairs - t.n)

    # hard termination when we take too long
    if time.time() - start_time > 3 * expl_learn_ratio * recent_learning_time:
        return True, last_progress_time
    if total_new_pairs >= max_new_pairs:
        return True, last_progress_time
    if total_new_pairs >= min_new_pairs:
        return time.time() - start_time >= expl_learn_ratio * recent_learning_time, last_progress_time
    return False, last_progress_time


@dataclass(frozen=True)
class ProblemInitData:
    slot_id: int
    name: str
    obs_dim: int
    act_dim: int
    dom_meta: DomainMeta  # might cause circular imports
    prob_meta: ProblemMeta  # might cause circular imports
    ssipp_dead_end_value: int


def collect_problem_dims_worker(inp: Any) -> ProblemInitData:
    """
    Runs in a fresh spawn process.

    Purpose:
        Build planner extensions / mdpsim for exactly one problem,
        extract grounded obs/action dimensions, return plain metadata.

    Must avoid importing TensorFlow here if possible.
    """
    spec = inp.spec
    CanonicalState.network_input_config(
        use_fluents=spec.use_fluents,
        use_comparisons=spec.use_comps,
    )
    pe = _build_planner_exts_from_spec(spec, 0)

    # Adapt these to your real fields.
    init_cstate = get_init_cstate(pe)
    obs = init_cstate.to_network_input()

    obs_dim = int(obs.shape[-1])
    act_dim = int(pe.problem_meta.num_acts)

    return ProblemInitData(
        slot_id=spec.slot_id,
        name=pe.current_problem_name,
        obs_dim=obs_dim,
        act_dim=act_dim,
        dom_meta=pe.domain_meta,
        prob_meta=pe.problem_meta,
        ssipp_dead_end_value=pe.ssipp_dead_end_value
    )
