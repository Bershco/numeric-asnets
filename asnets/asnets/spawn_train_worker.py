# asnets/spawn_train_worker.py
from __future__ import annotations

import cProfile
import os
import pstats
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, List

import numpy as np
from asnets.models import configure_tf_gpu_memory_growth
from post_training.action_selection_policy import build_action_policy

_T0 = time.time()
print(f"[WORKER_IMPORT] pid={os.getpid()} module start", flush=True)
t = time.time()
import tensorflow as tf

print(
    f"[WORKER_IMPORT] pid={os.getpid()} import tensorflow: {time.time() - t:.2f}s (since module start {time.time() - _T0:.2f}s)",
    flush=True)
import logging

t = time.time()
from asnets.models import PropNetworkWeights, PropNetwork

print(
    f"[WORKER_IMPORT] pid={os.getpid()} import asnets models: {time.time() - t:.2f}s (since module start {time.time() - _T0:.2f}s)",
    flush=True)
from asnets.spawn_context import LocalExploreContext
from asnets.state_reprs import CanonicalState
from asnets.supervised import PlannerExtensions
from asnets.utils.generator_utils import extract_domain_name_from_file, Domain
from asnets.utils.py_utils import set_random_seeds
from post_training.enhspwrapper import ENHSPEstimator
from post_training.training_mcts import TrainingMCTS

from enum import Enum, auto

LOGGER = logging.getLogger(__name__)


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class WorkerInput:
    spec: Any  # SpawnExploreSpec
    weights_np: dict  # PropNetworkWeights.export_numpy() result
    seed: Optional[int]
    dropout: float
    debug: bool
    policy_only: bool

    # loss cfg
    mse_coeff: float
    l2_reg_coeff: float
    l1_reg_coeff: float
    l1_l2_reg_coeff: float

    # logging
    log: bool = False

    # profiling
    PROFILE_DIR: Optional[str] = None

    # run corruption settings for corruption testing
    corrupt_pi: Optional[str] = None  # "shuffle" | "random" | "zero" | None
    corrupt_z: Optional[str] = None  # "shuffle" | "random" | "zero" | None


@dataclass
class WorkerOutput:
    hit_goal_mean: float
    n_samples: int
    loss_mean: float
    grads_np: list[np.ndarray]
    root_target_entropy: Optional[np.float64] = None
    root_pred_entropy: Optional[np.float64] = None
    root_kl: Optional[np.float64] = None


class DataSource(Enum):
    TRAJECTORY = auto()
    TREE_SAMPLE = auto()
    HEURISTIC_BOOTSTRAP = auto()
    GOAL_PATH = auto()


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


# -----------------------------
# Hook functions you must connect
# -----------------------------

def _build_planner_exts_from_spec(spec, seed_override: Optional[int]):
    """
    MUST return a PlannerExtensions that already points to a concrete instance
    (i.e., has mdpsim_problem/init_state ready).
    """
    # You already do this in multiple places.
    # Make it match your PlannerExtensions ctor signature.
    # spec.pddls is list of pddl files; spec.domain_type exists.
    # If you need Domain object: Domain.from_pddl_name(extract_domain_name_from_file(...))
    # But many codepaths already infer from pddls.
    domain = Domain.from_pddl_name(
        extract_domain_name_from_file(spec.pddls[0])
    )

    pe = PlannerExtensions(
        spec.pddls,
        domain,
        spec.domain_type,
        dg_ssipp_heuristic_name=spec.ssipp_dg_heuristic,
        dg_use_lm_cuts=spec.use_lm_cuts,
        dg_use_numeric_landmarks=spec.use_numeric_landmarks,
        dg_use_contributions=spec.use_contributions,
        dg_use_act_history=spec.use_act_history,
        difficulty=spec.difficulty,
        seed=seed_override if seed_override is not None else spec.random_seed,
        fixed_instance=spec.fixed_instance_pddl,
    )
    return pe


def _build_estimator(planner_exts, spec):
    """
    MUST return an object with:
        estimator.get_cstate_h(cstate) -> float
    """
    # Replace with your actual ENHSP estimator wrapper creation.
    return ENHSPEstimator(planner_exts, enhsp_config=spec.enhsp_config)


def _rebuild_weight_manager_local(prob_meta, weights_np: dict):
    """
    MUST return PropNetworkWeights instance whose variables are created locally,
    then assigned from weights_np.
    """
    # Your classmethod signature currently: from_numpy(cls, prob_meta, weights_np)
    wm = PropNetworkWeights.from_numpy(prob_meta, weights_np)
    return wm


def _build_network_local(weight_manager_local, prob_meta, dropout, debug, policy_only):
    """
    MUST return a Keras-callable network:
        if policy_only: pi_pred
        else: (pi_pred, v_pred)
    """
    net = PropNetwork(
        weight_manager=weight_manager_local,
        problem_meta=prob_meta,
        dropout=dropout,
        debug=debug,
        policy_network_only=policy_only,
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


def heuristic_bootstrapping(bootstrap_k: int, trajectory_info: list, ctx: LocalExploreContext) -> list:
    result = []
    traj_len = len(trajectory_info)
    if traj_len >= bootstrap_k:
        sampled_traj_indices: List[int] = np.random.choice(traj_len, size=bootstrap_k, replace=False)
    else:
        sampled_traj_indices: List[int] = [i for i in range(traj_len)]
    print('[HEURISTIC_BOOTSTRAPPING] Acquiring sampled states heuristics.')
    for ind in sampled_traj_indices:
        sampled_state = trajectory_info[ind]['state']
        sampled_state_v = ctx.get_state_h(sampled_state)
        logits = np.full(ctx.get_act_dim(), -np.inf, dtype=np.float32)
        # for act, child_node in mcts_tree.state_to_node[sampled_state].children.items():
        for act, child_state in trajectory_info[ind]['children']:
            logits[act] = -1 * ctx.estimator_value_conversion_lambda * ctx.get_state_h(child_state)

        # subtract max for stability (this handles -inf too)
        shifted = logits - np.max(logits)

        exp_vals = np.exp(shifted)
        sampled_state_softmax = exp_vals / np.sum(exp_vals)

        result.append(
            {'state': sampled_state, 'children': trajectory_info[ind]['children'], 'pi': sampled_state_softmax,
             'z': sampled_state_v})
    return result


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
def run_worker(inp: WorkerInput) -> WorkerOutput:
    """
    This runs fully inside a spawned process.
    It returns grads as numpy arrays aligned to local weight_manager_local.all_weights order.
    """
    set_random_seeds(inp.seed)
    worker_tag = f"[W{inp.seed}|{os.getpid()}]"
    configure_tf_gpu_memory_growth()
    _dbg_tf_threads(tag=f"{worker_tag} worker_start")
    # --- build instance infra ---
    CanonicalState.network_input_config(use_fluents=inp.spec.use_fluents, use_comparisons=inp.spec.use_comps)
    planner_exts = _build_planner_exts_from_spec(inp.spec, inp.seed)
    estimator = _build_estimator(planner_exts, inp.spec)
    action_policy = build_action_policy(
        base_policy=inp.spec.action_policy,
        worker_tag=worker_tag,
        distance_threshold=np.inf if inp.spec.action_policy_goal_chase_distance_threshold == -1 else inp.spec.action_policy_goal_chase_distance_threshold,
        epsilon=inp.spec.action_policy_epsilon,
        temperature=inp.spec.action_policy_temperature,
        decay_rate=inp.spec.action_policy_decay_rate,
    )
    act_dim = planner_exts.problem_meta.num_acts

    # local weight vars
    wm_local = _rebuild_weight_manager_local(planner_exts.problem_meta, inp.weights_np)
    if inp.log:
        w = wm_local.all_weights[0]
        print(f"{worker_tag} after rebuild:", float(tf.reduce_mean(w)), float(tf.math.reduce_std(w)),
              float(tf.linalg.norm(w)))
    # local network for THIS instance
    net = _build_network_local(wm_local, planner_exts.problem_meta, inp.dropout, inp.debug, inp.policy_only)

    # ctx for TrainingMCTS
    ctx = LocalExploreContext(
        planner_exts=planner_exts,
        estimator=estimator,
        estimator_h_to_v_coeff=inp.spec.estimator_h_to_v_coeff,
    )

    # --- run exploration ---
    # get init state
    cstate = ctx.get_init_state()
    # if inp.log:
    #     cstate.print_state_data()
    select_logging = False
    mcts = TrainingMCTS(
        network=net,
        ctx=ctx,
        iterations=inp.spec.training_mcts_iterations,
        expansion_k=inp.spec.mcts_expansion_k,
        exploration_weight=inp.spec.mcts_exploration_weight,
        sharpen_pi=0.1,
        log_visitations=False,
        select_logging=select_logging,
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

            if inp.policy_only:
                pi_net = net(obs_tf, training=False)
                z_net = None
            else:
                pi_net, z_net = net(obs_tf, training=False)

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
        )

    obs_batch, pi_tgt, z_tgt = collector.as_batches()

    from collections import Counter

    if inp.log:
        log_lines = []
        log_lines.append("\n=== WORKER DIAGNOSTICS ===")

        # ---- source masks & counts ----
        src_list = collector.sources
        all_sources = [
            DataSource.TRAJECTORY,
            DataSource.TREE_SAMPLE,
            DataSource.HEURISTIC_BOOTSTRAP,
            DataSource.GOAL_PATH,
        ]
        masks = {ds: np.asarray([s == ds for s in src_list], dtype=bool) for ds in all_sources}

        counts = Counter(src_list)
        counts_str = ", ".join(f"{ds.name}:{counts[ds]}" for ds in all_sources if counts.get(ds, 0) > 0)

        # Ensure shapes
        pi_tgt_2d = np.atleast_2d(pi_tgt)
        z_tgt_1d = np.asarray(z_tgt, dtype=np.float32)

        applicable = np.count_nonzero(pi_tgt_2d, axis=1)
        log_lines.append(
            f"Samples: {pi_tgt_2d.shape[0]}   [{counts_str}]   Hit:{collector.hit_goal}   AppActsμ:{applicable.mean():.2f}")

        # ---- NEW: batched predictions for ALL samples ----
        obs_tf = tf.convert_to_tensor(obs_batch, dtype=tf.float32)

        if inp.policy_only:
            pi_pred_all = net(obs_tf, training=False)
            v_pred_all = None
        else:
            pi_pred_all, v_pred_all = net(obs_tf, training=False)

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
        z_mb = z_tgt[idx] if not inp.policy_only else None

        with tf.GradientTape() as tape:
            if inp.policy_only:
                pi_pred = net(obs_mb, training=True)
                xent_loss = _policy_xent_loss(pi_pred, tf.convert_to_tensor(pi_mb, dtype=pi_pred.dtype))
                mse_loss = 0.0
            else:
                pi_pred, v_pred = net(obs_mb, training=True)
                xent_loss = _policy_xent_loss(pi_pred, tf.convert_to_tensor(pi_mb, dtype=pi_pred.dtype))
                mse_loss = tf.cast(inp.mse_coeff, xent_loss.dtype) * _value_mse_loss(v_pred, tf.convert_to_tensor(z_mb,
                                                                                                                  dtype=v_pred.dtype))

            reg_loss = _reg_terms(vars_, inp.l2_reg_coeff, inp.l1_reg_coeff, inp.l1_l2_reg_coeff)
            loss = xent_loss + mse_loss + reg_loss
            if inp.log:
                collector.add_losses(
                    xent_loss.numpy(),
                    float(mse_loss.numpy()) if not inp.policy_only else 0.0,
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
        print(
            f"\n{worker_tag} ".join(log_lines),
            flush=True
        )

    return WorkerOutput(
        hit_goal_mean=float(collector.hit_goal),
        n_samples=int(obs_batch.shape[0]),
        loss_mean=float(loss.numpy()),
        grads_np=grads_np,
        root_target_entropy=root_summary.get("root_target_entropy"),
        root_pred_entropy=root_summary.get("root_pred_entropy"),
        root_kl=root_summary.get("root_kl"),
    )


def run_worker_opt_profile(inp: WorkerInput) -> WorkerOutput:
    # Make sure this directory exists (spawn safe)
    prof = None
    t0 = time.time()

    if inp.PROFILE_DIR:
        os.makedirs(inp.PROFILE_DIR, exist_ok=True)
        prof = cProfile.Profile()
        prof.enable()

    try:
        out = run_worker(inp)
        return out
    finally:
        if prof is not None:
            prof.disable()
            pid = os.getpid()
            seed = getattr(inp, "seed", None)
            spec_name = getattr(getattr(inp, "spec", None), "name", None)
            tag = f"pid{pid}_seed{seed}_spec{spec_name}"
            path = os.path.join(inp.PROFILE_DIR, f"worker_{tag}.prof")
            prof.dump_stats(path)

            # Optional: also write a tiny human-readable "top 30" alongside it
            txt_path = os.path.join(inp.PROFILE_DIR, f"worker_{tag}.top.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                ps = pstats.Stats(prof, stream=f).sort_stats("cumtime")
                ps.print_stats(30)

        # Optional: coarse phase timings even without pstats
        print(f"[WORKER TIMING] pid={os.getpid()} total={time.time() - t0:.2f}s", flush=True)
