# asnets/spawn_train_worker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import tensorflow as tf
import logging
from asnets.models import PropNetworkWeights, PropNetwork
from asnets.spawn_context import LocalExploreContext
from asnets.state_reprs import CanonicalState
from asnets.supervised import PlannerExtensions
from asnets.utils.generator_utils import extract_domain_name_from_file, Domain
from asnets.utils.py_utils import set_random_seeds
from post_training.enhspwrapper import ENHSPEstimator
from post_training.training_mcts import TrainingMCTS



LOGGER = logging.getLogger(__name__)

# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class WorkerInput:
    spec: Any                      # SpawnExploreSpec
    weights_np: dict               # PropNetworkWeights.export_numpy() result
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

    # run corruption settings for corruption testing
    corrupt_pi: Optional[str] = None   # "shuffle" | "random" | "zero" | None
    corrupt_z: Optional[str] = None    # "shuffle" | "random" | "zero" | None


@dataclass
class WorkerOutput:
    hit_goal_mean: float
    n_samples: int
    loss_mean: float
    grads_np: list[np.ndarray]
    root_target_entropy: Optional[np.float64] = None
    root_pred_entropy: Optional[np.float64] = None
    root_kl: Optional[np.float64] = None


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
        reg += tf.add_n([tf.reduce_sum(tf.abs(v)) + tf.reduce_sum(tf.square(v)) for v in vars_]) * tf.cast(l1_l2, reg.dtype)
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


# -----------------------------
# Worker main
# -----------------------------
def run_worker(inp: WorkerInput) -> WorkerOutput:
    """
    This runs fully inside a spawned process.
    It returns grads as numpy arrays aligned to local weight_manager_local.all_weights order.
    """
    set_random_seeds(inp.seed)
    # --- build instance infra ---
    CanonicalState.network_input_config(use_fluents=inp.spec.use_fluents, use_comparisons=inp.spec.use_comps)
    planner_exts = _build_planner_exts_from_spec(inp.spec, inp.seed)
    estimator = _build_estimator(planner_exts, inp.spec)

    prob_meta = planner_exts.problem_meta
    act_dim = prob_meta.num_acts

    # local weight vars
    wm_local = _rebuild_weight_manager_local(prob_meta, inp.weights_np)
    if inp.log:
        w = wm_local.all_weights[0]
        print("WORKER after rebuild:", float(tf.reduce_mean(w)), float(tf.math.reduce_std(w)), float(tf.linalg.norm(w)))

    # local network for THIS instance
    net = _build_network_local(wm_local, prob_meta, inp.dropout, inp.debug, inp.policy_only)

    # ctx for TrainingMCTS
    ctx = LocalExploreContext(
        planner_exts=planner_exts,
        estimator=estimator,
        prob_meta=prob_meta,
        act_dim=act_dim,
    )

    # --- run exploration ---
    # get init state
    cstate = ctx.get_init_state()

    mcts = TrainingMCTS(
        network=net,
        ctx=ctx,
        iterations=inp.spec.training_mcts_iterations,
        expansion_k=inp.spec.mcts_expansion_k,
        exploration_weight=inp.spec.mcts_exploration_weight,
        sharpen_pi=0.1,
        log_visitations=False,
    )
    init_cstate_id, _ = mcts.initialise_tree(cstate)

    max_len = inp.spec.max_len

    obs_list = []
    cstate_id_list =[]
    cstate_id_list.append(init_cstate_id)
    pi_pred_list = []
    z_pred_list = []
    pi_tgt_list = []
    z_tgt_list = []
    hit_goal = 0.0

    # logging accumulators
    root_target_entropies = []
    root_pred_entropies = []
    root_kls = []

    # Episode
    for t in range(max_len):
        # terminal?
        if cstate.is_terminal:
            hit_goal = 1.0 if cstate.is_goal else 0.0
            break

        pi, z = mcts.run_search()  # pi: (act_dim,), z: scalar
        # store targets + obs (use your canonical state -> network input)
        obs = cstate.to_network_input()  # must be 1D (obs_dim,)
        obs_list.append(obs)
        pi_tgt_list.append(pi.astype(np.float32))
        z_tgt_list.append(float(z))
        # --- ROOT POLICY DIAGNOSTICS ---
        if inp.log:
            obs_tf = tf.expand_dims(tf.convert_to_tensor(obs, tf.float32), 0)
            if inp.policy_only:
                pi_net = net(obs_tf, training=False)
            else:
                pi_net, z_net = net(obs_tf, training=False)
                z_pred_list.append(z_net)

            pi_mcts = tf.stop_gradient(tf.convert_to_tensor(pi, tf.float32))
            pi_net = tf.stop_gradient(pi_net[0])
            pi_pred_list.append(pi_net.numpy())
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

            root_target_entropies.append(float(entropy_t.numpy()))
            root_pred_entropies.append(float(entropy_p.numpy()))
            root_kls.append(float(kl_t.numpy()))
        # sample action from pi masked by available children
        mask = mcts.get_children_mask(act_dim=act_dim)
        masked_pi = pi * mask
        s = masked_pi.sum()
        if s <= 0:
            # fallback uniform over valid
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            masked_pi = np.zeros_like(pi)
            masked_pi[valid] = 1.0 / len(valid)
        else:
            masked_pi = masked_pi / s

        # a = np.random.choice(np.arange(act_dim), p=masked_pi)

        a = np.argmax(masked_pi)

        sid, sh = mcts.step_forward(a)
        cstate_id_list.append(sid)
        cstate = ctx.get_state_from_identifiers(sid, sh)
        if inp.log and cstate.is_terminal:
            LOGGER.info(f"z_target:{ z}, v_pred: {net(cstate.to_network_input()[None, :],training=False)[1]}")

    # If ended due to max_len without terminal
    #TODO: make sure I don't miss successes if goal reached after max_len moves
    if not cstate.is_terminal:
        hit_goal = 1.0 if getattr(cstate, "is_goal", False) else 0.0

    if len(obs_list) == 0:
        # no data => zero grads
        zeros = [np.zeros(v.shape, dtype=np.float32) for v in wm_local.all_weights]
        return WorkerOutput(hit_goal_mean=hit_goal, n_samples=0, loss_mean=0.0, grads_np=zeros)

    obs_batch = np.asarray(obs_list)
    pi_pred=np.asarray(pi_pred_list)
    v_pred=np.asarray(z_pred_list,dtype=np.float32)
    pi_tgt = np.asarray(pi_tgt_list)
    z_tgt = np.asarray(z_tgt_list, dtype=np.float32)

    if inp.log:
        # 1. Calculate values
        max_tgt_vals = np.max(pi_tgt, axis=1)
        argmax_tgt = np.argmax(pi_tgt, axis=1)
        max_pred_vals = np.max(pi_pred, axis=1)
        argmax_pred = np.argmax(pi_pred, axis=1)
        app_actions = np.count_nonzero(pi_tgt, axis=1)

        print(f"Mean of max(pi_tgt): {np.mean(max_tgt_vals)} while num of applicable actions: {np.mean(app_actions)}")
        print(f"Mean of max(pi_pred): {np.mean(max_pred_vals)} while num of applicable actions: {np.mean(app_actions)}")
        print(f"argmax_tgt: {argmax_tgt}\nargmax_pred: {argmax_pred}, Ratio of pred==target: {np.mean(argmax_tgt == argmax_pred)}")
        print(f"z_target: mean={np.mean(z_tgt)}, std={np.std(z_tgt)}, min={np.min(z_tgt)}, max={np.max(z_tgt)}")
        print(f"v_pred: mean={np.mean(v_pred)}, std={np.std(v_pred)}, min={np.min(v_pred)}, max={np.max(v_pred)}")

        # 2. Print with specific formatting (3 decimal places, suppressed scientific notation)
        with np.printoptions(precision=3, suppress=True, linewidth=300):
            print(f"Max Target Probabilities:   {max_tgt_vals}")
            print(f"Max Predicted Probabilities:   {max_pred_vals}")
            print(f"Applicable Actions:  {app_actions}")

        invalid = 0
        for target_pi, cstate_id in zip(pi_tgt_list, cstate_id_list):
            target_action = np.argmax(target_pi)
            mask = mcts.get_children_mask(act_dim=act_dim, cstate_id=cstate_id)
            if mask[target_action] == 0:
                invalid += 1

        print(f"Invalid target check: invalid: {invalid}, total_steps: {len(pi_tgt)}, invalid_ratio: {invalid/len(pi_tgt) if len(pi_tgt)>0 else None}")
    if inp.corrupt_pi is not None or inp.corrupt_z is not None:
        pi_tgt, z_tgt = _corrupt_targets(inp, pi_tgt, z_tgt)

    # --- compute grads locally ---
    vars_ = wm_local.all_weights

    with tf.GradientTape() as tape:
        mse_loss = tf.constant(0.0)
        if inp.policy_only:
            pi_pred = net(obs_batch, training=True)
            xent_loss = _policy_xent_loss(pi_pred, tf.convert_to_tensor(pi_tgt, dtype=pi_pred.dtype))
        else:
            pi_pred, v_pred = net(obs_batch, training=True)
            xent_loss = _policy_xent_loss(pi_pred, tf.convert_to_tensor(pi_tgt, dtype=pi_pred.dtype))
            mse_loss = tf.cast(inp.mse_coeff, xent_loss.dtype) * _value_mse_loss(v_pred, tf.convert_to_tensor(z_tgt, dtype=v_pred.dtype))

        # reg on the SAME variables
        reg_loss = _reg_terms(vars_, inp.l2_reg_coeff, inp.l1_reg_coeff, inp.l1_l2_reg_coeff)
        loss = xent_loss + mse_loss + reg_loss

    grads = tape.gradient(loss, vars_)
    if inp.log:
        LOGGER.info(f"Xent loss: {xent_loss}, MSE loss: {mse_loss}, Reg loss: {reg_loss}")
    grads_np = []
    grad_stats = []
    for g, v in zip(grads, vars_):
        assert g is not None, f"Gradient is None for variable {v.name}"
        # assert tf.reduce_any(tf.math.not_equal(g, 0.0))
        assert not tf.reduce_any(tf.math.is_nan(g)), f"NaN gradient in {v.name}"
        assert not tf.reduce_any(tf.math.is_inf(g)), f"Inf gradient in {v.name}"
        # print(f"Absolute gradient mean: {tf.reduce_mean(tf.abs(g))}")
        if inp.log:
            grad_stats.append({
                "mean": tf.reduce_mean(tf.abs(g)),
                "max": tf.reduce_max(tf.abs(g)),
                "nnz": tf.reduce_mean(tf.cast(g != 0.0, tf.float32))
            })
        if g is None:
            # replace None grads with zeros (keeps apply_gradients happy)
            grads_np.append(np.zeros(v.shape, dtype=np.float32))
        else:
            grads_np.append(g.numpy().astype(np.float32))
    if inp.log:
        mean_grad = np.mean([s["mean"] for s in grad_stats])
        mean_nnz = np.mean([s["nnz"] for s in grad_stats])
        LOGGER.info(f"Worker.{inp.seed}: mean_grad:{mean_grad}, mean_nnz:{mean_nnz}")
    return WorkerOutput(
        hit_goal_mean=float(hit_goal),
        n_samples=int(obs_batch.shape[0]),
        loss_mean=float(loss.numpy()),
        grads_np=grads_np,
        root_target_entropy=np.mean(root_target_entropies) if root_target_entropies else None,
        root_pred_entropy=np.mean(root_pred_entropies) if root_pred_entropies else None,
        root_kl=np.mean(root_kls) if root_kls else None,
    )
