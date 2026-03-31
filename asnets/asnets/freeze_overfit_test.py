import logging
from time import time

import numpy as np
import tensorflow as tf
import tqdm.auto as tqdm

from asnets.spawn_context import LocalExploreContext
from asnets.spawn_train_worker import WorkerCollectorWithLogging, DataSource, heuristic_bootstrapping, \
    _build_network_local, _build_planner_exts_from_spec, _build_estimator, _policy_xent_loss, _value_mse_loss, \
    _reg_terms
from asnets.state_reprs import CanonicalState
from asnets.supervised import SupervisedTrainer, tf_and_log
from post_training.training_mcts import TrainingMCTS


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class FrozenSupervisedTrainer(SupervisedTrainer):

    def __init__(self,
                 # problems,
                 weight_manager,
                 summary_writer,
                 explorer,
                 strategy,
                 start_time,
                 scratch_dir,
                 snapshot_dir,
                 *,
                 batch_size=64,
                 lr=0.001,
                 lr_steps=[],
                 opt_batches_per_epoch=300,
                 l1_reg_coeff,
                 l2_reg_coeff,
                 l1_l2_reg_coeff,
                 mse_coeff,
                 save_training_set=None,
                 use_saved_training_set=None,
                 hide_progress=False,
                 use_fluents=False,
                 use_comps=False,
                 time_out=40,
                 early_stop=20,
                 save_every=20,
                 dk="dk",
                 policy_only=False,
                 planner_exts=None,
                 ):
        super().__init__(weight_manager=weight_manager,
                         summary_writer=summary_writer,
                         explorer=explorer,
                         strategy=strategy,
                         start_time=start_time,
                         scratch_dir=scratch_dir,
                         snapshot_dir=snapshot_dir,
                         batch_size=batch_size,
                         lr=lr,
                         lr_steps=lr_steps,
                         opt_batches_per_epoch=opt_batches_per_epoch,
                         l1_reg_coeff=l1_reg_coeff,
                         l2_reg_coeff=l2_reg_coeff,
                         l1_l2_reg_coeff=l1_l2_reg_coeff,
                         mse_coeff=mse_coeff,
                         save_training_set=save_training_set,
                         use_saved_training_set=use_saved_training_set,
                         hide_progress=hide_progress,
                         use_fluents=use_fluents,
                         use_comps=use_comps,
                         time_out=time_out,
                         early_stop=early_stop,
                         save_every=save_every,
                         dk=dk,
                         policy_only=policy_only,
                         )
        self._frozen_dataset = None
        self.planner_exts=planner_exts

    def train(self, max_epochs):
        best_rate = None
        iter_num = 0

        tr = tqdm.trange(max_epochs, desc='epoch', leave=True)
        epoch = tf.Variable(0, dtype=tf.int64)
        self.summary_writer.set_as_default(step=epoch)

        for epoch_num in tr:
            epoch.assign(epoch_num)

            # --------------------------------------------------
            # 1. EXPLORE (spawn workers, compute grads there)
            # --------------------------------------------------
            if self._frozen_dataset is None:
                self._collect_frozen_dataset()
            mean_loss, acc, mean_kl, n_states = self._train_on_frozen_dataset_one_epoch()

            tf_and_log("train-loss", mean_loss)
            tr.set_postfix(net_loss=mean_loss, acc=acc, kl=mean_kl, states=n_states, lr=self.optimiser.lr)

            elapsed_time = time() - self.start_time
            if self.timeout and elapsed_time > self.timeout * 0.95:
                LOGGER.info('[TIMING_TERMINATION] Timeout reached')
                break

        return best_rate, elapsed_time, iter_num

    def _collect_frozen_dataset(self):
        self._build_frozen_bundle_if_needed()
        B = self._frozen_bundle
        spec = self.explorer.specs[0]

        # Run EXACTLY the same episode logic as worker, but only collect samples
        cstate = B.ctx.get_init_state()

        mcts = TrainingMCTS(
            network=B.net,
            ctx=B.ctx,
            iterations=spec.mcts_iterations,
            expansion_k=spec.mcts_expansion_k,
            exploration_weight=spec.mcts_exploration_weight,
            sharpen_pi=0.1,
            log_visitations=False,
        )
        mcts.initialise_tree(cstate)

        collector = WorkerCollectorWithLogging()  # or plain WorkerCollector
        a = None

        for t in range(spec.max_len):
            if cstate.is_terminal:
                collector.hit_goal = 1.0 if cstate.is_goal else 0.0
                break

            pi, z = mcts.run_search()
            collector.add_sample(
                cstate=cstate,
                children=mcts.get_children_of(cstate),
                action=a,
                pi=pi,
                z=z,
                source=DataSource.TRAJECTORY,
            )

            if spec.sample_k_additional_states:
                sampled_data = mcts.sample_k_sufficient_nodes(k=spec.sample_k_additional_states)
                for item in sampled_data:
                    collector.add_sample(
                        cstate=item["node"].state,
                        children=None,
                        action=None,
                        pi=item["pi"],
                        z=item["z"],
                        source=DataSource.TREE_SAMPLE,
                    )

            # deterministic step like your worker (argmax)
            mask = mcts.get_children_mask(act_dim=B.act_dim)
            masked_pi = pi * mask
            s = masked_pi.sum()
            if s <= 0:
                valid = np.where(mask)[0]
                if len(valid) == 0:
                    break
                masked_pi = np.zeros_like(pi)
                masked_pi[valid] = 1.0 / len(valid)
            else:
                masked_pi = masked_pi / s

            a = int(np.argmax(masked_pi))
            cstate = mcts.step_forward(a)

        if not cstate.is_terminal:
            collector.hit_goal = 1.0 if cstate.is_goal else 0.0

        # optional heuristic bootstrapping
        if spec.heuristic_bootstrapping:
            trajectory_info = collector.get_trajectory_info_as_list()
            sampled_data = heuristic_bootstrapping(bootstrap_k=5, trajectory_info=trajectory_info, ctx=B.ctx,
                                                   mcts_tree=mcts)
            for item in sampled_data:
                collector.add_sample(
                    cstate=item["state"],
                    children=item["children"],
                    action=None,
                    pi=item["pi"],
                    z=item["z"],
                    source=DataSource.HEURISTIC_BOOTSTRAP,
                )

        obs_batch, pi_tgt, z_tgt = collector.as_batches()
        assert obs_batch.shape[0] > 0, "Frozen dataset ended up empty"

        # obs_batch = obs_batch[0:1]
        # pi_tgt = pi_tgt[0:1]
        # z_tgt = z_tgt[0:1]

        self._frozen_dataset = {
            "obs": obs_batch.astype(np.float32, copy=False),
            "pi_tgt": pi_tgt.astype(np.float32, copy=False),
            "z_tgt": (z_tgt.astype(np.float32, copy=False) if z_tgt is not None else None),
            "prob_meta": B.planner_exts.problem_meta,
        }

        print(
            f"[FREEZE] dataset cached: obs={obs_batch.shape} pi={pi_tgt.shape} "
            f"z={(None if z_tgt is None else z_tgt.shape)}",
            flush=True
        )

    def _ensure_freeze_net(self):
        """
        Build the per-instance network ONCE for freeze mode, using the cached problem_meta etc.
        Assumes self._frozen_dataset already exists.
        """
        if getattr(self, "_freeze_net", None) is not None:
            return

        fd = self._frozen_dataset
        # You MUST have stored prob_meta (and any needed config) when collecting frozen dataset.
        prob_meta = fd["prob_meta"]
        dropout = getattr(self, "dropout", 0.0)
        debug = getattr(self, "debug", False)
        policy_only = getattr(self, "policy_only", False)

        # build net directly on *trainer* weight_manager (NOT a rebuilt local copy)
        self._freeze_net = _build_network_local(
            weight_manager_local=self.weight_manager,
            prob_meta=prob_meta,
            dropout=dropout,
            debug=debug,
            policy_only=policy_only,
        )

        # Build once (Keras lazy build). Use one sample.
        obs0 = tf.convert_to_tensor(fd["obs"][:1], tf.float32)
        _ = self._freeze_net(obs0, training=False)

    def _train_on_frozen_dataset_one_epoch(self):
        """
        Overfit test: train ONLY on cached frozen dataset (obs, pi_tgt, z_tgt)
        and report strong diagnostics: accuracy, KL, max(pi_pred), entropies.
        """
        assert hasattr(self, "_frozen_dataset") and self._frozen_dataset is not None, \
            "Frozen dataset not collected yet."

        self._ensure_freeze_net()
        net = self._freeze_net

        fd = self._frozen_dataset
        obs = fd["obs"]
        pi_tgt = fd["pi_tgt"]
        z_tgt = fd.get("z_tgt", None)

        spec = self.explorer.specs[0]

        # ---- knobs for overfit ----
        batch_size = getattr(spec, "freeze_batch_size", 32)
        train_steps = getattr(spec, "freeze_train_steps", 10)  # inner steps per epoch
        shuffle = getattr(spec, "freeze_shuffle", True)

        # IMPORTANT for overfit: disable regularization in freeze mode
        disable_reg = True

        # Convert once
        obs_tf_all = tf.convert_to_tensor(obs, tf.float32)
        pi_tf_all = tf.convert_to_tensor(pi_tgt, tf.float32)
        z_tf_all = tf.convert_to_tensor(z_tgt, tf.float32) if (z_tgt is not None and not self.policy_only) else None

        n = obs.shape[0]
        idx_all = np.arange(n)

        vars_ = list(self.weight_manager.all_weights)  # the REAL shared weights
        total_loss = 0.0
        total_batches = 0

        for step in range(train_steps):
            if shuffle:
                np.random.shuffle(idx_all)

            # minibatch loop (covers ALL data every step)
            for start in range(0, n, batch_size):
                mb_idx = idx_all[start:start + batch_size]

                obs_mb = tf.gather(obs_tf_all, mb_idx, axis=0)
                pi_mb = tf.gather(pi_tf_all, mb_idx, axis=0)
                z_mb = tf.gather(z_tf_all, mb_idx, axis=0) if z_tf_all is not None else None

                with tf.GradientTape() as tape:
                    # Do NOT rely on Keras tracking; explicitly watch shared weights
                    tape.watch(vars_)

                    if self.policy_only:
                        pi_pred = net(obs_mb, training=True)
                        xent_loss = _policy_xent_loss(pi_pred, pi_mb)
                        mse_loss = tf.constant(0.0, dtype=xent_loss.dtype)
                    else:
                        pi_pred, v_pred = net(obs_mb, training=True)
                        xent_loss = _policy_xent_loss(pi_pred, pi_mb)
                        mse_loss = tf.cast(self.mse_coeff, xent_loss.dtype) * _value_mse_loss(v_pred, z_mb)

                    if disable_reg:
                        reg_loss = tf.constant(0.0, dtype=xent_loss.dtype)
                    else:
                        reg_loss = _reg_terms(vars_, self.explorer.l2_reg_coeff, self.explorer.l1_reg_coeff, self.explorer.l1_l2_reg_coeff)

                    loss = xent_loss + mse_loss + reg_loss

                grads = tape.gradient(loss, vars_)
                self.optimiser.apply_gradients(zip(grads, vars_))

                total_loss += float(loss.numpy())
                total_batches += 1

        mean_loss = total_loss / max(1, total_batches)

        # -------------------------------
        # Diagnostics on FULL dataset
        # -------------------------------
        with tf.device("/CPU:0"):
            if self.policy_only:
                pi_pred_all = net(obs_tf_all, training=False)
                v_pred_all = None
            else:
                pi_pred_all, v_pred_all = net(obs_tf_all, training=False)

        pi_pred_all = tf.stop_gradient(pi_pred_all).numpy()
        pi_tgt_2d = np.atleast_2d(pi_tgt)
        pi_pred_2d = np.atleast_2d(pi_pred_all)

        # Argmax accuracy
        acc = float(np.mean(np.argmax(pi_pred_2d, axis=1) == np.argmax(pi_tgt_2d, axis=1)))

        # KL(tgt || pred) per sample, then mean
        eps = 1e-8
        kl = pi_tgt_2d * (np.log(np.clip(pi_tgt_2d, eps, 1.0)) - np.log(np.clip(pi_pred_2d, eps, 1.0)))
        kl_mean = float(np.mean(np.sum(kl, axis=1)))

        # Entropies + max-prob
        H_tgt = float(np.mean(-np.sum(pi_tgt_2d * np.log(np.clip(pi_tgt_2d, eps, 1.0)), axis=1)))
        H_pred = float(np.mean(-np.sum(pi_pred_2d * np.log(np.clip(pi_pred_2d, eps, 1.0)), axis=1)))
        max_pi_tgt = float(np.mean(np.max(pi_tgt_2d, axis=1)))
        max_pi_pred = float(np.mean(np.max(pi_pred_2d, axis=1)))

        # Log them (same style you use elsewhere)
        tf_and_log("freeze/loss", mean_loss)
        tf_and_log("freeze/acc", acc)
        tf_and_log("freeze/kl", kl_mean)
        tf_and_log("freeze/H_tgt", H_tgt)
        tf_and_log("freeze/H_pred", H_pred)
        tf_and_log("freeze/max_pi_tgt", max_pi_tgt)
        tf_and_log("freeze/max_pi_pred", max_pi_pred)
        tf_and_log("freeze/states", int(n))
        print("weight norm:",
              float(tf.linalg.global_norm(self.weight_manager.all_weights)))
        print("grad norm:", float(tf.linalg.global_norm(grads)))

        return mean_loss, acc, kl_mean, int(n)

    def _build_frozen_bundle_if_needed(self):
        if getattr(self, "_frozen_bundle", None) is not None:
            return

        spec = self.explorer.specs[0]
        assert spec.fixed_instance_pddl, "freeze_train only supported with fixed_instance for now"

        # IMPORTANT: build exactly like the worker, but in MAIN PROCESS
        CanonicalState.network_input_config(use_fluents=spec.use_fluents, use_comparisons=spec.use_comps)

        planner_exts = self.planner_exts
        # planner_exts = _build_planner_exts_from_spec(spec, seed)
        estimator = _build_estimator(planner_exts, spec)

        ctx = LocalExploreContext(
            planner_exts=planner_exts,
            estimator=estimator,
            estimator_value_conversion_lambda=spec.estimator_value_conversion_lambda,
        )

        act_dim = planner_exts.problem_meta.num_acts

        # THE KEY: build a per-instance network that uses the TRAINER'S weight_manager vars
        # In worker you do _rebuild_weight_manager_local(...) because different process.
        # Here we are in same process, so just pass self.weight_manager directly.
        net = _build_network_local(
            weight_manager_local=self.weight_manager,  # <<<<< THIS is the change
            prob_meta=planner_exts.problem_meta,
            dropout=0,  # whatever you use
            debug=False,
            policy_only=self.policy_only,
        )

        self._frozen_bundle = FrozenInstanceBundle(planner_exts, estimator, ctx, net, act_dim)

class FrozenInstanceBundle:
    def __init__(self, planner_exts, estimator, ctx, net, act_dim):
        self.planner_exts = planner_exts
        self.estimator = estimator
        self.ctx = ctx
        self.net = net
        self.act_dim = act_dim
