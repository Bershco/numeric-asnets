from collections import Counter
from time import time, perf_counter
import math

import numpy as np
import tensorflow as tf
from typing import Optional, Any

from asnets.spawn_context import LocalExploreContext
from asnets.state_reprs import CanonicalState
from asnets.utils.pddl_utils import replace_init_state, hlist_to_sexprs
from .enhspwrapper import EstimatorMode
from .monte_carlo_tree_search import MCTS, wrapInMCTSNode, FixedChildMap, MCTSNode


class TrainingMCTS(MCTS):
    """Slim MCTS for training with policy+value network.
    Uses value head instead of rollouts, supports tree re-rooting.
    Used inside rpyc.Service rather than controlling it.
    """

    def __init__(self, network, ctx: LocalExploreContext,
                 iterations=10, expansion_k=5,
                 exploration_weight=1.0, sharpen_pi=1.0, one_hot_distance_gamma=0.999,
                 select_logging=False, estimator_coeff=0.0, puct_debug=False,
                 minimization=False, progressive_widening=False,
                 pw_min_width=2, pw_c=0.6, pw_alpha=0.5):
        super().__init__(exploration_weight, network=network, select_logging=select_logging, minimization=minimization)
        if expansion_k < 1:
            raise ValueError("expansion_k must be at least 1")
        if pw_min_width < 1 or pw_min_width > expansion_k:
            raise ValueError(
                "pw_min_width must be between 1 and expansion_k")
        if pw_c <= 0:
            raise ValueError("pw_c must be positive")
        if not 0 < pw_alpha <= 1:
            raise ValueError("pw_alpha must be in (0, 1]")
        self.ctx = ctx
        self.iterations = iterations
        self.k = expansion_k
        self.sharpen_pi_T = sharpen_pi
        self.one_hot_distance_gamma = one_hot_distance_gamma
        self.estimator_coeff = estimator_coeff
        self.estimator_mode = EstimatorMode.V_ONLY
        self.puct_debug = puct_debug
        self.progressive_widening = progressive_widening
        self.pw_min_width = pw_min_width
        self.pw_c = pw_c
        self.pw_alpha = pw_alpha
        self.widening_events = 0
        self.widening_width_hist = Counter()
        self.admitted_policy_rank_hist = Counter()
        self.successor_generation_seconds = 0.0
        self.network_inference_seconds = 0.0

    def get_single_node_policy_value(self, node, training=False):
        if self.network.value_network_enabled:
            act_dist, value = self.network(node.as_network_input, training=training)
            alpha = self.estimator_coeff
            if alpha:
                est_v, est_pi = self.ctx.get_state_v_pi_one_hot_est(node.state)
                # Identical logic: value = value * (1-alpha) + est_v * alpha
                value += alpha * (est_v - value)
                act_dist += alpha * (est_pi - act_dist)
            return act_dist, value
        else:
            return self.network(node.as_network_input, training=training)

    def sharpen_pi(self, pi):
        T = self.sharpen_pi_T

        if T == 1.0:
            return pi

        eps = 1e-8
        pi = np.clip(pi, eps, 1.0)
        pi_pow = np.power(pi, 1.0 / (T + eps))
        pi_pow_sum = pi_pow.sum()
        if pi_pow_sum <= 0:
            # fallback safeguard, also works when T = 0
            one_hot = np.zeros_like(pi)
            one_hot[np.argmax(pi)] = 1.0
            return one_hot
        return pi_pow / pi_pow_sum

    def _progressive_width(self, node: MCTSNode) -> int:
        scheduled = math.floor(
            self.pw_c * (max(1, node.visit_count) ** self.pw_alpha))
        return min(self.k, max(self.pw_min_width, scheduled))

    def _should_expand(self, node: MCTSNode) -> bool:
        if node.children is None or node.children.is_empty():
            return True
        return (
            self.progressive_widening
            and len(node.children) < self._progressive_width(node)
        )

    def _ranked_unexpanded_actions(self, node: MCTSNode) -> np.ndarray:
        valid = np.where(
            node.applicable_action_mask & (node.act_dist > 0.0))[0]
        if node.children is not None and not node.children.is_empty():
            valid = valid[
                ~np.isin(valid, node.children.actions_np, assume_unique=True)]
        if len(valid) == 0:
            return valid
        # lexsort makes action id the deterministic tie-breaker.
        order = np.lexsort((valid, -node.act_dist[valid]))
        return valid[order]

    def _expand(
            self, node: MCTSNode, *,
            force_single_admission: bool = False) -> Optional[MCTSNode]:
        """Expand fixed top-k children or admit progressive children.

        Progressive widening admits ``pw_min_width`` children on first
        expansion and exactly one child on subsequent widening events.  The
        highest-prior newly admitted child is returned for immediate
        evaluation and backpropagation.
        """

        if (node.children is not None
                and not self.progressive_widening
                and not force_single_admission):
            return None

        act_dist = node.act_dist

        actions, children_nodes, edge_priors = [], [], []
        children_network_repr = []

        ranked_actions = self._ranked_unexpanded_actions(node)
        if force_single_admission:
            add_count = 1
        elif self.progressive_widening:
            add_count = self.pw_min_width if node.children is None else 1
            allowed_remaining = max(
                0, self._progressive_width(node)
                   - (0 if node.children is None else len(node.children)))
            add_count = min(add_count, allowed_remaining)
        else:
            add_count = min(self.k, len(ranked_actions))
        selected_actions = ranked_actions[:add_count]

        if len(selected_actions) == 0:
            if node.children is None:
                node.children = FixedChildMap([], [], [])
            return None

        started = perf_counter()
        results = self.ctx.env_simulate_batch_steps(node.state, selected_actions)
        self.successor_generation_seconds += perf_counter() - started

        for (
                action_id,
                cstate,
                step_cost,
                is_goal,
                is_terminal,
                network_ready_repr,
                applicable_action_mask,
        ) in results:

            state_key = cstate.state_key

            node_entry = self.state_key_to_node.get(state_key)

            if node_entry is None:

                wrapped_output_cstate = wrapInMCTSNode(
                    state=cstate,
                    cost_until_now=node.cost_until_now + step_cost
                )
                self.state_key_to_node[state_key] = wrapped_output_cstate

            else:

                wrapped_output_cstate = node_entry
            if self.puct_debug:
                wrapped_output_cstate.add_parent(node, action_id)

            actions.append(action_id)
            edge_priors.append(float(act_dist[action_id]))
            children_nodes.append(wrapped_output_cstate)
            children_network_repr.append(wrapped_output_cstate.as_network_input)

        # Network inference only.  The selected child is estimator-evaluated
        # immediately by mcts_iteration_value_based.
        if len(children_network_repr) > 0:
            started = perf_counter()
            batch_tensor = tf.stack(children_network_repr)
            if self.network.value_head_enabled:
                pred_pi_batch, pred_v_batch = self.network(batch_tensor, training=False)
                pred_pi_batch = pred_pi_batch.numpy().astype(np.float32, copy=False)
                pred_v_batch = pred_v_batch.numpy().flatten()

                for i, child in enumerate(children_nodes):
                    child.act_dist = pred_pi_batch[i]
                    child.pred_value = float(pred_v_batch[i])
            else:
                pred_pi_batch = self.network(batch_tensor, training=False)
                pred_pi_batch = pred_pi_batch.numpy().astype(np.float32, copy=False)
                for i, child in enumerate(children_nodes):
                    child.act_dist = pred_pi_batch[i]
                    child.pred_value = self.worst_value() if not child.goal_state else self.best_value()
            self.network_inference_seconds += perf_counter() - started

        if node.children is None:
            node.children = FixedChildMap(
                actions, children_nodes, edge_priors)
        else:
            for action, child, prior in zip(
                    actions, children_nodes, edge_priors):
                node.children.append(action, child, prior)

        if self.progressive_widening:
            self.widening_events += 1
            self.widening_width_hist[len(node.children)] += 1
            policy_order = {
                int(action): rank
                for rank, action in enumerate(ranked_actions, 1)
            }
            for action in actions:
                self.admitted_policy_rank_hist[policy_order[int(action)]] += 1

        return (
            children_nodes[0]
            if self.progressive_widening or force_single_admission
            else None
        )

    def ensure_safe_root_child(self) -> bool:
        """Admit/evaluate actions until the root has a known safe child.

        This is called only by the opt-in MCTS-SAFE external selector. Normal
        training and evaluation preserve their existing expansion semantics.
        A forced admission may exceed the ordinary width schedule (and fixed
        top-k) only when every currently admitted root child is a known
        terminal non-goal.
        """
        root = self.curr_tree_root

        def has_safe_child() -> bool:
            return bool(
                root.children is not None
                and any(
                    child is not None
                    and (child.goal_state or not child.terminal_state)
                    for child in root.children.values()
                )
            )

        while not has_safe_child():
            if len(self._ranked_unexpanded_actions(root)) == 0:
                break
            child = self._expand(root, force_single_admission=True)
            if child is None:
                break
            # Complete the estimator blend as well as the batched network
            # prediction before the safety selector considers this child.
            self._evaluate_node(child)
        return has_safe_child()

    def _evaluate_node(self, node: MCTSNode) -> float:
        if node.goal_state:
            return self.best_value()
        if node.terminal_state:
            return self.worst_value()
        net_v = node.pred_value
        alpha = self.estimator_coeff
        if alpha > 0.0 and self.estimator_mode in (EstimatorMode.V_ONLY, EstimatorMode.BOTH):
            estimator = self.ctx.estimator
            key = node.state.state_key
            cached = estimator.state_key_cache.get(key)
            if cached is not None:
                est_v = cached[0]
            else:
                est_v = get_est_v(estimator, node.state.to_tup_state(), self.ctx.estimator_h_to_v_coeff, self.minimization)
                estimator.state_key_cache[key] = (est_v, None)
            value = (1.0 - alpha) * net_v + alpha * est_v
            node.pred_value = value  # overwrite prior with refined estimate
        else:
            value = net_v
        return value

    def _rollout(self, node, horizon=0):
        """Use value head for evaluation instead of random rollout."""
        return self.get_value_from_mcts_node(node)

    def initialise_tree(self, cstate) -> None:
        """Start a new tree for a fresh episode."""
        self.curr_tree_root = wrapInMCTSNode(state=cstate, cost_until_now=0)
        self.curr_tree_root.on_trajectory = True
        self.original_tree_root = self.curr_tree_root
        self.ensure_root_act_dist_value()
        self.state_key_to_node[cstate.state_key] = self.curr_tree_root

    def compute_pi_z_for_node(self, node: MCTSNode, act_dim) -> tuple[np.ndarray, float]:
        assert node.children is not None
        pi = np.zeros(act_dim, dtype=np.float32)
        z_partial = np.zeros(act_dim, dtype=np.float32)
        for act, Qsa, Nsa in node.children.get_qsa_nsa_list():
            pi[act] = Nsa
            z_partial[act] = Nsa * Qsa
        pi_sum = pi.sum()
        if pi_sum > 0:
            pi /= pi_sum
            z = z_partial.sum() / pi_sum
        else:
            mask = node.applicable_action_mask
            valid = np.where(mask)[0]
            if len(valid) > 0:
                pi[valid] = 1.0 / len(valid)
            else:
                pi[:] = 1.0 / act_dim
            z = 0.0
        if self.sharpen_pi_T is not None:
            pi = self.sharpen_pi(pi)
        return pi, z

    def compute_pi_z_one_hot(self, node: MCTSNode, act_dim: int) -> tuple[np.ndarray, float]:
        assert node.children is not None
        assert node.best_goal_child is not None
        assert node.known_distance_to_goal < np.inf
        pi = np.zeros(act_dim, dtype=np.float32)
        # find the action leading to best_goal_child
        best_action = None
        for action, child in node.children.items():
            if child is node.best_goal_child:
                best_action = action
                break
        assert best_action is not None, "best_goal_child not found among node.children"
        # one-hot policy target
        pi[best_action] = 1.0
        # value target based on distance to goal
        # subtract 1 because we supervise the action that moves to the child
        distance = node.known_distance_to_goal - 1
        z = float(self.one_hot_distance_gamma ** distance)

        return pi, z

    def run_search(self, remaining_horizon=None) -> tuple[np.ndarray, float]:
        """Run N simulations on current root and return π."""
        if remaining_horizon is not None and remaining_horizon < 0:
            raise ValueError("remaining_horizon cannot be negative")
        root = self.curr_tree_root
        self.search_calls += 1
        for iteration in range(self.iterations):
            self.mcts_iteration_value_based(
                root,
                remaining_horizon=remaining_horizon,
            )

            if (
                    self.puct_debug
                    and root.children is not None
                    and not root.children.is_empty()
                    and iteration % 10 == 0
            ):
                self.log_puct_snapshot(
                    root,
                    label="root",
                    iteration=iteration,
                    selection_depth=0,
                    top_k=10,
                    print_rows=True,
                    return_dict=False,
                    skip_zero_q_range=True,
                )

        root_width = (
            0 if root.children is None else len(root.children))
        self.root_width_hist[root_width] += 1
        act_dim = self.ctx.get_act_dim()
        return self.compute_pi_z_for_node(root, act_dim)

    @staticmethod
    def _hist_summary(hist: Counter) -> str:
        if not hist:
            return "count=0"
        count = sum(hist.values())
        mean = sum(value * freq for value, freq in hist.items()) / count
        ordered = sorted(hist.items())

        def percentile(fraction):
            target = max(1, math.ceil(fraction * count))
            seen = 0
            for value, freq in ordered:
                seen += freq
                if seen >= target:
                    return value
            return ordered[-1][0]

        return (
            f"count={count} min={ordered[0][0]} mean={mean:.2f} "
            f"median={percentile(0.5)} p90={percentile(0.9)} "
            f"p95={percentile(0.95)} max={ordered[-1][0]}"
        )

    def print_search_diagnostics(self) -> None:
        final_width_hist = Counter(
            0 if node.children is None else len(node.children)
            for node in self.state_key_to_node.values()
        )
        print(
            "[MCTS SEARCH SUMMARY] "
            f"progressive_widening={self.progressive_widening} "
            f"iterations_per_search={self.iterations} "
            f"search_calls={self.search_calls} "
            f"nodes={len(self.state_key_to_node)} "
            f"peak_nodes={self.peak_node_count} "
            f"widening_events={self.widening_events}",
            flush=True,
        )
        print(
            "[MCTS WIDTH SUMMARY] "
            f"final_widths=({self._hist_summary(final_width_hist)}) "
            f"root_widths=({self._hist_summary(self.root_width_hist)}) "
            f"widen_event_widths=("
            f"{self._hist_summary(self.widening_width_hist)}) "
            f"admitted_policy_ranks=("
            f"{self._hist_summary(self.admitted_policy_rank_hist)})",
            flush=True,
        )
        for depth_band, hist in self.width_by_depth_band.items():
            print(
                "[MCTS WIDTH BY DEPTH] "
                f"depth={depth_band} {self._hist_summary(hist)}",
                flush=True,
            )
        print(
            "[MCTS DEPTH SUMMARY] "
            f"selection_depths=("
            f"{self._hist_summary(self.selection_depth_hist)}) "
            f"horizon_cutoffs=("
            f"{self._hist_summary(self.horizon_cutoff_depth_hist)})",
            flush=True,
        )
        print(
            "[MCTS ACTION SUMMARY] "
            f"selected_raw_policy_ranks=("
            f"{self._hist_summary(self.selected_policy_rank_hist)}) "
            f"known_goal_decisions={self.known_goal_decisions} "
            f"infeasible_known_goal_decisions="
            f"{self.infeasible_known_goal_decisions}",
            flush=True,
        )
        print(
            "[MCTS TIME SUMMARY] "
            f"selection={self.selection_seconds:.6f}s "
            f"expansion={self.expansion_seconds:.6f}s "
            f"successor_generation={self.successor_generation_seconds:.6f}s "
            f"network_inference={self.network_inference_seconds:.6f}s "
            f"evaluation={self.evaluation_seconds:.6f}s "
            f"backpropagation={self.backpropagation_seconds:.6f}s",
            flush=True,
        )

    def step_forward(self, action_id):
        """Re-root at chosen child and prune irrelevant branches."""
        parent = self.curr_tree_root
        next_node = parent.children[action_id]
        # self.prune_children_except(parent, action_id)
        self.curr_tree_root = next_node
        self.curr_tree_root.on_trajectory = True
        if hasattr(self, "times_moved_forward"):
            self.times_moved_forward += 1
        return self.curr_tree_root.state

    def get_children_mask(self, act_dim=None, node=None,
                          # cstate_id=None
                          cstate=None,
                          ):
        assert act_dim is not None, "Can't get a mask without a size!"
        if node is None:
            if cstate is None:
                node = self.curr_tree_root
            else:
                node = self.state_key_to_node[cstate.state_key]
        assert node.children is not None, "No children, no mask!"
        mask = np.zeros(act_dim, dtype=bool)

        for action in node.children.keys():
            mask[int(action)] = True

        return mask

    def sample_k_sufficient_nodes(self, k, min_visitations: int = 5,
                                  power_law_weight: float = 2.0) -> tuple[list, dict]:
        # 1. Filter eligible nodes
        nodes_examined = len(self.state_key_to_node)
        eligible = [(node, node.visit_count) for node in self.state_key_to_node.values() if
                    node.visit_count > min_visitations]
        if not eligible:
            return [], {
                "nodes_examined": nodes_examined,
                "eligible": 0,
                "emitted": 0,
            }

        nodes, counts = zip(*eligible)

        # 2. Sample nodes
        if k == -1:
            sampled_nodes = nodes
        elif k > 0:
            counts_array = np.array(counts, dtype=np.float32)
            counts_array **= power_law_weight
            probs = counts_array / counts_array.sum()
            num_to_sample = min(k, len(nodes))
            sampled_nodes = np.random.choice(nodes, size=num_to_sample, replace=False, p=probs)
        else:
            raise ValueError(f"k must be positive or -1, got {k}")

        act_dim = self.ctx.get_act_dim()
        results = []

        # 3. Calculate pi and z for each sampled node
        for node in sampled_nodes:
            pi, z = self.compute_pi_z_for_node(node, act_dim)

            results.append({
                'node': node,
                'pi': pi,
                'z': z
            })

        return results, {
            "nodes_examined": nodes_examined,
            "eligible": len(eligible),
            "emitted": len(results),
        }

    def get_children_of(self, cstate: CanonicalState) -> list:
        return [(act, child_node.state) for act, child_node in
                self.state_key_to_node[cstate.state_key].children.items()]

    def count_subtrees_with_goal(self):
        return sum(1 for node in self.state_key_to_node.values() if node.known_distance_to_goal)

    def reconstruct_goal_path(
            self,
            start_node: MCTSNode,
            seen_states: Optional[set[CanonicalState]] = None,
            one_hot_pi_z: bool = False,
    ) -> list[dict[str, Any]]:
        assert start_node.known_distance_to_goal < np.inf
        path = []
        node = start_node
        act_dim = self.ctx.get_act_dim()
        while node.known_distance_to_goal > 1:
            best_child = node.best_goal_child
            assert best_child is not None, (
                "Goal path reconstruction failed: best_goal_child is None\n"
                f"Node distance: {node.known_distance_to_goal}\n"
                f"Children distances: "
                f"{[child.known_distance_to_goal if child is not None else None for child in node.children.values()]}"
            )
            if seen_states is not None and best_child.state in seen_states:
                break
            if one_hot_pi_z:
                best_child_pi, best_child_z = self.compute_pi_z_one_hot(best_child, act_dim)
            else:
                best_child_pi, best_child_z = self.compute_pi_z_for_node(best_child, act_dim)
            path.append({
                'state': best_child.state,
                'children': [g.state for g in best_child.children.values()],
                'pi': best_child_pi,
                'z': best_child_z,
            })
            node = best_child
        return path

    def reconstruct_goal_path_closest(self, trajectory_info, one_hot_pi_z: bool = False) -> list[dict[str, Any]]:
        closest: Optional[MCTSNode] = None
        closest_ind: int = -1
        for i, item in enumerate(trajectory_info):
            node = self.state_key_to_node[item['state'].state_key]
            if node.known_distance_to_goal < np.inf:
                if closest is None or node.known_distance_to_goal < closest.known_distance_to_goal:
                    closest = node
                    closest_ind = i
        if closest is None:
            return []
        print(f"[DEBUG_RECONSTRUCT_GOAL_PATH] The closest node to goal was on step {closest_ind}")
        return self.reconstruct_goal_path(closest, one_hot_pi_z=one_hot_pi_z)

    def reconstruct_goal_paths_from_trajectory(self, trajectory_info, one_hot_pi_z: bool = False) -> list[
        dict[str, Any]]:
        seen_states: set[CanonicalState] = set()
        all_paths = []
        for i, item in enumerate(trajectory_info):
            node = self.state_key_to_node[item['state'].state_key]
            if node.known_distance_to_goal == np.inf:
                continue
            path = self.reconstruct_goal_path(node, seen_states, one_hot_pi_z=one_hot_pi_z)
            for step in path:
                state = step['state']
                if state in seen_states:
                    continue
                seen_states.add(state)
                all_paths.append(step)
        return all_paths

    def ensure_root_act_dist_value(self):
        assert self.curr_tree_root is not None
        self.curr_tree_root.as_network_input = self.curr_tree_root.state.to_network_input()
        if self.network.value_head_enabled:
            act_dist, value = self.network(self.curr_tree_root.as_network_input)
            self.curr_tree_root.pred_value = float(value.numpy().squeeze())
        else:
            act_dist = self.network(self.curr_tree_root.as_network_input)
            self.curr_tree_root.pred_value = self.worst_value() if not self.curr_tree_root.goal_state else self.best_value()
        self.curr_tree_root.act_dist = tf.squeeze(act_dist).numpy().astype(np.float32)

def get_est_v(estimator, state_tup, est_h_to_v_coeff, minimization):
    problem_hlist = replace_init_state(
        estimator._problem_hlist,
        state_tup
    )
    oneliner = hlist_to_sexprs(problem_hlist)
    (h, _) = estimator.get_estimate_batched(
        [oneliner],
        EstimatorMode.V_ONLY
    )[0]
    if minimization:
        est_v = h
    else:
        coeff = est_h_to_v_coeff
        est_v = float(np.exp(-coeff * h))
    return est_v
