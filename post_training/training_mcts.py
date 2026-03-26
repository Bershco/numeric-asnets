from time import time

import numpy as np
import tensorflow as tf
from typing import Optional, Any

from asnets.spawn_context import LocalExploreContext
from asnets.state_reprs import CanonicalState
from asnets.utils.pddl_utils import replace_init_state, hlist_to_sexprs
from .monte_carlo_tree_search import MCTS, wrapInMCTSNode, FixedChildMap, MCTSNode


class TrainingMCTS(MCTS):
    """Slim MCTS for training with policy+value network.
    Uses value head instead of rollouts, supports tree re-rooting.
    Used inside rpyc.Service rather than controlling it.
    """

    def __init__(self, network, ctx: LocalExploreContext,
                 iterations=10, expansion_k=5,
                 exploration_weight=1.0, sharpen_pi=1.0, one_hot_distance_gamma=0.999, use_batched_inference=True,
                 log_visitations=False, select_logging=False, estimator_decay=False, ):
        super().__init__(exploration_weight, network=network, select_logging=select_logging)
        self.use_batched_inference = use_batched_inference
        self.ctx = ctx
        self.iterations = iterations
        self.k = expansion_k
        self.sharpen_pi_T = sharpen_pi
        self.one_hot_distance_gamma = one_hot_distance_gamma
        self.log_visitations = log_visitations
        self.estimator_decay = estimator_decay
        self.estimator_curr_coeff = 0.0 if not estimator_decay else 1.0
        if estimator_decay:
            self.estimator_coeff_tup = (1.0,0.6,0.2)

    def get_single_node_policy_value(self, node, training=False):
        act_dist, value = self.network(node.as_network_input, training=training)
        if self.estimator_decay:
            est_v, est_pi = self.ctx.get_state_v_pi_one_hot_est(node.state)
            alpha = self.estimator_curr_coeff
            # Identical logic: value = value * (1-alpha) + est_v * alpha
            value += alpha * (est_v - value)
            act_dist += alpha * (est_pi - act_dist)
        return act_dist, value

    def decay_estimator_coeff(self):
        assert self.estimator_decay, "Can't decay estimator coefficient if estimator decay if off"
        current_idx = self.estimator_coeff_tup.index(self.estimator_curr_coeff)
        if current_idx + 1 < len(self.estimator_coeff_tup):
            self.estimator_curr_coeff = self.estimator_coeff_tup[current_idx + 1]

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

    def _expand(self, node: MCTSNode):
        """Expand children using policy priors (top-k by prob)."""
        if node.children is not None:
            return
        act_dist = node.act_dist
        mask = node.applicable_action_mask
        actions, children_nodes, edge_priors = [], [], []
        children_network_repr = []
        valid = np.where(mask & (act_dist > 0.0))[0]
        if len(valid) > self.k:
            topk = np.argpartition(-act_dist[valid], self.k)[:self.k]
            selected_actions = valid[topk]
        else:
            selected_actions = valid
        results = self.ctx.env_simulate_batch_steps(node.state, selected_actions)
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
            actions.append(action_id)
            edge_priors.append(float(act_dist[action_id]))
            children_nodes.append(wrapped_output_cstate)
            children_network_repr.append(wrapped_output_cstate.as_network_input)
        if self.use_batched_inference and len(children_network_repr) > 0:
            # 1. Network Inference (The GPU part)
            batch_tensor = tf.stack(children_network_repr)
            pred_pi_batch, pred_v_batch = self.network(batch_tensor, training=False)
            # Convert to numpy arrays immediately (squeezing v to be 1D)
            pred_pi_batch = pred_pi_batch.numpy().astype(np.float32, copy=False)
            pred_v_batch = pred_v_batch.numpy().flatten()  # (Batch,)
            # 2. Handle Estimator Decay (The Hybrid part)
            alpha = self.estimator_curr_coeff
            if self.estimator_decay and alpha > 0.0:
                num_children = len(children_nodes)
                est_v_batch = np.empty(num_children, dtype=np.float32)
                est_pi_batch = np.empty_like(pred_pi_batch)
                estimator = self.ctx.estimator
                state_cache = estimator.state_key_cache
                uncached_indices = []
                uncached_oneliners = []
                # first pass: check cache
                for i, child in enumerate(children_nodes):
                    key = child.state.state_key
                    cached = state_cache.get(key)
                    if cached is not None:
                        est_v_batch[i], est_pi_batch[i] = cached
                    else:
                        uncached_indices.append(i)
                        problem_hlist = replace_init_state(
                            estimator._problem_hlist,
                            child.state.to_tup_state()
                        )
                        uncached_oneliners.append(
                            hlist_to_sexprs(problem_hlist)
                        )
                # batch call only for uncached states
                if uncached_oneliners:
                    batch_results = estimator.get_heuristic_and_pi_batched(
                        uncached_oneliners
                    )
                    coeff = self.ctx.estimator_h_to_v_coeff
                    for idx, (h, pi) in zip(uncached_indices, batch_results):
                        v = float(np.exp(-coeff * h))
                        est_v_batch[idx] = v
                        est_pi_batch[idx] = pi
                        key = children_nodes[idx].state.state_key
                        state_cache[key] = (v, pi)
                # blend network + estimator
                pred_v_batch += alpha * (est_v_batch - pred_v_batch)
                pred_pi_batch += alpha * (est_pi_batch - pred_pi_batch)
            # 3. Assign back to nodes
            for i, child in enumerate(children_nodes):
                child.act_dist = pred_pi_batch[i]
                child.pred_value = float(pred_v_batch[i])
        else:
            # Fallback for single/small nodes
            for child in children_nodes:
                act_dist, value = self.get_single_node_policy_value(child)
                child.act_dist = np.array(act_dist).flatten().astype(np.float32)
                child.pred_value = float(value)
        node.children = FixedChildMap(actions, children_nodes, edge_priors)

    def _rollout(self, node, horizon=0):
        """Use value head for evaluation instead of random rollout."""
        return self.get_value_from_mcts_node(node)

    def initialise_tree(self, cstate) -> None:
        """Start a new tree for a fresh episode."""
        self.curr_tree_root = wrapInMCTSNode(state=cstate, cost_until_now=0)
        self.ensure_root_act_dist_value()
        self.state_key_to_node[cstate.state_key] = self.curr_tree_root

    def compute_pi_z_for_node(self, node: MCTSNode, act_dim) -> tuple[np.ndarray, float]:
        assert node.children is not None
        pi = np.zeros(act_dim, dtype=np.float32)
        z_partial = np.zeros(act_dim, dtype=np.float32)
        for action, child in node.children.items():
            visits = child.visit_count
            pi[action] = visits
            z_partial[action] = visits * child.Q_value
        pi_sum = pi.sum()
        if pi_sum > 0:
            pi /= pi_sum
            z = z_partial.sum() / pi_sum
        else:
            # mask = self.get_applicable_action_mask(node)
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

    def run_search(self) -> tuple[np.ndarray, float]:
        """Run N simulations on current root and return π."""
        root = self.curr_tree_root
        for _ in range(self.iterations):
            self.mcts_iteration_value_based(root)

        act_dim = self.ctx.get_act_dim()
        return self.compute_pi_z_for_node(root, act_dim)

    def step_forward(self, action_id):
        """Re-root at chosen child and prune irrelevant branches."""
        parent = self.curr_tree_root
        next_node = parent.children[action_id]
        # self.prune_children_except(parent, action_id)
        self.curr_tree_root = next_node
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

    def sample_k_sufficient_nodes(self, k, min_visitations: int = 5, power_law_weight: float = 2.0) -> list:
        # 1. Filter eligible nodes
        # eligible = [(node, self.N[node]) for node, count in self.N.items() if count > min_visitations]
        eligible = [(node, node.visit_count) for node in self.state_key_to_node.values() if
                    node.visit_count > min_visitations]
        if not eligible:
            return []

        nodes, counts = zip(*eligible)
        counts_array = np.array(counts, dtype=np.float32)
        counts_array **= power_law_weight
        probs = counts_array / counts_array.sum()

        # 2. Sample nodes
        num_to_sample = min(k, len(nodes))
        sampled_nodes = np.random.choice(nodes, size=num_to_sample, replace=False, p=probs)

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

        return results

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
        act_dist, value = self.network(self.curr_tree_root.as_network_input)
        self.curr_tree_root.act_dist = tf.squeeze(act_dist).numpy().astype(np.float32)
        self.curr_tree_root.pred_value = float(value.numpy().squeeze())
