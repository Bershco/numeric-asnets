from time import time

import numpy as np
import tensorflow as tf
from typing import Optional, Any

from asnets.spawn_context import LocalExploreContext
from asnets.state_reprs import CanonicalState
from .monte_carlo_tree_search import MCTS, wrapInMCTSNode, FixedChildMap, MCTSNode


class TrainingMCTS(MCTS):
    """Slim MCTS for training with policy+value network.
    Uses value head instead of rollouts, supports tree re-rooting.
    Used inside rpyc.Service rather than controlling it.
    """

    def __init__(self, network, ctx: LocalExploreContext, #problem_service,
                 iterations=10, expansion_k=5,
                 exploration_weight=1.0, sharpen_pi=1.0, use_batched_inference=True, log_visitations=False):
        super().__init__(exploration_weight, network=network)
        # self.problem_service = problem_service
        self.use_batched_inference = use_batched_inference
        self.ctx=ctx
        self.iterations = iterations
        self.k = expansion_k
        self.sharpen_pi_T=sharpen_pi
        self.log_visitations=log_visitations

    def sharpen_pi(self, pi):
        T = self.sharpen_pi_T

        if T == 1.0:
            return pi

        eps = 1e-8
        pi = np.clip(pi, eps, 1.0)
        pi_pow = np.power(pi, 1.0 / (T+eps))
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

        # get priors from network
        act_dist = self.get_act_dist_from_mcts_node(node)
        act_dist = tf.squeeze(act_dist).numpy()

        # mask invalid actions
        mask = self.get_applicable_action_mask(node)
        sorted_indices = sorted(range(len(act_dist)), key=lambda i: act_dist[i], reverse=True)

        actions, children_nodes = [], []
        children_network_repr = []
        selected_actions = []
        for i in sorted_indices:
            if len(selected_actions) >= self.k:
                break
            if not mask[i] or act_dist[i] == 0.0:
                continue
            selected_actions.append(i)

        # results = self.problem_service.exposed_env_simulate_batch_steps(node.state_id, hash(node), selected_actions)
        # results = self.ctx.env_simulate_batch_steps(node.state_id, hash(node), selected_actions)
        results = self.ctx.env_simulate_batch_steps(node.state, selected_actions)
        for (action_id,
             # cstate_after_action_i_id, cstate_after_action_i_hash,
             cstate,
             step_cost, is_goal, is_terminal,
             network_ready_repr, applicable_action_mask
             ) in results:
            # wrapped_output_cstate = wrapInMCTSNode(
            #     cstate_id=cstate_after_action_i_id,
            #     cost_until_now=node.cost_until_now + step_cost,
            #     previous_action=action_id,
            #     is_goal=is_goal,
            #     is_terminal=is_terminal,
            #     as_network_input=network_ready_repr, applicable_action_mask=applicable_action_mask,
            #     hashed_state=cstate_after_action_i_hash,
            #     parent=node,
            # )
            if cstate not in self.state_to_node.keys():
                wrapped_output_cstate = wrapInMCTSNode(state=cstate, cost_until_now=node.cost_until_now + step_cost,
                                                       previous_action=action_id, parent=node)
                self.state_to_node[cstate] = wrapped_output_cstate
            else:
                wrapped_output_cstate = self.state_to_node[cstate]
            actions.append(action_id)
            children_nodes.append(wrapped_output_cstate)
            children_network_repr.append(wrapped_output_cstate.as_network_input)

        if self.use_batched_inference and len(children_network_repr) > 0:
            batch_tensor = tf.stack(children_network_repr)  # or tf.convert_to_tensor
            pred_pi_batch, pred_v_batch = self.network(batch_tensor, training=False)
            pred_pi_batch = pred_pi_batch.numpy()
            pred_v_batch = pred_v_batch.numpy()
            for i, child in enumerate(children_nodes):
                child.act_dist = pred_pi_batch[i]
                child.pred_value = pred_v_batch[i][0]
        node.children = FixedChildMap(actions,children_nodes)

    def _rollout(self, node, horizon=0):
        """Use value head for evaluation instead of random rollout."""
        return self.get_value_from_mcts_node(node)

    def initialise_tree(self, cstate) -> None:
        """Start a new tree for a fresh episode."""
        # cstate_id, cstate_hash = self.problem_service.internal_get_state_identifiers(cstate)
        # cstate_id, cstate_hash = self.ctx.get_state_identifiers(cstate)
        # self.curr_tree_root = wrapInMCTSNode(cstate_id=cstate_id,
        #                                      previous_action=None,
        #                                      cost_until_now=0,
        #                                      is_goal=cstate.is_goal,
        #                                      is_terminal=cstate.is_terminal,
        #                                      # These next method calls are possible because TrainingMCTS is inside the service
        #                                      # as_network_input=self.problem_service.internal_to_network_input(cstate),
        #                                      # applicable_action_mask=self.problem_service.internal_get_applicable_action_mask(cstate),
        #                                      as_network_input=self.ctx.to_network_input(cstate_id, cstate_hash),
        #                                      applicable_action_mask=self.ctx.get_applicable_action_mask(cstate_id, cstate_hash),
        #                                      hashed_state = cstate_hash)
        self.curr_tree_root = wrapInMCTSNode(state=cstate, previous_action=None, cost_until_now=0)
        self.state_to_node[self.curr_tree_root.state] = self.curr_tree_root
        self.N.clear()

    def get_act_dist_from_mcts_node(self, node: MCTSNode):
        if node.act_dist is None:
            if node.as_network_input is None:
                # node.as_network_input = self.problem_service.exposed_to_network_input(*node.get_identifiers())
                # node.as_network_input = self.ctx.to_network_input(*node.get_identifiers())
                node.as_network_input = self.ctx.to_network_input(node.state)
            if self.policy_only:
                node.act_dist = self.network(node.as_network_input)
            else:
                node.act_dist, value_tensor = self.network(node.as_network_input)
                node.pred_value = float(value_tensor.numpy().squeeze())
        return tf.squeeze(node.act_dist)

    def compute_pi_z_for_node(self, node: MCTSNode, act_dim) -> tuple[np.ndarray, float]:
        assert node.children is not None
        pi = np.zeros(act_dim, dtype=np.float32)
        z_partial = np.zeros(act_dim, dtype=np.float32)
        for action, child in node.children.items():
            visits = self.N.get(child, 0)
            pi[action] = visits
            z_partial[action] = visits * child.Q_value
        pi_sum = pi.sum()
        if pi_sum > 0:
            pi /= pi_sum
            z = z_partial.sum() / pi_sum
        else:
            mask = self.get_applicable_action_mask(node)
            valid = np.where(mask)[0]
            if len(valid) > 0:
                pi[valid] = 1.0 / len(valid)
            else:
                pi[:] = 1.0 / act_dim
            z = 0.0
        if self.sharpen_pi_T is not None:
            pi = self.sharpen_pi(pi)
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
        self.current_trajectory.append(parent)
        next_node = parent.children[action_id]
        # self.prune_children_except(parent, action_id)
        self.curr_tree_root = next_node
        return self.curr_tree_root.state

    def get_children_mask(self, act_dim=None, node=None,
                          # cstate_id=None
                          cstate=None,
                          ):
        assert act_dim is not None, "Can't get a mask without a size!"
        if node is None:
            if cstate is None:
                node=self.curr_tree_root
            else:
                node=self.state_to_node[cstate]
        # assert node in self.children, "No children, no mask!"
        assert node.children is not None, "No children, no mask!"
        mask = np.zeros(act_dim, dtype=bool)

        # for action in self.children.get(node):
        for action in node.children.keys():
            mask[int(action)] = True

        return mask

    def sample_k_sufficient_nodes(self, k, min_visitations=5) -> list:
        # 1. Filter eligible nodes
        eligible = [(node, self.N[node]) for node, count in self.N.items() if count > min_visitations]

        if not eligible:
            return []

        nodes, counts = zip(*eligible)
        counts_array = np.array(counts, dtype=np.float32)
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
        return [(act, child_node.state) for act, child_node in self.state_to_node[cstate].children.items()]

    def count_subtrees_with_goal(self):
        return sum(1 for node in self.state_to_node.values() if node.known_distance_to_goal)

    def reconstruct_goal_path_if_applicable(self, trajectory_info) -> list[dict[str,Any]]:
        closest = None
        for item in trajectory_info:
            node = self.state_to_node[item['state']]
            if node.known_distance_to_goal < np.inf:
                if closest is None or node.known_distance_to_goal < closest.known_distance_to_goal:
                    closest = node
        if closest is None:
            return []
        return self.reconstruct_goal_path(closest)

    def reconstruct_goal_path(self, start_node: MCTSNode) -> list[dict[str, Any]]:
        assert start_node.known_distance_to_goal < np.inf
        path = []
        node = start_node
        act_dim = self.ctx.get_act_dim()
        while node.known_distance_to_goal > 0:
            best_child = node.best_goal_child

            assert best_child is not None, (
                "Goal path reconstruction failed: best_goal_child is None\n"
                f"Node distance: {node.known_distance_to_goal}\n"
                f"Children distances: "
                f"{[child.known_distance_to_goal if child is not None else None for child in node.children.values()]}"
            )

            best_child_pi, best_child_z = self.compute_pi_z_for_node(best_child, act_dim)

            path.append({
                'state': best_child.state,
                'children': [grandchild.state for grandchild in best_child.children.values()],
                'pi': best_child_pi,
                'z': best_child_z,
            })
            node = best_child
        return path