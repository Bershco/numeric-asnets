from time import time

import numpy as np
import tensorflow as tf

from asnets.spawn_context import LocalExploreContext
from .monte_carlo_tree_search import MCTS, wrapInMCTSNode, FixedChildMap, MCTSNode


class TrainingMCTS(MCTS):
    """Slim MCTS for training with policy+value network.
    Uses value head instead of rollouts, supports tree re-rooting.
    Used inside rpyc.Service rather than controlling it.
    """

    def __init__(self, network, ctx: LocalExploreContext, #problem_service,
                 iterations=10, expansion_k=5,
                 exploration_weight=1.0, sharpen_pi=1.0, log_visitations=False):
        super().__init__(exploration_weight, network=network)
        # self.problem_service = problem_service
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

        keys, values = [], []
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
            keys.append(action_id)
            values.append(wrapped_output_cstate)

        node.children = FixedChildMap(keys,values)

    def _rollout(self, node, horizon=0):
        """Use value head for evaluation instead of random rollout."""
        return self.get_value_from_mcts_node(node)

    def _backpropagate(self, path, reward):
        for node in reversed(path):
            n = self.N[node] + 1
            self.N[node] = n

            q_old = node.Q_value
            node.Q_value = q_old + (reward - q_old) / n
        if self.debug_time_mcts_iterations:
            self.end_times.append(time())

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

    def run_search(self) -> tuple[np.ndarray, float]:
        """Run N simulations on current root and return π."""
        root = self.curr_tree_root
        for _ in range(self.iterations):
            self.mcts_iteration_value_based(root)

        # act_dim = self.problem_service.exposed_get_act_dim()
        act_dim = self.ctx.get_act_dim()
        pi = np.zeros(act_dim, dtype=np.float32)
        z_partial = np.zeros(act_dim, dtype=np.float32)

        children = root.children
        for action, child in children.items():
            # The following comments say "pretty much" because they're approximate, it's not N(s,a)
            # but rather N(s_child) - could be child of other states as well
            pi[action] = self.N.get(child, 0) # N(s,a) (pretty much)
            z_partial[action] = pi[action] * child.Q_value # N(s,a) * Q(s,a) (again, pretty much)

        if self.log_visitations:
            max_visits_action = np.argmax(pi)
            max_visits = int(np.max(pi))
            sum_visits = int(pi.sum())
            print(f"[INSIDE_WORKER: run_search] Max visits: action no.{max_visits_action} with {max_visits} visitations. Sum visits: {sum_visits}. Ratio: {max_visits/sum_visits if sum_visits>0 else None}, Random ratio: {1/np.count_nonzero(pi) if np.count_nonzero(pi)>0 else None}")

        # Normalize
        pi_sum = pi.sum()
        if pi_sum > 0:
            pi /= pi_sum
            z_partial_norm = z_partial/pi_sum
            z = z_partial_norm.sum()
        else: # Fallback to uniform distribution if stuff broke
            mask = self.get_applicable_action_mask(root)
            valid = np.where(mask)[0]
            if len(valid) > 0:
                pi[valid] = 1.0 / len(valid)
            else:
                pi[:] = 1.0 / act_dim
            z = 0
        if self.sharpen_pi_T is not None:
            pi = self.sharpen_pi(pi)
        return pi, z


    def step_forward(self, action_id):
        """Re-root at chosen child and prune irrelevant branches."""
        parent = self.curr_tree_root
        # next_node = self.children[parent][action_id]
        next_node = parent.children[action_id]
        # self.prune_children_except(parent, action_id)
        self.curr_tree_root = next_node
        return self.curr_tree_root.state
        # return self.curr_tree_root.state_id, hash(self.curr_tree_root)

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
            pi = np.zeros(act_dim, dtype=np.float32)
            z_partial = np.zeros(act_dim, dtype=np.float32)

            children = node.children  # Assuming node has a .children dict
            for action, child in children.items():
                # Get visit count for the edge (node -> child)
                # Using self.N.get(child, 0) as per your run_search logic
                visits = self.N.get(child, 0)
                pi[action] = visits
                z_partial[action] = visits * child.Q_value

            pi_sum = pi.sum()
            if pi_sum > 0:
                pi /= pi_sum
                z = (z_partial / pi_sum).sum()
            else:
                # Fallback logic
                mask = self.get_applicable_action_mask(node)
                valid = np.where(mask)[0]
                if len(valid) > 0:
                    pi[valid] = 1.0 / len(valid)
                else:
                    pi[:] = 1.0 / act_dim
                z = 0.0

            if self.sharpen_pi_T is not None:
                pi = self.sharpen_pi(pi)

            results.append({
                'node': node,
                'pi': pi,
                'z': z
            })

        return results