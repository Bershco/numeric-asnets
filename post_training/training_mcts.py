import numpy as np
import tensorflow as tf

from asnets.multiprob import to_local
from .monte_carlo_tree_search import MCTS, wrapInMCTSNode, FixedChildMap, MCTSNode


class TrainingMCTS(MCTS):
    """Slim MCTS for training with policy+value network.
    Uses value head instead of rollouts, supports tree re-rooting.
    Used inside rpyc.Service rather than controlling it.
    """

    def __init__(self, network, problem_service,
                 iterations=10, expansion_k=5,
                 exploration_weight=1.0):
        super().__init__(exploration_weight, network=network)
        self.problem_service = problem_service
        self.iterations = iterations
        self.k = expansion_k

    def _expand(self, node: MCTSNode):
        """Expand children using policy priors (top-k by prob)."""
        if node.children is not None:
            return

        # get priors from network
        act_dist = self.get_act_dist_from_mcts_node(node)
        act_dist = to_local(act_dist)
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

        results = self.problem_service.exposed_env_simulate_batch_steps(node.state_id, hash(node), selected_actions)

        for (action_id, cstate_after_action_i_id, cstate_after_action_i_hash,
             step_cost, is_goal, is_terminal,
             network_ready_repr, applicable_action_mask
             ) in results:
            wrapped_output_cstate = wrapInMCTSNode(
                cstate_id=cstate_after_action_i_id,
                cost_until_now=node.cost_until_now + step_cost,
                previous_action=action_id,
                is_goal=is_goal,
                is_terminal=is_terminal,
                as_network_input=network_ready_repr, applicable_action_mask=applicable_action_mask,
                hashed_state=cstate_after_action_i_hash,
                parent=node,
            )
            self.state_id_to_node[cstate_after_action_i_id] = wrapped_output_cstate
            keys.append(action_id)
            values.append(wrapped_output_cstate)

        node.children = FixedChildMap(keys,values)

    def _rollout(self, node, horizon=0):
        """Use value head for evaluation instead of random rollout."""
        return self.get_value_from_mcts_node(node)

    def initialise_tree(self, cstate):
        """Start a new tree for a fresh episode."""
        cstate_id, cstate_hash = self.problem_service.internal_get_state_identifiers(cstate)
        self.curr_tree_root = wrapInMCTSNode(cstate_id=cstate_id,
                                             previous_action=None,
                                             cost_until_now=0,
                                             is_goal=cstate.is_goal,
                                             is_terminal=cstate.is_terminal,
                                             # These next method calls are possible because TrainingMCTS is inside the service
                                             as_network_input=self.problem_service.internal_to_network_input(cstate),
                                             applicable_action_mask=self.problem_service.internal_get_applicable_action_mask(cstate),
                                             hashed_state = cstate_hash)
        self.state_id_to_node[self.curr_tree_root.state_id] = self.curr_tree_root
        # self.children.clear()
        self.N.clear()
        # self.Q.clear()
        # self.act_dist_per_node.clear()

    def get_act_dist_from_mcts_node(self, node: MCTSNode):
        if node.act_dist is None:
            if node.as_network_input is None:
                node.as_network_input = self.problem_service.exposed_to_network_input(*node.get_identifiers())
            if self.policy_only:
                node.act_dist = to_local(self.network(node.as_network_input))
            else:
                node.act_dist, value_tensor = to_local(self.network(node.as_network_input))
                # value_tensor = to_local(value_tensor)
                node.value = float(value_tensor.numpy().squeeze())
        return tf.squeeze(node.act_dist)

    def run_search(self) -> np.ndarray:
        """Run N simulations on current root and return π."""
        root = self.curr_tree_root
        for _ in range(self.iterations):
            self.mcts_iteration_value_based(root)

        act_dim = self.problem_service.exposed_get_act_dim()
        pi = np.zeros(act_dim, dtype=np.float32)
        # children = self.children.get(root, {})
        children = root.children
        for action, child in children.items():
            pi[action] = self.N.get(child, 0)
        # Normalize visit counts
        if pi.sum() > 0:
            pi /= pi.sum()
        else: # Fallback to uniform distribution if stuff broke
            mask = self.get_applicable_action_mask(root)
            valid = np.where(mask)[0]
            if len(valid) > 0:
                pi[valid] = 1.0 / len(valid)
            else:
                pi[:] = 1.0 / act_dim
        return pi

    def step_forward(self, action_id):
        """Re-root at chosen child and prune irrelevant branches."""
        parent = self.curr_tree_root
        # next_node = self.children[parent][action_id]
        next_node = parent.children[action_id]
        # self.prune_children_except(parent, action_id)
        self.curr_tree_root = next_node
        return self.curr_tree_root.state_id, hash(self.curr_tree_root)

    def get_children_mask(self, act_dim=None, node=None):
        assert act_dim is not None, "Can't get a mask without a size!"
        if node is None:
            node=self.curr_tree_root
        # assert node in self.children, "No children, no mask!"
        assert node.children is not None, "No children, no mask!"
        mask = np.zeros(act_dim, dtype=bool)

        # for action in self.children.get(node):
        for action in node.children.keys():
            mask[int(action)] = True

        return mask

    def get_children_states(self, state_id: int):
        return self.state_id_to_node[state_id].children
