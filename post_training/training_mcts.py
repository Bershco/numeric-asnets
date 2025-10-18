import numpy as np
import tensorflow as tf

from asnets.multiprob import to_local
from .monte_carlo_tree_search import MCTS, wrapInMCTSNode, FixedChildMap


class TrainingMCTS(MCTS):
    """Slim MCTS for training with policy+value network.
    Uses value head instead of rollouts, supports tree re-rooting.
    """

    def __init__(self, network, problem_service,
                 iterations=10, expansion_k=5,
                 exploration_weight=1.0):
        super().__init__(exploration_weight, network=network)
        self.problem_service = problem_service
        self.iterations = iterations
        self.k = expansion_k
        self.state_to_node = {}
        self.act_dist_per_node = {}

    def _expand(self, node):
        """Expand children using policy priors (top-k by prob)."""
        # if node in self.children:
        if node.children is not None:
            return

        # get priors from network
        act_dist = self.get_act_dist_from_mcts_node(node)
        act_dist = to_local(act_dist)
        act_dist = tf.squeeze(act_dist).numpy()
        self.act_dist_per_node[node] = act_dist

        # mask invalid actions
        mask = [node.is_applicable_action(i) for i in range(len(act_dist))]
        sorted_indices = sorted(range(len(act_dist)), key=lambda i: act_dist[i], reverse=True)

        keys, values = [], []
        selected = 0
        for i in sorted_indices:
            if selected >= self.k:
                break
            if not mask[i] or act_dist[i] == 0.0:
                continue
            next_state, step_cost = node.simulate_step(i, self.problem_service)
            child = wrapInMCTSNode(
                next_state,
                cost_until_now=node.cost_until_now + step_cost,
                previous_action=i
            )
            keys.append(i)
            values.append(child)
            self.state_to_node[child.state] = child
            selected += 1

        # self.children[node] = dict(zip(keys, values)) #TODO: change to FixedChildMap
        node.children = FixedChildMap(keys,values)

    def _evaluate_node(self, node) -> float:
        return self._rollout(node)

    def _rollout(self, node, horizon=0):
        """Use value head for evaluation instead of random rollout."""
        value = self.get_value_from_mcts_node(node)
        value = to_local(value)
        return float(tf.squeeze(value).numpy())

    def initialise_tree(self, cstate):
        """Start a new tree for a fresh episode."""
        self.curr_tree_root = wrapInMCTSNode(cstate, cost_until_now=0, previous_action=None)
        self.state_to_node[self.curr_tree_root.state] = self.curr_tree_root
        # self.children.clear()
        self.N.clear()
        self.Q.clear()
        self.act_dist_per_node.clear()

    def run_search(self):
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
        else:
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
        self.prune_children_except(parent, action_id)
        self.curr_tree_root = next_node
        return self.curr_tree_root.state

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
