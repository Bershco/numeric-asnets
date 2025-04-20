"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import numpy as np
import tensorflow as tf


class MCTS:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # actions and children output of each node. structure is (action,result_state)
        self.exploration_weight = exploration_weight

    def do_rollout(self, node, policy_network):
        """Make the tree one layer better. (Train for one iteration.)"""
        path = self._select(node, policy_network)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node, policy_network):
        """Find an unexplored descendent of `node`"""
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                # Roee: should still work as intended even though we're messing with action_nodes tuples and not just
                # nodes,because "not self.children[node]" means that there are no applicable actions_node tuples,
                # hence node is terminal.
                return path
            #unexplored = self.children[node][1] - self.children.keys()
            unexplored = {action_state_tuple[1] for action_state_tuple in self.children[node]} - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            # node = self._uct_select(node)  # descend a layer deeper
            node = self._puct_select(node, policy_network) # same as the above line, but use the policy network

    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node, horizon=10):
        """Returns the reward for a random simulation (to a certain horizon) of `node`"""
        invert_reward = True
        for _ in range(horizon):
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_child_by_policy()
            invert_reward = not invert_reward
        return 0

    def _backpropagate(self, path, reward):
        """Send the reward back up to the ancestors of the leaf"""
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        """Select a child of node, balancing exploration & exploitation"""

        # All children of node should already be expanded:
        #assert all(n in self.children for n in self.children[node][1]) - this won't work, because self.children[node] is a set and is unsubscriptable
        assert all(action_cstate_tuple[1] in self.children for action_cstate_tuple in self.children[node]) #Roee: should work this way, same as changed in puct

        log_N_vertex = math.log(self.N[node])

        def uct(pair):
            """Upper confidence bound for trees"""
            _, n = pair  # Extract node from the (action, node) tuple

            if self.N[n] == 0:
                return float("inf")  # Encourage exploration of unseen moves

            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

    def _puct_select(self, node, policy_network):
        """Sample a child of `node` using PUCT scores as softmax logits."""

        # All children of node should already be expanded
        #assert all(n in self.children for n in self.children[node][1]) - this won't work, because self.children[node] is a set and is unsubscriptable
        assert all(action_cstate_tuple[1] in self.children for action_cstate_tuple in self.children[node]) #Roee: should work this way, same as changed in uct

        # Get the prior probabilities from the policy network
        priors = policy_network(node.to_network_input())  # returns an eagertensor
        priors = tf.squeeze(priors) # makes sure the tensor is (num_of_actions,) and not (1,num_of_actions)
        total_visits = self.N[node]

        scores = []
        actions_nodes = []

        for action, child in self.children[node]:
            # Use prior if available, otherwise assume 0 (or small epsilon if you prefer)
            if not isinstance(action, int):
                raise ValueError(f"Action must be an int, got {type(action)}")
            prior = float(priors[action]) if isinstance(action, int) and action < priors.shape[0] else 0.0
            if self.N[child] == 0:
                score = float("inf")  # Encourage at least one visit
            else:
                q_value = self.Q[child] / self.N[child]
                exploration = self.exploration_weight * prior * math.sqrt(total_visits) / (1 + self.N[child])
                score = q_value + exploration

            scores.append(score)
            actions_nodes.append((action, child))

        # If any node is unvisited (inf score), choose uniformly among them
        if any(s == float("inf") for s in scores):
            unexplored = [an for s, an in zip(scores, actions_nodes) if s == float("inf")]
            return np.random.choice(unexplored)

        # Convert scores to probabilities via softmax
        scores = np.array(scores)
        probs = np.exp(scores - np.max(scores))  # subtract max for numerical stability
        probs = probs / probs.sum()

        # Sample an index from the softmax
        idx = np.random.choice(len(actions_nodes), p=probs)
        return actions_nodes[idx][1]


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        """All possible successors of this board state"""
        return set()

    @abstractmethod
    def find_child_by_policy(self):
        """Random successor of this board state (for more efficient simulation)"""
        return None

    @abstractmethod
    def is_terminal(self):
        """Returns True if the node has no children"""
        return True

    @abstractmethod
    def reward(self):
        """Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"""
        return 0

    @abstractmethod
    def __hash__(self):
        """Nodes must be hashable"""
        return 123456789

    @abstractmethod
    def __eq__(self, node2):
        """Nodes must be comparable"""
        return True