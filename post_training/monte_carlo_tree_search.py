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
import logging

class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

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


class MCTS:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children: dict[Node, dict[int, Node]] = dict()  # actions and children output of each node. structure is (action,result_state)
        self.exploration_weight = exploration_weight
        self.path_until_goal = None

    def mcts_iteration(self, node, horizon):
        """Make the tree one layer better. (Train for one iteration.)"""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._rollout(leaf, horizon=horizon)
        self._backpropagate(path, reward)
        if self.path_until_goal is not None:
            self.path_until_goal = self.reconstructSelectionPath(path) + self.path_until_goal

    def _select(self, node: Node):
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
            # unexplored = {action_state_tuple[1] for action_state_tuple in self.children[node]} - self.children.keys()
            unexplored = set(self.children[node].values()) - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            # node = self._uct_select(node)  # descend a layer deeper
            node = self._puct_select_no_cycle(node, set(path))

    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        raise NotImplemented


    def _rollout(self, mcts_node, horizon=10):
        """Returns the reward for a random simulation (to a certain horizon) of `node`"""
        raise NotImplemented

    def _backpropagate(self, path, reward):
        """Send the reward back up to the ancestors of the leaf"""
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _puct_select_no_cycle(self, node, path_set):
        """Sample a child of `node` using PUCT scores as softmax logits and make sure to not get into cycles"""

        # All children of node should already be generated
        assert all(child_node in self.children for child_node in self.children[node].values())

        # Get the prior probabilities from the policy network
        # priors = policy_network(node.to_network_input())  # returns an eagertensor
        priors = self.get_act_dist_from_mcts_node(node) # makes sure the tensor is (num_of_actions,) and not (1,num_of_actions)
        priors = priors.numpy() if hasattr(priors, "numpy") else priors  # if running eagerly

        total_visits = self.N[node]
        scores = []
        actions_nodes = list(self.children[node].items())  # List of (action, child_node)

        for action, child in actions_nodes:
            if child in path_set:  # as to not create a cycle
                scores.append(float("-inf"))
                continue
            # Use prior if available, otherwise assume 0 (or small epsilon if you prefer)
            prior = float(priors[action]) if 0 <= action < len(priors) else 0.0
            if self.N[child] == 0:
                score = float("inf")  # Encourage at least one visit
            else:
                q_value = self.Q[child] / self.N[child]
                exploration = self.exploration_weight * prior * math.sqrt(total_visits) / (1 + self.N[child])
                score = q_value + exploration

            scores.append(score)

        # If any node is unvisited (inf score), choose uniformly among them
        if any(np.isposinf(score) for score in scores):
            unexplored = [child for score, (_, child) in zip(scores, actions_nodes) if np.isinf(score)]
            np.random.seed(self.seed)
            return np.random.choice(unexplored)

        # Convert scores to probabilities via softmax
        scores = np.array(scores, dtype=np.float64)
        exp_probs = np.exp(scores - np.max(scores))  # subtract max for numerical stability
        probs = exp_probs / np.sum(exp_probs)

        # Sample an index from the softmax
        np.random.seed(self.seed)
        idx = np.random.choice(len(actions_nodes), p=probs)
        # logging.getLogger(__name__).debug(f"PUCT probs: {probs}, selected idx: {idx}, action: {actions_nodes[idx][0]}")
        print(f"PUCT probs: {probs}, selected idx: {idx}, action: {actions_nodes[idx][0]}")
        return actions_nodes[idx][1]

    def reconstructSelectionPath(self, path):
        output_path = [(None, self.curr_tree_root)]
        for mcts_node in path:
            if mcts_node == self.curr_tree_root:
                continue
            assert mcts_node in self.children[output_path[-1][1]].values()
            for action, next_node in self.children[output_path[-1][1]].items():
                if mcts_node == next_node:
                    output_path.append((action, mcts_node))
        return output_path[1:]
