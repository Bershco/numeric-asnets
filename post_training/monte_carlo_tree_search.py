"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from typing import Any
from time import time
import numpy as np

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

class SelectProbe:
    EPS = 1e-12
    def __init__(self):
        self.events = 0
        self.exploration_share_sum = 0.0
        self.flip = 0  # argmax(Q+U) != argmax(Q)

    def log(self, q_list, u_list, chosen_idx):
        # Only log when all edges were "visited" (no +inf first-visit forcing)
        if any(v is None for v in u_list):  # mark unvisited edges with None below
            return
        # who wins by score vs by Q alone?
        scores = [q + u for q, u in zip(q_list, u_list)]
        idx_score = int(np.argmax(scores))
        idx_q     = int(np.argmax(q_list))
        # share of exploration at chosen edge
        Qc, Uc = q_list[chosen_idx], u_list[chosen_idx]
        share = Uc / (abs(Qc) + abs(Uc) + self.EPS)

        self.events += 1
        self.exploration_share_sum += share
        if idx_score != idx_q:
            self.flip += 1

    def summary(self):
        if self.events == 0:
            return {"events": 0}
        return {
            "events": self.events,
            "avg_exploration_share": self.exploration_share_sum / self.events,
            "pct_argmax_flipped_by_U": 100.0 * self.flip / self.events,
        }

class MCTS:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, exploration_weight=1):
        self.curr_tree_root = None
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.Nsa = defaultdict(int)
        self.children: dict[Node, Any] = dict()  # actions and children output of each node. structure is (action,result_state)
        self.exploration_weight = exploration_weight
        self.path_until_goal = None
        self.time_debug_mcts_iterations = False
        if self.time_debug_mcts_iterations:
            self.start_times = []
            self.after_selection_times = []
            self.after_expansion_times = []
            self.after_eval_times = []
            self.end_times = []
        self.debug_comparison_exploration_exploitation = True
        self._probe = None
        if self.debug_comparison_exploration_exploitation:
            self._probe = SelectProbe()

    def mcts_iteration_classic(self, node, horizon):
        """Make the tree one layer better. (Train for one iteration.)"""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._rollout(leaf, horizon=horizon)
        self._backpropagate(path, reward)
        if self.path_until_goal is not None:
            self.path_until_goal = self.reconstructSelectionPath(path) + self.path_until_goal

    def mcts_iteration_value_based(self, node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = 1 / (1 + self._evaluate_node(leaf))
        # numbers might be too low or insignificant?? I think it would be okay...
        # theoretically and practically it SHOULD not be lower than 1/10001 which isn't that low.
        self._backpropagate(path,reward)

    def _select(self, node: "Node"):
        """Find an unexplored descendent of `node` (returns path including final leaf or frontier child)."""
        if self.time_debug_mcts_iterations:
            self.start_times.append(time())
        path = []
        while True:
            path.append(node)

            # If node has no generated children, it's unexplored or terminal.
            if node not in self.children or not self.children[node]:
                if self.time_debug_mcts_iterations:
                    self.after_selection_times.append(time())
                return path

            # Prefer an unexplored child if any (child not yet in self.children keys)
            actions_nodes = list(self.children[node].items())  # [(action, child_node), ...]
            unexplored_edges = [(a, c) for (a, c) in actions_nodes if c not in self.children]

            if unexplored_edges:
                a, n = unexplored_edges[np.random.randint(len(unexplored_edges))]
                # count traversed edge
                self.Nsa[(node, a)] += 1
                path.append(n)
                if self.time_debug_mcts_iterations:
                    self.after_selection_times.append(time())
                return path

            # Otherwise pick via PUCT (with no-cycle)
            a_next, n_next = self._puct_select_no_cycle(node, set(path))
            # count traversed edge
            self.Nsa[(node, a_next)] += 1
            node = n_next

    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        raise NotImplemented

    def _rollout(self, mcts_node, horizon=10):
        """Returns the reward for a random simulation (to a certain horizon) of `node`"""
        raise NotImplemented

    def _backpropagate(self, path, reward):
        """Backpropagate the reward through the visited nodes in reverse."""
        for node in reversed(path):
            self.N[node] += 1
            q = self.Q.get(node, 0.0)
            n = self.N[node]
            self.Q[node] = q + (reward - q) / n  # running average
        if self.time_debug_mcts_iterations:
            self.end_times.append(time())


    def _evaluate_node(self, node) -> float:
        """Use the teacher's (or another) heuristic to evaluate a specific node, in order to use value-based mcts"""
        raise NotImplemented

    def _puct_select_no_cycle(self, node, path_set):
        """Sample a child of `node` using PUCT scores as softmax logits while avoiding cycles.
           Returns (action, child_node).  Vectorized over children to reduce Python overhead."""

        # 0) Sanity: children should already be generated for this node
        children_map = self.children[node]  # dict: action -> child_node
        assert all(child in self.children for child in children_map.values())
        actions_nodes = list(children_map.items())  # [(a, child), ...]
        n_children = len(actions_nodes)
        assert n_children > 0, "PUCT select called on a node with no children"

        # 1) Priors from policy (vector over action indices)
        priors = self.get_act_dist_from_mcts_node(node)
        priors = priors.numpy() if hasattr(priors, "numpy") else priors

        # Build parallel arrays once
        import numpy as _np, math as _math
        actions = _np.fromiter((a for a, _ in actions_nodes), dtype=_np.int32, count=n_children)
        child_list = [c for _, c in actions_nodes]

        # 2) Masks (vectorized)
        #   - cycle mask: child already on current path => invalidate
        cycle = _np.fromiter((c in path_set for c in child_list), dtype=bool, count=n_children)

        # 3) Prior lookup (vectorized, with bounds check)
        prior = _np.zeros(n_children, dtype=_np.float32)
        if _np.ndim(priors) == 1 and priors.size > 0:
            valid = (actions >= 0) & (actions < priors.size)
            if valid.any():
                prior[valid] = _np.asarray(priors, dtype=_np.float32)[actions[valid]]
        else:
            # extremely defensive: if priors is scalar/empty, leave zeros
            pass

        # 4) Edge visits Nsa(s,a) and child Q(s') in one sweep
        edge_visits = _np.fromiter((self.Nsa[(node, int(a))] for a in actions),
                                   dtype=_np.int32, count=n_children)
        Q_child = _np.fromiter((self.Q.get(c, 0.0) for c in child_list),
                               dtype=_np.float32, count=n_children)

        # 5) Forced first visits: if any edge is unvisited and not a cycle, pick among them
        unvisited = (edge_visits == 0) & (~cycle)
        if _np.any(unvisited):
            cand = _np.flatnonzero(unvisited)
            # bias by prior if it has any mass, else uniform among unvisited
            w = prior[cand].astype(_np.float64)
            s = float(w.sum())
            if _np.isfinite(s) and s > 0.0:
                w /= s
                idx = int(_np.random.choice(cand, p=w))
            else:
                idx = int(_np.random.choice(cand))
            a, child = actions_nodes[idx]
            # (as in your code: skip probe here; exploration is trivially dominating)
            return a, child

        # 6) Compute U and score = Q + U (vectorized)
        N_parent = float(self.N.get(node, 0))
        sqrtN = _math.sqrt(max(1.0, N_parent))
        U = self.exploration_weight * prior * (sqrtN / (1.0 + edge_visits.astype(_np.float32)))

        # Invalidate cycles
        U[cycle] = 0.0
        score = Q_child + U
        score[cycle] = -_np.inf

        # 7) Sample by softmax over (Q+U) with numerical stability
        x = score.astype(_np.float64)
        x -= _np.max(x)
        w = _np.exp(x)
        w[~_np.isfinite(w)] = 0.0
        s = float(w.sum())
        if (not _np.isfinite(s)) or s <= 0.0:
            idx = int(_np.argmax(score))  # fallback if all weights underflow
        else:
            p = w / s
            idx = int(_np.random.choice(n_children, p=p))

        a, child = actions_nodes[idx]

        # 8) Probe (keep your semantics: log q_list, u_list, chosen_idx)
        self._probe.log(Q_child.tolist(), U.tolist(), idx)

        return a, child

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

    def get_act_dist_from_mcts_node(self, node):
        raise NotImplemented
