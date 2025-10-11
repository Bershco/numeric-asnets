"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
import logging
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
    """Lightweight, backward-compatible probe.
    - Keep existing summary from log(Q,U,idx)
    - Add small counters/averages you can print at the end
    """
    EPS = 1e-12

    def __init__(self):
        # legacy summary fields (driven by log(Q,U,idx))
        self.events = 0
        self.exploration_share_sum = 0.0
        self.flip = 0  # argmax(Q+U) != argmax(Q)

        # new, small counters (no heavy work)
        self.sel_forced_unvisited = 0
        self.sel_softmax = 0
        self.sel_cycle_present = 0
        self.sel_prior_entropy_sum = 0.0
        self.sel_chosen_p_sum = 0.0
        self.sel_chosen_p_count = 0
        self.sel_edge_visits_chosen_sum = 0
        self.sel_edge_visits_chosen_count = 0

        self.expand_calls = 0
        self.expand_children_sum = 0
        self.expand_actdim_sum = 0

        self.eval_calls = 0
        self.eval_cold = 0
        self.eval_ms_sum = 0.0

        self.backprop_calls = 0
        self.backprop_pathlen_sum = 0

        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

    # -------- old callsite (keep working) --------
    def log(self, q_list, u_list, chosen_idx):
        import numpy as np
        scores = [q + u for q, u in zip(q_list, u_list)]
        idx_score = int(np.argmax(scores))
        idx_q     = int(np.argmax(q_list))
        Qc, Uc = q_list[chosen_idx], u_list[chosen_idx]
        share = Uc / (abs(Qc) + abs(Uc) + self.EPS)
        self.events += 1
        self.exploration_share_sum += share
        if idx_score != idx_q:
            self.flip += 1

    # -------- small helpers you’ll call where noted --------
    def log_select_unvisited(self):
        self.sel_forced_unvisited += 1

    def log_select_softmax(self, *, prior_entropy=None, chosen_p=None, edge_visits_chosen=None, cycle_present=False):
        self.sel_softmax += 1
        if cycle_present:
            self.sel_cycle_present += 1
        if prior_entropy is not None:
            self.sel_prior_entropy_sum += float(prior_entropy)
        if chosen_p is not None:
            self.sel_chosen_p_sum += float(chosen_p)
            self.sel_chosen_p_count += 1
        if edge_visits_chosen is not None:
            self.sel_edge_visits_chosen_sum += int(edge_visits_chosen)
            self.sel_edge_visits_chosen_count += 1

    def log_expand(self, *, act_dim, children_created):
        self.expand_calls += 1
        self.expand_children_sum += int(children_created)
        if act_dim is not None:
            self.expand_actdim_sum += int(act_dim)

    def log_eval(self, *, ms, cold):
        self.eval_calls += 1
        self.eval_ms_sum += float(ms)
        if cold:
            self.eval_cold += 1

    def log_backprop(self, *, path_len):
        self.backprop_calls += 1
        self.backprop_pathlen_sum += int(path_len)

    # -------- printing --------
    def print_exploration_exploitation_comparison(self):
        """Keeps your original summary behavior + prints the new counters concisely."""
        self.logger.debug("=== Exploration/Exploitation (selection) ===")
        if self.events == 0:
            self.logger.debug("no softmax selections recorded (events=0)")
        else:
            avg_share = self.exploration_share_sum / max(1, self.events)
            flip_pct = 100.0 * self.flip / max(1, self.events)
            self.logger.debug(f"events={self.events}  avg_U_share_on_chosen={avg_share:.3f}  pct_argmax_flipped_by_U={flip_pct:.1f}%")

        self.logger.debug("=== Selection counters ===")
        self.logger.debug(f"forced_first_visit={self.sel_forced_unvisited}  softmax={self.sel_softmax}  cycles_present={self.sel_cycle_present}")
        if self.sel_softmax > 0:
            avg_ent = self.sel_prior_entropy_sum / max(1, self.sel_softmax)
            self.logger.debug(f"avg_prior_entropy(softmax)={avg_ent:.3f}")
        if self.sel_chosen_p_count > 0:
            avg_p = self.sel_chosen_p_sum / self.sel_chosen_p_count
            self.logger.debug(f"avg_chosen_softmax_p={avg_p:.3f}")
        if self.sel_edge_visits_chosen_count > 0:
            avg_vis = self.sel_edge_visits_chosen_sum / self.sel_edge_visits_chosen_count
            self.logger.debug(f"avg_edge_visits_on_chosen={avg_vis:.2f}")

        self.logger.debug("=== Expand/Eval/Backprop ===")
        if self.expand_calls > 0:
            self.logger.debug(f"expand_calls={self.expand_calls}  avg_children_created={self.expand_children_sum/self.expand_calls:.2f}  "
                  f"avg_act_dim≈{self.expand_actdim_sum/self.expand_calls:.1f}")
        if self.eval_calls > 0:
            self.logger.debug(f"eval_calls={self.eval_calls}  cold_starts={self.eval_cold}  avg_eval_ms={self.eval_ms_sum/self.eval_calls:.2f}")
        if self.backprop_calls > 0:
            self.logger.debug(f"backprop_calls={self.backprop_calls}  avg_path_len={self.backprop_pathlen_sum/self.backprop_calls:.2f}")

class MCTS:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, exploration_weight=1,
                 debug_memory = False,
                 debug_time_mcts_iterations = False,
                 debug_comparison_exploration_exploitation = False):
        self.curr_tree_root = None
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.Nsa = defaultdict(int)
        self.children: dict[Node, Any] = dict()  # actions and children output of each node. structure is (action,result_state)
        self.exploration_weight = exploration_weight
        self.path_until_goal = None
        self.debug_memory = debug_memory
        self.debug_time_mcts_iterations = debug_time_mcts_iterations
        if self.debug_time_mcts_iterations:
            self.start_times = []
            self.after_selection_times = []
            self.after_expansion_times = []
            self.after_eval_times = []
            self.end_times = []
        self.debug_comparison_exploration_exploitation = debug_comparison_exploration_exploitation
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
        if self.debug_time_mcts_iterations:
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
                if self.debug_time_mcts_iterations:
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
        N_local = self.N  # local refs to cut attribute lookups
        Q_local = self.Q
        for node in reversed(path):
            n = N_local[node] + 1
            N_local[node] = n
            q = Q_local.get(node, 0.0)
            Q_local[node] = q + (reward - q) / n  # running average
        if self.debug_time_mcts_iterations:
            self.end_times.append(time())

    def _evaluate_node(self, node) -> float:
        """Use the teacher's (or another) heuristic to evaluate a specific node, in order to use value-based mcts"""
        raise NotImplemented

    def _puct_select_no_cycle(self, node, path_set):
        """Sample a child of `node` using PUCT scores as softmax logits while avoiding cycles.
           Returns (action, child_node).  Vectorized over children to reduce Python overhead."""

        # 0) Sanity: children should already be generated for this node
        children_map = self.children[node]  # dict: action -> child_node
        # assert all(child in self.children for child in children_map.values())
        actions_nodes = list(children_map.items())  # [(a, child), ...]
        n_children = len(actions_nodes)
        assert n_children > 0, "PUCT select called on a node with no children"

        # 1) Priors from policy (vector over action indices)
        priors = self.get_act_dist_from_mcts_node(node)
        priors = priors.numpy() if hasattr(priors, "numpy") else priors

        # Build parallel arrays once
        actions = np.fromiter((a for a, _ in actions_nodes), dtype=np.int32, count=n_children)
        child_list = [c for _, c in actions_nodes]

        # 2) Masks (vectorized)
        #   - cycle mask: child already on current path => invalidate
        cycle = np.fromiter((c in path_set for c in child_list), dtype=bool, count=n_children)

        # 3) Prior lookup (vectorized, with bounds check)
        prior = np.zeros(n_children, dtype=np.float32)
        if np.ndim(priors) == 1 and priors.size > 0:
            valid = (actions >= 0) & (actions < priors.size)
            if valid.any():
                prior[valid] = np.asarray(priors, dtype=np.float32)[actions[valid]]
        else:
            # extremely defensive: if priors is scalar/empty, leave zeros
            pass

        # 4) Edge visits Nsa(s,a) and child Q(s') in one sweep
        edge_visits = np.fromiter((self.Nsa[(node, int(a))] for a in actions),
                                   dtype=np.int32, count=n_children)
        Q_child = np.fromiter((self.Q.get(c, 0.0) for c in child_list),
                               dtype=np.float32, count=n_children)

        # 5) Forced first visits: if any edge is unvisited and not a cycle, pick among them
        unvisited = (edge_visits == 0) & (~cycle)
        if np.any(unvisited):
            cand = np.flatnonzero(unvisited)
            # bias by prior if it has any mass, else uniform among unvisited
            w = prior[cand].astype(np.float64)
            s = float(w.sum())
            if np.isfinite(s) and s > 0.0:
                w /= s
                idx = int(np.random.choice(cand, p=w))
            else:
                idx = int(np.random.choice(cand))
            a, child = actions_nodes[idx]
            if self._probe: self._probe.log_select_unvisited()
            return a, child

        # 6) Compute U and score = Q + U (vectorized)
        N_parent = float(self.N.get(node, 0))
        sqrtN = math.sqrt(max(1.0, N_parent))
        U = self.exploration_weight * prior * (sqrtN / (1.0 + edge_visits.astype(np.float32)))

        # Invalidate cycles
        U[cycle] = 0.0
        score = Q_child + U
        score[cycle] = -np.inf

        # 7) Sample by softmax over (Q+U) with numerical stability
        x = score.astype(np.float64)
        x -= np.max(x)
        w = np.exp(x)
        w[~np.isfinite(w)] = 0.0
        s = float(w.sum())
        if (not np.isfinite(s)) or s <= 0.0:
            idx = int(np.argmax(score))  # fallback if all weights underflow
        else:
            p = w / s
            idx = int(np.random.choice(n_children, p=p))

        a, child = actions_nodes[idx]

        # small O(n) helpers
        cycle_present = bool(cycle.any())
        valid = ~cycle
        pv = prior[valid]
        pv_sum = float(pv.sum())
        prior_entropy = None
        if pv_sum > 0 and np.isfinite(pv_sum):
            pv = pv / pv_sum
            prior_entropy = float(-(pv * np.log(pv + 1e-12)).sum())

        chosen_p = float(p[idx]) if 'p' in locals() else None
        edge_visits_chosen = int(edge_visits[idx])

        # minimal event
        if self._probe: self._probe.log_select_softmax(
            prior_entropy=prior_entropy,
            chosen_p=chosen_p,
            edge_visits_chosen=edge_visits_chosen,
            cycle_present=cycle_present
        )

        # keep your existing summary call (DON'T remove this)
        if self._probe:
            try:
                self._probe.log(Q_child.tolist(), U.tolist(), idx)
            except Exception:
                pass

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
