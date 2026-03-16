import gc
import logging
import random
from abc import ABC, abstractmethod
from array import array
import bisect
from collections import defaultdict
import math
from typing import Any, List, Optional, Iterator, Tuple
from time import time
import numpy as np
from rpyc import BaseNetref

from asnets.state_reprs import CanonicalState

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class MCTSNode:
    delete_counter = 0
    __slots__ = (
        "state", "cost_until_now", "reward_weight", "children", "goal_state", "terminal_state", "as_network_input",
        "applicable_action_mask", "act_dist", "pred_value", "Q_value", "known_distance_to_goal", "best_goal_child",
        "visit_count", "last_select_id"
    )

    def __init__(self,
                 state,
                 cost_until_now, reward_weight=1000,
                 is_goal=False, is_terminal=False, as_network_input=None, applicable_action_mask=None):
        self.state = state
        self.cost_until_now = cost_until_now
        self.reward_weight = reward_weight
        self.children = None
        self.goal_state = is_goal
        self.terminal_state = is_terminal
        self.as_network_input = as_network_input
        self.applicable_action_mask = applicable_action_mask
        self.act_dist = None
        self.pred_value = None
        self.Q_value = 0
        self.known_distance_to_goal = 0 if is_goal else np.inf
        self.best_goal_child = None
        self.visit_count = 0
        self.last_select_id = -1

    def simulate_step(self, action_id, problem_service):
        if hasattr(problem_service, "env_simulate_step"):
            return problem_service.env_simulate_step(self.state_id, self._hash, int(action_id))
        return problem_service.exposed_env_simulate_step(self.state_id, self._hash, int(action_id))

    def is_terminal(self):
        """Returns True if the node has no children"""
        return self.terminal_state

    def is_goal(self):
        """Return True if the current not is a goal"""
        return self.goal_state

    def reward(self):
        if self.is_goal():
            return self.reward_weight / self.cost_until_now
        return 0

    def to_network_input(self):
        """Make the cstate represented by 'this' MCTSNode to be compatible for the policy network, and transposes it"""
        return self.as_network_input

    def get_identifiers(self) -> tuple[int, int]:
        # return self.state_id, self._hash
        raise NotImplementedError("Changed to using states in nodes instead of identifiers")

    @property
    def state_key(self) -> bytes:
        return self.state.state_key


class FixedChildMap:
    __slots__ = ("_keys", "_values", "_visits", "_actions_np")

    def __init__(self, keys: List[int], values: List[Any]):
        assert len(keys) == len(values), "Keys and values must match in length"

        sorted_pairs = sorted(zip(keys, values))
        self._keys = array('H', (k for k, _ in sorted_pairs))
        self._values = [v for _, v in sorted_pairs]
        self._visits = np.zeros(len(self._keys), dtype=np.int32)

        # cached numpy version
        self._actions_np = np.asarray(self._keys, dtype=np.int32)

    def _index_of(self, key: int) -> int:
        """Return the index of the action key."""
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            return idx
        raise KeyError(key)

    def get(self, key: int, default: Optional[Any] = None) -> Optional[Any]:
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            return self._values[idx]
        return default

    def __getitem__(self, key: int) -> Any:
        idx = self._index_of(key)
        return self._values[idx]

    def __contains__(self, key: int) -> bool:
        idx = bisect.bisect_left(self._keys, key)
        return idx < len(self._keys) and self._keys[idx] == key

    def items(self) -> Iterator[Tuple[int, Any]]:
        return zip(self._keys, self._values)

    def keys(self) -> Iterator[int]:
        return iter(self._keys)

    def values(self) -> Iterator[Any]:
        return iter(self._values)

    @property
    def visits(self):
        return self._visits

    @property
    def actions_np(self):
        return self._actions_np

    def visit_count(self, key: int) -> int:
        return int(self._visits[self._index_of(key)])

    def increment_visit(self, key: int):
        """Increment visit count for the given action key."""
        idx = self._index_of(key)
        self._visits[idx] += 1

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[int]:
        return iter(self._keys)

    def __repr__(self) -> str:
        items = ', '.join(f"{k}: {v}" for k, v in self.items())
        return f"FixedChildMap({{{items}}})"

    def is_empty(self) -> bool:
        return len(self._keys) == 0


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
        scores = [q + u for q, u in zip(q_list, u_list)]
        idx_score = int(np.argmax(scores))
        idx_q = int(np.argmax(q_list))
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
            self.logger.debug(
                f"events={self.events}  avg_U_share_on_chosen={avg_share:.3f}  pct_argmax_flipped_by_U={flip_pct:.1f}%")

        self.logger.debug("=== Selection counters ===")
        self.logger.debug(
            f"forced_first_visit={self.sel_forced_unvisited}  softmax={self.sel_softmax}  cycles_present={self.sel_cycle_present}")
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
            self.logger.debug(
                f"expand_calls={self.expand_calls}  avg_children_created={self.expand_children_sum / self.expand_calls:.2f}  "
                f"avg_act_dim≈{self.expand_actdim_sum / self.expand_calls:.1f}")
        if self.eval_calls > 0:
            self.logger.debug(
                f"eval_calls={self.eval_calls}  cold_starts={self.eval_cold}  avg_eval_ms={self.eval_ms_sum / self.eval_calls:.2f}")
        if self.backprop_calls > 0:
            self.logger.debug(
                f"backprop_calls={self.backprop_calls}  avg_path_len={self.backprop_pathlen_sum / self.backprop_calls:.2f}")


class MCTS:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, exploration_weight=1,
                 network=None,
                 problem_service=None,
                 debug_memory=False,
                 debug_time_mcts_iterations=False,
                 debug_comparison_exploration_exploitation=False,
                 use_numpy_sampler=False):
        self.curr_tree_root: Optional[MCTSNode] = None
        self.exploration_weight = exploration_weight
        self.path_until_goal = None
        self.state_key_to_node: dict[bytes, MCTSNode] = {}
        self.problem_service = problem_service
        self.network = network
        self.policy_only = self.network.policy_only()

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
        self._select_counter = -1

        # -- select path --
        self._init_selector(use_numpy_sampler=use_numpy_sampler)

    def _init_selector(self, use_numpy_sampler: bool):

        if self._probe:
            if use_numpy_sampler:
                raise NotImplementedError
            else:
                self._puct_select_no_cycle = self._puct_select_probe_python
        else:
            if use_numpy_sampler:
                self._puct_select_no_cycle = self._puct_select_fast_numpy
            else:
                self._puct_select_no_cycle = self._puct_select_fast_python

    def mcts_iteration_classic(self, node, horizon):
        """Make the tree one layer better. (Train for one iteration.)"""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._rollout(leaf, horizon=horizon)
        self._backpropagate(path, reward, leaf.goal_state)
        if self.path_until_goal is not None:
            self.path_until_goal = self.reconstructSelectionPath(path) + self.path_until_goal

    def mcts_iteration_value_based(self, node):
        path = self._select(node)
        if hasattr(self, "total_select_depth"):
            self.total_select_depth += len(path)
        leaf = path[-1]
        self._expand(leaf)
        # reward = 1 / (1 + self._evaluate_node(leaf))
        reward = self._evaluate_node(leaf)
        # numbers might be too low or insignificant?? I think it would be okay...
        # theoretically and practically it SHOULD not be lower than 1/10001 which isn't that low.
        self._backpropagate(path, reward, leaf.goal_state)

    def _select(self, node: MCTSNode):
        """Find an unexplored descendant of `node`."""
        if self.debug_time_mcts_iterations:
            self.start_times.append(time())
        node_path = []
        self._select_counter += 1
        while True:
            node_path.append(node)
            node.last_select_id = self._select_counter
            childmap = node.children
            if childmap is None or childmap.is_empty():
                if self.debug_time_mcts_iterations:
                    self.after_selection_times.append(time())
                return node_path
            action, child = self._puct_select_no_cycle(node)
            # increment edge visit by ACTION KEY
            if child is None:
                if self.debug_time_mcts_iterations:
                    self.after_selection_times.append(time())
                return node_path
            childmap.increment_visit(action)
            # all children are cyclic on this path -> stop here
            node = child

    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        raise NotImplemented

    def _rollout(self, mcts_node, horizon=10):
        """Returns the reward for a random simulation (to a certain horizon) of `node`"""
        raise NotImplemented

    def _backpropagate(self, path: list[MCTSNode], reward: float, subtree_contains_goal: bool):
        distance_from_goal = 0 if subtree_contains_goal else None
        child_toward_goal = None
        for node in reversed(path):
            n = node.visit_count + 1
            node.visit_count = n
            q_old = node.Q_value
            node.Q_value = q_old + (reward - q_old) / n
            if distance_from_goal is not None:
                old_dist = node.known_distance_to_goal
                update = False
                if distance_from_goal < old_dist:
                    update = True
                elif distance_from_goal == old_dist and node.best_goal_child is not None:
                    # tie-breaking
                    prev_child = node.best_goal_child
                    new_child = child_toward_goal
                    prev_visits = prev_child.visit_count
                    new_visits = new_child.visit_count
                    if new_visits > prev_visits:
                        update = True
                    elif new_visits == prev_visits and new_child.Q_value > prev_child.Q_value:
                        update = True
                if update:
                    node.known_distance_to_goal = distance_from_goal
                    node.best_goal_child = child_toward_goal
                distance_from_goal += 1
                child_toward_goal = node
        if self.debug_time_mcts_iterations:
            self.end_times.append(time())

    def _evaluate_node(self, node: MCTSNode) -> float:
        """Use the teacher's (or another) heuristic to evaluate a specific node, in order to use value-based mcts"""
        # value = self.problem_service.get_state_h(*node.get_identifiers())
        value = self.get_value_from_mcts_node(node)
        if self.debug_time_mcts_iterations:
            self.after_eval_times.append(time())
        return value

    # def _puct_select_no_cycle(self, node):
    #     children = node.children
    #
    #     # _keys are the REAL action IDs, not compact positions in the global action space
    #     actions = children.actions_np
    #     child_list = children._values
    #     edge_visits = children.visits
    #
    #     n_children = len(actions)
    #     assert n_children > 0, "PUCT select called on node with no children"
    #
    #     prior = node.act_dist[actions]
    #
    #     # --- child Q values ---
    #     Q_child = np.empty(n_children, dtype=np.float32)
    #     for i, child in enumerate(child_list):
    #         Q_child[i] = child.Q_value
    #
    #     # # --- cycle mask ---
    #     cycle = np.empty(n_children, dtype=bool)
    #     for i, child in enumerate(child_list):
    #         cycle[i] = (child.last_select_id == self._select_counter)
    #
    #     # --- PUCT ---
    #     sqrtN = math.sqrt(max(1.0, node.visit_count))
    #
    #     U = self.exploration_weight * prior * (
    #             sqrtN / (1.0 + edge_visits)
    #     )
    #     score = Q_child + U
    #
    #     valid_mask = ~cycle
    #
    #     # no non-cyclic child available on this path
    #     if not valid_mask.any():
    #         if self._probe:
    #             self._probe.log_select_softmax(
    #                 prior_entropy=None,
    #                 chosen_p=None,
    #                 edge_visits_chosen=None,
    #                 cycle_present=True
    #             )
    #             try:
    #                 self._probe.log(Q_child.tolist(), U.tolist(), -1)
    #             except Exception:
    #                 pass
    #
    #         return None, None
    #
    #     valid_indices = np.flatnonzero(valid_mask)
    #     score_valid = score[valid_mask]
    #
    #     # --- stable softmax on valid children only ---
    #     x = score_valid.astype(np.float64)
    #     x -= x.max()
    #     w = np.exp(x)
    #     s = float(w.sum())
    #
    #     if (not np.isfinite(s)) or s <= 0.0:
    #         idx_local = int(np.argmax(score_valid))
    #         p = None
    #     else:
    #         p = w / s
    #         idx_local = int(np.random.choice(len(valid_indices), p=p))
    #
    #     idx = int(valid_indices[idx_local])
    #     action = int(actions[idx])
    #     child = child_list[idx]
    #
    #     if self._probe:
    #         cycle_present = bool(cycle.any())
    #         pv = prior[valid_mask]
    #         pv_sum = float(pv.sum())
    #         prior_entropy = None
    #         if pv_sum > 0.0 and np.isfinite(pv_sum):
    #             pv = pv / pv_sum
    #             prior_entropy = float(-(pv * np.log(pv + 1e-12)).sum())
    #         chosen_p = float(p[idx_local]) if p is not None else None
    #         edge_visits_chosen = int(edge_visits[idx])
    #
    #         self._probe.log_select_softmax(
    #             prior_entropy=prior_entropy,
    #             chosen_p=chosen_p,
    #             edge_visits_chosen=edge_visits_chosen,
    #             cycle_present=cycle_present
    #         )
    #         try:
    #             self._probe.log(Q_child.tolist(), U.tolist(), idx)
    #         except Exception:
    #             pass
    #
    #     return action, child

    def reconstructSelectionPath(self, path):
        output_path = [(None, self.curr_tree_root)]
        for mcts_node in path:
            if mcts_node == self.curr_tree_root:
                continue
            assert output_path[-1][1].children is not None
            assert mcts_node in output_path[-1][1].children.values()
            for action, next_node in output_path[-1][1].children.items():
                if mcts_node == next_node:
                    output_path.append((action, mcts_node))
        return output_path[1:]

    def get_value_from_mcts_node(self, node: MCTSNode) -> float:
        if node.goal_state:
            return 1.0
        else:
            return node.pred_value

    def _delete_subtree(self, node, recursive=True):
        # Recursively delete the subtree rooted at this node
        if recursive:
            if node.children is not None:
                for _, child in node.children.items():
                    self._delete_subtree(child)
        node.children = None

    def log_node_count(self, label=""):
        gc.collect()

        count = 0
        for obj in gc.get_objects():
            # Filter out remote RPyC references explicitly
            if isinstance(obj, BaseNetref):
                continue
            if isinstance(obj, MCTSNode):
                count += 1

        print(f"{label} - Live MCTSNode instances: {count}")

    def prune_children_except(self, parent_node, keep_action):
        children_dict = parent_node.children
        if children_dict is None:
            return
        keep_child = None
        if self.debug_memory:
            self.log_node_count("Before deleting old root's irrelevant children")
        for action, child_node in list(children_dict.items()):
            if keep_child is None and action == keep_action:
                keep_child = child_node
                continue
            self._delete_subtree(child_node)
        if self.debug_memory:
            self.log_node_count("After deleting old root's irrelevant children")
        assert keep_child is not None
        # Replace children dict with just the one we kept
        parent_node.children = FixedChildMap([keep_action], [keep_child])

    def get_applicable_action_mask(self, node: MCTSNode):
        if node.applicable_action_mask is None:  # Fallback
            node.applicable_action_mask = self.problem_service.get_applicable_action_mask(*node.get_identifiers())
        return node.applicable_action_mask

    def _puct_select_fast_python(self, node):
        children = node.children
        actions = children.actions_np
        child_list = children._values
        # next 4 lines are for readability and are not 100% optimized
        edge_visits = children.visits
        priors = node.act_dist
        sqrtN = math.sqrt(max(1.0, node.visit_count))
        c = self.exploration_weight

        sid = self._select_counter
        best_max = -math.inf
        any_valid = False
        scores = []
        for i, child in enumerate(child_list):
            u = c * priors[actions[i]] * (sqrtN / (1.0 + edge_visits[i]))
            s = child.Q_value + u
            scores.append(s)
            if child.last_select_id != sid:
                any_valid = True
                if s > best_max:
                    best_max = s
        if not any_valid:
            return None, None
        total = 0.0
        weights = []
        for i, child in enumerate(child_list):
            if child.last_select_id == sid:
                weights.append(0.0)
                continue
            w = math.exp(scores[i] - best_max)
            weights.append(w)
            total += w
        if total <= 0 or not math.isfinite(total):
            best = -math.inf
            idx = -1
            for i, child in enumerate(child_list):
                if child.last_select_id == sid:
                    continue
                s = scores[i]
                if s > best:
                    best = s
                    idx = i
        else:
            r = random.random() * total
            acc = 0.0
            idx = -1
            for i, child in enumerate(child_list):
                if child.last_select_id == sid:
                    continue
                acc += weights[i]
                if acc >= r:
                    idx = i
                    break
        return int(actions[idx]), child_list[idx]

    def _puct_select_fast_numpy(self, node):
        children = node.children
        actions = children.actions_np
        child_list = children._values
        edge_visits = children.visits
        priors = node.act_dist
        n = len(actions)
        sid = self._select_counter
        cycle = np.empty(n, dtype=bool)
        Q = np.empty(n, dtype=np.float32)
        for i, child in enumerate(child_list):
            Q[i] = child.Q_value
            cycle[i] = child.last_select_id == sid
        prior = priors[actions]
        sqrtN = math.sqrt(max(1.0, node.visit_count))
        U = self.exploration_weight * prior * (sqrtN / (1.0 + edge_visits))
        score = Q + U
        valid_mask = ~cycle
        if not valid_mask.any():
            return None, None
        score_valid = score[valid_mask]
        x = score_valid - score_valid.max()
        w = np.exp(x)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            idx_local = int(np.argmax(score_valid))
        else:
            p = w / s
            idx_local = int(np.random.choice(len(score_valid), p=p))
        idx = np.flatnonzero(valid_mask)[idx_local]
        return int(actions[idx]), child_list[idx]

    def _puct_select_probe_python(self, node):
        children = node.children
        actions = children.actions_np
        child_list = children._values
        edge_visits = children.visits
        priors = node.act_dist
        sqrtN = math.sqrt(max(1.0, node.visit_count))
        c = self.exploration_weight
        sid = self._select_counter
        n = len(actions)
        Q = np.empty(n, dtype=np.float32)
        U = np.empty(n, dtype=np.float32)
        scores = [0.0] * n
        cycle = [False] * n
        max_score = -math.inf
        any_valid = False
        for i, child in enumerate(child_list):
            q = child.Q_value
            u = c * priors[actions[i]] * (sqrtN / (1.0 + edge_visits[i]))
            Q[i] = q
            U[i] = u
            s = q + u
            scores[i] = s
            cyc = child.last_select_id == sid
            cycle[i] = cyc
            if not cyc:
                any_valid = True
                if s > max_score:
                    max_score = s
        if not any_valid:
            return None, None


def wrapInMCTSNode(state: CanonicalState, cost_until_now=float('inf')):
    return MCTSNode(state=state, cost_until_now=cost_until_now, is_goal=state.is_goal,
                    is_terminal=state.is_terminal, as_network_input=state.to_network_input(),
                    applicable_action_mask=state.get_applicable_action_mask(), )
