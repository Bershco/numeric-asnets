import gc
import hashlib
import logging
from array import array
import bisect
import math
from collections import Counter
from time import perf_counter
from typing import Any, List, Optional, Iterator, Tuple
import numpy as np
from rpyc import BaseNetref

from asnets.state_reprs import CanonicalState

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def action_history_digest(state: CanonicalState) -> bytes | None:
    """Hash only the per-ground-action ``action_count`` feature."""
    try:
        action_counts = state.get_aux_data_dimension("action_count")
    except (AttributeError, KeyError, ValueError):
        return None
    canonical = np.asarray(action_counts, dtype="<f4").tobytes(order="C")
    return hashlib.blake2b(
        canonical,
        digest_size=16,
        person=b"asnet-act-count",
    ).digest()


class MCTSNode:
    delete_counter = 0
    __slots__ = (
        "state", "cost_until_now", "children", "goal_state", "terminal_state", "as_network_input",
        "applicable_action_mask", "act_dist", "pred_value", "Q_value", "known_distance_to_goal", "best_goal_child",
        "visit_count", "last_select_id", "parents", "root_visit_count", "on_trajectory",
        "action_history_digest",
    )

    def __init__(self,
                 state,
                 cost_until_now,
                 is_goal=False, is_terminal=False, as_network_input=None, applicable_action_mask=None):
        self.state = state
        self.cost_until_now = cost_until_now
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
        self.root_visit_count = 0
        self.last_select_id = -1
        self.parents: list[tuple["MCTSNode", int]] = [] # list of tuples of (parent, action)
        self.on_trajectory = False
        self.action_history_digest = None

    def is_terminal(self):
        """Returns True if the node has no children"""
        return self.terminal_state

    def is_goal(self):
        """Return True if the current not is a goal"""
        return self.goal_state

    def to_network_input(self):
        """Make the cstate represented by 'this' MCTSNode to be compatible for the policy network, and transposes it"""
        return self.as_network_input

    def get_identifiers(self) -> tuple[int, int]:
        # return self.state_id, self._hash
        raise NotImplementedError("Changed to using states in nodes instead of identifiers")

    @property
    def state_key(self) -> bytes:
        return self.state.state_key

    def add_parent(self, node: "MCTSNode", act: int):
        if not any(parent is node and parent_act == act for parent, parent_act in self.parents):
            self.parents.append((node, act))

    def get_child_on_trajectory_mask(self) -> np.ndarray:
        """
        Returns a float32 mask aligned with children.actions_np:
            1.0 -> child is on trajectory
            0.0 -> child is not on trajectory
        """
        if self.children is None or self.children.is_empty():
            return np.empty(0, dtype=np.float32)

        return np.asarray(
            [
                float(child is not None and child.on_trajectory)
                for child in self.children.values()
            ],
            dtype=np.float32,
        )

class FixedChildMap:
    __slots__ = ("_keys", "_values", "_visits", "_actions_np", "_priors")

    def __init__(self, keys: List[int], values: List[Any], priors: List[float]):
        assert len(keys) == len(values), "Keys and values must match in length"
        assert len(keys) == len(priors), "Keys and priors must match in length"

        sorted_triples = sorted(zip(keys, values, priors))
        self._keys = array('H', (k for k, _, _ in sorted_triples))
        self._values = [v for _, v, _ in sorted_triples]
        self._priors = np.asarray([p for _, _, p in sorted_triples], dtype=np.float32)

        self._visits = np.zeros(len(self._keys), dtype=np.int32)
        self._actions_np = np.asarray(self._keys, dtype=np.int32)

    def _find_index(self, key: int) -> Optional[int]:
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            return idx
        return None

    def _index_of(self, key: int) -> int:
        idx = self._find_index(key)
        if idx is None:
            raise KeyError(key)
        return idx

    def get(self, key: int, default: Optional[Any] = None) -> Optional[Any]:
        idx = self._find_index(key)
        return self._values[idx] if idx is not None else default

    def __getitem__(self, key: int) -> Any:
        return self._values[self._index_of(key)]

    def __contains__(self, key: int) -> bool:
        return self._find_index(key) is not None

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
    def priors(self):
        return self._priors

    @property
    def actions_np(self):
        return self._actions_np

    def increment_visit(self, key: int):
        self._visits[self._index_of(key)] += 1

    def increment_visit_by_child(self, child: Any):
        for i, value in enumerate(self._values):
            if value is child:
                self._visits[i] += 1
                return
        raise KeyError("Child not found in FixedChildMap")

    def append(self, key: int, value: Any, prior: float) -> None:
        """Insert one child without disturbing existing edge statistics."""
        if self._find_index(key) is not None:
            raise ValueError(f"Action {key} already exists in child map")
        if key < 0 or key > 65535:
            raise ValueError(f"Action {key} does not fit compact uint16 storage")

        idx = bisect.bisect_left(self._keys, key)
        self._keys.insert(idx, key)
        self._values.insert(idx, value)
        self._priors = np.insert(
            self._priors, idx, np.float32(prior)).astype(
                np.float32, copy=False)
        self._visits = np.insert(
            self._visits, idx, np.int32(0)).astype(
                np.int32, copy=False)
        self._actions_np = np.asarray(self._keys, dtype=np.int32)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[int]:
        return iter(self._keys)

    def __repr__(self) -> str:
        items = ', '.join(f"{k}: {v}" for k, v in self.items())
        return f"FixedChildMap({{{items}}})"

    def is_empty(self) -> bool:
        return len(self._keys) == 0

    def get_qsa_nsa_list(self) -> list[tuple[int, float, int]]:
        return [
            (act, child.Q_value, edge_Nsa)
            for act, child, edge_Nsa in sorted(
                zip(self._keys, self._values, self._visits),
                key=lambda x: x[0],
            )
        ]


class MCTS:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, exploration_weight=1,
                 network=None,
                 debug_memory=False,
                 select_logging=False,
                 minimization=False,
                 ):
        self.curr_tree_root: Optional[MCTSNode] = None
        self.original_tree_root: Optional[MCTSNode] = None
        self.exploration_weight = exploration_weight
        self.path_until_goal = None
        self.state_key_to_node: dict[Any, MCTSNode] = {}
        self.network = network

        self.sign = -1 if minimization else 1 # This turns selection to argmax(U-Q) instead of argmax(Q+U) in the maximization setting
        self.minimization = minimization

        self.debug_memory = debug_memory
        self._select_counter = -1
        self.selection_depth_hist = Counter()
        self.horizon_cutoff_depth_hist = Counter()
        self.root_width_hist = Counter()
        self.width_by_depth_band = {}
        self.selected_policy_rank_hist = Counter()
        self.known_goal_decisions = 0
        self.infeasible_known_goal_decisions = 0
        self.selection_seconds = 0.0
        self.expansion_seconds = 0.0
        self.evaluation_seconds = 0.0
        self.backpropagation_seconds = 0.0
        self.search_calls = 0
        self.peak_node_count = 0

        self.select_logging = select_logging
        if select_logging:
            self.select_depths = []
            self.deep_select_applicable_actions = []
            self.times_moved_forward = 0
            self.select_depth_limit = 80
            self.num_applicable_action_limit = 2
            self.same_action_streaks = []
            self.effective_branching = []
            self.select_stop_frontier = 0
            self.select_stop_cycle_blocked = 0
            self.cycle_blocked_depths = []

    def node_registry_key(self, state: CanonicalState):
        """Return the node-statistics identity for a canonical state.

        The default remains the physical-state key.  Context-aware subclasses
        may override this while retaining the physical key for cycle safety.
        """
        return state.state_key

    def get_select_depth_stats(self):
        import numpy as np

        depths = self.select_depths

        if not depths:
            print("[SELECT STATS] No data collected yet.")
            return

        d = np.asarray(depths, dtype=np.int32)

        count = d.size
        mean = float(d.mean())
        median = float(np.median(d))
        std = float(d.std())

        d_min = int(d.min())
        d_max = int(d.max())

        p90 = float(np.percentile(d, 90))
        p95 = float(np.percentile(d, 95))
        p99 = float(np.percentile(d, 99))

        # per decision step (very useful signal)
        steps = max(1, self.times_moved_forward)
        per_step = count / steps

        print("\n=== SELECT DEPTH STATS ===")
        print(f"Samples: {count}   Decision steps: {steps}   Selects/step: {per_step:.2f}")
        print(f"Min: {d_min}   Max: {d_max}")
        print(f"Mean: {mean:.2f}   Median: {median:.2f}   Std: {std:.2f}")
        print(f"P90: {p90:.2f}   P95: {p95:.2f}   P99: {p99:.2f}")

        # --- histogram (coarse but very informative) ---
        bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
        hist, edges = np.histogram(d, bins=bins)

        print("\nDepth distribution:")
        for i in range(len(hist)):
            lo = edges[i]
            hi = edges[i + 1]
            print(f"[{int(lo):>4}, {int(hi):>4}): {hist[i]}")

        # --- sanity hints ---
        if p95 > 150:
            print("\n[NOTE] Very deep selection paths (p95 > 150). Likely long corridors / low branching.")

        if std > mean:
            print("[NOTE] High variance in depth — search behavior is unstable across iterations.")

        if median < mean:
            print("[NOTE] Long tail detected (mean > median). Some very deep paths dominate runtime.")

        print("Top 10 deepest paths:", sorted(depths)[-10:])
        print("==========================")
        print("Deep applicable actions mean:",
              np.mean(self.deep_select_applicable_actions) if self.deep_select_applicable_actions else "None")
        print("Deep applicable actions P90:",
              np.percentile(self.deep_select_applicable_actions, 90) if self.deep_select_applicable_actions else "None")
        print("Same-action streak mean:", np.mean(self.same_action_streaks))
        print("Same-action streak max:", np.max(self.same_action_streaks))
        print("Effective branching mean:", np.mean(self.effective_branching))
        print("Effective branching P90:", np.percentile(self.effective_branching, 90))
        print("==========================")
        print(f"Select stop frontier: {self.select_stop_frontier}")
        print(f"Select stop cycle-blocked: {self.select_stop_cycle_blocked}")
        if self.cycle_blocked_depths:
            cd = np.asarray(self.cycle_blocked_depths)
            print(f"Cycle-blocked depth mean: {cd.mean():.2f}, median: {np.median(cd):.2f}, max: {cd.max()}")
        print("==========================")

    def mcts_iteration_classic(self, node, horizon):
        """Make the tree one layer better. (Train for one iteration.)"""
        path, _ = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        value = self._rollout(leaf, horizon=horizon)
        self._backpropagate(path, value, leaf.goal_state)
        if self.path_until_goal is not None:
            self.path_until_goal = self.reconstructSelectionPath(path) + self.path_until_goal

    def mcts_iteration_value_based(self, node, remaining_horizon=None):
        node.root_visit_count += 1
        started = perf_counter()
        path, stop_reason = self._select(
            node,
            max_depth=remaining_horizon,
        )
        self.selection_seconds += perf_counter() - started
        selection_depth = len(path) - 1
        self.selection_depth_hist[selection_depth] += 1
        if self.select_logging:
            self.select_depths.append(len(path))
        leaf = path[-1]
        if stop_reason == "horizon":
            self.horizon_cutoff_depth_hist[selection_depth] += 1
            evaluation_node = leaf
        else:
            started = perf_counter()
            expanded_child = self._expand(leaf)
            self.expansion_seconds += perf_counter() - started
            if (expanded_child is not None
                    and expanded_child.last_select_id != self._select_counter):
                path.append(expanded_child)
                evaluation_node = expanded_child
            else:
                evaluation_node = leaf
        self.peak_node_count = max(
            self.peak_node_count, len(self.state_key_to_node))
        started = perf_counter()
        value = self._evaluate_node(evaluation_node)
        self.evaluation_seconds += perf_counter() - started
        # numbers might be too low or insignificant?? I think it would be okay...
        # theoretically and practically it SHOULD not be lower than 1/10001 which isn't that low.
        started = perf_counter()
        self._backpropagate(path, value, evaluation_node.goal_state)
        self.backpropagation_seconds += perf_counter() - started

    def _select(self, node: MCTSNode, max_depth=None):
        """Find an unexplored descendant of `node`."""
        if self.select_logging:
            prev_action = None
            same_action_streak = 0
            max_same_action_streak = 0
        node_path = []
        physical_path = set()
        self._select_counter += 1
        depth = 0
        while True:
            node_path.append(node)
            physical_path.add(node.state_key)
            selection_depth = depth
            if node.children is not None:
                band = self._depth_band(selection_depth)
                self.width_by_depth_band.setdefault(band, Counter())[
                    len(node.children)] += 1
            self.maybe_log_puct_on_selection_path(
                node,
                selection_depth=selection_depth,
                max_depth=-1,
                every=25,
            )
            depth += 1
            if self.select_logging and depth > self.select_depth_limit:
                self.deep_select_applicable_actions.append(sum(node.applicable_action_mask))
            node.last_select_id = self._select_counter
            if (max_depth is not None
                    and selection_depth >= max_depth):
                return node_path, "horizon"
            if self._should_expand(node):
                if self.select_logging:
                    self.same_action_streaks.append(max_same_action_streak)
                    self.select_stop_frontier += 1
                return node_path, "expand"
            childmap = node.children
            if childmap is None or childmap.is_empty():
                return node_path, "frontier"
            action, child = self._puct_select_argmax(
                node,
                forbidden_physical_keys=(
                    physical_path
                    if getattr(self, "contextual_nodes", False)
                    else None
                ),
            )

            if self.select_logging:
                if prev_action is not None and action == prev_action:
                    same_action_streak += 1
                else:
                    same_action_streak = 1

                max_same_action_streak = max(max_same_action_streak, same_action_streak)
                prev_action = action

            # increment edge visit by ACTION KEY
            if child is None:
                # all children are cyclic on this path -> stop here
                if self.select_logging:
                    self.same_action_streaks.append(max_same_action_streak)
                    self.select_stop_cycle_blocked += 1
                    self.cycle_blocked_depths.append(depth)
                return node_path, "cycle"
            # childmap.increment_visit(action)
            node = child

    @staticmethod
    def _depth_band(depth: int) -> str:
        if depth <= 2:
            return str(depth)
        for lower, upper in (
                (3, 4), (5, 9), (10, 19), (20, 49),
                (50, 99), (100, 199)):
            if lower <= depth <= upper:
                return f"{lower}-{upper}"
        return "200+"

    def record_goal_feasibility(self, remaining_horizon: int) -> None:
        root = self.curr_tree_root
        if root.known_distance_to_goal == np.inf:
            return
        self.known_goal_decisions += 1
        if root.known_distance_to_goal > remaining_horizon:
            self.infeasible_known_goal_decisions += 1

    def record_selected_policy_rank(self, action_id: int) -> None:
        root = self.curr_tree_root
        valid = np.where(root.applicable_action_mask)[0]
        if len(valid) == 0:
            return
        order = np.lexsort((valid, -root.act_dist[valid]))
        ranked = valid[order]
        matches = np.where(ranked == action_id)[0]
        if len(matches):
            self.selected_policy_rank_hist[int(matches[0]) + 1] += 1

    def _should_expand(self, node: MCTSNode) -> bool:
        """Return whether selection should stop so this node can expand."""
        return node.children is None or node.children.is_empty()

    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        raise NotImplemented

    def _rollout(self, mcts_node, horizon=10):
        """Returns the reward for a random simulation (to a certain horizon) of `node`"""
        raise NotImplemented

    def _backpropagate(self, path: list[MCTSNode], value: float, subtree_contains_goal: bool):
        for parent, child in zip(path, path[1:]):
            if parent.children is None:
                raise RuntimeError("Backprop path contains parent with no children")

            parent.children.increment_visit_by_child(child)
        distance_from_goal = 0 if subtree_contains_goal else None
        child_toward_goal = None
        for node in reversed(path):
            node.visit_count += 1
            q_old = node.Q_value
            node.Q_value = q_old + ((value - q_old) / node.visit_count)
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
                    elif (new_visits == prev_visits and
                          (new_child.Q_value > prev_child.Q_value if not self.minimization
                          else new_child.Q_value < prev_child.Q_value)):
                        update = True
                if update:
                    node.known_distance_to_goal = distance_from_goal
                    node.best_goal_child = child_toward_goal
                distance_from_goal += 1
                child_toward_goal = node

    def _evaluate_node(self, node: MCTSNode) -> float:
        """Use the teacher's (or another) heuristic to evaluate a specific node, in order to use value-based mcts"""
        return self.get_value_from_mcts_node(node)

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
            return self.best_value()
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
        return node.applicable_action_mask

    def _puct_select_argmax(self, node, forbidden_physical_keys=None):
        children = node.children
        actions = children.actions_np
        child_list = children._values
        edge_visits = children.visits
        priors = children.priors
        sqrtN = math.sqrt(max(1.0, node.visit_count))
        c = self.exploration_weight
        sid = self._select_counter
        best_score = -math.inf
        best_idx = -1
        scores = [] if self.select_logging else None
        for i, child in enumerate(child_list):
            u = c * priors[i] * (sqrtN / (1.0 + edge_visits[i]))
            s = self.sign * child.Q_value + u
            if scores is not None:
                scores.append(s)
            if child.last_select_id == sid:
                continue
            if (forbidden_physical_keys is not None
                    and child.state_key in forbidden_physical_keys):
                continue
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx < 0:
            return None, None
        if self.select_logging:
            threshold = best_score - 1e-1
            effective_branching = 0
            for i, child in enumerate(child_list):
                if (child.last_select_id == sid
                        or (forbidden_physical_keys is not None
                            and child.state_key in forbidden_physical_keys)):
                    continue
                if scores[i] >= threshold:
                    effective_branching += 1
            self.effective_branching.append(effective_branching)
        return int(actions[best_idx]), child_list[best_idx]

    def log_puct_snapshot(
            self,
            node: "MCTSNode",
            *,
            label: str = "",
            iteration: int | None = None,
            selection_depth: int | None = None,
            top_k: int = 10,
            print_rows: bool = True,
            return_dict: bool = False,
            skip_zero_q_range: bool = False,
            q_range_eps: float = 1e-8,
    ):
        """
        Log a compact PUCT diagnostic snapshot for a single node.

        Designed for realistic use:
          - call on root every K MCTS iterations
          - optionally call on selected-path nodes by local selection depth

        Uses FixedChildMap fields:
          - children.actions_np
          - children._values
          - children.visits
          - children.priors
        """

        children = node.children

        if children is None or children.is_empty():
            msg = (
                f"[PUCT] label={label} iter={iteration} depth={selection_depth} "
                f"EMPTY children | node_N={getattr(node, 'visit_count', None)}"
            )
            print(msg)
            if return_dict:
                return {
                    "label": label,
                    "iteration": iteration,
                    "selection_depth": selection_depth,
                    "num_children": 0,
                    "node_N": getattr(node, "visit_count", None),
                }
            return None

        eps = 1e-8

        actions = children.actions_np
        child_list = children._values
        edge_visits = children.visits
        priors = children.priors

        node_N = int(getattr(node, "visit_count", 0))
        root_N = int(getattr(node, "root_visit_count", 0)) if hasattr(node, "root_visit_count") else None
        sum_Nsa = int(np.sum(edge_visits))

        sqrtN = math.sqrt(max(1.0, node_N))
        c = float(self.exploration_weight)

        Q = np.empty(len(actions), dtype=np.float32)

        for i, child in enumerate(child_list):
            Q[i] = float(child.Q_value)

        U = c * priors * (sqrtN / (1.0 + edge_visits))
        score = self.sign * Q + U

        q_min = float(np.min(Q))
        q_max = float(np.max(Q))
        q_mean = float(np.mean(Q))
        q_std = float(np.std(Q))
        q_range = q_max - q_min
        if skip_zero_q_range and q_range <= q_range_eps:
            return None

        u_min = float(np.min(U))
        u_max = float(np.max(U))
        u_mean = float(np.mean(U))
        u_std = float(np.std(U))
        u_range = u_max - u_min

        score_min = float(np.min(score))
        score_max = float(np.max(score))
        score_mean = float(np.mean(score))
        score_std = float(np.std(score))
        score_range = score_max - score_min

        p_min = float(np.min(priors))
        p_max = float(np.max(priors))
        p_mean = float(np.mean(priors))
        p_std = float(np.std(priors))

        best_q_idx = int(np.argmax(Q))
        best_u_idx = int(np.argmax(U))
        best_score_idx = int(np.argmax(score))

        best_by_Q = int(actions[best_q_idx])
        best_by_U = int(actions[best_u_idx])
        best_by_score = int(actions[best_score_idx])

        U_changed_Q_choice = best_by_score != best_by_Q

        def _top_margin(arr: np.ndarray) -> float:
            if len(arr) < 2:
                return 0.0
            # cheaper than full sort
            top2 = np.partition(arr, -2)[-2:]
            return float(np.max(top2) - np.min(top2))

        q_margin = _top_margin(Q)
        u_margin = _top_margin(U)
        score_margin = _top_margin(score)

        mean_abs_U_over_mean_abs_Q = float(np.mean(np.abs(U)) / (np.mean(np.abs(Q)) + eps))
        U_range_over_Q_range = float(u_range / (q_range + eps))

        selected_Q = float(Q[best_score_idx])
        selected_U = float(U[best_score_idx])
        selected_score = float(score[best_score_idx])
        selected_P = float(priors[best_score_idx])
        selected_Nsa = int(edge_visits[best_score_idx])
        selected_abs_U_over_abs_Q = float(abs(selected_U) / (abs(selected_Q) + eps))

        summary = {
            "label": label,
            "iteration": iteration,
            "selection_depth": selection_depth,
            "num_children": int(len(actions)),

            "node_N": node_N,
            "root_N": root_N,
            "sum_Nsa": sum_Nsa,
            "node_N_minus_sum_Nsa": int(node_N - sum_Nsa),

            "c": c,

            "Q_min": q_min,
            "Q_max": q_max,
            "Q_mean": q_mean,
            "Q_std": q_std,
            "Q_range": q_range,

            "U_min": u_min,
            "U_max": u_max,
            "U_mean": u_mean,
            "U_std": u_std,
            "U_range": u_range,

            "score_min": score_min,
            "score_max": score_max,
            "score_mean": score_mean,
            "score_std": score_std,
            "score_range": score_range,

            "P_min": p_min,
            "P_max": p_max,
            "P_mean": p_mean,
            "P_std": p_std,

            "U_range_over_Q_range": U_range_over_Q_range,
            "mean_abs_U_over_mean_abs_Q": mean_abs_U_over_mean_abs_Q,

            "best_by_Q": best_by_Q,
            "best_by_U": best_by_U,
            "best_by_score": best_by_score,
            "U_changed_Q_choice": U_changed_Q_choice,

            "Q_margin": q_margin,
            "U_margin": u_margin,
            "score_margin": score_margin,

            "selected_Q": selected_Q,
            "selected_U": selected_U,
            "selected_score": selected_score,
            "selected_P": selected_P,
            "selected_Nsa": selected_Nsa,
            "selected_abs_U_over_abs_Q": selected_abs_U_over_abs_Q,
        }

        root_part = f" root_N={root_N}" if root_N is not None else ""

        print(
            f"[PUCT] label={label} iter={iteration} depth={selection_depth} "
            f"children={len(actions)} node_N={node_N}{root_part} "
            f"sum_Nsa={sum_Nsa} node_N-sum_Nsa={node_N - sum_Nsa} c={c:.4g}"
        )

        print(
            f"  ranges: "
            f"Q={q_range:.6f} "
            f"U={u_range:.6f} "
            f"score={score_range:.6f} "
            f"U_range/Q_range={U_range_over_Q_range:.4f} "
            f"mean|U|/mean|Q|={mean_abs_U_over_mean_abs_Q:.4f}"
        )

        print(
            f"  choices: "
            f"best_Q={best_by_Q} "
            f"best_U={best_by_U} "
            f"best_score={best_by_score} "
            f"U_changed_Q_choice={U_changed_Q_choice} "
            f"Q_margin={q_margin:.6f} "
            f"U_margin={u_margin:.6f} "
            f"score_margin={score_margin:.6f}"
        )

        print(
            f"  selected: "
            f"act={best_by_score} "
            f"Nsa={selected_Nsa} "
            f"P={selected_P:.6f} "
            f"Q={selected_Q:.6f} "
            f"U={selected_U:.6f} "
            f"Q+U={selected_score:.6f} "
            f"|U|/|Q|={selected_abs_U_over_abs_Q:.4f}"
        )

        if print_rows and top_k > 0:
            order = np.argsort(-score)
            order = order[:min(top_k, len(order))]

            print(f"  top_{len(order)}_by_score:")
            print("    act      Nsa          P          Q          U        Q+U    |U|/|Q|")
            print("    --------------------------------------------------------------------")

            for idx in order:
                q = float(Q[idx])
                u = float(U[idx])
                s = float(score[idx])
                p = float(priors[idx])
                nsa = int(edge_visits[idx])
                act = int(actions[idx])
                ratio = abs(u) / (abs(q) + eps)

                print(
                    f"    {act:>3} "
                    f"{nsa:>8d} "
                    f"{p:>10.6f} "
                    f"{q:>10.6f} "
                    f"{u:>10.6f} "
                    f"{s:>10.6f} "
                    f"{ratio:>10.4f}"
                )

        if return_dict:
            return summary

        return None

    def maybe_log_puct_on_selection_path(
            self,
            node: "MCTSNode",
            *,
            selection_depth: int,
            max_depth: int = 2,
            every: int = 25,
    ):
        if not getattr(self, "puct_debug", False):
            return

        if selection_depth > max_depth:
            return

        if self._select_counter % every != 0:
            return

        if node.children is None or node.children.is_empty():
            return

        self.log_puct_snapshot(
            node,
            label=f"select_depth_{selection_depth}",
            iteration=self._select_counter,
            selection_depth=selection_depth,
            top_k=0,
            print_rows=False,
            return_dict=False,
            skip_zero_q_range=True,
        )

    def best_value(self):
        if self.minimization:
            return 0.0
        return 1.0

    def worst_value(self):
        if self.minimization:
            return 10000000.0 #np.inf raises warnings in multiplications and averaging
        return 0.0

def wrapInMCTSNode(state: CanonicalState, cost_until_now=float('inf')):
    return MCTSNode(state=state, cost_until_now=cost_until_now, is_goal=state.is_goal,
                    is_terminal=state.is_terminal, as_network_input=state.to_network_input(),
                    applicable_action_mask=state.get_applicable_action_mask(), )
