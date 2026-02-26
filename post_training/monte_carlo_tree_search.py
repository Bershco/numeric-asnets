"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
import gc
import logging
from abc import ABC, abstractmethod
from array import array
import bisect
from collections import defaultdict
import math
from typing import Any, List, Optional, Iterator, Tuple
from time import time
import numpy as np
import rpyc
from rpyc import BaseNetref
from typing_extensions import Self
import tensorflow as tf

from asnets.state_reprs import CanonicalState
from asnets.utils.rpyc_utils import to_local


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """
    __slots__ = ()

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

class MCTSNode(Node):
    delete_counter = 0
    __slots__ = (
        # "state_id", "_hash",
        "state",
        "cost_until_now", "reward_weight",
        "previous_action",  "children", "parent",
        "goal_state", "terminal_state", "as_network_input",
        "applicable_action_mask", "act_dist", "pred_value", "Q_value"
    )

    def __init__(self,
                 # state_id,
                 state,
                 cost_until_now, previous_action, reward_weight = 1000,
        is_goal = False, is_terminal = False, as_network_input = None, applicable_action_mask = None,
                 # hashed_state = -1,
                 parent = None):
        # self.state_id = state_id
        # self._hash = hashed_state
        self.state = state
        self.cost_until_now = cost_until_now
        self.reward_weight = reward_weight
        self.previous_action = previous_action
        self.parent = parent
        self.children = None
        self.goal_state = is_goal
        self.terminal_state = is_terminal
        self.as_network_input = as_network_input
        self.applicable_action_mask = applicable_action_mask
        self.act_dist = None
        self.pred_value = None
        self.Q_value = 0

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
        # return 1 if self.is_terminal() else 0
        if self.is_goal():
            return self.reward_weight / self.cost_until_now
        return 0

    def to_network_input(self):
        """Make the cstate represented by 'this' MCTSNode to be compatible for the policy network, and transposes it"""
        return self.as_network_input

    def __hash__(self):
        """Nodes must be hashable"""
        return hash(self.state)

    def __eq__(self, node2: Self) -> bool:
        """
        Nodes must be comparable.
        That being said, employing rpyc interception, pickling, serialization etc. just for equality seems redundant.
        At least if their hashes already don't fit.
        """
        return hash(self) == hash(node2) and self.state == node2.state

    def get_identifiers(self) -> tuple[int,int]:
        # return self.state_id, self._hash
        raise NotImplementedError("Changed to using states in nodes instead of identifiers")

    def env_state_key(self) -> bytes:
        return self.state.env_state_key()

class FixedChildMap:
    def __init__(self, keys: List[int], values: List[Any]):
        assert len(keys) == len(values), "Keys and values must match in length"
        sorted_pairs = sorted(zip(keys, values))
        self._keys = array('H', (k for k, _ in sorted_pairs))   # unsigned short
        self._values = [v for _, v in sorted_pairs]

    def get(self, key: int, default: Optional[Any] = None) -> Optional[Any]:
        idx = bisect.bisect_left(self._keys, key)
        if idx < len(self._keys) and self._keys[idx] == key:
            return self._values[idx]
        return default

    def __getitem__(self, key: int) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __contains__(self, key: int) -> bool:
        return self.get(key) is not None

    def items(self) -> Iterator[Tuple[int, Any]]:
        return zip(self._keys, self._values)

    def keys(self) -> Iterator[int]:
        return iter(self._keys)

    def values(self) -> Iterator[Any]:
        return iter(self._values)

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
                 network = None,
                 problem_service = None,
                 debug_memory = False,
                 debug_time_mcts_iterations = False,
                 debug_comparison_exploration_exploitation = False):
        self.curr_tree_root: MCTSNode = None
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.Nsa = defaultdict(int)
        # self.children: dict[Node, Any] = dict()  # actions and children output of each node. structure is (action,result_state)
        self.exploration_weight = exploration_weight
        self.path_until_goal = None
        self.state_to_node: dict[CanonicalState,MCTSNode] = {}
        # self.act_dist_per_node: dict[MCTSNode,np.ndarray] = {}
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
        # reward = 1 / (1 + self._evaluate_node(leaf))
        reward = self._evaluate_node(leaf)
        # numbers might be too low or insignificant?? I think it would be okay...
        # theoretically and practically it SHOULD not be lower than 1/10001 which isn't that low.
        self._backpropagate(path,reward)

    def _select(self, node: "MCTSNode"):
        """Find an unexplored descendent of `node` (returns path including final leaf or frontier child)."""
        if self.debug_time_mcts_iterations:
            self.start_times.append(time())
        node_path = []
        while True:
            node_path.append(node)

            childmap: FixedChildMap | None = node.children

            # If node has no generated children (i.e. node not in children dict, or is in dict with None value) -
            # it's unexplored or terminal.
            if childmap is None or childmap.is_empty():
                if self.debug_time_mcts_iterations:
                    self.after_selection_times.append(time())
                return node_path

            # Otherwise pick via PUCT (with no-cycle)
            a_next, n_next = self._puct_select_no_cycle(node, set(node_path))
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
        # Q_local = self.Q
        for node in reversed(path):
            n = N_local[node] + 1
            N_local[node] = n
            # q = Q_local.get(node, 0.0)
            q = self.get_value_from_mcts_node(node)
            # Q_local[node] = q + (reward - q) / n  # running average
            node.Q_value = q + (reward - q) / n # running average
        if self.debug_time_mcts_iterations:
            self.end_times.append(time())

    def _evaluate_node(self, node: MCTSNode) -> float:
        """Use the teacher's (or another) heuristic to evaluate a specific node, in order to use value-based mcts"""
        # value = self.problem_service.get_state_h(*node.get_identifiers())
        value = self.get_value_from_mcts_node(node)
        if self.debug_time_mcts_iterations:
            self.after_eval_times.append(time())
        return value

    def _puct_select_no_cycle(self, node, path_set):
        """Sample a child of `node` using PUCT scores as softmax logits while avoiding cycles.
           Returns (action, child_node).  Vectorized over children to reduce Python overhead."""

        # 0) Sanity: children should already be generated for this node
        children_map = node.children
        actions_nodes = list(children_map.items())  # [(a, child), ...]
        n_children = len(actions_nodes)
        assert n_children > 0, "PUCT select called on a node with no children"

        # 1) Priors from policy (vector over action indices)
        priors = self.get_act_dist_from_mcts_node(node)
        priors = priors.numpy() if hasattr(priors, "numpy") else priors

        # Build parallel arrays once
        actions, child_list = zip(*actions_nodes)

        # 2) Masks (vectorized)
        #   - cycle mask: child already on current path => invalidate
        actions = np.frombuffer(np.array(actions, dtype=np.int32), dtype=np.int32)
        path_keys = {n.env_state_key() for n in path_set}
        cycle = np.array([child.env_state_key() in path_keys for child in child_list])
        if cycle.any():
            child_arr = np.array(child_list)
            culprits = child_arr[cycle]
            culprit_nodes = [child for child in culprits]
            for node in culprit_nodes:
                LOGGER.debug(f"Cycle found, {node} is present in {path_set}")
        # 3) Prior lookup (vectorized, with bounds check)
        prior = np.zeros(n_children, dtype=np.float32)
        if np.ndim(priors) == 1 and priors.size > 0:
            valid = (actions >= 0) & (actions < priors.size)
            if valid.any():
                prior[valid] = np.asarray(priors, dtype=np.float32)[actions[valid]]

        # 4) Edge visits Nsa(s,a) and child Q(s') in one sweep
        edge_visits = np.array(
            [self.Nsa.get((node, int(a)), 0) for a in actions],
            dtype=np.int32
        )
        Q_child = np.array(
            # [self.Q.get(c, 0.0) for c in child_list],
            [child.Q_value for child in child_list],
            dtype=np.float32
        )

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

        a, child = actions[int(idx)], child_list[int(idx)]

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
        if self._probe:
            self._probe.log_select_softmax(
                prior_entropy=prior_entropy,
                chosen_p=chosen_p,
                edge_visits_chosen=edge_visits_chosen,
                cycle_present=cycle_present
            )

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
            # assert mcts_node in self.children[output_path[-1][1]].values()
            assert output_path[-1][1].children is not None
            assert mcts_node in output_path[-1][1].children.values()
            # for action, next_node in self.children[output_path[-1][1]].items():
            for action, next_node in output_path[-1][1].children.items():
                if mcts_node == next_node:
                    output_path.append((action, mcts_node))
        return output_path[1:]

    def get_act_dist_from_mcts_node(self, node: MCTSNode):
        if node.act_dist is None:
            if node.as_network_input is None:
                node.as_network_input = self.problem_service.to_network_input(*node.get_identifiers())
            if self.policy_only:
                node.act_dist = self.network(node.as_network_input)
            else:
                node.act_dist, value_tensor = self.network(node.as_network_input)
                node.pred_value = float(value_tensor.numpy().squeeze())
        return tf.squeeze(node.act_dist)

    def get_value_from_mcts_node(self, node: MCTSNode):
        if node.goal_state:
            return 1
        if self.policy_only:
            # Heuristic value (received by get_state_h) would be the distance to goal,
            # reward \ value for a node\state would be 'how good it is' - hence the inverse
            return 1 / (1 + self.problem_service.get_state_h(node.state_id,hash(node)))
        else:
            if node.pred_value is None:
                if node.as_network_input is None:
                    node.as_network_input = self.problem_service.to_network_input(*node.get_identifiers())
                node.act_dist, value_tensor = self.network(node.as_network_input)
                node.pred_value = float(value_tensor.numpy().squeeze())
            return node.pred_value

    def _delete_subtree(self, node, recursive=True):
        # Recursively delete the subtree rooted at this node
        if recursive:
            # for _, child in self.children.get(node, {}).items():
            if node.children is not None:
                for _, child in node.children.items():
                    self._delete_subtree(child)
        # self.children.pop(node, None)
        node.children = None
        self.N.pop(node, None)
        # self.Q.pop(node, None)
        # self.state_to_node.pop(node.state_id, None)
        self.state_to_node.pop(node.state)
        # self.act_dist_per_node.pop(node, None)

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
        # children_dict = self.children.get(parent_node)
        children_dict = parent_node.children
        if children_dict is None:
            return
        keep_child = None
        if self.debug_memory:
            self.log_node_count("Before deleting old root's irrelevant children")
        for action, child_node in list(children_dict.items()):
            if action == keep_action:
                keep_child = child_node
                continue
            self._delete_subtree(child_node)
        if self.debug_memory:
            self.log_node_count("After deleting old root's irrelevant children")
        assert keep_child is not None
        # Replace children dict with just the one we kept
        # self.children[parent_node] = FixedChildMap([keep_action], [keep_child])
        parent_node.children = FixedChildMap([keep_action],[keep_child])

    def get_applicable_action_mask(self, node: MCTSNode):
        if node.applicable_action_mask is None:  # Fallback
            node.applicable_action_mask = self.problem_service.get_applicable_action_mask(*node.get_identifiers())
        return node.applicable_action_mask

# def wrapInMCTSNode(cstate_id: int, previous_action, cost_until_now=float('inf'), is_goal=False,
#                    is_terminal=False, as_network_input=None, applicable_action_mask=None, hashed_state = -1, parent = None):
#     return MCTSNode(state_id=cstate_id, cost_until_now=cost_until_now, previous_action=previous_action,is_goal=is_goal,
#                     is_terminal=is_terminal, as_network_input=as_network_input,
#                     applicable_action_mask=applicable_action_mask, hashed_state=hashed_state, parent=parent)

def wrapInMCTSNode(state: CanonicalState, previous_action, cost_until_now=float('inf'),
                   as_network_input=None, applicable_action_mask=None, parent=None):
    return MCTSNode(state=state, previous_action=previous_action, cost_until_now=cost_until_now, is_goal=state.is_goal,
                    is_terminal=state.is_terminal, as_network_input=state.to_network_input(),
                    applicable_action_mask=state.get_applicable_action_mask(), parent=parent,)
