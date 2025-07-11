#!/usr/bin/env python3

import argparse
import atexit
from copy import deepcopy
from json import dump
import logging
from logging import Logger
from os import makedirs, path
from pathlib import Path
import random
import signal
import sys
from time import time
from pympler import muppy, summary, asizeof
from array import array
import bisect
from typing import Any, Iterator, List, Optional, Tuple, Set
import numpy
from pympler.asizeof import asized
import gc

from rpyc import BaseNetref

from asnets.interfaces.enhsp_interface import ENHSP_CONFIGS


from tensorflow.python.ops.gen_nn_ops import top_k

from asnets.prob_dom_meta import DomainType
from asnets.state_reprs import CanonicalState, sample_next_state
from collections import defaultdict

import joblib
import numpy as np
import rpyc
import tensorflow as tf
# for some reason "import tensorflow.python.debug" doesn't work (maybe it's a
# class or something?)
from tensorflow.python import debug as tfdbg
# import tqdm
import tqdm.auto as tqdm

from asnets.explorer import StaticExplorer, DynamicExplorer
from asnets.interfaces.enhsp_interface import ENHSP_CONFIGS
from asnets.models import PropNetworkWeights, PropNetwork
from asnets.supervised import SupervisedTrainer, SupervisedObjective, \
    ProblemServiceConfig, make_problem_service
from asnets.multiprob import ProblemServer, to_local, parent_death_pact
from asnets.utils.prof_utils import can_profile
from asnets.utils.py_utils import set_random_seeds

# from post_training.enhspforhwrapper import ENHSPForHWrapper

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    stream=sys.stdout
)

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


class CachingPolicyEvaluator(object):
    """Can be used to ensure that we evaluate policy only once for each state
    encountered at test time."""

    def __init__(self, policy, det_sample):
        self.det_sample = det_sample
        self.policy = policy
        self.cache = {}
        self._misses = 0
        self._hits = 0

    def get_action(self, obs):
        assert obs.ndim == 1
        obs_key = obs.tobytes()
        if obs_key in self.cache:
            self._hits += 1
            act_dist = self.cache[obs_key]
        else:
            self._misses += 1
            in_obs = obs[None, :]
            act_dist = self.policy(in_obs, training=False)
            self.cache[obs_key] = act_dist
        # we cache action *distribution* instead of action so that we can draw
        # a different random sample each time (caching should be transparent!)
        if self.det_sample:
            action = int(np.argmax(act_dist))
        else:
            num_actions = act_dist.shape[-1]
            act_indices = np.arange(num_actions)
            action = int(np.random.choice(act_indices, p=act_dist))
        return action

    def get_action_from_cstate(self, cstate):
        return self.get_action(cstate.to_network_input())


from post_training.monte_carlo_tree_search import Node
class MCTSNode(Node):
    delete_counter = 0
    __slots__ = ("state", "cost_until_now", "reward_weight", "previous_action")

    def __init__(self, state, cost_until_now, previous_action, reward_weight = 1000):
        self.state = to_local(state)
        self.cost_until_now = cost_until_now
        self.reward_weight = reward_weight
        self.previous_action = previous_action

    def simulate_step(self, action_id, problem_service):
        return problem_service.env_simulate_step(self.state, int(action_id))

    def is_terminal(self):
        """Returns True if the node has no children"""
        return self.state.exposed_is_terminal()

    def is_goal(self):
        """Return True if the current not is a goal"""
        return self.state.exposed_is_goal()

    def reward(self):
        # return 1 if self.is_terminal() else 0
        if self.is_goal():
            return self.reward_weight / self.cost_until_now
        return 0

    def to_network_input(self):
        """Make the cstate represented by 'this' MCTSNode to be compatible for the policy network, and transposes it"""
        return self.state.to_network_input()[None, :]

    def is_applicable_action(self, action_num):
        _, applicable = self.state.acts_enabled[action_num]
        return applicable

    def __hash__(self):
        """Nodes must be hashable"""
        return self.state.__hash__()

    def __eq__(self, node2):
        """Nodes must be comparable"""
        return self.state.__eq__(node2.state)

    def __repr__(self):
        return self.state.__repr__()

    def __del__(self):
        MCTSNode.delete_counter += 1
        if MCTSNode.delete_counter % 100 == 0:
            print(f"Deleted {MCTSNode.delete_counter} MCTSNodes - and counting!")

def wrapInMCTSNode(inner_node: CanonicalState, previous_action, cost_until_now=float('inf')):
    return MCTSNode(state=inner_node, cost_until_now=cost_until_now, previous_action=previous_action)

from post_training.monte_carlo_tree_search import MCTS
class MonteCarloPolicyEvaluator(MCTS):

    def sanitize_node(self, node):
        """Deepcopy and strip aux_data from CanonicalState"""
        try:
            node_copy = deepcopy(node)
            if hasattr(node_copy, "state") and hasattr(node_copy.state, "_aux_data"):
                node_copy.state._aux_data = None
            return node_copy
        except Exception as e:
            print(f"Error copying/sanitizing node: {e}")
            return None

    def profile_children_dict(self, sample_size=20):
        total_size = 0
        count = 0

        for parent, child_dict in self.children.items():
            if count >= sample_size:
                break
            try:
                parent_clean = self.sanitize_node(parent)
                if parent_clean is None:
                    continue

                # Clean each child node
                cleaned_child_dict = {}
                for action_id, child_node in child_dict.items():
                    child_clean = self.sanitize_node(child_node)
                    if child_clean is not None:
                        cleaned_child_dict[action_id] = child_clean

                # Bundle together for sizing
                structure = (parent_clean, cleaned_child_dict)
                total_size += asized(structure).size
                count += 1

            except Exception as e:
                print(f"Failed at sample {count}: {e}")
                continue

        estimated_total = total_size * len(self.children) / count if count else 0
        print(f"Estimated total memory for all children: {estimated_total / 1024 ** 2:.2f} MB")

    def profile_state_to_node(self):
        total = 0
        for i, node in enumerate(self.state_to_node.values()):
            try:
                node_copy = deepcopy(node)
                node_copy.state._aux_data = None
                total += asized(node_copy).size
            except:
                continue
            if i >= 20:
                break
        estimated_total = total * len(self.state_to_node) / 20
        print(f"Estimated total memory for all nodes in state_to_node dictionary: {estimated_total / 1024 ** 2:.2f} MB")

    def print_memory_summary(self):
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1, limit=3)
        self.profile_state_to_node()
        self.profile_children_dict()
        self.safe_asizeof(self.visited_cstates_hashes, name="visited_cstates_hashes")

    def safe_asizeof(self, obj, name):
        try:
            size = asizeof.asizeof(obj)
            print(f"Size of {name}: {size / 1024 ** 2:.6f} MB")
        except Exception as e:
            print(f"Error sizing {name}: {e}")

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

    def __init__(self, policy, problem_service, horizon=0, exploration_weight=1, iterations=10, seed=42,
                 num_cstates_to_expand=5, use_value_based=False):
        super().__init__(exploration_weight)
        self.policy = policy
        self.problem_service = problem_service
        self.iterations = iterations
        self.horizon = horizon
        self.seed = seed
        self.k = num_cstates_to_expand
        self.curr_tree_root = None
        self.debug_orig_root = None
        self.state_to_node = {}     #This might benefit memory-wise from being 'state_hash_to_node' dict instead
        self.visited_cstates_hashes: Set[int] = set()
        self.revisit_counter = 0
        self.act_dist_per_node: dict[MCTSNode,numpy.ndarray] = {}
        self.use_value_based=use_value_based

    def get_action(self, obs):
        raise Exception("Sorry, wrong usage in code, try using get_action_from_cstate instead.")

    def get_action_from_cstate(self, cstate, cost): #cstate is non-terminal
        if self.curr_tree_root is None:
            self.curr_tree_root = wrapInMCTSNode(cstate, cost_until_now=0, previous_action=None)
            self.debug_orig_root = self.curr_tree_root
            self.visited_cstates_hashes.add(self.curr_tree_root.__hash__())
        if self.use_value_based:
            for _ in range(self.iterations):
                self.mcts_iteration_value_based(self.curr_tree_root)
        else:
            for _ in range(self.iterations):
                if self.path_until_goal is None:
                    self.mcts_iteration_classic(self.curr_tree_root, self.horizon)
            if self.path_until_goal is not None:
                next_action, next_mcts_node = self.path_until_goal[0]
                self.path_until_goal = self.path_until_goal[1:]
                if self.state_to_node[cstate] not in self.children:
                    self.children[self.state_to_node[cstate]] = dict()
                self.children[self.state_to_node[cstate]][next_action] = next_mcts_node
                self.state_to_node[next_mcts_node.state] = next_mcts_node
                return next_action

        def node_ranking(node):
            if self.N[node] == 0:
                return float("-inf")
            return self.Q[node] / self.N[node]

        def custom_tiebreaker(node):
            return self.N[node]

        best_action, best_node = max(
            self.children[self.curr_tree_root].items(),
            key=lambda item: (node_ranking(item[1]), custom_tiebreaker(item[1]))
        )
        print(f'[get_action_from_cstate] - chosen action: {best_action}')
        self.visited_cstates_hashes.add(best_node.__hash__())
        self.print_memory_summary()
        return best_action

    def progress_to(self, action_id, cstate, cost):
        next_node = self.get_corresponding_mcts_node(cstate)
        assert next_node in self.children[self.curr_tree_root].values(), \
            f"Assertion failed: next_node ({next_node}) is not one of current root's children"
        assert next_node == self.children[self.curr_tree_root][action_id], \
            f"Assertion failed: next_node ({next_node}) != expected ({self.children[self.curr_tree_root][action_id]})"
        self.prune_children_except(self.curr_tree_root, action_id)
        if next_node is None:
            logging.getLogger(__name__).info('Next node is not available, creating a new tree.')
            self.curr_tree_root = wrapInMCTSNode(cstate, cost_until_now=cost,previous_action=action_id)
        else:
            _temp = self.curr_tree_root
            self.curr_tree_root = next_node
            #This explicit 'recursive=False' means that only the node would be properly deleted, subtree left as-is
            self._delete_subtree(_temp, recursive=False)
            # logging.getLogger(__name__).info(f'Next node is available, it has been visited %s times.', self.N[self.curr_tree_root])

    def get_corresponding_mcts_node(self, cstate):
        return self.state_to_node.get(cstate, None)

    def _expand(self, node):
        if node in self.children:
            return
        self.children[node] = self.find_children(node)
        self.state_to_node[node.state] = node
        for child_node in self.children[node].values():
            assert isinstance(child_node, MCTSNode)
            self.state_to_node[child_node.state] = child_node

    def _rollout(self, mcts_node, horizon=10):
        """Returns the reward for a random simulation (to a certain horizon) of `node`"""
        action_following_state_path = []
        for _ in range(horizon):
            if mcts_node.is_goal():
                print("\n\n============================================\nGoal was found during rollout\n============================================\n")
                action_path = []
                curr_mcts_node = self.curr_tree_root
                for action_from_path, mcts_node_from_path in action_following_state_path:
                    if curr_mcts_node not in self.children:
                        self.children[curr_mcts_node] = dict()
                    self.children[curr_mcts_node][action_from_path] = mcts_node_from_path
                    self.state_to_node[curr_mcts_node.state] = curr_mcts_node
                    curr_mcts_node = mcts_node_from_path
                    action_path.append(action_from_path)
                print(f"Next actions are: {action_path}")
                self.path_until_goal = action_following_state_path
                break
            best_action, mcts_node = self.find_child_by_policy(mcts_node)
            action_following_state_path.append((best_action, mcts_node))
        return mcts_node.reward()

    def _evaluate_node(self, node):
        """Use the teacher's (or another) heuristic to evaluate a specific node, in order to use value-based mcts"""
        return self.problem_service.get_state_h_value(cstate=node.state)

    def prune_children_except(self, parent_node, keep_action):
        print(f"Pruning parent_node: {hash(parent_node)} of all children except:")
        children_dict = self.children.get(parent_node)
        if children_dict is None:
            return
        keep_child = None
        self.log_node_count("Before deleting old root's irrelevant children")
        for action, child_node in list(children_dict.items()):
            if action == keep_action:
                print(f"The chosen child_node: {hash(child_node)}")
                keep_child = child_node
                continue
            self._delete_subtree(child_node)
        self.log_node_count("After deleting old root's irrelevant children")

        assert keep_child is not None
        # Replace children dict with just the one we kept
        self.children[parent_node] = FixedChildMap([keep_action],[keep_child])

    def _delete_subtree(self, node, recursive=True):
        # Recursively delete the subtree rooted at this node
        if recursive:
            for _, child in self.children.get(node, {}).items():
                self._delete_subtree(child)
        self.children.pop(node, None)
        self.N.pop(node, None)
        self.Q.pop(node, None)
        self.state_to_node.pop(node.state, None)
        self.act_dist_per_node.pop(node, None)

    def find_children(self, parent_node: MCTSNode):
        """Find up to k successors of parent_node that are applicable and not yet visited"""
        act_dist = self.get_act_dist_from_mcts_node(parent_node).numpy()
        mask = [parent_node.is_applicable_action(i) for i in range(len(act_dist))]
        # Rank actions by descending policy probability
        sorted_indices = sorted(range(len(act_dist)), key=lambda i: act_dist[i], reverse=True)
        # output = dict()
        keys = []
        values = []
        selected = 0
        for i in sorted_indices:
            if selected >= self.k:
                break
            if not mask[i] or act_dist[i] == 0.0:
                continue
            if self.problem_service is None:
                raise RuntimeError("problem_service is None — was it shut down?")
            # Simulate step only now (expensive!)
            cstate_after_action_i, step_cost = parent_node.simulate_step(i, self.problem_service)
            if hash(cstate_after_action_i) in self.visited_cstates_hashes:
                self.revisit_counter += 1
                if self.revisit_counter % 100 == 0:
                    print(f"========>>There has been {self.revisit_counter} re-visitations in canonical states so far.")
                continue  # skip visited cstates
            wrapped_output_cstate = wrapInMCTSNode(
                cstate_after_action_i,
                cost_until_now=parent_node.cost_until_now + step_cost,
                previous_action=i
            )
            # output[i] = wrapped_output_cstate
            keys.append(i)
            values.append(wrapped_output_cstate)
            selected += 1
        # return output
        return FixedChildMap(keys, values)

    def find_child_by_policy(self, parent_node: MCTSNode):
        """Random successor of this board state (for more efficient simulation)"""
        # input_format_cstate = self.to_network_input()                                                                 # negligible
        # act_dist = parent_node.get_act_dist_from_policy(self.policy)
        act_dist = self.get_act_dist_from_mcts_node(parent_node)
        # act_dist is a vector of shape (1,n) of distribution of action possibilities
        # next_action_ind = np.argmax(act_dist[0]) - this would have just generated a single trajectory from the select-
        # -ed MCTSNode, changing this to the distribution that is given by the policy network.
        mask = tf.convert_to_tensor([parent_node.is_applicable_action(i) for i in range(act_dist.shape[0])], dtype=tf.bool)    # tf - negligible
        masked_act_dist = tf.where(mask, act_dist, tf.zeros_like(act_dist))                                             # tf - negligible
        total = tf.reduce_sum(masked_act_dist)                                                                          # tf - negligible
        normalized_act_dist = tf.cond(                                                                                  # tf - negligible
            tf.greater(total,0),
            lambda: masked_act_dist / total,
            lambda: tf.zeros_like(act_dist)
        )
        norm_act_dist_np = normalized_act_dist.numpy()                                                                  # negligible
        np.random.seed(self.seed)
        next_action_ind = np.random.choice(len(norm_act_dist_np), p=norm_act_dist_np)                                   # negligible
        if self.problem_service is None:
            raise RuntimeError("problem_service is None — was it shut down?")
        best_cstate, step_cost = parent_node.simulate_step(next_action_ind, self.problem_service)
        return next_action_ind, wrapInMCTSNode(best_cstate, cost_until_now=parent_node.cost_until_now + step_cost,
                                               previous_action=next_action_ind)                                         # around 6.9% of function time is wrapInMCTSNode

    def get_act_dist_from_mcts_node(self, node: MCTSNode):
        act_dist = self.act_dist_per_node.get(node)
        if act_dist is None:
            node_as_network_input = node.to_network_input()
            act_dist = self.policy(node_as_network_input)
            self.act_dist_per_node[node] = act_dist
        return tf.squeeze(act_dist)


@can_profile
def run_trial(policy_evaluator, problem_server, limit=1000, det_sample=False, graceful_timeout=300):
    """Run policy on problem. Returns (cost, path), where cost may be None if
    goal not reached before horizon."""
    print(f'\n-------------> Graceful_timeout is set to {graceful_timeout}\n')
    print(f'\n-------------> Limit is set to {limit}\n')
    trial_start_time = time()
    problem_service = problem_server.service
    curr_cstate = to_local(problem_service.env_reset())
    # total cost of this run
    cost = 0
    path = []
    for i in range(1, limit):
        if time() - trial_start_time > graceful_timeout:
            print('Graceful_timeout has been reached :)')
            break
        action = policy_evaluator.get_action_from_cstate(curr_cstate, cost)
        curr_cstate, step_cost = to_local(problem_service.env_step(action))
        policy_evaluator.progress_to(action, curr_cstate, cost+step_cost)
        path.append(to_local(problem_service.action_name(action)))
        cost += step_cost
        if curr_cstate.is_goal:
            # path.append('GOAL! :D')
            return cost, True, path
        # we can run out of time or run out of actions to take
        if curr_cstate.is_terminal:
            break
        if i == limit-1:
            print(" I actually reached the end, something weird is happening, only some actions were chosen but limit was reached? ")
    # path.append('FAIL! D:')
    return cost, False, path


def run_trials(policy, problem_server, trials, iterations, horizon=None, limit=1000, det_sample=False,
               single_trial_graceful_timeout_sec=300, seed=42, num_cstates_to_expand=5, use_value_based=False, state_value_heuristic=None):
    # policy_evaluator = CachingPolicyEvaluator(policy=policy, det_sample=det_sample)
    policy_evaluator = MonteCarloPolicyEvaluator(policy=policy, problem_service=problem_server.service,
                                                 iterations=iterations, horizon=horizon, seed=seed,
                                                 num_cstates_to_expand=num_cstates_to_expand,
                                                 use_value_based=use_value_based,
                                                 )
    all_exec_times = []
    all_costs = []
    all_goal_reached = []
    paths = []
    print(f'\n-------------> MCTS iterations number: {iterations}\n')
    print(f'\n-------------> MCTS rollout horizon length: {horizon}\n')
    for _ in tqdm.trange(trials, desc='trials', leave=True):
        start = time()
        cost, goal_reached, path = run_trial(policy_evaluator, problem_server,
                                             limit, det_sample,graceful_timeout=single_trial_graceful_timeout_sec)
        elapsed = time() - start
        paths.append(path)
        all_exec_times.append(elapsed)
        all_costs.append(cost)
        all_goal_reached.append(goal_reached)
        print("%d trials of length %d took %fs" % (trials, limit, elapsed))

    meta_dict = {
        'turn_limit': limit,
        'trials': trials,
        'all_goal_reached': all_goal_reached,
        'all_exec_times': all_exec_times,
        'all_costs': all_costs,
    }
    return meta_dict, paths


def unique_name(args, digits=6):
    rand_num = random.randint(1, (1 << (4 * (digits + 1)) - 1))
    suffix = '{:x}'.format(rand_num).zfill(digits)
    if args.timeout is None:
        time_str = 'inf'
    else:
        time_str = '%d' % round(args.timeout)
    mo_str = ','.join('%s=%s' % (k, v) for k, v in args.model_opts.items())
    if args.problems:
        all_probs_comma = ','.join(args.problems)
        if len(all_probs_comma) > 50:
            all_probs_comma = all_probs_comma[:47] + '...'
        start = 'P[{}]'.format(all_probs_comma)
    else:
        names = []
        for pf in args.pddls:
            # remove directory path
            bn = path.basename(pf)
            pf_suffix = '.pddl'
            if bn.endswith(pf_suffix):
                # chop off extension
                bn = bn[:-len(pf_suffix)]
            if bn:
                names.append(bn)
        all_names_comma = ','.join(names)
        if len(all_names_comma) > 50:
            all_names_comma = all_names_comma[:47] + '...'
        start = 'P[%s]' % all_names_comma

    teacher_config_str = ''
    if args.teacher_planner == 'ssipp':
        teacher_config_str = args.ssipp_teacher_heuristic
    elif args.teacher_planner == 'fd':
        teacher_config_str = args.fd_teacher_heuristic
    elif args.teacher_planner == 'enhsp':
        teacher_config_str = f'enhsp-{args.enhsp_config}'

    prefix = '{}-S[{},{},{}]-MO[{}]-T[{}]'.format(
        start, args.supervised_lr, args.supervised_bs, teacher_config_str,
        mo_str, time_str)
    start_time_str = str(int(time() / 60 - 24881866)).zfill(8)
    return prefix + '-' + start_time_str + '-' + suffix


def opt_str(in_str):
    rv = {}
    for item in in_str.split(','):
        item = item.strip()
        if not item:
            continue
        name, value = item.split('=', 1)
        rv[name] = value
    return rv


def sup_objective_str(in_str):
    return SupervisedObjective[in_str]


def int_or_float(arg_str):
    """Convert string to non-negative integer (preferred) or float."""
    if arg_str.isnumeric():
        return int(arg_str)
    try:
        result = float(arg_str)
        if result < 0:
            raise ValueError("value can't be negative")
        return result
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Could not convert argument '%s' to non-negative int or float" %
            (arg_str,))


parser = argparse.ArgumentParser(description='Trainer for ASNets')
parser.add_argument(
    '-p',
    '--problem',
    dest='problems',
    action='append',
    help='name of problem to solve (can use this flag many times)')
parser.add_argument(
    '--domain-type',
    type=DomainType.argparse,
    choices=list(DomainType),
    help='type of the domain'
)
parser.add_argument(
    '--opt-patience',
    type=int,
    default=10,
    help="if best observed undiscounted mean reward is >=1, *and* there has "
         "been no improvement for this many epochs, then we stop.")
parser.add_argument(
    '--max-opt-epochs',
    type=int,
    default=100,
    help="absolute maximum number of epochs to do optimisation for")
parser.add_argument(
    '--supervised-lr',
    type=float,
    default=0.0005,
    help='learning rate for supervised learning')
parser.add_argument(
    '--lr-step',
    nargs=2,
    action='append',
    type=int_or_float,
    default=[],
    dest='lr_steps',
    help='specifying "k r" will step down to LR `r` after `k` epochs (can be '
         'given multiple times)')
parser.add_argument(
    '--supervised-bs',
    type=int,
    default=128,
    help='batch size for supervised learning')
parser.add_argument(
    '--ssipp-teacher-heuristic',
    default='lm-cut',
    choices=['lm-cut', 'h-add', 'h-max', 'simpleZero', 'smartZero'],
    help='heuristic to use for SSiPP teacher in supervised mode')
parser.add_argument(
    '--fd-teacher-heuristic',
    default='astar-hadd',
    choices=['astar-hadd', 'lama-2011', 'lama-first',
             'lama-w5', 'lama-w3', 'lama-w2', 'lama-w1',
             'astar-lmcut', 'astar-lmcount', 'astar-hadd',
             'gbf-lmcut', 'gbf-hadd'],
    help='heuristic to use for fd teacher in supervised mode')
parser.add_argument(
    '--enhsp-config',
    default='hadd-gbfs',
    choices=ENHSP_CONFIGS.keys(),
    help='configuration to use for ENHSP'
)
parser.add_argument(
    '--supervised-early-stop',
    type=int,
    default=12,
    help='halt after this many epochs with succ. rate >0.8 & no increase (0 '
         'disables)')
parser.add_argument(
    '--save-every',
    type=int,
    default=0,
    metavar='N',
    help='save models every N epochs, in addition to normal saves for best '
         'success rate')
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help='base random seed to use for main proc & subprocs')
parser.add_argument(
    '-A',
    '--optimiser-opts',
    default={},
    type=opt_str,
    help='additional arguments for optimiser')
parser.add_argument(
    '--resume-from', default=None, help='snapshot pickle to resume from')
parser.add_argument(
    '-t',
    '--timeout',
    type=float,
    default=None,
    help='maximum training time (disabled by default)')
parser.add_argument(
    '-O',
    '--model-opts',
    type=opt_str,
    default={},
    help='options for model (e.g. p1=v1,p2=v2,p3=v3)')
parser.add_argument(
    '--no-skip',
    action='store_false',
    dest='skip',
    help='disable skip connections')
parser.add_argument(
    '--num-layers', type=int, default=2, help='number of layers for network')
parser.add_argument(
    '--hidden-size',
    type=int,
    default=16,
    help='hidden size of latent representation')
parser.add_argument(
    '--dropout',
    type=int_or_float,
    default=0.0,
    help='enable dropout during both learning & rollouts')
parser.add_argument(
    '--sup-objective',
    type=sup_objective_str,
    default=SupervisedObjective.ANY_GOOD_ACTION,
    help='objective for supervised training (choices: %s)' % ', '.join(
        [obj.name for obj in SupervisedObjective]))
parser.add_argument(
    '--no-use-teacher-envelope',
    dest='use_teacher_envelope',
    default=True,
    action='store_false',
    help='disable pulling entire envelope of teacher policy '
         'into experience buffer each time ASNet visits a state, '
         'and instead pull in just a single rollout under the '
         'teacher policy')
parser.add_argument(
    '--det-eval',
    action='store_true',
    default=False,
    help='use deterministic action selection for evaluation')
parser.add_argument(
    '--ssipp-dg-heuristic',
    type=str,
    default=None,
    help='SSiPP heuristic to give to ASNet')
parser.add_argument(
    '--minimal-file-saves',
    default=False,
    action='store_true',
    help="don't create TB files, final snapshot, or other extraneous "
         "(and expensive) run info")
parser.add_argument(
    '--no-use-lm-cuts',
    dest='use_lm_cuts',
    default=True,
    action='store_false',
    help="don't add flags indicating which actions are in lm-cut cuts. On "
         "numeric domains, lm-cuts are produced by numeric relaxing the domain.")
parser.add_argument(
    '--use-numeric-landmarks',
    dest='use_numeric_landmarks',
    default=False,
    action='store_true',
    help='add flags indicating which actions are in numeric landmarks')
parser.add_argument(
    '--use-contributions',
    dest='use_contributions',
    default=False,
    action='store_true',
    help='use contributions for numeric landmarks')
parser.add_argument(
    '--use-act-history',
    default=False,
    action='store_true',
    help='add features for past execution count of each action')
parser.add_argument(
    '--save-training-set',
    default=None,
    help='save pickled training set to this file')
parser.add_argument(
    '--use-saved-training-set',
    default=None,
    help='instead of collecting experience, used this pickled training set '
         '(produced by --save-training-set)')
parser.add_argument(
    '-R', '--rounds-eval', type=int, default=100, help='number of eval rounds')
parser.add_argument(
    '-L', '--limit-turns', type=int, default=100, help='max turns per round')
parser.add_argument(
    '--training-limit-turns', type=int, default=50,
    help='max turns per round during training')
parser.add_argument(
    '-e', '--expt-dir', default=None, help='path to store experiments in')
parser.add_argument(
    '--dK', default='dk', help='prefix of the domain knowledge file'
)
parser.add_argument(
    '--debug',
    default=False,
    action='store_true',
    help='enable tensorflow debugger')
parser.add_argument(
    '--no-train',
    default=False,
    action='store_true',
    help="don't train, just evaluate")
parser.add_argument(
    '--l1-reg', type=float, default=0.0, help='l1 regulariser weight')
parser.add_argument(
    # start with token regulariser to ensure opt problem is bounded below
    '--l2-reg',
    type=float,
    default=1e-5,
    help='l2 regulariser weight')
parser.add_argument(
    # this encourages equations to go to zero completely unless they're
    # actually needed (ideally use this in conjunction with a larger --l1-reg)
    '--l1-l2-reg',
    type=float,
    default=0.0,
    help='l1-l2 (group sparse) regulariser weight')
parser.add_argument(
    '--teacher-planner',
    choices=('ssipp', 'fd', 'domain-specific', 'enhsp', 'metricff'),
    default='ssipp',
    help='choose between several different teacher planners')
parser.add_argument(
    '--opt-batch-per-epoch',
    default=1000,
    type=int,
    help='number of batches of optimisation per epoch')
parser.add_argument(
    '--net-debug',
    action='store_true',
    default=False,
    help='put in place additional assertions etc. to help debug network')
parser.add_argument(
    '--exploration-algorithm',
    choices=('static', 'dynamic'),
    default='static',
    help='The exploration algorithm to use. Static exploration is the '
         'original ASNets algorithm. Dynamic exploration is the algorithm '
         'proposed for numeric planning.')
parser.add_argument(
    '--rollouts',
    type=int,
    default=75,
    help='Number of rollouts per problem per epoch. For static exploration, '
         'this is the number of rollouts per problem. For dynamic exploration, '
         'this is the number of rollouts initially performed per problem.')
parser.add_argument(
    '--min-explored',
    type=int,
    default=10,
    help='Minimum number of new states to add per epoch. Only used for dynamic'
         ' exploration.')
parser.add_argument(
    '--max-explored',
    type=int,
    default=1000,
    help='Maximum number of new states to add per epoch. Only used for dynamic'
         ' exploration.')
parser.add_argument(
    '--exploration-learning-ratio',
    type=float,
    default=1,
    help='The ratio of time spent exploring to time spent learning. Only used'
         ' for dynamic exploration.')
parser.add_argument(
    '--max-replay-size',
    type=int,
    default=10000,
    help='Maximum size of the replay buffer. Only used for dynamic exploration')
parser.add_argument(
    '--teacher-timeout-s',
    type=int,
    # default is small b/c anything less than "nearly instant" is going to take
    # a lot of cumulative time
    default=10,
    help='teacher timeout, in seconds (must be >0; default 10)')
parser.add_argument(
    '--plan-file-name',
    default='plan_sas',
    help="plan output file name")
parser.add_argument(
    '--limit-train-obs-size',
    default=700,
    help="limit the problem size. If it is too big, skip the problem.")
parser.add_argument(
    '--use-fluents',
    action='store_true',
    default=False,
    help='include fluent modules in the network.')
parser.add_argument(
    '--use-comparisons',
    action='store_true',
    default=False,
    help='include comparison modules in the network.')
parser.add_argument(
    'pddls',
    nargs='+',
    help='paths to PDDL domain/problem definitions')
parser.add_argument(
    '--mcts-iterations',
    type=int,
    default=3,
    help='Number of nodes to select->expand->rollout->backpropagate.')
parser.add_argument(
    '--mcts-rollout-horizon',
    type=int,
    default=3,
    help='How far should the mcts rollout go for.')
parser.add_argument(
    '--graceful-timeout',
    type=int,
    default=3000000,
    help='Number of seconds to gracefully timeout after.')
parser.add_argument(
    '--random-seed',
    type=int,
    default=None,
    help='Seed.')
parser.add_argument(
    '--mcts-expansion-size',
    type=int,
    default=5,
    help='Number of MCTS Nodes to generate upon MCTS parent node expansion.')
parser.add_argument(
    '--no-eval',
    action='store_true',
    default=False,
    help='Disable evaluation after training.')
parser.add_argument(
    '--mcts-value-based',
    action='store_true',
    default=False,
    help='Use value-based mcts instead of rollout-based mcts.')
parser.add_argument(
    '--mcts-heuristic',
    choices=list(ENHSP_CONFIGS.keys()),
    default='hadd-gbfs',
    help='When value-based mcts runs, this would be the state-value heuristic function.')
parser.add_argument(
    '--mcts-heuristic-horizon',
    type=int,
    default=10,
    help='Upon using ENHSP as a value estimator - what should the horizon be.')


def eval_single(args, policy, problem_server, unique_prefix, elapsed_time,
                iter_num, weight_manager, scratch_dir):
    LOGGER = logging.getLogger(__name__)
    # now we evaluate the learned policy
    LOGGER.info('Evaluating policy')
    trial_results, paths = run_trials(
        policy,
        problem_server,
        args.rounds_eval,
        limit=args.limit_turns,
        det_sample=args.det_eval,
        iterations=args.mcts_iterations,
        horizon=args.mcts_rollout_horizon,
        single_trial_graceful_timeout_sec=args.graceful_timeout,
        seed=args.random_seed,
        num_cstates_to_expand=args.mcts_expansion_size,
        use_value_based=args.mcts_value_based,
    )

    # print('Trial results:')
    LOGGER.info('Trial results')
    # print('\n'.join('%s: %s' % (k, v) for k, v in trial_results.items()))
    LOGGER.info('\n'.join('%s: %s' % (k, v) for k, v in trial_results.items()))
    out_dict = {
        'no_train': args.no_train,
        'args_problems': args.problems,
        'problem': to_local(problem_server.service.get_current_problem_name()),
        'timeout': args.timeout,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'all_args': sys.argv[1:],
        # elapsed_* also includes time/iterations spent looking for better
        # results after converging
        'elapsed_opt_time': elapsed_time,
        'elapsed_opt_iters': iter_num,
        'trial_paths': paths
    }
    out_dict.update(trial_results)
    result_path = path.join(scratch_dir, 'results.json')
    with open(result_path, 'w') as fp:
        dump(out_dict, fp, indent=2)
    # also write out lists of actions taken during final trial
    actions_path = path.join(args.plan_file_name)
    for i, alist in enumerate(paths):
        if trial_results["all_goal_reached"][i]:
            with open(f'{actions_path}.{i}', 'w') as fp:
                fp.write('(')
                fp.write(')\n('.join(alist))
                fp.write(')')


@can_profile
def make_policy(args,
                obs_dim,
                act_dim,
                dom_meta,
                prob_meta,
                dg_extra_dim=None,
                weight_manager=None):
    # size of input and output
    obs_dim = int(obs_dim)
    act_dim = int(act_dim)

    # can make normal FC MLP or an action/proposition network
    hs = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout
    print('hidden_size: %d, num_layers: %d, dropout: %f' % (hs, num_layers,
                                                            dropout))
    if weight_manager is not None:
        print('Re-using same weight manager')
    elif args.resume_from:
        print('Reloading weight manager (resuming training)')
        resume_from_str = args.resume_from
        print(f'\n\n[model-loading] - Resuming from: {args.resume_from}\n\n')
        resume_from_str = resume_from_str.replace("\\",'/') # for Windows support, do not delete.
        resume_from_path_obj = Path(resume_from_str)
        resume_from_path_obj = resume_from_path_obj.resolve(strict=False)
        weight_manager = joblib.load(resume_from_path_obj)
    else:
        print('Creating new weight manager (not resuming)')
        # TODO: should save all network metadata with the network weights or
        # within a separate config class, INCLUDING heuristic configuration
        weight_manager = PropNetworkWeights(
            dom_meta,
            hidden_sizes=[(hs, hs)] * num_layers,
            # extra inputs to each action module from data generators
            extra_dim=dg_extra_dim,
            skip=args.skip,
            use_fluents=args.use_fluents,
            use_comparisons=args.use_comparisons)
    custom_network = PropNetwork(
        weight_manager, prob_meta, dropout=dropout, debug=args.net_debug)

    # weight_manager will sometimes be None
    return custom_network, weight_manager


def get_problem_names(pddl_files, domain_type, teacher_planner):
    """Return a list of problem names from some PDDL files by spooling up
    background process."""
    config = ProblemServiceConfig(
        pddl_files, None, domain_type=domain_type,
        teacher_planner=teacher_planner, random_seed=None)
    server = ProblemServer(config)
    try:
        server.service.initialise()
        names = to_local(server.service.get_problem_names())
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)
        names = [name.strip() for name in names]
    finally:
        server.stop()
    return names


class SingleProblem(object):
    """Wrapper to store all information relevant to training on a single
    problem."""

    def __init__(self, name, problem_server):
        self.name = name
        # need a handle to problem server so that it doesn't get GC'd (which
        # would kill the child process!)
        self.problem_server = problem_server
        self.problem_service = problem_server.service
        self.prob_meta, self.dom_meta = to_local(
            self.problem_service.get_meta())
        self.obs_dim = to_local(self.problem_service.get_obs_dim())
        self.act_dim = to_local(self.problem_service.get_act_dim())
        self.dg_extra_dim = to_local(self.problem_service.get_dg_extra_dim())
        # will get filled in later
        self.policy = None


@can_profile
def make_services(args):
    """Make a ProblemService for each relevant problem."""
    # first get names
    if not args.problems:
        print("No problem name given, will use all discovered problems")
        problem_names = get_problem_names(args.pddls, args.domain_type,
                                          args.teacher_planner)
    else:
        problem_names = args.problems
    print("Loading problems %s" % ', '.join(problem_names))

    # now get contexts for each problem and a manager for their weights
    servers = []

    def kill_servers():
        for server in servers:
            try:
                server.stop()
            except Exception as e:
                print("Got exception %r while trying to stop %r" % (e, server))

    atexit.register(kill_servers)

    only_one_good_action = args.sup_objective \
                           == SupervisedObjective.THERE_CAN_ONLY_BE_ONE
    async_calls = []
    for prob_id, problem_name in enumerate(problem_names, start=1):
        random_seed = None if args.seed is None \
            else args.seed + prob_id
        service_config = ProblemServiceConfig(
            args.pddls,
            problem_name,
            args.domain_type,
            random_seed=random_seed,
            ssipp_dg_heuristic=args.ssipp_dg_heuristic,
            use_lm_cuts=args.use_lm_cuts,
            use_numeric_landmarks=args.use_numeric_landmarks,
            use_contributions=args.use_contributions,
            use_act_history=args.use_act_history,
            fd_heuristic=args.fd_teacher_heuristic,
            ssipp_teacher_heuristic=args.ssipp_teacher_heuristic,
            enhsp_config=args.enhsp_config,
            teacher_planner=args.teacher_planner,
            teacher_timeout_s=args.teacher_timeout_s,
            only_one_good_action=only_one_good_action,
            use_teacher_envelope=args.use_teacher_envelope,
            max_len=args.training_limit_turns)
        problem_server = ProblemServer(service_config)
        servers.append(problem_server)
        # must call initialise()
        init_method = rpyc.async_(problem_server.service.initialise)
        init_method_2 = rpyc.async_(problem_server.service.initialise_estimator)
        async_calls.append(init_method())
        async_calls.append(init_method_2(enhsp_config=args.mcts_heuristic,horizon=args.mcts_heuristic_horizon))

    # wait for initialise() calls to finish
    for async_call in async_calls:
        async_call.wait()
        # this property lookup is necessary to trigger any exceptions that
        # might have occurred during init (.wait() will not throw exceptions
        # from the child process; it only throws an exception on timeout)
        async_call.value

    # do this as a separate loop so that we can wait for services to spool
    # up in background
    problems = []
    weight_manager = None
    for problem_name, problem_server in zip(problem_names, servers):
        problem = SingleProblem(problem_name, problem_server)

        if not args.no_train and \
                problem.obs_dim > int(args.limit_train_obs_size):
            print(
                f'Skipping {problem_name} for training because it has obs_dim {problem.obs_dim} > {args.limit_train_obs_size}')

        print('Setting up policy and weight manager for %s' % problem_name)
        problem.policy, weight_manager = make_policy(
            args,
            problem.obs_dim,
            problem.act_dim,
            problem.dom_meta,
            problem.prob_meta,
            problem.dg_extra_dim,
            weight_manager=weight_manager)
        problems.append(problem)

    return problems, weight_manager


@can_profile
def main_supervised(args, unique_prefix, snapshot_dir, scratch_dir):
    print('Training supervised')

    start_time = time()

    # configure network input
    CanonicalState.network_input_config(use_fluents=args.use_fluents,
                                        use_comparisons=args.use_comparisons)

    problems, weight_manager = make_services(args)

    # need to create FileWriter *after* creating the policy network itself, or
    # the network will not show up in TB (I assume that the `Graph` view is
    # just a snapshot of the global TF op graph at the time a given
    # `FileWriter` is instantiated)
    summary_path = path.join(scratch_dir, 'tensorboard')
    if args.minimal_file_saves:
        sample_writer = None
    else:
        sample_writer = tf.summary.create_file_writer(summary_path)

    if not args.no_train:
        print('Training supervised with strategy %r and heuristic %r' %
              (args.sup_objective, args.fd_teacher_heuristic))
        if args.exploration_algorithm == 'static':
            explorer = StaticExplorer(problems, args.rollouts)
        elif args.exploration_algorithm == 'dynamic':
            explorer = DynamicExplorer(
                problems,
                init_trajs_per_problem=args.rollouts,
                min_new_pairs=args.min_explored,
                max_new_pairs=args.max_explored,
                expl_learn_ratio=args.exploration_learning_ratio,
                max_replay_size=args.max_replay_size)
        else:
            raise ValueError(
                f'Unknown exploration algorithm: {args.exploration_algorithm}')

        sup_trainer = SupervisedTrainer(
            problems=problems,
            weight_manager=weight_manager,
            summary_writer=sample_writer,
            explorer=explorer,
            strategy=args.sup_objective,
            batch_size=args.supervised_bs,
            lr=args.supervised_lr,
            lr_steps=args.lr_steps,
            l1_reg_coeff=args.l1_reg,
            l2_reg_coeff=args.l2_reg,
            l1_l2_reg_coeff=args.l1_l2_reg,
            opt_batches_per_epoch=args.opt_batch_per_epoch,
            save_training_set=args.save_training_set,
            use_saved_training_set=args.use_saved_training_set,
            start_time=start_time,
            early_stop=args.supervised_early_stop,
            save_every=args.save_every,
            scratch_dir=scratch_dir,
            snapshot_dir=snapshot_dir,
            dk=args.dK,
        )
        best_rate, elapsed_time, iter_num = sup_trainer.train(
            max_epochs=args.max_opt_epochs)
    else:
        assert not args.dropout, \
            f"--no-train provided, but we have dropout of {args.dropout}?"
        # need to fill up stats values with garbage :P
        elapsed_time = iter_num = None
        # normally trainers do this
        # sess.run(tf.compat.v1.global_variables_initializer())

    if args.no_eval:
        return

    # evaluate
    if weight_manager is not None and not args.minimal_file_saves:
        weight_manager.save(path.join(snapshot_dir, 'snapshot_final.pkl'))
    for problem in tqdm.tqdm(problems, desc='Evaluation'):
        print('Solving %s' % problem.name)
        eval_single(args, problem.policy, problem.problem_server,
                    unique_prefix + '-' + problem.name, elapsed_time,
                    iter_num, weight_manager, scratch_dir)


def main():
    rpyc.core.protocol.DEFAULT_CONFIG.update({
        # this is required for rpyc to allow pickling
        'allow_pickle': True,
        # required for some large problems where get_action() (passed as
        # synchronous callback to child processes) can take a very long time
        # the first time it is called
        'sync_request_timeout': 1800,
    })

    # ALWAYS die when parent dies; useful when running under run_experiment
    # etc. (this should never outlive run_experiment!)
    parent_death_pact(signal.SIGKILL)

    args = parser.parse_args()
    LOGGER = logging.getLogger(__name__)
    LOGGER.info('Arguments are: %s', args)

    if args.seed is not None:
        set_random_seeds(args.seed)
    else:
        # here "defaults" probably just means seeding based on time (although
        # possibly each library might be a little different)
        print("No random seed provided; defaults will be used")

    unique_prefix = unique_name(args)
    print('Unique prefix:', unique_prefix)

    if args.minimal_file_saves:
        # --minimal-file-saves is mostly there to avoid writing out a
        # checkpoint & TB file for each evaluation run when doing *many*
        # evaluations, so it doesn't make much sense to specify it on training
        # runs, where checkpoints are always written anyway (they have to be!)
        assert args.no_train, \
            "--minimal-file-saves without --no-train is weird; is this a bug?"

    if args.expt_dir is None:
        args.expt_dir = 'experiment-results'
    scratch_dir = path.join(args.expt_dir, unique_prefix)
    makedirs(scratch_dir, exist_ok=True)

    # where to save models
    snapshot_dir = path.join(scratch_dir, 'snapshots')
    makedirs(snapshot_dir, exist_ok=True)
    print('Snapshot directory:', snapshot_dir)

    if args.random_seed is None:
        args.random_seed = np.random.randint(int(1e6))
        print(f'\nRandom seed was not set, so it is now {args.random_seed}\n')


    main_supervised(args, unique_prefix, snapshot_dir, scratch_dir)


def _main():
    global prof_utils

    # these will be useful for nefarious hacking when running under kernprof
    from asnets.utils import prof_utils
    prof_utils._run_asnets_globals = globals()

    # now run actual program
    main()


if __name__ == '__main__':
    _main()
