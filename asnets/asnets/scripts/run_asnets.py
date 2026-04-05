#!/usr/bin/env python3

import argparse
import atexit
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from json import dump
import logging
from os import makedirs, path
import random
import signal
import sys
from time import time

from pympler import muppy, summary, asizeof
from typing import Set, Any
from pympler.asizeof import asized

from asnets.explorer_spawn_grads import ParallelMCTSExplorerGrads, ParallelMCTSExplorerEval
from asnets.freeze_overfit_test import FrozenSupervisedTrainer
from asnets.models import make_weight_manager, configure_tf_gpu_memory_growth
from asnets.prob_dom_meta import DomainType
from asnets.state_reprs import CanonicalState

import numpy as np
import rpyc, gc
import tensorflow as tf
import multiprocessing
import tqdm.auto as tqdm

from asnets.explorer import StaticExplorer, DynamicExplorer
from asnets.interfaces.enhsp_interface import ENHSP_CONFIGS
from asnets.supervised import SupervisedTrainer, SupervisedObjective, \
    ProblemServiceConfig, PlannerExtensions
from asnets.multiprob import ProblemServer, to_local, parent_death_pact
from asnets.utils.generator_utils import Domain, extract_domain_name_from_file, InstanceDifficulty
from asnets.utils.prof_utils import can_profile
from asnets.utils.py_utils import set_random_seeds

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    stream=sys.stdout
)

LOGGER = logging.getLogger(__name__)


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
            act_dist, _ = self.policy(in_obs, training=False)
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


from post_training.monte_carlo_tree_search import MCTSNode, wrapInMCTSNode, MCTS, FixedChildMap


class MonteCarloPolicyEvaluator(MCTS):

    def __init__(self, network, problem_service, horizon=0, exploration_weight=1, iterations=10,
                 num_cstates_to_generate_per_expansion=5, batch_expansion_call=True,
                 progressive_widening=False, problem_server=None,
                 debug_memory=False, debug_time_mcts_iterations=False,
                 debug_comparison_exploration_exploitation=False, ):
        super().__init__(exploration_weight, network=network,
                         problem_service=problem_service,
                         debug_memory=debug_memory,
                         debug_time_mcts_iterations=debug_time_mcts_iterations,
                         debug_comparison_exploration_exploitation=debug_comparison_exploration_exploitation)
        self.iterations = iterations
        self.horizon = horizon
        self.k = num_cstates_to_generate_per_expansion
        self.curr_tree_root = None
        self.debug_orig_root = None
        self.visited_cstates_hashes: Set[int] = set()
        self.revisit_counter = 0
        self.debug_memory = debug_memory
        self.progressive_widening = progressive_widening
        self.batch_expansion_call = batch_expansion_call

    def is_comparing_exploration_exploitation(self):
        return self._probe is not None

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

    def profile_state_id_to_node(self):
        total = 0
        for i, node in enumerate(self.state_key_to_node.values()):
            try:
                node_copy = deepcopy(node)
                node_copy.state._aux_data = None
                total += asized(node_copy).size
            except:
                continue
            if i >= 20:
                break
        estimated_total = total * len(self.state_key_to_node) / 20
        print(f"Estimated total memory for all nodes in state_to_node dictionary: {estimated_total / 1024 ** 2:.2f} MB")

    def print_memory_summary(self):
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1, limit=3)
        self.profile_state_id_to_node()
        self.safe_asizeof(self.visited_cstates_hashes, name="visited_cstates_hashes")

    def safe_asizeof(self, obj, name):
        try:
            size = asizeof.asizeof(obj)
            print(f"Size of {name}: {size / 1024 ** 2:.6f} MB")
        except Exception as e:
            print(f"Error sizing {name}: {e}")

    def get_action(self, obs):
        raise Exception("Sorry, wrong usage in code, try using get_action_from_cstate instead.")

    def get_action_from_cstate_id_hash(self, cstate_id, cstate_hash, cost):  # cstate is non-terminal
        if self.curr_tree_root is None:
            self.curr_tree_root = wrapInMCTSNode(cstate_id=cstate_id, hashed_state=cstate_hash, cost_until_now=0,
                                                 previous_action=None)
            self.state_key_to_node[cstate_id] = self.curr_tree_root
            self.debug_orig_root = self.curr_tree_root
            self.visited_cstates_hashes.add(self.curr_tree_root.__hash__())
        if self.use_value_based:
            for i in range(self.iterations):
                if i % 5 == 0:
                    gc.collect()
                self.mcts_iteration_value_based(self.curr_tree_root)
        else:
            for i in range(self.iterations):
                if self.path_until_goal is None:
                    self.mcts_iteration_classic(self.curr_tree_root, self.horizon)
            if self.path_until_goal is not None:
                next_action, next_mcts_node = self.path_until_goal[0]
                self.path_until_goal = self.path_until_goal[1:]
                # if self.state_to_node[cstate] not in self.children:
                #     self.children[self.state_to_node[cstate]] = dict()
                # self.children[self.state_to_node[cstate]][next_action] = next_mcts_node
                self.state_key_to_node[cstate_id].children = FixedChildMap([next_action], [next_mcts_node])
                self.state_key_to_node[next_mcts_node.state_id] = next_mcts_node
                return next_action

        def node_priority_by_n(node):
            return node.visit_count

        def tiebreak_by_q(node):
            # return self.Q.get(node,0.0)
            return node.Q_value

        best_action, best_node = max(
            # self.children[self.curr_tree_root].items(),
            self.curr_tree_root.children.items(),
            key=lambda item: (node_priority_by_n(item[1]), tiebreak_by_q(item[1]))
        )
        # LOGGER.info(f'chosen action: {best_action}')
        self.visited_cstates_hashes.add(best_node.__hash__())
        if self.debug_memory:
            self.print_memory_summary()
        return best_action

    def progress_to(self, action_id, cstate, cost):
        next_node = self.get_corresponding_mcts_node(cstate)
        # assert next_node in self.children[self.curr_tree_root].values(), \
        assert self.curr_tree_root.children is not None
        assert next_node in self.curr_tree_root.children.values(), \
            f"Assertion failed: next_node ({next_node}) is not one of current root's children"
        # assert next_node == self.children[self.curr_tree_root][action_id], \
        # f"Assertion failed: next_node ({next_node}) != expected ({self.children[self.curr_tree_root][action_id]})"
        assert next_node == self.curr_tree_root.children[action_id], \
            f"Assertion failed: next_node ({next_node} != expected ({self.curr_tree_root.children[action_id]})"
        # TODO: these two assertions above might be redundant
        self.prune_children_except(self.curr_tree_root, action_id)
        if next_node is None:
            LOGGER.info('Next node is not available, creating a new tree.')
            self.curr_tree_root = wrapInMCTSNode(cstate, cost_until_now=cost, previous_action=action_id)
        else:
            _temp = self.curr_tree_root
            self.curr_tree_root = next_node
            # This explicit 'recursive=False' means that only the node would be properly deleted, subtree left as-is
            self._delete_subtree(_temp, recursive=False)

    def progress_to_without_cstate(self, action_id, cost):
        # next_node = self.children[self.curr_tree_root][action_id]
        next_node = self.curr_tree_root.children[action_id]
        # self.prune_children_except(self.curr_tree_root, action_id) TODO: check if memory explodes again, or if this is ok to drop entirely
        assert next_node is not None, "Somehow need to progress to a non-generated node."
        _temp = self.curr_tree_root
        self.curr_tree_root = next_node
        # This explicit 'recursive=False' means that only the node would be properly deleted, subtree left as-is
        self._delete_subtree(_temp, recursive=False)
        return self.curr_tree_root.state_id, hash(
            self.curr_tree_root), 1, self.curr_tree_root.goal_state, self.curr_tree_root.terminal_state

    def get_corresponding_mcts_node(self, cstate):
        return self.state_key_to_node.get(cstate, None)

    def _expand(self, node):
        if node.children is not None:
            return
        node.children = self.find_children(node)
        self.state_key_to_node[node.state_id] = node
        if self._probe:
            try:
                act_dim = None
                try:
                    pri = self.get_act_dist_from_mcts_node(node)
                    act_dim = len(pri) if pri is not None else None
                except Exception:
                    pass
                self._probe.log_expand(act_dim=act_dim, children_created=len(node.children))
            except Exception:
                pass
        for child_node in node.children.values():
            assert isinstance(child_node, MCTSNode)
            self.state_key_to_node[child_node.state_id] = child_node
        if self.debug_time_mcts_iterations:
            self.after_expansion_times.append(time())

    def _rollout(self, mcts_node, horizon=10):
        """Returns the reward for a random simulation (to a certain horizon) of `node`"""
        action_following_state_path = []
        for _ in range(horizon):
            if mcts_node.is_goal():
                print(
                    "\n\n============================================\nGoal was found during rollout\n============================================\n")
                action_path = []
                curr_mcts_node = self.curr_tree_root
                for action_from_path, mcts_node_from_path in action_following_state_path:
                    # if curr_mcts_node not in self.children:
                    # if curr_mcts_node.children is None:
                    # self.children[curr_mcts_node] = dict()
                    # curr_mcts_node.children = None
                    # self.children[curr_mcts_node][action_from_path] = mcts_node_from_path
                    curr_mcts_node.children = FixedChildMap([action_from_path], [mcts_node_from_path])
                    self.state_key_to_node[curr_mcts_node.state_id] = curr_mcts_node
                    curr_mcts_node = mcts_node_from_path
                    action_path.append(action_from_path)
                print(f"Next actions are: {action_path}")
                self.path_until_goal = action_following_state_path
                break
            best_action, mcts_node = self.find_child_by_policy(mcts_node)
            action_following_state_path.append((best_action, mcts_node))
        return mcts_node.reward()

    def find_children(self, parent_node: MCTSNode) -> FixedChildMap:
        """Find up to k successors of parent_node that are applicable and not yet visited"""
        act_dist = self.get_act_dist_from_mcts_node(parent_node).numpy()
        mask = self.get_applicable_action_mask(parent_node)
        # Rank actions by descending policy probability
        sorted_indices = sorted(range(len(act_dist)), key=lambda i: act_dist[i], reverse=True)
        keys, values = self.get_actions_and_nodes(parent_node, sorted_indices, mask, act_dist)
        return FixedChildMap(keys, values)

    def get_actions_and_nodes(self, parent_node, sorted_indices, mask, act_dist) -> tuple[list[Any], list[Any]]:
        actions = []
        nodes = []
        if self.batch_expansion_call:
            selected_actions = []
            for i in sorted_indices:
                if len(selected_actions) >= self.k:
                    break
                if not mask[i] or act_dist[i] == 0.0:
                    continue
                selected_actions.append(i)

            # Single RPC for all k selected actions
            results = self.problem_service.env_simulate_batch_steps(*parent_node.get_identifiers(), selected_actions)

            generated_ids = []
            for (action_id, cstate_after_action_i_id, cstate_after_action_i_hash,
                 step_cost, is_goal, is_terminal,
                 network_ready_repr, applicable_action_mask
                 ) in results:
                generated_ids.append(cstate_after_action_i_id)
                wrapped_output_cstate = wrapInMCTSNode(
                    cstate_id=cstate_after_action_i_id,
                    cost_until_now=parent_node.cost_until_now + step_cost,
                    previous_action=action_id,
                    is_goal=is_goal,
                    is_terminal=is_terminal,
                    as_network_input=network_ready_repr, applicable_action_mask=applicable_action_mask,
                    hashed_state=cstate_after_action_i_hash,
                    parent=parent_node,
                )
                self.state_key_to_node[cstate_after_action_i_id] = wrapped_output_cstate
                actions.append(action_id)
                nodes.append(wrapped_output_cstate)
            ids = ",".join([str(i) for i in generated_ids])
            print(f"Generated nodes with ids: {ids}")

        else:
            selected = 0
            for i in sorted_indices:
                if selected >= self.k:
                    break
                if not mask[i] or act_dist[i] == 0.0:
                    continue
                if self.problem_service is None:
                    raise RuntimeError("problem_service is None — was it shut down?")
                # Simulate step only now (expensive!)
                cstate_after_action_id, step_cost, is_goal, is_terminal, network_ready_repr, \
                    applicable_action_mask, state_hash = parent_node.simulate_step(i, self.problem_service)
                wrapped_output_cstate = wrapInMCTSNode(
                    cstate_after_action_id,
                    cost_until_now=parent_node.cost_until_now + step_cost,
                    previous_action=i,
                    is_goal=is_goal,
                    is_terminal=is_terminal,
                    as_network_input=network_ready_repr,
                    applicable_action_mask=applicable_action_mask,
                    hashed_state=state_hash,
                    parent=parent_node,
                )
                self.state_key_to_node[cstate_after_action_id] = wrapped_output_cstate
                # output[i] = wrapped_output_cstate
                actions.append(i)
                nodes.append(wrapped_output_cstate)
                selected += 1
        return actions, nodes

    def find_child_by_policy(self, parent_node: MCTSNode):
        """Random successor of this board state (for more efficient simulation)"""
        act_dist = self.get_act_dist_from_mcts_node(parent_node)
        mask = self.get_applicable_action_mask(parent_node)
        masked_act_dist = tf.where(mask, act_dist, tf.zeros_like(act_dist))
        total = tf.reduce_sum(masked_act_dist)
        normalized_act_dist = tf.cond(
            tf.greater(total, 0),
            lambda: masked_act_dist / total,
            lambda: tf.zeros_like(act_dist)
        )
        norm_act_dist_np = normalized_act_dist.numpy()
        next_action_ind = np.random.choice(len(norm_act_dist_np), p=norm_act_dist_np)
        if self.problem_service is None:
            raise RuntimeError("problem_service is None — was it shut down?")
        best_cstate, step_cost = parent_node.simulate_step(next_action_ind, self.problem_service)
        return next_action_ind, wrapInMCTSNode(best_cstate, cost_until_now=parent_node.cost_until_now + step_cost,
                                               previous_action=next_action_ind, parent=parent_node)

    def print_exploration_exploitation_comparison(self):
        self._probe.print_exploration_exploitation_comparison()


@can_profile
def run_trial(policy_evaluator, problem_server, limit=1000, det_sample=False, graceful_timeout=300):
    """Run policy on problem. Returns (cost, path), where cost may be None if
    goal not reached before horizon."""
    print(f'\n-------------> Graceful_timeout is set to {graceful_timeout}\n')
    print(f'\n-------------> Limit is set to {limit}\n')
    trial_start_time = time()
    problem_service = problem_server.service
    # curr_state = to_local(problem_service.env_reset())
    curr_state_id, curr_state_hash = to_local(problem_service.env_reset())

    # total cost of this run
    cost = 0
    path = []
    for i in range(1, limit):
        if time() - trial_start_time > graceful_timeout:
            print('Graceful_timeout has been reached :)')
            break
        action = policy_evaluator.get_action_from_cstate_id_hash(curr_state_id, curr_state_hash, cost)
        path.append(to_local(problem_service.action_name(action)))
        curr_state_id, curr_sate_hash, step_cost, is_goal, is_terminal = move_to_next_state(problem_service,
                                                                                            policy_evaluator, action,
                                                                                            cost, current_code=False)
        cost += step_cost
        if is_goal:
            if policy_evaluator.is_comparing_exploration_exploitation():
                print("Exploration-Exploitation comparison:")
                policy_evaluator.print_exploration_exploitation_comparison()
            return cost, True, path
        # we can run out of time or run out of actions to take
        if is_terminal:
            break
        if i == limit - 1:
            print(
                " I actually reached the end, something weird is happening, only some actions were chosen but limit was reached? ")
    # path.append('FAIL! D:')
    if policy_evaluator.is_comparing_exploration_exploitation():
        print("Exploration-Exploitation comparison:")
        policy_evaluator.print_exploration_exploitation_comparison()
    return cost, False, path


def move_to_next_state(problem_service, policy_evaluator, action, cost, current_code=True):
    if current_code:
        curr_state, step_cost = to_local(problem_service.env_step(action))
        policy_evaluator.progress_to(action, curr_state, cost + step_cost)
        return curr_state, step_cost  # FIXME: this currently does not work, must return also if the curr_state is goal
    else:
        assert isinstance(policy_evaluator, MonteCarloPolicyEvaluator)
        return policy_evaluator.progress_to_without_cstate(action,
                                                           cost)  # TODO: env_step on the problem_service is irrelevant


def run_trials(network, problem_server, trials, iterations, horizon=None, limit=1000, det_sample=False,
               single_trial_graceful_timeout_sec=300, num_cstates_to_expand=5,
               debug_memory=False, mcts_exploration_weight=1, mcts_smart_expansions=False):
    # policy_evaluator = CachingPolicyEvaluator(policy=network, det_sample=det_sample)
    k = min(iterations - 1, num_cstates_to_expand)
    policy_evaluator = MonteCarloPolicyEvaluator(network=network, problem_service=problem_server.service,
                                                 problem_server=problem_server,
                                                 iterations=iterations, horizon=horizon,
                                                 num_cstates_to_generate_per_expansion=k,
                                                 debug_memory=debug_memory,
                                                 exploration_weight=mcts_exploration_weight,
                                                 progressive_widening=mcts_smart_expansions,
                                                 )
    all_exec_times = []
    all_costs = []
    all_goal_reached = []
    paths = []
    print(f'\n-------------> MCTS iterations number: {iterations}\n')
    # print(f'\n-------------> MCTS rollout horizon length: {horizon}\n')
    for _ in tqdm.trange(trials, desc='trials', leave=True):
        start = time()
        cost, goal_reached, path = run_trial(policy_evaluator, problem_server,
                                             limit, det_sample, graceful_timeout=single_trial_graceful_timeout_sec)
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
    '-R', '--rounds-eval',
    type=int,
    default=100,
    help='number of eval rounds')
parser.add_argument(
    '-L', '--limit-turns',
    type=int,
    default=100,
    help='max turns per round')
parser.add_argument(
    '--search-max-length',
    type=int,
    default=50,
    help='Maximum number of action decision steps.')
parser.add_argument(
    '-e', '--expt-dir',
    default=None,
    help='path to store experiments in')
parser.add_argument(
    '--dK',
    default='dk',
    help='prefix of the domain knowledge file'
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
    '--mse',
    type=float,
    default=1e-3,
    help='mse coefficient for loss'
)
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
    choices=('static', 'dynamic', 'mcts'),
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
    '--mcts-expansion-size',
    type=int,
    default=20,
    help='Number of MCTS Nodes to generate upon MCTS parent node expansion.')
parser.add_argument(
    '--no-eval',
    action='store_true',
    default=False,
    help='Disable evaluation after training.')
parser.add_argument(
    '--mcts-heuristic',
    choices=list(ENHSP_CONFIGS.keys()),
    default='hadd-gbfs',
    help='When value-based mcts runs, this would be the state-value heuristic function.')
parser.add_argument(
    '--debug-memory',
    default=False,
    help='Enable memory debugging.')
parser.add_argument(
    '--mcts-exploration-weight',
    type=float,
    default=1.0,
    help='PUCT exploration weight (c value).'
)
parser.add_argument(
    '--mcts-smart-expansions',
    action='store_true',
    default=False,
    help='Enable smart expansions, progressive widening (or "unpruning"),'
         ' otherwise only limits number of generated children nodes to be min(mcts_expansion_size,(mcts_iterations - 1))'
)
parser.add_argument(
    '--policy-network-only',
    action='store_true',
    default=False,
    help='Revert to policy network only instead of the new dual-head network (for ablation study)'
)
parser.add_argument(
    '--mcts-iterations',
    type=int,
    default=0,
    help='Number of MCTS iterations done during training, default is f(act_dim)'
)
parser.add_argument(
    '--heuristic-bootstrapping',
    action='store_true',
    default=False,
    help='Enable heuristic bootstrapping during training.'
)
parser.add_argument(
    '--mcts-her-strategy',
    action='store_true',
    default=False,
    help='Enable hindsight experience replay strategy where states are sampled from the training-based mcts tree and trajectories are decalred her goals.'
)
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Set the number of problem slots for the trainer\evaluator'
)
parser.add_argument(
    '--slurm-job-id',
    type=int,
    default=0,
    help='Set the slurm job id for inner logic'
)
parser.add_argument(
    '--worker-logs',
    action='store_true',
    default=False,
    help='Enable worker logging.'
)
parser.add_argument(
    '--corrupt-pi',
    choices=('shuffle', 'random'),
    default=None,
    help='Enable pi (target policy) corruption during training for corruption sanity test'
)
parser.add_argument(
    '--corrupt-z',
    choices=('shuffle', 'random', 'zero'),
    default=None,
    help='Enable z (target value) corruption during training for corruption sanity test'
)
parser.add_argument(
    '--fixed-instance',
    action='store_true',
    default=False,
    help='Single instance overfit test.'
)
parser.add_argument(
    '--original-training-set',
    action='store_true',
    default=False,
    help='Set the training set to be the original of Numeric ASNets paper, this overrides fixed-instance.'
)
parser.add_argument(
    '--sample-k-additional-states',
    type=int,
    default=0,
    help='Set the amount of additional states sampled during training'
)
parser.add_argument(
    '--profile-dir',
    default=None,
    help='Path to profile directory, default is not profiling at all.'
)
parser.add_argument(
    '--freeze-train',
    action='store_true',
    default=False,
    help='Freeze training on one single exploration to make sure network is learning SOMETHING.'
)
parser.add_argument(
    '--goal-path-reconstruction',
    choices=('all', 'closest'),
    default=None,
    help='Enable goal path reconstruction during training.'
)
parser.add_argument(
    '--action-policy',
    choices=('argmax', 'sample', 'visit'),
    default=None,
    help='Set action policy to use during MCTS action decision.'
)
parser.add_argument(
    '--action-policy-goal-chase-distance-threshold',
    type=int,
    default=None,
    help='Set goal chase distance threshold in MCTS action decision, if goal is closer than the threshold,'
         ' MCTS decision-making process will exploit consistently.'
         'default is None - do not goal chase.'
         '-1 is infinite - i.e. if goal is visible - run for it.'
)
parser.add_argument(
    '--action-policy-epsilon',
    type=float,
    default=None,
    help='Set epsilon greedy mixin for MCTS action policy.'
)
parser.add_argument(
    '--action-policy-temperature',
    type=float,
    default=None,
    help='Set temperature mixin for MCTS action policy.'
)
parser.add_argument(
    '--action-policy-decay-rate',
    type=float,
    default=None,
    help='Set decay rate mixin for MCTS action policy.'
)
parser.add_argument(
    '--estimator-h-to-v-coeff',
    type=float,
    default=1.0,
    help='Set "k" coefficient for e^{-k*h(s)} in conversion from estimator h value to canonical state value.'
)
parser.add_argument(
    '--estimator-decay',
    action='store_true',
    default=False,
    help='Enable estimator decay, when on, each node will be estimated by an estimator (ENHSP) during training,'
         ' for MCTS exploration and policy+value targets,'
         ' this "help" will decay in favor of the network output along the run.'
)
parser.add_argument(
    '--estimator-decay-epochs',
    type=int,
    default=None,
    help='Set the amount of epochs estimator decays from est_coeff_start to est_coeff_end, default value is 20% of all epochs.'
)
parser.add_argument(
    '--estimator-decay-coeff-start',
    type=float,
    default=1.0,
    help='Set est_coeff_start value.'
)
parser.add_argument(
    '--estimator-decay-coeff-end',
    type=float,
    default=0.2,
    help='Set est_coeff_end value.'
)


def eval_single(args, network, problem_server, unique_prefix, elapsed_time,
                iter_num, weight_manager, scratch_dir):
    # now we evaluate the learned network
    LOGGER.info('Evaluating network')
    trial_results, paths = run_trials(
        network,
        problem_server,
        args.rounds_eval,
        limit=args.limit_turns,
        det_sample=args.det_eval,
        iterations=args.mcts_iterations,
        horizon=args.mcts_rollout_horizon,
        single_trial_graceful_timeout_sec=args.graceful_timeout,
        num_cstates_to_expand=args.mcts_expansion_size,
        debug_memory=args.debug_memory,
        mcts_exploration_weight=args.mcts_exploration_weight,
        mcts_smart_expansions=args.mcts_smart_expansions,
    )

    LOGGER.info('Trial results')
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

    @property
    def network(self):
        return self.problem_server.network

    @network.setter
    def network(self, network):
        self.problem_server.network = network


@can_profile
def make_services(args):
    """Make a ProblemService for each relevant problem."""
    servers = []

    def kill_servers():
        for server in servers:
            try:
                server.stop()
            except Exception as e:
                print("Got exception %r while trying to stop %r" % (e, server))

    atexit.register(kill_servers)

    only_one_good_action = args.sup_objective == SupervisedObjective.THERE_CAN_ONLY_BE_ONE or args.sup_objective == SupervisedObjective.MCTS_POLICY_DIST

    domain = Domain.from_pddl_name(extract_domain_name_from_file(args.pddls[0]))
    LOGGER.info(f"Starting to initialize {args.num_workers} problem servers")
    for slot_id in range(args.num_workers):
        random_seed = None if args.seed is None \
            else args.seed + slot_id
        service_config = ProblemServiceConfig(
            args.pddls,
            args.domain_type,
            domain=domain,
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
            max_len=args.search_max_length,
            training_mcts_iterations=args.mcts_iterations,
            heuristic_bootstrapping=args.heuristic_bootstrapping,
            mcts_her_strategy=args.mcts_her_strategy,
            mcts_expansion_k=args.mcts_expansion_size,
            use_fluents=args.use_fluents,
            use_comps=args.use_comparisons,
            slot_id=slot_id,
        )
        servers.append(ProblemServer(service_config))
    with ThreadPoolExecutor(max_workers=min(32, len(servers))) as ex:
        futs = [ex.submit(s.connect) for s in servers]
        for f in as_completed(futs):
            f.result()  # raises immediately on connect failure

    # Dispatch initialise() for ALL servers
    init_results = []
    for s in servers:
        init_async = rpyc.async_(s.service.initialise)  # netref lookup happens once here
        init_results.append(init_async())

    # Ensure initialise() completed everywhere (and surface remote exceptions)
    for ar in init_results:
        _ = ar.value

    step2_results = []
    step3_results = []

    for s in servers:
        # estimator init after initialise barrier
        init_est_async = rpyc.async_(s.service.initialise_estimator)
        step2_results.append(init_est_async(enhsp_config=args.mcts_heuristic))

        # local setter ok (not RPyC)
        s.set_enhsp_config(args.mcts_heuristic)

        set_pol_async = rpyc.async_(s.service.set_policy_only)
        step3_results.append(set_pol_async(bool(args.policy_network_only)))

        # local setter ok
        s.set_policy_only(bool(args.policy_network_only))

    # Barrier 2: wait + surface remote exceptions
    for ar in step2_results:
        _ = ar.value
    for ar in step3_results:
        _ = ar.value
    LOGGER.info("Finished initializing problem servers")
    # do this as a separate loop so that we can wait for services to spool
    # up in background
    weight_manager = None
    for problem_server in servers:
        weight_manager = problem_server.register_network(weight_manager, args)
    return servers, weight_manager


@can_profile
def main_supervised_no_rpyc(args, unique_prefix, snapshot_dir, scratch_dir):
    print('Training supervised on random instances (SPAWN, NO RPyC, NO REPLAY BUFFER)')
    print(f"Instances: {args.pddls}")
    start_time = time()

    # ------------------------------------------------------------
    # Configure network input
    # ------------------------------------------------------------
    CanonicalState.network_input_config(
        use_fluents=args.use_fluents,
        use_comparisons=args.use_comparisons
    )
    configure_tf_gpu_memory_growth()

    only_one_good_action = (
            args.sup_objective == SupervisedObjective.THERE_CAN_ONLY_BE_ONE
            or args.sup_objective == SupervisedObjective.MCTS_POLICY_DIST
    )

    # ------------------------------------------------------------
    # Build SpawnExploreSpec list (one per slot)
    # ------------------------------------------------------------
    from asnets.parllel_explore_spawn_grads import SpawnExploreSpec

    specs = []
    for slot_id in range(args.num_workers):
        specs.append(
            SpawnExploreSpec(
                pddls=args.pddls,
                domain_type=args.domain_type,
                trainer_seed=args.seed,
                slot_id=slot_id,
                num_slots=args.num_workers,
                ssipp_dg_heuristic=args.ssipp_dg_heuristic,
                use_lm_cuts=args.use_lm_cuts,
                use_numeric_landmarks=args.use_numeric_landmarks,
                use_contributions=args.use_contributions,
                use_act_history=args.use_act_history,
                fd_heuristic=args.fd_teacher_heuristic,
                ssipp_teacher_heuristic=args.ssipp_teacher_heuristic,
                enhsp_config=args.enhsp_config,
                estimator_h_to_v_coeff=args.estimator_h_to_v_coeff,
                teacher_planner=args.teacher_planner,
                teacher_timeout_s=args.teacher_timeout_s,
                only_one_good_action=only_one_good_action,
                use_teacher_envelope=args.use_teacher_envelope,
                max_len=args.search_max_length,
                mcts_iterations=args.mcts_iterations,
                heuristic_bootstrapping=args.heuristic_bootstrapping,
                mcts_her_strategy=args.mcts_her_strategy,
                mcts_expansion_k=args.mcts_expansion_size,
                use_fluents=args.use_fluents,
                use_comps=args.use_comparisons,
                difficulty=InstanceDifficulty.EASY,
                fixed_instance_pddl=args.fixed_instance,
                mcts_exploration_weight=args.mcts_exploration_weight,
                sample_k_additional_states=args.sample_k_additional_states,
                goal_path_reconstruction=args.goal_path_reconstruction,
                action_policy=args.action_policy,
                action_policy_goal_chase_distance_threshold=args.action_policy_goal_chase_distance_threshold,
                action_policy_epsilon=args.action_policy_epsilon,
                action_policy_temperature=args.action_policy_temperature,
                action_policy_decay_rate=args.action_policy_decay_rate,
                original_training_set=args.original_training_set,
                estimator_decay=args.estimator_decay,
                estimator_decay_coeff_start=args.estimator_decay_coeff_start,
                estimator_decay_coeff_end=args.estimator_decay_coeff_end,
                estimator_decay_epochs=args.estimator_decay_epochs if args.estimator_decay_epochs is not None else int(
                    args.max_opt_epochs / 3),
            )
        )

    # ------------------------------------------------------------
    # Build planner ONCE (for shapes / network construction)
    # ------------------------------------------------------------
    p = PlannerExtensions(
        args.pddls,
        args.domain_type,
        dg_ssipp_heuristic_name=args.ssipp_dg_heuristic,
        dg_use_lm_cuts=args.use_lm_cuts,
        dg_use_numeric_landmarks=args.use_numeric_landmarks,
        dg_use_contributions=args.use_contributions,
        dg_use_act_history=args.use_act_history,
    )
    dg_extra_dim = sum(g.extra_dim for g in p.data_gens)

    # ------------------------------------------------------------
    # Weight manager
    # ------------------------------------------------------------
    weight_manager = make_weight_manager(
        args, p.domain_meta, dg_extra_dim
    )

    summary_path = path.join(scratch_dir, 'tensorboard')
    LOGGER.info(f'Tensorboard summary path: {summary_path}')

    if args.minimal_file_saves:
        sample_writer = None
    else:
        sample_writer = tf.summary.create_file_writer(summary_path)

    # ------------------------------------------------------------
    # Explorer (SPAWN-BASED, SERVERLESS)
    # ------------------------------------------------------------

    if args.corrupt_pi:
        LOGGER.info(f'Set corrupt_pi to {args.corrupt_pi}')
    if args.corrupt_z:
        LOGGER.info(f'Set corrupt_z to {args.corrupt_z}')
    explorer = ParallelMCTSExplorerGrads(
        specs=specs,
        dropout=args.dropout,
        debug=args.debug_memory,
        policy_only=args.policy_network_only,
        log=args.worker_logs,
        PROFILE_DIR=args.profile_dir,
        corrupt_pi=args.corrupt_pi,
        corrupt_z=args.corrupt_z,
        mse_coeff=args.mse,
        l2_reg_coeff=args.l2_reg,
        l1_reg_coeff=args.l1_reg,
        l1_l2_reg_coeff=args.l1_l2_reg,
        max_workers=args.num_workers,
    )
    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
    if not args.no_train:
        strategy = (
            SupervisedObjective.ANY_GOOD_ACTION
            if args.sup_objective == SupervisedObjective.MCTS_POLICY_DIST
               and args.policy_network_only
            else args.sup_objective
        )
        if not args.freeze_train:
            sup_trainer = SupervisedTrainer(
                weight_manager=weight_manager,
                summary_writer=sample_writer,
                explorer=explorer,
                strategy=strategy,
                batch_size=args.supervised_bs,
                lr=args.supervised_lr,
                lr_steps=args.lr_steps,
                l1_reg_coeff=args.l1_reg,
                l2_reg_coeff=args.l2_reg,
                l1_l2_reg_coeff=args.l1_l2_reg,
                mse_coeff=args.mse,
                opt_batches_per_epoch=args.opt_batch_per_epoch,
                start_time=start_time,
                early_stop=args.supervised_early_stop,
                save_every=args.save_every,
                scratch_dir=scratch_dir,
                snapshot_dir=snapshot_dir,
                dk=args.dK,
                time_out=args.timeout,
                use_fluents=args.use_fluents,
                use_comps=args.use_comparisons,
                policy_only=args.policy_network_only,
            )
        else:
            sup_trainer = FrozenSupervisedTrainer(
                weight_manager=weight_manager,
                summary_writer=sample_writer,
                explorer=explorer,
                strategy=strategy,
                batch_size=args.supervised_bs,
                lr=args.supervised_lr,
                lr_steps=args.lr_steps,
                l1_reg_coeff=args.l1_reg,
                l2_reg_coeff=args.l2_reg,
                l1_l2_reg_coeff=args.l1_l2_reg,
                mse_coeff=args.mse,
                opt_batches_per_epoch=args.opt_batch_per_epoch,
                start_time=start_time,
                early_stop=args.supervised_early_stop,
                save_every=args.save_every,
                scratch_dir=scratch_dir,
                snapshot_dir=snapshot_dir,
                dk=args.dK,
                time_out=args.timeout,
                use_fluents=args.use_fluents,
                use_comps=args.use_comparisons,
                policy_only=args.policy_network_only,
                planner_exts=p,
            )

        best_rate, elapsed_time, iter_num = sup_trainer.train(
            max_epochs=args.max_opt_epochs
        )
    else:
        elapsed_time = iter_num = None

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    if args.no_eval:
        return

    weights_np = weight_manager.export_numpy()
    assert len(specs) == 1, f"Currently evaluation only works serially, {len(specs)}"
    specs[0].evaluation_instance_index = 0
    eval_explorer = ParallelMCTSExplorerEval(
        specs=specs,
        max_workers=args.num_workers,
    )
    eval_start_time = time()
    success_rate, outs = eval_explorer.evaluate(weights_np)
    print("spec: ", specs[0])
    print(f"Inference success rate: {success_rate}, took: {time() - eval_start_time}")


@can_profile
def main_supervised(args, unique_prefix, snapshot_dir, scratch_dir):
    if args.exploration_algorithm == 'mcts':
        main_supervised_no_rpyc(args, unique_prefix, snapshot_dir, scratch_dir)
        return
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
    LOGGER.info(f'Tensorboard summary path: {summary_path}')
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
                max_replay_size=args.max_replay_size,
                debug_memory=args.debug_memory)
        elif args.exploration_algorithm == 'mcts':
            # explorer = MCTSExplorer(
            raise NotImplementedError("This is weird, should have arrived in a different code location.")
        else:
            raise ValueError(
                f'Unknown exploration algorithm: {args.exploration_algorithm}')

        # we maintain the old loss for usage of policy network only (instead of dual-head using the new loss)
        strategy = SupervisedObjective.ANY_GOOD_ACTION if args.sup_objective == SupervisedObjective.MCTS_POLICY_DIST and args.policy_network_only else args.sup_objective
        sup_trainer = SupervisedTrainer(
            problems=problems,
            weight_manager=weight_manager,
            summary_writer=sample_writer,
            explorer=explorer,
            strategy=strategy,
            batch_size=args.supervised_bs,
            lr=args.supervised_lr,
            lr_steps=args.lr_steps,
            l1_reg_coeff=args.l1_reg,
            l2_reg_coeff=args.l2_reg,
            l1_l2_reg_coeff=args.l1_l2_reg,
            mse_coeff=args.mse,
            opt_batches_per_epoch=args.opt_batch_per_epoch,
            save_training_set=args.save_training_set,
            use_saved_training_set=args.use_saved_training_set,
            start_time=start_time,
            early_stop=args.supervised_early_stop,
            save_every=args.save_every,
            scratch_dir=scratch_dir,
            snapshot_dir=snapshot_dir,
            dk=args.dK,
            time_out=args.timeout,
            use_fluents=args.use_fluents,
            use_comps=args.use_comparisons,
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
        eval_single(args, problem.network, problem.problem_server,
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
    LOGGER.info('Arguments are: %s', args)

    if args.seed is not None:
        set_random_seeds(args.seed)
    else:
        # if seed was not set, we will create a universal seed through time
        SEED = int(time() * 1000) % (2 ** 32)
        set_random_seeds(SEED)
        args.seed = SEED
        LOGGER.info(f'Seed was not manually set, so it was automatically set to {SEED}')

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

    main_supervised(args, unique_prefix, snapshot_dir, scratch_dir)


def _main():
    global prof_utils

    # these will be useful for nefarious hacking when running under kernprof
    from asnets.utils import prof_utils
    prof_utils._run_asnets_globals = globals()

    # now run actual program
    main()


if __name__ == "__main__":
    USE_GPU = os.environ.get("ASN_GPU", "0") == "1"
    multiprocessing.set_start_method("forkserver", force=True)
    if USE_GPU:
        multiprocessing.set_forkserver_preload([
            "asnets.tf_preload",
            "asnets.models",
        ])
    else:
        multiprocessing.set_forkserver_preload([
            "asnets.tf_cpu_preload",
            "asnets.models",
        ])
    _main()
