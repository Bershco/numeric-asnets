#!/usr/bin/env python3

import argparse
import copy
import ctypes
import os
from copy import deepcopy
from json import dump
import logging
from os import makedirs, path
import random
import signal
import sys
from time import time

from pympler import muppy, summary, asizeof
from typing import Set, Any, Optional
from pympler.asizeof import asized

from asnets.checkpointing import save_checkpoint_dir
from asnets.explorer_spawn_grads import ParallelMCTSExplorerGrads, ParallelEvaluator
from asnets.models import make_weight_manager, PropNetworkWeights, PropNetwork, make_network
from asnets.parllel_explore_spawn_grads import make_specs
from asnets.utils.tf_utils import configure_tf_gpu_memory_growth
from asnets.prob_dom_meta import DomainType
from asnets.state_reprs import CanonicalState
from asnets.spawn_train_worker import run_worker_eval_policy_only, run_worker_eval_mcts, run_worker_eval_enhsp

import numpy as np
import rpyc, gc
import tensorflow as tf
import multiprocessing
import tqdm.auto as tqdm

from asnets.explorer import StaticExplorer, DynamicExplorer, SingleProblem, run_parallel_problem_init_data_collection
from asnets.interfaces.enhsp_interface import ENHSP_CONFIGS
from asnets.supervised import SupervisedTrainer, SupervisedObjective, \
    PlannerExtensions, OriginalSupervisedTrainer
from asnets.utils.generator_utils import InstanceDifficulty
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
    '--optimizer-opts',
    default={},
    type=opt_str,
    help='additional arguments for optimizer')
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
    '-e', '--expt-dir',
    default=None,
    help='path to store experiments in')
parser.add_argument(
    '--dK',
    default='dk',
    help='prefix of the domain knowledge file'
)
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
    '--exploration-algorithm',
    choices=('static', 'dynamic', 'mcts', 'enhsp','mcts_valid','policy_valid'),
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
    default=86400, # 1 day is the default
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
    '--disable-value-head',
    action='store_true',
    default=False,
    help='Disable the usage of value head, meaning policy network only instead of two-headed.'
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
    '--action-policy-duplicate-penalty',
    type=float,
    default=0.0,
    help='Set duplicate penalty rate (for trajectory cycle skipping) mixin for MCTS action policy.'
)
parser.add_argument(
    '--estimator-h-to-v-coeff',
    type=float,
    default=1.0,
    help='Set "k" coefficient for e^{-k*h(s)} in conversion from estimator h value to canonical state value.'
)
parser.add_argument(
    '--use-estimator-decay',
    action='store_true',
    default=False,
    help='Enable estimator decay, when on, each node will be estimated by an estimator (ENHSP) during training,'
         ' for MCTS exploration and policy+value targets,'
         ' this "help" will decay in favor of the network output along the run.'
)
parser.add_argument(
    '--use-estimator',
    type=float,
    default=0.0,
    help='Enable estimator, input a floating point number from 0.0 to 1.0, never decay, use as heuristic service'
)
parser.add_argument(
    '--discard-failed-runs',
    action='store_true',
    default=False,
    help='Discard failed runs from training data, only use successful runs (important only in mcts exploration)'
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
parser.add_argument(
    '--validation-pddls-easy',
    nargs='+',
    default=[],
)
parser.add_argument(
    '--validation-pddls-medium',
    nargs='+',
    default=[],
)
parser.add_argument(
    '--validation-pddls-hard',
    nargs='+',
    default=[],
)
parser.add_argument(
    '--validation-on-test-instances',
    action='store_true',
    default=False,
    help='Have the test set also be the validation set'
)

@can_profile
def main_supervised_no_rpyc(args, unique_prefix, snapshot_dir, scratch_dir):
    print('Training supervised on random instances (SPAWN, NO RPyC)')
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

    if args.corrupt_pi:
        LOGGER.info(f'Set corrupt_pi to {args.corrupt_pi}')
    if args.corrupt_z:
        LOGGER.info(f'Set corrupt_z to {args.corrupt_z}')

    if not args.no_train:
        specs = make_specs(args)

        def make_replay_bucket(init_data):
            if init_data.dom_meta != weight_manager.dom_meta:
                raise ValueError(
                    "Worker bucket domain metadata is incompatible with the "
                    "global weight manager"
                )

            problem = SingleProblem(spec=None)
            problem.name = init_data.name
            problem.obs_dim = init_data.obs_dim
            problem.act_dim = init_data.act_dim
            problem.dom_meta = init_data.dom_meta
            problem.problem_meta = init_data.prob_meta
            problem.ssipp_dead_end_value = init_data.ssipp_dead_end_value
            problem.network, bucket_weight_manager = make_network(
                args, problem, dg_extra_dim, weight_manager,
            )
            if bucket_weight_manager is not weight_manager:
                raise AssertionError(
                    "Replay bucket did not reuse the global weight manager"
                )
            return problem

        explorer = ParallelMCTSExplorerGrads(
            problems=[],
            specs=specs,
            log=args.worker_logs,
            bucket_factory=make_replay_bucket,
            PROFILE_DIR=args.profile_dir,
            corrupt_pi=args.corrupt_pi,
            corrupt_z=args.corrupt_z,
            max_workers=args.num_workers,
            max_replay_size=args.max_replay_size,
        )
        validation_sets = {
            "easy": args.validation_pddls_easy,
            "medium": args.validation_pddls_medium,
            "hard": args.validation_pddls_hard,
        }
        validation_sets = {k: v for k, v in validation_sets.items() if v}
        all_validation_specs = []
        for diff_name, instances in validation_sets.items():
            diff_enum = InstanceDifficulty[diff_name.upper()]
            specs = make_specs(
                args,
                specific_instances=instances,
                evaluation_mode=True,
                difficulty=diff_enum,
            )
            all_validation_specs.extend(specs)
        validator = ParallelEvaluator(specs=all_validation_specs,
                                      max_workers=min(args.num_workers, len(all_validation_specs)),
                                      worker_fn=run_worker_eval_policy_only,
                                      wave_threshold=0)
        # this is policy driven inference because validation checks how good the policy is
        # and because mcts validation takes way too long
        sup_trainer = SupervisedTrainer(
            weight_manager=weight_manager,
            summary_writer=sample_writer,
            explorer=explorer,
            validator=validator,
            lr=args.supervised_lr,
            lr_steps=args.lr_steps,
            l1_reg_coeff=args.l1_reg,
            l2_reg_coeff=args.l2_reg,
            l1_l2_reg_coeff=args.l1_l2_reg,
            mse_coeff=args.mse,
            batch_size=args.supervised_bs,
            train_steps_per_epoch=1,
            main_road_fraction=0.75,
            grad_clip_norm=5.0,
            start_time=start_time,
            early_stop=args.supervised_early_stop,
            save_every=args.save_every,
            snapshot_dir=snapshot_dir,
            time_out=args.timeout,
            discard_failed_runs=args.discard_failed_runs,
            resume_from=args.resume_from,
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

    instances = args.pddls[1:]

    specs = make_specs(args, specific_instances=instances, evaluation_mode=True)
    weights_np = weight_manager.export_numpy()
    eval_explorer = ParallelEvaluator(
        specs=specs,
        max_workers=args.num_workers,
        worker_fn=run_worker_eval_mcts,
    )
    eval_start_time = time()
    _, success_rate, outs = eval_explorer.evaluate(weights_np)
    print("spec: ", specs[0])
    print(f"Inference success rate: {success_rate}, took: {time() - eval_start_time}s")

def validating_validation(args):
    if args.exploration_algorithm == 'enhsp':
        print("Validating validation instances using ENHSP proper planning")
    else:
        print(f"Validating validation instances using {args.exploration_algorithm} algorithm")
    validation_sets = {
        "easy": args.validation_pddls_easy,
        "medium": args.validation_pddls_medium,
        "hard": args.validation_pddls_hard,
    }
    validation_sets = {k: v for k, v in validation_sets.items() if v}
    all_validation_specs = []
    for diff_name, instances in validation_sets.items():
        diff_enum = InstanceDifficulty[diff_name.upper()]
        specs = make_specs(
            args,
            specific_instances=instances,
            evaluation_mode=True,
            difficulty=diff_enum,
        )
        all_validation_specs.extend(specs)
    assert args.exploration_algorithm in ['enhsp', 'mcts_valid', 'policy_valid']
    weights_np = None
    if args.exploration_algorithm == 'enhsp':
        curr_worker_fn = run_worker_eval_enhsp
    else:
        if args.exploration_algorithm == 'mcts_valid':
            curr_worker_fn = run_worker_eval_mcts
        else:
            curr_worker_fn = run_worker_eval_policy_only
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
        weight_manager = make_weight_manager(
            args, p.domain_meta, dg_extra_dim
        )
        weights_np = weight_manager.export_numpy()

    eval_explorer = ParallelEvaluator(
        specs=all_validation_specs,
        max_workers=args.num_workers,
        worker_fn=curr_worker_fn,
    )
    _, success_rate, outs = eval_explorer.evaluate(weights_np)


@can_profile
def main_supervised(args, unique_prefix, snapshot_dir, scratch_dir):
    if args.exploration_algorithm in ['enhsp', 'mcts_valid', 'policy_valid']:
        validating_validation(args)
        return
    if args.exploration_algorithm == 'mcts':
        main_supervised_no_rpyc(args, unique_prefix, snapshot_dir, scratch_dir)
        return
    print('Training/Testing supervised - not mcts')

    start_time = time()

    # configure network input
    CanonicalState.network_input_config(use_fluents=args.use_fluents,
                                        use_comparisons=args.use_comparisons)

    # problems, weight_manager = make_services(args)
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

    if not args.no_train:
        specs = make_specs(args)
        problems, weight_manager = make_problems(args, dg_extra_dim, specs, weight_manager)

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
        print('Training supervised with strategy %r and heuristic %r' %
              (args.sup_objective, args.fd_teacher_heuristic))
        if args.exploration_algorithm == 'static':
            explorer = StaticExplorer(problems, args.rollouts, args.max_replay_size)
        elif args.exploration_algorithm == 'dynamic':
            explorer = DynamicExplorer(
                problems,
                init_trajs_per_problem=args.rollouts,
                min_new_pairs=args.min_explored,
                max_new_pairs=args.max_explored,
                expl_learn_ratio=args.exploration_learning_ratio,
                max_replay_size=args.max_replay_size)
        elif args.exploration_algorithm == 'mcts':
            raise NotImplementedError("This is weird, should have arrived in a different code location.")
        else:
            raise ValueError(
                f'Unknown exploration algorithm: {args.exploration_algorithm}')
        validation_sets = {
            "easy": args.validation_pddls_easy,
            "medium": args.validation_pddls_medium,
            "hard": args.validation_pddls_hard,
        } if not args.validation_on_test_instances else {
            "test_instances": args.pddls[1:]
        }
        validation_sets = {k: v for k, v in validation_sets.items() if v}
        all_validation_specs = []
        for diff_name, instances in validation_sets.items():
            diff_enum = InstanceDifficulty[diff_name.upper()]
            specs = make_specs(
                args,
                specific_instances=instances,
                evaluation_mode=True,
                difficulty=diff_enum,
            )
            all_validation_specs.extend(specs)
        validator = ParallelEvaluator(specs=all_validation_specs,
                                      max_workers=min(args.num_workers, len(all_validation_specs)),
                                      worker_fn=run_worker_eval_policy_only, wave_threshold=0)
        sup_trainer = OriginalSupervisedTrainer(
            problems=problems,
            weight_manager=weight_manager,
            summary_writer=sample_writer,
            explorer=explorer,
            validator=validator,
            batch_size=args.supervised_bs,
            lr=args.supervised_lr,
            lr_steps=args.lr_steps,
            l1_reg_coeff=args.l1_reg,
            l2_reg_coeff=args.l2_reg,
            l1_l2_reg_coeff=args.l1_l2_reg,
            opt_batches_per_epoch=args.opt_batch_per_epoch,
            save_training_set=args.save_training_set,
            use_saved_training_set=args.use_saved_training_set,
            resume_from=args.resume_from,  # the default is None, and will not load any optimizer weights,
            # if resume_from is not None and code reached here,
            # that means we want to re-train the network so we load the optimizer weights
            start_time=start_time,
            early_stop=args.supervised_early_stop,
            save_every=args.save_every,
            scratch_dir=scratch_dir,
            snapshot_dir=snapshot_dir,
            time_out=args.timeout,
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
        save_checkpoint_dir(
            snapshot_dir=snapshot_dir,
            snapshot_name="snapshot_final",
            weight_manager=weight_manager,
            optimizer=sup_trainer.optimizer if not args.no_train else None,
        )
    print("\n[EVAL] Running final evaluation (parallel)")
    instances = args.pddls[1:]
    evaluation_specs = make_specs(
        args,
        specific_instances=instances,
        evaluation_mode=True,
    )
    weights_np = weight_manager.export_numpy()
    eval_explorer = ParallelEvaluator(
        specs=evaluation_specs,
        max_workers=min(args.num_workers, len(evaluation_specs)),
        worker_fn=run_worker_eval_policy_only,
        wave_threshold=0.0,
    )

    _, success_rate, outs = eval_explorer.evaluate(weights_np)


def make_problems(args, dg_extra_dim, specs, weight_manager):
    problems = [SingleProblem(spec) for spec in specs]
    before_dim_init = time()
    init_data = run_parallel_problem_init_data_collection(
        specs=[problem.spec for problem in problems], max_workers=args.num_workers
    )
    for id in init_data:
        problem = problems[id.slot_id]
        problem.name = id.name
        problem.obs_dim = id.obs_dim
        problem.act_dim = id.act_dim
        problem.dom_meta = id.dom_meta
        problem.problem_meta = id.prob_meta
        problem.ssipp_dead_end_value = id.ssipp_dead_end_value
    print(f"[EXPLORER_DIM_CACHE] dimension initialization done, took {time() - before_dim_init} seconds")
    for problem in problems:
        problem.network, weight_manager = make_network(
            args, problem, dg_extra_dim, weight_manager,
        )
    return problems, weight_manager


def parent_death_pact(signal: signal.Signals = signal.SIGINT) -> None:
    """Commit to kill current process when parent process dies. This function
    only works on linux for now. Specifically, it calls prctl() with the
    operation PR_SET_PDEATHSIG, which is documented in the kernel source code
    in include/uapi/linux/prctl.h. This operation is available for
    Linux>=2.1.57.

    Args:
        signal: the signal to send to the current process when the parent
        process dies. Defaults to SIGINT.
    """
    assert sys.platform == 'linux', \
        "this fn only works on Linux right now"
    libc = ctypes.CDLL("libc.so.6")
    # see include/uapi/linux/prctl.h in kernel
    PR_SET_PDEATHSIG = 1
    # last three args are unused for PR_SET_PDEATHSIG
    retcode = libc.prctl(PR_SET_PDEATHSIG, signal, 0, 0, 0)
    if retcode != 0:
        raise Exception("prctl() returned nonzero retcode %d" % retcode)


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
