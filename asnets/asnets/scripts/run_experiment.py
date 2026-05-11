#!/usr/bin/env python3
"""Run an experiment using the standard Python-based configuration format (see
`experiments/` subdirectory for example.)"""

import argparse
import datetime
from hashlib import md5
from importlib import import_module
from os import path, makedirs, listdir, getcwd, environ
from shutil import copytree
from subprocess import Popen, PIPE, TimeoutExpired
import sys
from time import time

import ray

from asnets.interfaces.enhsp_interface import ENHSP_CONFIGS

THIS_DIR = path.dirname(path.abspath(__file__))
PLANNER_ROOT = path.abspath(path.join(THIS_DIR, '..', '..'))
# hack to ensure we can find 'experiments' module
sys.path.append(PLANNER_ROOT)


def extract_by_prefix(lines, prefix):
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):]


def run_asnets_local(flags, root_dir, need_snapshot, timeout, is_train,
                     cwd, profiling=False, train_only=False, memory_profiling=False):
    assert not profiling or not memory_profiling, "Cannot profile memory and efficiency at the same time."
    cmdline = []
    if profiling:
        print("[run_experiment_setup] timing profiling is on.")
        cmdline.extend([
                           sys.executable, '-m', 'cProfile', '-o', 'profile_output.prof',
                           '-m', 'asnets.scripts.run_asnets'
                       ] + flags)
        # for graceful timeout of a single trial - this is specifically for profiling, but can obviously be used otherwise
        cmdline.extend(['--graceful-timeout', str(timeout - 300)])
    elif memory_profiling:
        # If MEMRAY=1 in the environment, wrap run_asnets with memray and save to a stable dir
        print("[run_experiment_setup] memory profiling is on.")
        logdir = path.expanduser("~/training_new_domains/alphazero_training/memray_logs")
        makedirs(logdir, exist_ok=True)
        job = environ.get("SLURM_JOB_ID", "noj")
        task = environ.get("SLURM_PROCID", "0")
        # timestamp helps avoid collisions across retries
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"runasn-{job}-{task}-{ts}"
        outfile = path.join(logdir, f"{base}.bin")
        cmdline.extend([sys.executable, '-m', 'memray', 'run', '--follow-fork', '-o', outfile,
                        '-m', 'asnets.scripts.run_asnets'] + flags
                       + ['--graceful-timeout', str((3 * 60 * 60))])
    else:
        cmdline.extend([sys.executable, '-u', '-m', 'asnets.scripts.run_asnets'] + flags)

    if train_only:
        cmdline.append('--no-eval')
    print('Running command line "%s"' % ' '.join(cmdline))

    # we use this for logging
    unique_suffix = md5(' '.join(cmdline).encode('utf8')).hexdigest()
    dest_dir = path.join(root_dir, 'runs', unique_suffix)
    print('Will write results to %s' % dest_dir)
    makedirs(dest_dir, exist_ok=True)
    with open(path.join(dest_dir, 'cmdline'), 'w') as fp:
        fp.write(' '.join(cmdline))
    stdout_path = path.join(dest_dir, 'stdout')
    stderr_path = path.join(dest_dir, 'stderr')

    dfpg_proc = tee_out_proc = tee_err_proc = None
    start_time = time()
    try:
        # print to stdout/stderr *and* save as well
        dfpg_proc = Popen(cmdline, stdout=PIPE, stderr=PIPE, cwd=cwd)
        # first tee for stdout
        tee_out_proc = Popen(['tee', stdout_path], stdin=dfpg_proc.stdout)
        # second tee for stderr
        tee_err_proc = Popen(['tee', stderr_path], stdin=dfpg_proc.stderr)

        # close descriptors from this proc (they confuse child error handling);
        # see https://stackoverflow.com/q/23074705
        dfpg_proc.stdout.close()
        dfpg_proc.stderr.close()

        # twiddle, twiddle, twiddle
        timed_out = False
        bad_retcode = False
        try:
            dfpg_proc.wait(timeout=max(timeout * 1.25, timeout + 300))
        except TimeoutExpired:
            # uh, oops; better kill everything
            print('Run timed out after %ss!' % timeout)
            timed_out = True
    finally:
        # "cleanup"
        for proc in [tee_out_proc, tee_err_proc, dfpg_proc]:
            if proc is None:
                continue
            proc.poll()
            if proc.returncode is None:
                # make sure it's dead
                print('Force-killing a process')
                proc.terminate()
            proc.wait()
            retcode = proc.returncode
            if retcode != 0:
                print('Process exited with code %s: %s' %
                      (retcode, ' '.join(proc.args)))
                bad_retcode = True

    # write out extra info
    elapsed_time = time() - start_time
    with open(path.join(dest_dir, 'elapsed_secs'), 'w') as fp:
        fp.write('%f\n' % elapsed_time)
    with open(path.join(dest_dir, 'termination_status'), 'w') as fp:
        fp.write('timed_out: %s\nbad_retcode: %s\n' % (timed_out, bad_retcode))
    if is_train:
        with open(path.join(dest_dir, 'is_train'), 'w') as fp:
            # presence of 'is_train' file (in this case containing just a
            # newline) is sufficient to indicate that this was a train run
            print('', file=fp)

    # get stdout for... reasons
    with open(stdout_path, 'r') as fp:
        stdout = fp.read()
    lines = stdout.splitlines()

    # copy all info in custom dir into original prog's output dir (makes it
    # easier to associated)
    run_subdir = extract_by_prefix(lines, 'Unique prefix: ')
    if run_subdir is None:
        raise Exception("Couldn't find unique prefix for problem!")
    run_dir = path.join(root_dir, run_subdir)
    copytree(dest_dir, path.join(run_dir, 'run-info'))

    if need_snapshot:
        # parse output to figure out where it put the last checkpoint
        final_checkpoint_dir = extract_by_prefix(lines, 'Snapshot directory: ')
        if final_checkpoint_dir is None:
            msg = "cannot find final snapshot from stdout; check logs!"
            raise Exception(msg)
        # choose latest snapshot
        by_num = {}
        snaps = [
            path.join(final_checkpoint_dir, bn)
            for bn in listdir(final_checkpoint_dir)
            if bn.startswith('snapshot_')
        ]
        for snap in snaps:
            bn = path.basename(snap)
            num_s = bn.split('_')[1].rsplit('.', 1)[0]
            if num_s == 'final':
                # always choose this
                num = float('inf')
            else:
                num = int(num_s)
            by_num[num] = snap
        if len(by_num) == 0:
            msg = "could not find any snapshots in '%s'" % final_checkpoint_dir
            raise Exception(msg)
        # if this fails then we don't have any snapshots
        final_checkpoint_path = by_num[max(by_num.keys())]

        return final_checkpoint_path


def build_arch_flags(arch_mod, is_train, override_enhsp_config=None, override_mse_coeff=None, override_epoch_num=None, override_sup_lr=None):
    """Build flags which control model arch and training strategy."""
    flags = []
    assert arch_mod.SUPERVISED, "only supervised training supported atm"
    if is_train:
        flags.extend(['--dropout', str(arch_mod.DROPOUT)])
    if not arch_mod.SKIP:
        flags.append('--no-skip')
    if arch_mod.DET_EVAL:
        flags.append('--det-eval')
    if not arch_mod.USE_LMCUT_FEATURES:
        flags.append('--no-use-lm-cuts')
    if arch_mod.USE_ACT_HISTORY_FEATURES:
        flags.append('--use-act-history')
    if arch_mod.TEACHER_EXPERIENCE_MODE == 'ROLLOUT':
        flags.append('--no-use-teacher-envelope')
    elif arch_mod.TEACHER_EXPERIENCE_MODE != 'ENVELOPE':
        raise ValueError(
            f"Unknown experience mode '{arch_mod.TEACHER_EXPERIENCE_MODE}'; "
            "try 'ROLLOUT' or 'ENVELOPE'")
    if arch_mod.L1_REG:
        assert isinstance(arch_mod.L1_REG, (float, int))
        l1_reg = str(arch_mod.L1_REG)
    else:
        l1_reg = '0.0'
    if arch_mod.L2_REG:
        assert isinstance(arch_mod.L2_REG, (float, int))
        l2_reg = str(arch_mod.L2_REG)
    else:
        l2_reg = '0.0'
    if override_mse_coeff:
        mse = str(override_mse_coeff)
    else:
        if arch_mod.MSE:
            assert isinstance(arch_mod.MSE, (float, int))
            mse = str(arch_mod.MSE)
        else:
            mse = '0.0'

    # optional flags
    if hasattr(arch_mod, 'MAX_OPT_EPOCHS'):
        max_opt_epochs = override_epoch_num if override_epoch_num is not None else arch_mod.MAX_OPT_EPOCHS
        assert isinstance(max_opt_epochs, int)
        flags.extend(['--max-opt-epochs', str(max_opt_epochs)])
    elif override_epoch_num is not None:
        assert isinstance(override_epoch_num, int)
        flags.extend(['--max-opt-epochs', str(override_epoch_num)])
    if hasattr(arch_mod, 'TEACHER_TIMEOUT_S'):
        assert isinstance(arch_mod.TEACHER_TIMEOUT_S, int)
        flags.extend(['--teacher-timeout-s', str(arch_mod.TEACHER_TIMEOUT_S)])
    if hasattr(arch_mod, 'USE_COMPARISONS') and arch_mod.USE_COMPARISONS:
        flags.append('--use-comparisons')
    if hasattr(arch_mod, 'USE_FLUENTS') and arch_mod.USE_FLUENTS:
        flags.append('--use-fluents')
    if hasattr(arch_mod, 'USE_NUMERIC_LANDMARKS') and arch_mod.USE_NUMERIC_LANDMARKS:
        flags.append('--use-numeric-landmarks')
    if hasattr(arch_mod, 'USE_CONTRIBUTIONS') and arch_mod.USE_CONTRIBUTIONS:
        flags.append('--use-contributions')
    if hasattr(arch_mod, 'SSIPP_TEACHER_HEURISTIC'):
        flags.extend(['--ssipp-teacher-heuristic',
                      arch_mod.SSIPP_TEACHER_HEURISTIC])
    if hasattr(arch_mod, 'FD_TEACHER_HEURISTIC'):
        flags.extend(['--fd-teacher-heuristic',
                      arch_mod.FD_TEACHER_HEURISTIC])
    if hasattr(arch_mod, 'ENHSP_CONFIG') or override_enhsp_config:
        flags.extend(['--enhsp-config', override_enhsp_config
        if override_enhsp_config else arch_mod.ENHSP_CONFIG])
    if hasattr(arch_mod, 'LIMIT_TRAIN_OBS_SIZE'):
        assert isinstance(arch_mod.LIMIT_TRAIN_OBS_SIZE, int)
        flags.extend(['--limit-train-obs-size',
                      str(arch_mod.LIMIT_TRAIN_OBS_SIZE)])
    if hasattr(arch_mod, 'EXPLORATION_ALGORITHM'):
        assert isinstance(arch_mod.EXPLORATION_ALGORITHM, str)
        flags.extend(['--exploration-algorithm',
                      arch_mod.EXPLORATION_ALGORITHM])
    if hasattr(arch_mod, 'ROLLOUTS'):
        assert isinstance(arch_mod.ROLLOUTS, int)
        flags.extend(['--rollouts', str(arch_mod.ROLLOUTS)])
    if hasattr(arch_mod, 'MIN_EXPLORED'):
        assert isinstance(arch_mod.MIN_EXPLORED, int)
        flags.extend(['--min-explored', str(arch_mod.MIN_EXPLORED)])
    if hasattr(arch_mod, 'MAX_EXPLORED'):
        assert isinstance(arch_mod.MAX_EXPLORED, int)
        flags.extend(['--max-explored', str(arch_mod.MAX_EXPLORED)])
    if hasattr(arch_mod, 'EXPLORATION_LEARNING_RATIO'):
        assert isinstance(arch_mod.EXPLORATION_LEARNING_RATIO, int)
        flags.extend(['--exploration-learning-ratio',
                      str(arch_mod.EXPLORATION_LEARNING_RATIO)])
    if hasattr(arch_mod, 'MAX_REPLAY_SIZE'):
        assert isinstance(arch_mod.MAX_REPLAY_SIZE, int)
        flags.extend(['--max-replay-size', str(arch_mod.MAX_REPLAY_SIZE)])

    # action policy flags
    if hasattr(arch_mod, 'ACTION_POLICY'):
        assert isinstance(arch_mod.ACTION_POLICY, str) and arch_mod.ACTION_POLICY in ("argmax", "sample", "visit")
        flags.extend(['--action-policy', arch_mod.ACTION_POLICY])
    if hasattr(arch_mod, 'ACTION_POLICY_GOAL_CHASE_DISTANCE_THRESHOLD'):
        assert isinstance(arch_mod.ACTION_POLICY_GOAL_CHASE_DISTANCE_THRESHOLD, int)
        flags.extend(['--action-policy-goal-chase-distance-threshold',
                      str(arch_mod.ACTION_POLICY_GOAL_CHASE_DISTANCE_THRESHOLD)])
    if hasattr(arch_mod, 'ACTION_POLICY_EPSILON'):
        assert isinstance(arch_mod.ACTION_POLICY_EPSILON, float | None)
        if arch_mod.ACTION_POLICY_EPSILON is not None:
            flags.extend(['--action-policy-epsilon', str(arch_mod.ACTION_POLICY_EPSILON)])
    if hasattr(arch_mod, 'ACTION_POLICY_TEMPERATURE'):
        assert isinstance(arch_mod.ACTION_POLICY_TEMPERATURE, float | None)
        if arch_mod.ACTION_POLICY_TEMPERATURE is not None:
            flags.extend(['--action-policy-temperature', str(arch_mod.ACTION_POLICY_TEMPERATURE)])
    if hasattr(arch_mod, 'ACTION_POLICY_DECAY_RATE'):
        assert isinstance(arch_mod.ACTION_POLICY_DECAY_RATE, float | None)
        if arch_mod.ACTION_POLICY_DECAY_RATE is not None:
            flags.extend(['--action-policy-decay-rate', str(arch_mod.ACTION_POLICY_DECAY_RATE)])

    # compulsory flags
    sup_lr_flag = override_sup_lr if override_sup_lr is not None else arch_mod.SUPERVISED_LEARNING_RATE
    flags.extend([
        '--domain-type', str(arch_mod.DOMAIN_TYPE),
        '--num-layers', str(arch_mod.NUM_LAYERS),
        '--hidden-size', str(arch_mod.HIDDEN_SIZE),
        '--l2-reg', l2_reg,
        '--l1-reg', l1_reg,
        '--mse', mse,
        '-R', str(arch_mod.EVAL_ROUNDS),
        '-L', str(arch_mod.ROUND_TURN_LIMIT) if is_train else str(arch_mod.EVAL_ROUND_TURN_LIMIT),
        '-t', str(arch_mod.TIME_LIMIT_SECONDS),
        '--supervised-lr', str(sup_lr_flag),
        '--supervised-bs', str(arch_mod.SUPERVISED_BATCH_SIZE),
        '--supervised-early-stop', str(arch_mod.SUPERVISED_EARLY_STOP),
        '--save-every', str(arch_mod.SAVE_EVERY_N_EPOCHS),
        '--opt-batch-per-epoch', str(arch_mod.OPT_BATCH_PER_EPOCH),
        '--teacher-planner', arch_mod.TEACHER_PLANNER,
        '--sup-objective', arch_mod.TRAINING_STRATEGY,
    ])  # yapf: disable
    if arch_mod.LEARNING_RATE_STEPS:
        for k, r in arch_mod.LEARNING_RATE_STEPS:
            assert k > 0, r > 0
            assert isinstance(k, int)
            assert isinstance(k, (int, float))
            flags.extend(['--lr-step', str(k), str(r)])
    return flags


def add_prefix(prefix, filenames):
    """Add a prefix directory to a bunch of filenames."""
    return [path.join(prefix, fn) for fn in filenames]


def build_prob_flags_train(prob_mod):
    """Build up some train flags for ASNets."""
    pddls = add_prefix(prob_mod.PDDL_DIR, prob_mod.COMMON_PDDLS)
    train_pddls = add_prefix(prob_mod.PDDL_DIR, prob_mod.TRAIN_PDDLS)
    pddls.extend(train_pddls)
    other_flags = []
    if prob_mod.TRAIN_NAMES:
        for tn in prob_mod.TRAIN_NAMES:
            other_flags.extend(['-p', tn])
    return other_flags + pddls

def build_prob_flags_validation(prob_mod):
    """Build validation PDDL flags grouped by difficulty."""
    flags = []

    if not hasattr(prob_mod, "VALIDATION_PDDLS"):
        return flags

    for diff, pddls in prob_mod.VALIDATION_PDDLS.items():
        prefixed = add_prefix(prob_mod.PDDL_DIR, pddls)
        flags.append(f'--validation-pddls-{diff}')
        flags.extend(prefixed)
    return flags


def build_prob_flags_test(prob_mod, allowed_idxs=None):
    """Build a list of flag sets, with one flag set for each requested
    experiment."""
    pddls = add_prefix(prob_mod.PDDL_DIR, prob_mod.COMMON_PDDLS)
    rv = []
    for idx, path_and_name in enumerate(prob_mod.TEST_RUNS):
        pddl_paths, prob_name = path_and_name
        if allowed_idxs is not None and idx not in allowed_idxs:
            print('Will skip item %d: %s' % (idx, path_and_name))
            continue
        prob_flag = []
        if prob_name is not None:
            prob_flag = ['-p', prob_name]
        these_pddls = add_prefix(prob_mod.PDDL_DIR, pddl_paths)
        rv.append((idx, prob_flag + pddls + these_pddls))
    return rv


def get_prefix_dir(checkpoint_path):
    """Turn path like experiments-results/experiments.actprop_2l-.../.../... into
    experiment-results/experiments.actprop_2l.../"""
    real_path = path.abspath(checkpoint_path)
    parts = real_path.split(path.sep)
    for idx in range(len(parts))[::-1]:
        part = parts[idx]
        if part.startswith('experiments.') or part.startswith('experiments_numeric.'):
            return path.sep.join(parts[:idx + 1])
    raise ValueError("Couldn't find experiments. or experiments_numeric. prefix in '%s'" %
                     checkpoint_path)


def parse_idx_list(idx_list):
    idx_strs = [int(s) for s in idx_list.split(',') if s.strip()]
    return idx_strs


parser = argparse.ArgumentParser(description='Run an experiment with ASNets')
parser.add_argument('--resume-from',
                    default=None,
                    help='resume experiment from given checkpoint path')
parser.add_argument(
    '--no-eval',
    default=False,
    action='store_true',
    help='do not run evaluation (only train)')
parser.add_argument(
    '--profiling',
    default=False,
    action='store_true',
    help='run cProfile on subprocesses running "run_asnets.py - do not confuse with --profile-dir which profiles exploration workers"')
parser.add_argument(
    '--memory-profiling',
    default=False,
    action='store_true',
    help='run memray profiling for memory usage across main process and rpyc workers')
parser.add_argument(
    '--serial-test',
    default=False,
    action='store_true',
    help='run test problems serially (default is to run them in parallel) '
         'subject to hardware limitations. These hardware limitations might not work'
         'correctly on all systems.')
parser.add_argument(
    '--restrict-test-probs',
    default=None,
    type=parse_idx_list,
    help='takes comma-separated list of evaluation problem numbers to test')
parser.add_argument(
    '--override-enhsp-config',
    default=None,
    help='override the ENHSP config file with this one (useful for '
         'changing ENHSP heuristic/search algorithm for different domains')
parser.add_argument(
    'arch_module',
    metavar='arch-module',
    help='import path for Python file with architecture config (e.g. '
         '"experiments.actprop_1l")')
parser.add_argument(
    'prob_module',
    metavar='prob-module',
    help='import path for Python file with problem config (e.g. '
         '"experiments.ex_blocksworld")')
parser.add_argument(
    '--random-seed',
    type=int,
    default=None,
    help='Seed.')
parser.add_argument(
    '--mcts-expansion-size',
    type=int,
    default=None,
    help='Number of MCTS Nodes to generate upon MCTS parent node expansion.')
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
    '--override-mse-coeff',
    type=float,
    default=None,
    help='Override architecture mse coefficient.'
)
parser.add_argument(
    '--max-opt-epochs',
    type=int,
    default=None,
    help='Override architecture maximum epoch count.'
)
parser.add_argument(
    '--supervised-lr',
    type=float,
    default=None,
    help='Override architecture supervised learning rate.'
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
    '--num-workers',
    type=int,
    default=4,
    help='Set the number of problem slots for the trainer\evaluator'
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
    '--estimator-h-to-v-coeff',
    type=float,
    default=None,
    help='Set "k" coefficient for e^{-k*h(s)} in conversion from estimator h value to canonical state value.'
)
parser.add_argument(
    '--use-estimator',
    action='store_true',
    default=False,
    help='Enable estimator decay, when on, each node will be estimated by an estiamtor (ENHSP) during training,'
         ' for MCTS exploration and policy+value targets,'
         ' this "help" will decay in favor of the network output along the run.'
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
    help='Set the amount of epochs estimator decays from est_coeff_start to est_coeff_end.'
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
    '--resume-train',
    action='store_true',
    default=False,
    help='Resume training instead of only evaluation when using --resume-from'
)


def main():
    args = parser.parse_args()

    # 1. load config
    print('Importing architecture from %s' % args.arch_module)
    arch_mod = import_module(args.arch_module)
    print('Importing problem from %s' % args.prob_module)
    prob_mod = import_module(args.prob_module)

    main_inner(arch_mod=arch_mod,
               prob_mod=prob_mod,
               resume_from=args.resume_from,
               resume_train=args.resume_train,
               restrict_test_probs=args.restrict_test_probs,
               override_enhsp_config=args.override_enhsp_config,
               override_mse_coeff=args.override_mse_coeff,
               override_epoch_num=args.max_opt_epochs,
               override_sup_lr=args.supervised_lr,
               serial_test=args.serial_test,
               no_eval=args.no_eval,
               profiling=args.profiling,
               memory_profiling=args.memory_profiling,
               random_seed=args.random_seed,
               mcts_expansion_size=args.mcts_expansion_size,
               train_only=args.no_eval,
               mcts_heuristic=args.mcts_heuristic,
               mcts_exploration_weight=args.mcts_exploration_weight,
               mcts_smart_expansions=args.mcts_smart_expansions,
               disable_value_head=args.disable_value_head,
               mcts_iterations=args.mcts_iterations,
               heuristic_bootstrapping=args.heuristic_bootstrapping,
               corrupt_pi=args.corrupt_pi,
               corrupt_z=args.corrupt_z,
               worker_logs=args.worker_logs,
               fixed_instance=args.fixed_instance,
               freeze_train=args.freeze_train,
               num_workers=args.num_workers,
               sample_k_additional_states=args.sample_k_additional_states,
               profile_dir=args.profile_dir,
               estimator_h_to_v_coeff=args.estimator_h_to_v_coeff,
               use_estimator=args.use_estimator,
               estimator_decay_coeff_start=args.estimator_decay_coeff_start,
               estimator_decay_coeff_end=args.estimator_decay_coeff_end,
               estimator_decay_epochs=args.estimator_decay_epochs,
               discard_failed_runs=args.discard_failed_runs,
               original_training_set=args.original_training_set,
               )
    print('Fin :-)')


def main_inner(*,
               arch_mod,
               prob_mod,
               resume_from=None,
               resume_train=None,
               restrict_test_probs=None,
               override_enhsp_config=None,
               override_mse_coeff=None,
               override_epoch_num=None,
               override_sup_lr=None,
               serial_test=None,
               no_eval=None,
               no_valid=None,
               profiling=False,
               memory_profiling=False,
               random_seed=None,
               mcts_expansion_size=None,
               train_only=False,
               mcts_heuristic=None,
               mcts_exploration_weight=1,
               mcts_smart_expansions=False,
               disable_value_head=False,
               mcts_iterations=None,
               heuristic_bootstrapping=False,
               corrupt_pi=None,
               corrupt_z=None,
               worker_logs=None,
               fixed_instance=False,
               freeze_train=False,
               num_workers=None,
               sample_k_additional_states=0,
               profile_dir=None,
               estimator_h_to_v_coeff=None,
               use_estimator=False,
               estimator_decay_coeff_start=None,
               estimator_decay_coeff_end=None,
               estimator_decay_epochs=None,
               discard_failed_runs=False,
               original_training_set=False,
               ):
    root_cwd = getcwd()

    arch_name = arch_mod.__name__
    prob_name = prob_mod.__name__
    if resume_from is None or resume_train:
        time_str = datetime.datetime.now().isoformat()
        prefix_dir = 'experiment-results/%s-%s-%s' % (prob_name, arch_name,
                                                      time_str)
        prefix_dir = path.join(root_cwd, prefix_dir)
        print('Will put everything in %s' % prefix_dir)

        # 3. train network
        print('\n\n\nTraining network')
        if train_only:
            print("--and only training, no evaluation afterwards")
        train_flags = [
            # log and snapshot dirs
            '-e', prefix_dir,
        ]  # yapf: disable
        train_flags.extend(build_arch_flags(
            arch_mod, is_train=True,
            override_enhsp_config=override_enhsp_config,
            override_mse_coeff=override_mse_coeff,
            override_epoch_num=override_epoch_num,
            override_sup_lr=override_sup_lr,
        ))
        train_flags.extend(build_prob_flags_train(prob_mod))
        if not no_valid:
            train_flags.extend(build_prob_flags_validation(prob_mod))
        print(f'''
========================================================
Starting to train network with the following parameters:
problem name: {prob_name}
flags = {train_flags}
cwd = {root_cwd}
root_dir = {prefix_dir}
timeout = {arch_mod.TIME_LIMIT_SECONDS}
evaluation = {"off" if no_eval else "on"}
========================================================
        ''', flush=True)
        if mcts_iterations:
            train_flags.extend(['--mcts-iterations', str(mcts_iterations)])
        if heuristic_bootstrapping:
            train_flags.append('--heuristic-bootstrapping')
        if mcts_exploration_weight:
            train_flags.extend(['--mcts-exploration-weight', str(mcts_exploration_weight)])
        if mcts_expansion_size:
            train_flags.extend(['--mcts-expansion-size', str(mcts_expansion_size)])
        if corrupt_pi:
            train_flags.extend(['--corrupt-pi', str(corrupt_pi)])
        if corrupt_z:
            train_flags.extend(['--corrupt-z', str(corrupt_z)])
        if worker_logs:
            train_flags.append('--worker-logs')
        if fixed_instance:
            train_flags.append('--fixed-instance')
        if freeze_train:
            train_flags.append('--freeze-train')
        if num_workers:
            train_flags.extend(['--num-workers', str(num_workers)])
        if sample_k_additional_states:
            train_flags.extend(['--sample-k-additional-states', str(sample_k_additional_states)])
        if profile_dir:
            train_flags.extend(['--profile-dir', str(profile_dir)])
        if estimator_h_to_v_coeff:
            train_flags.extend(['--estimator-h-to-v-coeff', str(estimator_h_to_v_coeff)])
        if use_estimator:
            train_flags.append('--use-estimator')
        if estimator_decay_coeff_start:
            train_flags.extend(['--estimator-decay-coeff-start', str(estimator_decay_coeff_start)])
        if estimator_decay_coeff_end:
            train_flags.extend(['--estimator-decay-coeff-end', str(estimator_decay_coeff_end)])
        if estimator_decay_epochs:
            train_flags.extend(['--estimator-decay-epochs', str(estimator_decay_epochs)])
        if discard_failed_runs:
            train_flags.append('--discard-failed-runs')
        if disable_value_head:
            train_flags.append('--disable-value-head')
        if original_training_set:
            train_flags.append('--original-training-set')
        if resume_from is not None:
            train_flags.extend(['--resume-from', resume_from])
        final_checkpoint = run_asnets_local(
            flags=train_flags,
            cwd=root_cwd,
            root_dir=prefix_dir,
            need_snapshot=True,
            is_train=True,
            timeout=arch_mod.TIME_LIMIT_SECONDS,
            train_only=train_only,
            profiling=profiling,
            memory_profiling=memory_profiling,
        )
        print('Last valid checkpoint is %s' % final_checkpoint)
    else:
        final_checkpoint = resume_from
        prefix_dir = get_prefix_dir(final_checkpoint)
        print('Resuming from checkpoint "%s"' % final_checkpoint)
        print('Using experiment dir "%s"' % prefix_dir)

    if no_eval:
        print('Skipping evaluation')
        return prefix_dir

    # 4. test network
    print('\n\n\n\n\n\nTesting network')
    main_test_flags = [
        '--no-train',
        # avoid writing extra snapshot & TB files
        '--minimal-file-saves',
        '--resume-from', final_checkpoint,
        '-e', prefix_dir,
    ]  # yapf: disable
    main_test_flags.extend(build_arch_flags(
        arch_mod, is_train=False,
        override_enhsp_config=override_enhsp_config))

    if random_seed is not None:
        main_test_flags.extend(['--seed', str(random_seed)])
    if mcts_expansion_size is not None:
        main_test_flags.extend(['--mcts-expansion-size', str(mcts_expansion_size)])
    if mcts_heuristic is not None:
        main_test_flags.extend(['--mcts-heuristic', str(mcts_heuristic)])
    if mcts_exploration_weight != 1:
        main_test_flags.extend(['--mcts-exploration-weight', str(mcts_exploration_weight)])
    if mcts_smart_expansions:
        main_test_flags.append('--mcts-smart-expansions')
    if disable_value_head:
        main_test_flags.append('--disable-value-head')
    if num_workers:
        main_test_flags.extend(['--num-workers', str(num_workers)])
    if mcts_iterations:
        main_test_flags.extend(['--mcts-iterations', str(mcts_iterations)])

    prob_flag_list = build_prob_flags_test(prob_mod, restrict_test_probs)
    if serial_test:
        print('Starting serial test loop')

        for prob_idx, test_prob_flags in prob_flag_list:
            print('Launching test on problem %d' % (prob_idx + 1))
            full_flags = main_test_flags + test_prob_flags

            # do not place a memory limit on the serial test
            run_asnets_local(
                flags=full_flags,
                root_dir=prefix_dir,
                cwd=root_cwd,
                need_snapshot=False,
                is_train=False,
                # run_asnets.py has its own timeout which it should obey, so
                # give it some slack
                timeout=arch_mod.EVAL_TIME_LIMIT_SECONDS + 30,
                profiling=profiling,
                memory_profiling=memory_profiling,
            )
    else:
        all_test_flags = list(main_test_flags)

        domain = None
        instances = []

        for _, test_prob_flags in prob_flag_list:
            if domain is None:
                domain = test_prob_flags[0]
            instances.append(test_prob_flags[1])

        all_test_flags.append(domain)
        all_test_flags.extend(instances)

        run_asnets_local(
            flags=all_test_flags,
            root_dir=prefix_dir,
            cwd=root_cwd,
            need_snapshot=False,
            is_train=False,
            timeout=arch_mod.EVAL_TIME_LIMIT_SECONDS + 30,
            profiling=profiling,
            memory_profiling=memory_profiling,
        )
    # return the prefix_dir because hype.py needs that to figure out where to
    # point collate_results at
    return prefix_dir


if __name__ == '__main__':
    main()
