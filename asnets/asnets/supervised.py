import queue
import threading
import traceback
from collections import Counter, deque
from enum import Enum
from functools import lru_cache
from itertools import repeat
import joblib
import logging
import numpy as np
import os
import rpyc
import setproctitle
import shutil
import tensorflow as tf
from time import time
import tqdm.auto as tqdm
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set
import cProfile
import datetime
from asnets.heur_inputs import ActionCountDataGenerator, \
    HeuristicDataGenerator, LMCutDataGenerator, RelaxedDeadendDetector, \
    NumericLandmarkGenerator
from asnets.models import PropNetworkWeights, PropNetwork
from asnets.utils.generator_utils import InstanceDifficulty, get_problem_names
from asnets.utils.mdpsim_utils import parse_problem_args
from asnets.utils.rpyc_utils import to_local, find_netrefs
from asnets.prob_dom_meta import BoundAction, DomainType, get_domain_meta, \
    get_problem_meta
from asnets.interfaces.jpddl_interface import start_jvm
from asnets.interfaces.ssipp_interface import set_up_ssipp
from asnets.state_reprs import compute_observation_dim, compute_action_dim, \
    get_action_name, sample_next_state, get_init_cstate, CanonicalState
from asnets.teacher import DomainSpecificTeacher, FDTeacher, MetricFFTeacher, \
    SSiPPTeacher, Teacher, TeacherException, ENHSPTeacher
from asnets.utils.prof_utils import can_profile
from asnets.utils.pddl_utils import get_domain_file
from asnets.utils.py_utils import RandomPopContainer, TimerContext, \
    strip_parens, weak_ref_to
from asnets.utils.tf_utils import cross_entropy, mean_squared_error
from post_training.monte_carlo_tree_search import MCTSNode
from post_training.training_mcts import TrainingMCTS
import jpype
import jpype.imports
import sys

J_PDDLDomain = None
J_PDDLProblem = None

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# ---- diagnostics: catch everything that reaches top-level ----

def _global_excepthook(exc_type, exc_value, exc_tb):
    import os, sys, traceback, multiprocessing

    # Capture the traceback text
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

    # Detect whether this came from a worker (via RPyC)
    is_remote = "========= Remote Traceback" in tb_text

    # Determine role
    proc = multiprocessing.current_process()
    role = "worker" if is_remote else (proc.name.lower() or "trainer")

    # Print nicely
    print(f"[{role.upper()} EXCEPTION PID={os.getpid()}]")
    print(tb_text)
    sys.stderr.flush()


sys.excepthook = _global_excepthook


def format_seconds_as_dhm(seconds_float):
    # Create a timedelta object
    timedelta_duration = datetime.timedelta(seconds=seconds_float)
    # The string representation is close to the desired format
    return str(timedelta_duration)


@jpype.onJVMStart
def _import_java_classes() -> None:
    """Import Java classes that will be used by this module. This is called
    automatically upon JVM start-up.
    """
    global J_PDDLDomain, J_PDDLProblem

    J_PDDLDomain = jpype.JPackage('com').hstairs.ppmajal.domain.PDDLDomain
    J_PDDLProblem = jpype.JPackage('com').hstairs.ppmajal.problem.PDDLProblem


class WeightedReplayBuffer:
    """Replay buffer for previously-encountered states. The 'weighted' in the
    name comes from the fact that it's really a multiset that lets you sample
    states weighted by multiplicity."""

    def __init__(self):
        """Initialize the replay buffer."""
        self.counter = Counter()
        self.added_items = deque()

    def update(self, new_elems: Iterable[Any]) -> None:
        """Add new elements to the replay buffer.

        Args:
            new_elems (Iterable[Any]): New elements to add to the replay
            buffer.
        """
        item_counter = Counter(new_elems)
        self.counter.update(item_counter)
        self.added_items.append(item_counter)

    def __len__(self) -> int:
        """Get the number of unique elements in the replay buffer.

        Returns:
            int: Number of unique elements in the replay buffer.
        """
        return len(self.counter)

    def get_full_dataset(self) -> Tuple[List[Any], List[int]]:
        """Get the full dataset stored in the replay buffer.

        Returns:
            Tuple[List[Any], List[int]]: List of elements in the replay buffer
            and list of their counts.
        """
        rich_dataset = list(self.counter)
        counts = [self.counter[item] for item in rich_dataset]
        return rich_dataset, counts

    def remove_oldest(self):
        """Remove the oldest element from the replay buffer."""
        # make sure we do not empty the replay buffer
        if len(self.added_items) <= 1:
            return

        item_counter = self.added_items.popleft()
        self.counter.subtract(item_counter)
        self.counter += Counter()  # remove zero and negative counts


class ProblemServiceConfig(object):
    """Configuration for a ProblemService. This is a separate class so that
    the config can be serialised and sent to the remote server."""

    def __init__(
            self,
            pddl_files: List[str],
            # init_problem_name: str,
            domain_type: DomainType,
            *,
            domain=None,
            ssipp_dg_heuristic: str = None,
            use_lm_cuts: bool = False,
            use_numeric_landmarks: bool = False,
            use_contributions: bool = False,
            use_act_history: bool = False,
            # ??? what does this do?
            # Oh, it controls the maximum length of training trajectories! That
            # explains why I'm not able to solve some certain big training
            # problems.
            # FIXME: this max_len should be adjusted based on the V(s0)
            # calculated by the teacher planner! Maybe add a separate method
            # for that (like "exposed_find_path_length") that plans on the
            # first state & uses the result to figure out what length should
            # be.
            fd_heuristic="astar-hadd",
            ssipp_teacher_heuristic: str = 'lm-cut',
            enhsp_config: str = 'hadd-gbfs',
            max_len: int = 50,
            her_k: int = 0,
            training_mcts_iterations: int = 10,
            planner_bootstrapping: bool = False,
            planner_bootstrapping_her: bool = False,
            heuristic_bootstrapping: bool = False,
            difficulty: InstanceDifficulty = InstanceDifficulty.EASY,
            bootstrap_k: int = 3,
            mcts_expansion_k: int = 10,
            mcts_her_strategy: bool = False,
            teacher_planner: str,
            random_seed: int = None,
            teacher_timeout_s: int = 1800,
            only_one_good_action: bool = False,
            use_teacher_envelope: bool = True,
            use_fluents: bool = False,
            use_comps: bool = False,
            slot_id: int = None,
    ):
        """Initialise a ProblemServiceConfig. This Config will allow
        initialisation of a ProblemService, which involves:
        - Initialising mdpsim and ssipp (requires pddl_files, problem_name)
        - Initialising data generators. This might be easiest to achieve with
          just a list of generator class names and arguments (although I
          still need to make sure those are actually deep copied, grumble
          grumble).

        Args:
            pdll_files (List[str]): List of PDDL files to load.
            init_problem_name (str): Name of the problem to load.
            domain_type (DomainType): Type of the domain.
            ssipp_dg_heuristic (str, optional): Name of the heuristic to use.
            Defaults to None.
            use_lm_cuts (bool, optional): Whether to use lm-cut heuristic.
            Defaults to False.
            use_act_history (bool, optional): Whether to use action history
            as input to the heuristic. Defaults to False.
            fd_heuristic (str, optional): Name of the heuristic to use for 
            FastDownward. Defaults to 'astar-hadd'.
            ssipp_teacher_heuristic (str, optional): Name of the heuristic to
            use for SSiPP. Defaults to 'lm-cut'.
            enhsp_config (str, optional): Name of the configuration to use for
            unified-planning when using ENHSP. Defaults to None.
            max_len (int, optional): Maximum length of training trajectories.
            Defaults to 50.
            teacher_planner (str, optional): Name of the planner to use for
            teacher. Defaults to None.
            random_seed (int, optional): Random seed to use. Defaults to None.
            only_one_good_action (bool, optional): Whether to only use the
            teacher action as a positive example. Controls whether planner
            should return accurate Q-values (False) or return Q-values that only
            make its favourite action look good (True). Defaults to False.
            use_teacher_envelope (bool, optional): Whether to use an entire
            policy envelope from teacher (True), or just a rollout (False).
            Defaults to True.
        """
        self.pddl_files = pddl_files
        # self.init_problem_name = init_problem_name
        self.domain = domain
        self.difficulty = difficulty
        self.domain_type = domain_type
        self.ssipp_dg_heuristic = ssipp_dg_heuristic
        self.use_lm_cuts = use_lm_cuts
        self.use_numeric_landmarks = use_numeric_landmarks
        self.use_contributions = use_contributions
        self.use_act_history = use_act_history
        self.fd_heuristic = fd_heuristic
        self.ssipp_teacher_heuristic = ssipp_teacher_heuristic
        self.enhsp_config = enhsp_config
        self.max_len = max_len
        self.random_seed = random_seed
        self.teacher_planner = teacher_planner
        self.teacher_timeout_s = teacher_timeout_s
        self.only_one_good_action = only_one_good_action
        self.use_teacher_envelope = use_teacher_envelope
        self.her_k = her_k
        self.training_mcts_iterations = training_mcts_iterations
        self.planner_bootstrapping = planner_bootstrapping
        self.planner_bootstrapping_her = planner_bootstrapping_her
        self.heuristic_bootstrapping = heuristic_bootstrapping
        self.mcts_her_strategy = mcts_her_strategy
        self.bootstrap_k = bootstrap_k
        self.mcts_expansion_k = mcts_expansion_k
        self.use_fluents = use_fluents
        self.use_comps = use_comps
        self.slot_id = slot_id


class PlannerExtensions(object):
    """Wrapper to hold references to SSiPP and MDPSim modules, and references
    to the relevant loaded problems (like the old ModuleSandbox). Mostly
    keeping this because it makes it convenient to pass stuff around, as I
    often need SSiPP and MDPSim at the same time."""

    def __init__(self,
                 pddl_files: List[str],
                 domain_type: DomainType,
                 *,
                 dg_ssipp_heuristic_name: str = None,
                 dg_use_lm_cuts: bool = False,
                 dg_use_numeric_landmarks: bool = False,
                 dg_use_contributions: bool = False,
                 dg_use_act_history: bool = False,):
        """Initialise a PlannerExtensions object.

        Args:
            pddl_files (List[str]): The PDDL files to load.
            init_problem_name (str): The name of the problem to load.
            domain_type (DomainType): The type of the domain.
            dg_ssipp_heuristic_name (str, optional): The heuristic feature
            generator to use. Defaults to None.
            dg_use_lm_cuts (bool, optional): Whether to use the lm-cut heuristic
            feature generator. If the domain is numeric, will perform numeric
            relaxation. Defaults to False.
            dg_use_numeric_landmarks (bool, optional): Whether to use the
            additive numeric landmarks feature generator. Defaults to False.
            dg_use_act_history (bool, optional): Whether to use the action count
            data generator. Defaults to False.
        """
        self.pddl_files = pddl_files # domain file
        self.domain_type = domain_type

        import mdpsim  # noqa: F811
        import ssipp  # noqa: F811
        current_problem = pddl_files[1]
        current_problem_name = get_problem_names([current_problem])[0]

        print(f'Starting to parse mdpsim problem: {current_problem_name}')
        # MDPSim stuff
        self.mdpsim: ModuleType = mdpsim
        self.mdpsim_problem = parse_problem_args(self.mdpsim, self.pddl_files, current_problem_name)
        self.problem_name: str = self.mdpsim_problem.name.strip()

        print(f'Finished parsing mdpsim problem: {self.problem_name}')

        # Maps to PyGroundAction object in MDPSim. Cannot use type hint.
        self.act_ident_to_mdpsim_act: Dict[str, Any] = {
            strip_parens(a.identifier): a
            for a in self.mdpsim_problem.ground_actions
        }
        self.act_ident_to_ind: Dict [str, int] = {
            "("+a+")" : i for i, a in enumerate(self.act_ident_to_mdpsim_act.keys())
        }
        LOGGER.debug(f'Python-side extra data')
        # Python-side extra data
        self.domain_meta = get_domain_meta(self.mdpsim_problem.domain)
        self.problem_meta = get_problem_meta(self.mdpsim_problem,
                                             self.domain_meta)

        LOGGER.debug(f'Using domain type: {self.domain_type}')
        # Either use JPDDL (numeric) or SSiPP (otherwise), ugly!
        if self.domain_type == DomainType.NUMERIC:
            domain_file = get_domain_file(self.pddl_files)

            LOGGER.debug(f"Process {os.getpid()} Starting JVM...")
            start_jvm()

            LOGGER.debug("Creating J_PDDLDomain...")
            self.j_domain = J_PDDLDomain(domain_file)

            LOGGER.debug("Creating J_PDDLProblem...")
            self.j_problem = J_PDDLProblem(str(current_problem), self.j_domain)
            LOGGER.debug("Calling prepareForSearch...")

            self.j_problem.prepareForSearch(
                True,  # enable AIBR preprocessing
                False  # stop after grounding
            )
            LOGGER.debug("JPDDL init done.")

            if dg_use_lm_cuts:
                # set up SSiPP using numeric relaxed problems
                self.ssipp: ModuleType = ssipp
                self.ssipp_problem = set_up_ssipp(
                    self.ssipp, self.pddl_files, self.problem_name,
                    use_numeric_relaxation=True)

                self.ssipp_ssp_iface = ssipp.SSPfromPPDDL(self.ssipp_problem)

        elif self.domain_type == DomainType.PROBABILISTIC:
            # SSiPP stuff
            self.ssipp: ModuleType = ssipp
            self.ssipp_problem = set_up_ssipp(self.ssipp, self.pddl_files,
                                              self.problem_name)
            # this leaks for some reason; will store it here so I don't have to
            # reconstruct
            #
            # This is an object of the SSPfromPPDDL class, which inherits the
            # interface SSPIface supporting the following methods:
            # - s0: Get initial state for problem
            # - isGoal: Check whether given state is a goal state
            # - applicableActions: List of actions applicable in this state.
            self.ssipp_ssp_iface = ssipp.SSPfromPPDDL(self.ssipp_problem)

        # now set up data generators
        data_gens = [
        ]

        # Domain type specific data generators
        if self.domain_type == DomainType.PROBABILISTIC:
            data_gens.append(RelaxedDeadendDetector(weak_ref_to(self)))

            if dg_ssipp_heuristic_name is not None:
                heur_gen = HeuristicDataGenerator(
                    weak_ref_to(self), dg_ssipp_heuristic_name)
                data_gens.append(heur_gen)

        elif self.domain_type == DomainType.NUMERIC and \
                dg_use_numeric_landmarks:
            numeric_landmark_gen = NumericLandmarkGenerator(
                weak_ref_to(self),
                dg_use_contributions,
                verbose=False)
            data_gens.append(numeric_landmark_gen)

        # Generic data generators
        if dg_use_act_history:
            ad_data_gen = ActionCountDataGenerator(self.problem_meta)
            data_gens.append(ad_data_gen)

        if dg_use_lm_cuts:
            lm_cut_gen = LMCutDataGenerator(weak_ref_to(self))
            data_gens.append(lm_cut_gen)

        self.data_gens = data_gens

    @property
    def ssipp_dead_end_value(self) -> int:
        """Get the value of the dead end state in SSiPP.

        Returns:
            int: The value of the dead end state in SSiPP.
        """
        # HACK We no longer always initialise SSiPP as it is problematic
        # (especially for numeric domains). SSiPP just hardcodes this as 500.
        return 500

    def __del__(self):
        import jpype, os
        print(f"[DEBUG GC] JVM running at del? {jpype.isJVMStarted()} PID={os.getpid()}")

    def update_difficulty(self, difficulty: int):
        assert 0 <= difficulty <= 2
        assert type(difficulty) == int
        self.difficulty = InstanceDifficulty.EASY if difficulty == 0 else InstanceDifficulty.MEDIUM if difficulty == 1 else InstanceDifficulty.HARD

def cosine_similarity(p, q):
    # p, q: numpy vectors representing policies
    dot = np.dot(p, q)
    norm = np.linalg.norm(p) * np.linalg.norm(q)
    if norm == 0:
        return 0.0
    return dot / norm


tf_logger = logging.getLogger('TF_SUMMARY_SCALAR_LOG')


def log_value_preds(value_pred_by_prob):
    combined_tensor = tf.concat(value_pred_by_prob, axis=0)
    LOGGER.info(
        f"[VALUE_PRED_LOG - across problems] mean: {tf.reduce_mean(combined_tensor)}, min: {tf.reduce_min(combined_tensor)}, max: {tf.reduce_max(combined_tensor)}")


def log_grad_norms(grads_and_vars):
    policy_grads = []
    value_grads = []

    for grad, var in grads_and_vars:
        if "final_act" in var.name.lower():  # policy head
            if grad is not None:
                policy_grads.append(tf.norm(grad))
        if "value_out" in var.name.lower():  # value head
            if grad is not None:
                value_grads.append(tf.norm(grad))
        if grad is None:
            LOGGER.error(f"GRADIENT IS NONE for variable {var.name}")
        else:
            LOGGER.warning(
                f"Grad stats for {var.name}: mean={tf.reduce_mean(tf.abs(grad)).numpy()}, max={tf.reduce_max(tf.abs(grad)).numpy()}")

    policy_grad_norm = tf.reduce_mean(policy_grads)
    value_grad_norm = tf.reduce_mean(value_grads)

    base_name = tf.get_current_name_scope()
    tf_logger.info(
        f"[TF_GRAD_NORMS_LOG] {base_name + '/' if base_name is not None else ''}policy_grad_norm : {policy_grad_norm}")
    tf_logger.info(
        f"[TF_GRAD_NORMS_LOG] {base_name + '/' if base_name is not None else ''}value_grad_norm : {value_grad_norm}")


def log_policy_target(pi_target_batch, problem: 'SingleProblem'):
    sampled_pi_targets_ind = np.random.choice(pi_target_batch.shape[0], size=3, replace=False)
    for pi_target_ind in sampled_pi_targets_ind:
        pi_target = pi_target_batch[pi_target_ind]
        pi_target_first_ten_entries = pi_target[:10]
        pi_target_sum = np.sum(pi_target)
        pi_target_argmax = np.argmax(pi_target)
        pi_target_argmax_name = problem.prob_meta.bound_acts_ordered[pi_target_argmax].__str__()
        LOGGER.info(
            f"[POLICY_TARGET_LOG - {problem.name}] pi_target (first 10 entries): {pi_target_first_ten_entries}, sum: {pi_target_sum} argmax: {pi_target_argmax}|{pi_target_argmax_name}")


def tf_and_log(name: str, value):
    tf.summary.scalar(name, value)
    base_name = tf.get_current_name_scope()
    # print(f"[TF_SUMMARY_SCALAR_LOG] {base_name + '/' if base_name is not None else ''}{name} : {value}")
    tf_logger.info(f"[TF_SUMMARY_SCALAR_LOG] {base_name + '/' if base_name is not None else ''}{name} : {value}")

@lru_cache(None)
def mock_qvalues(planner: Teacher,
                 planner_exts: PlannerExtensions,
                 action: Optional[str]):
    prob_meta = planner_exts.problem_meta
    if action is None:
        # no good action
        num_acts = len(prob_meta.bound_acts_ordered)
        q_values = [planner.dead_end_value] * num_acts
    else:
        assert action is not None
        planner_action_ident = action.strip('()')
        assert not planner_action_ident.startswith(')') \
               and not planner_action_ident.endswith(')')
        q_values = []
        found = False
        unique_idents = [
            ba.unique_ident for ba in prob_meta.bound_acts_ordered
        ]
        for unique_ident in unique_idents:
            if unique_ident == planner_action_ident:
                q_values.append(0)
                found = True
            else:
                q_values.append(planner.dead_end_value)
        assert found, \
            "no match for '%s' in '%s'" \
            % (planner_action_ident, ", ".join(unique_idents))

    return q_values


@can_profile
def planner_trace(planner: Teacher,
                  planner_exts: PlannerExtensions,
                  root_cstate: CanonicalState,
                  only_one_good_action: bool,
                  use_teacher_envelope: bool) \
        -> List[Tuple[CanonicalState, List[Tuple[BoundAction, float]]]]:
    """Extract (s, [q*]) pairs for all s reachable from (state) under some
    (arbitrary) optimal policy.

    Args:
        planner (Teacher): The teacher object to use for planning.
        planner_exts (PlannerExtensions): The planner extensions object.
        root_cstate (CanonicalState): The root state to start planning from.
        only_one_good_action (bool): If True, only the best action will be
        used for each state. This makes planning much faster, but may have an
        effect on learning (either good or bad) in some domains.
        use_teacher_envelope (bool): If True, the expert policy envelope will be
        used for planning. If False, the expert policy rollout will be used.

    Returns:
        List[Tuple[CanonicalState, List[Tuple[BoundAction, float]]]]: A list of
        states with their corresponding Q-values.
    """
    # TODO: do I need to explicitly cache this, or is extract_policy_envelope
    # fast enough?
    prob_meta = planner_exts.problem_meta
    pairs = []
    # not sure how expensive this is, but IIRC not very, so it shouldn't matter
    # if we do it on every epoch
    if use_teacher_envelope:
        pol_list = planner.extract_policy_envelope(root_cstate)
    else:
        pol_list = planner.expert_policy_rollout(root_cstate)
    for i, new_cstate in enumerate(pol_list):
        if only_one_good_action:
            # Shortcut: we get the planner to give us just the single best
            # action, and then construct a vector of pseudo-Q-values which will
            # favour that action. This makes planning much faster, and may have
            # an effect on learning (either good or bad) in some domains.
            planner_action_raw = planner.single_action_label(new_cstate)
            q_values = mock_qvalues(planner, planner_exts, planner_action_raw)
        else:
            # otherwise, get real q-values for all enabled actions; rest get
            # dead_end_value
            en_indices = []
            en_act_names = []
            for idx, (ba, en) in enumerate(new_cstate.acts_enabled):
                if not en:
                    continue
                en_indices.append(idx)
                en_act_names.append('(%s)' % ba.unique_ident)
            en_q_values = planner.q_values(new_cstate, en_act_names)
            assert len(en_q_values) == len(en_indices)
            q_values = [planner.dead_end_value] * len(new_cstate.acts_enabled)
            for idx, value in zip(en_indices, en_q_values):
                q_values[idx] = value

        assert len(prob_meta.bound_acts_ordered) == len(q_values)
        qv_tuple = tuple(zip(prob_meta.bound_acts_ordered, q_values))
        pairs.append((new_cstate, qv_tuple))

    return pairs


class SupervisedObjective(Enum):
    # use xent loss to choose any action with minimal Q-value
    ANY_GOOD_ACTION = 0
    # maximise expected teacher advantage of action taken by policy
    MAX_ADVANTAGE = 1
    # get the teacher to give you an arbitrary good action and use xent loss to
    # match exactly that action (& not the others); makes planning faster!
    THERE_CAN_ONLY_BE_ONE = 2
    # Use MCTS policy distribution instead of a teacher altogether
    MCTS_POLICY_DIST = 3


class SupervisedTrainer:
    @can_profile
    def __init__(self,
                 # problems,
                 weight_manager,
                 summary_writer,
                 explorer,
                 strategy,
                 start_time,
                 scratch_dir,
                 snapshot_dir,
                 *,
                 batch_size=64,
                 lr=0.001,
                 lr_steps=[],
                 opt_batches_per_epoch=300,
                 l1_reg_coeff,
                 l2_reg_coeff,
                 l1_l2_reg_coeff,
                 mse_coeff,
                 save_training_set=None,
                 use_saved_training_set=None,
                 hide_progress=False,
                 use_fluents=False,
                 use_comps=False,
                 time_out=40,
                 early_stop=20,
                 save_every=20,
                 dk="dk",
                 policy_only=False,
                 balanced_success_rate=True,
                 ):
        # gets incremented to deal with TF
        self.batches_seen = 0
        self.policy_only = policy_only
        self.balanced_success_rate = balanced_success_rate
        self.weight_manager = weight_manager
        # may be None if no summaries tuple()should be written
        self.summary_writer = summary_writer
        self.explorer = explorer
        self.batch_size_per_problem = max(batch_size // self.explorer.num_slots(), 1)
        self.opt_batches_per_epoch = opt_batches_per_epoch
        self.hide_progress = hide_progress
        self.strategy = strategy
        self.tf_init_done = False
        self.lr = lr
        self.l1_reg_coeff = l1_reg_coeff
        self.l2_reg_coeff = l2_reg_coeff
        self.mse_coeff = mse_coeff
        self.l1_l2_reg_coeff = l1_l2_reg_coeff
        self.timer = TimerContext()
        self.save_training_set = save_training_set
        self.use_saved_training_set = use_saved_training_set
        if use_saved_training_set:
            LOGGER.info("Loading saved training set from '%s'",
                        use_saved_training_set)
            self.loaded_training_set = joblib.load(use_saved_training_set)
        lr_steps = [(0, lr)] + sorted(lr_steps)
        for k, lr in lr_steps:
            assert k >= 0, "one of the steps was negative (?)"
            assert isinstance(k, int), \
                "one of the LR step epoch nums (%s) was not an int" % (k,)
            assert lr > 0, \
                "one of the given learning rates was not positive (?)"
        self.lr_steps = lr_steps
        self.lr_steps_remaining = list(lr_steps)
        self.start_time = start_time
        self.timeout = time_out
        self.early_stop = early_stop
        self.save_every = save_every
        self.scratch_dir = scratch_dir
        self.snapshot_dir = snapshot_dir
        self.dk = dk
        self.use_fluents = use_fluents
        self.use_comps = use_comps
        self._init_tf()


    @can_profile
    def _init_tf(self):
        """Do setup necessary for network (e.g. initialising weights)."""
        assert not self.tf_init_done, \
            "this class is not designed to be initialised twice"

        LOGGER.info('Initialising network structure')

        if len(self.lr_steps) > 1:
            # using a scheduler to control the learning rate
            boundaries = [i[0] for i in self.lr_steps[1:]]
            values = [i[1] for i in self.lr_steps]
            lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries, values)
            self.optimiser = tf.keras.optimizers.Adam(
                learning_rate=lr_scheduler)
        else:
            self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.lr)
        # self.optimiser.build(self.weight_manager.all_weights)
        # assert len(self.optimiser.variables) > 1, 'optimiser build wasn\'t successful'
        # self.loss_fn = ManualLoss(
        #     problems=self.problems,
        #     weight_manager=self.weight_manager,
        #     summary_writer=self.summary_writer,
        #     l1_reg_coeff=self.l1_reg_coeff,
        #     l2_reg_coeff=self.l2_reg_coeff,
        #     l1_l2_reg_coeff=self.l1_l2_reg_coeff,
        #     mse_coeff=self.mse_coeff,
        #     name="loss_fn",
        #     # strategy=SupervisedObjective.ANY_GOOD_ACTION
        #     strategy=self.strategy,
        # )
        # tensorboard ops
        self._log_ops = {}

        # self.sess.graph.finalize()
        self.tf_init_done = True

    def _optimise(self, n_batches):
        params = self.weight_manager.all_weights

        all_batches_iter = self._make_batches(n_batches)
        tr = tqdm.tqdm(all_batches_iter, desc='batch', total=n_batches)

        sample_indices = np.random.choice(n_batches, size=min(3, n_batches), replace=False)

        start_time = time()
        losses = []

        for feed_dict in tr:
            with tf.name_scope('grads_opt'):
                with tf.GradientTape() as tape:
                    per_prob_losses = []
                    per_prob_weights = []

                    for i, problem in enumerate(self.problems):
                        batch = feed_dict[i]

                        # empty_feed_value(...) should return arrays with correct shapes
                        # but may represent "no real data". Detect and skip.
                        if batch is None:
                            continue

                        if self.policy_only:
                            obs, pi_target = batch
                        else:
                            obs, pi_target, z_target = batch

                        # Detect “empty” batch (your empty_feed_value likely returns zeros)
                        # If you have a cleaner sentinel, use that instead.
                        if obs is None or len(obs) == 0:
                            continue

                        # Ensure batch dims
                        obs = np.asarray(obs)
                        if obs.ndim == 1:
                            obs = np.expand_dims(obs, axis=0)

                        if tr.n in sample_indices:
                            # optional debug hook
                            try:
                                log_policy_target(pi_target, problem)
                            except Exception:
                                pass

                        if self.policy_only:
                            pi_pred = problem.network(obs)
                            loss_i = self.loss_fn([pi_pred], [pi_target])
                        else:
                            pi_pred, v_pred = problem.network(obs)
                            loss_i = self.loss_fn(
                                [pi_pred], [pi_target],
                                target_values=[z_target],
                                pred_values=[v_pred],
                            )

                        # Weight by how many samples contributed in this slot-batch
                        bs_i = int(obs.shape[0])
                        per_prob_losses.append(loss_i * bs_i)
                        per_prob_weights.append(bs_i)

                    # Pool across slots (epoch-pooled)
                    assert len(per_prob_weights) > 0, "No non-empty batches — epoch_data had nothing usable."
                    total_w = tf.cast(tf.add_n([tf.constant(w, dtype=tf.float32) for w in per_prob_weights]),
                                      tf.float32)
                    loss = tf.add_n(per_prob_losses) / total_w

                grads = tape.gradient(loss, params)
                self.optimiser.apply_gradients(zip(grads, params))

            tr.set_postfix(loss=float(loss))
            losses.append(float(loss))
            if (self.batches_seen % 10) == 0:
                tf_and_log('train-loss', loss)
            self.batches_seen += 1

        self.explorer.update_learning_time(time() - start_time)
        return float(np.mean(losses))

    def train(self, max_epochs):
        best_rate = None
        time_since_best = 0
        solve_thresh = 0.8
        early_stop_first_epoch = self.explorer.estimator_decay_end_epoch() + self.early_stop if self.early_stop else 0
        good_epoch_thresh = 0.6
        good_epoch_num = 0
        good_epoch_cap = 30

        tr = tqdm.trange(max_epochs, desc='epoch', leave=True)
        epoch = tf.Variable(0, dtype=tf.int64)
        self.summary_writer.set_as_default(step=epoch)

        for epoch_num in tr:
            epoch.assign(epoch_num)

            # --------------------------------------------------
            # 1. EXPLORE (spawn workers, compute grads there)
            # --------------------------------------------------
            t_explore = time()
            weights_np = self.weight_manager.export_numpy()
            worker_outs = self.explorer.explore(weights_np)
            print(f"[EXPLORE TIMING] pid={os.getpid()} total={time() - t_explore:.2f}s", flush=True)

            if not worker_outs:
                LOGGER.warning("No worker outputs this epoch")
                continue

            # --------------------------------------------------
            # 1.1. LOGGING (output logs from trajectories)
            # --------------------------------------------------
            if getattr(self.explorer, "log", False):
                target_entropies = [
                    w.root_target_entropy for w in worker_outs
                    if w.root_target_entropy is not None
                ]

                pred_entropies = [
                    w.root_pred_entropy for w in worker_outs
                    if w.root_pred_entropy is not None
                ]

                kls = [
                    w.root_kl for w in worker_outs
                    if w.root_kl is not None
                ]

                if target_entropies:
                    mean_entropy = float(np.mean(target_entropies))
                    tf_and_log("mcts/root_target_entropy", mean_entropy)

                if pred_entropies:
                    mean_entropy = float(np.mean(pred_entropies))
                    tf_and_log("mcts/root_pred_entropy", mean_entropy)

                if kls:
                    mean_kl = float(np.mean(kls))
                    tf_and_log("mcts/root_kl", mean_kl)

            # --------------------------------------------------
            # 2. APPLY GRADIENTS (MAIN PROCESS ONLY)
            # --------------------------------------------------
            W0 = [w.numpy().copy() for w in self.weight_manager.all_weights]
            mean_loss, total_succ_rate, n_states = self.apply_worker_grads(worker_outs)
            succ_rate_easy, succ_rate_medium, succ_rate_hard = self.calculate_balanced_succ_rate(worker_outs)
            if getattr(self.explorer, "log", False):
                w = self.weight_manager.all_weights[0]
                print("MAIN after update:", float(tf.reduce_mean(w)), float(tf.math.reduce_std(w)),
                      float(tf.linalg.norm(w)))
            W1 = self.weight_manager.all_weights
            deltas = [np.mean(np.abs(w1.numpy() - w0)) for w0, w1 in zip(W0, W1)]
            tf_and_log("weight-delta/mean", np.mean(deltas))
            tf_and_log("weight-delta/max", np.max(deltas))
            tf_and_log('train-loss', mean_loss)
            tf_and_log('succ-rate/mean', total_succ_rate)

            present_diffs = {o.instance_diff for o in worker_outs}
            active_rates = []
            if InstanceDifficulty.EASY in present_diffs:
                tf_and_log('succ-rate/easy', succ_rate_easy)
                active_rates.append(succ_rate_easy)
            if InstanceDifficulty.MEDIUM in present_diffs:
                tf_and_log('succ-rate/medium', succ_rate_medium)
                active_rates.append(succ_rate_medium)
            if InstanceDifficulty.HARD in present_diffs:
                tf_and_log('succ-rate/hard', succ_rate_hard)
                active_rates.append(succ_rate_hard)
            if active_rates:
                balanced_rate = sum(active_rates) / len(active_rates)
                tf_and_log('succ-rate/balanced', balanced_rate)

            tf_and_log('states', n_states)
            tf_and_log('lr', self.optimiser.lr)

            if active_rates:
                total_succ_rate = balanced_rate # if we want to balance rates, this is the real deal
                #TODO: make sure this doesnt fuck up later
            tr.set_postfix(
                succ_rate=total_succ_rate,
                net_loss=mean_loss,
                states=n_states,
                lr=self.optimiser.lr,
            )

            # --------------------------------------------------
            # 3. EARLY STOP / SNAPSHOT LOGIC (unchanged)
            # --------------------------------------------------
            if best_rate is None or total_succ_rate > best_rate + 1e-4:
                time_since_best = 0
            else:
                time_since_best += 1

            should_save = (
                    best_rate is None
                    or total_succ_rate >= best_rate
                    or (self.save_every and epoch_num % self.save_every == 0)
                    or epoch_num == 1
            )

            if should_save:
                best_rate = total_succ_rate
                snapshot_path = os.path.join(
                    self.snapshot_dir,
                    f'snapshot_{epoch_num}_{total_succ_rate:.4f}.pkl'
                )
                self.weight_manager.save(snapshot_path)
                shutil.copy(snapshot_path, self.dk)
            tf.summary.flush()

            elapsed_time = time() - self.start_time
            if self.timeout and elapsed_time > self.timeout * 0.95:
                LOGGER.info('[TIMING_TERMINATION] Timeout reached')
                break

            if (
                    self.early_stop
                    and epoch_num >= early_stop_first_epoch
                    and time_since_best >= self.early_stop
                    and best_rate >= solve_thresh
            ):
                LOGGER.info('Terminating early (early stop condition met)')
                break

            if good_epoch_cap:
                if total_succ_rate > good_epoch_thresh:
                    good_epoch_num += 1
                if good_epoch_num >= good_epoch_cap:
                    self.explorer.advance_progression_level()
                    good_epoch_num = 0

        return best_rate, elapsed_time, int(epoch)

    def apply_worker_grads(self, worker_outs):
        params = self.weight_manager.all_weights
        if not worker_outs:
            raise RuntimeError("No worker outputs.")

        # init accumulators
        grads_sum = [np.zeros(v.shape, dtype=np.float32) for v in params]
        total = 0
        losses = []
        succs = []

        for out in worker_outs:
            losses.append(out.loss_mean)
            succs.append(out.hit_goal_mean)
            if out.n_samples <= 0:
                continue
            total += out.n_samples
            for i, g in enumerate(out.grads_np):
                grads_sum[i] += g * out.n_samples

        if total == 0:
            # no samples => skip update
            return 0.0, float(sum(succs) / max(1, len(succs))), 0

        mean_grads = [g / total for g in grads_sum]
        mean_grads_tf = [tf.convert_to_tensor(g, dtype=v.dtype) for g, v in zip(mean_grads, params)]
        self.optimiser.apply_gradients(zip(mean_grads_tf, params))

        return float(sum(losses) / len(losses)), float(sum(succs) / len(succs)), int(total)

    @can_profile
    def _make_batches(self, n_batches: int):
        """
        Epoch-pooled batching:
        - Consumes ONLY the current epoch's exploration data (self._epoch_data).
        - No replay buffer.
        - Supports variable obs/act dims by batching PER problem slot (network instance),
          then pooling at the loss level in _optimise.
        """
        assert hasattr(self, "_epoch_data") and self._epoch_data is not None, \
            "self._epoch_data missing. In train(): succs_probs, epoch_data = explorer.extend_replay(); self._epoch_data = epoch_data"

        cached_shapes = {p.name: (p.obs_dim, p.act_dim) for p in self.problems}

        # Build per-problem iterators that sample from that problem's epoch data
        batch_iters = []
        for problem in self.problems:
            obs_dim, act_dim = cached_shapes[problem.name]

            if problem not in self._epoch_data or self._epoch_data[problem] is None:
                # no samples this epoch for this slot
                batch_iters.append(repeat(None))
                continue

            prob_obs, prob_pi, prob_z = self._epoch_data[problem]

            # Safety: allow z to be (N,1)
            prob_z = np.asarray(prob_z)
            if prob_z.ndim == 2 and prob_z.shape[1] == 1:
                prob_z = prob_z[:, 0]

            # If the explorer returned python lists, normalize to arrays
            prob_obs = np.asarray(prob_obs)
            prob_pi = np.asarray(prob_pi)
            prob_z = np.asarray(prob_z)

            # Basic sanity
            if len(prob_obs) == 0:
                batch_iters.append(repeat(None))
                continue

            N = len(prob_obs)
            bs = self.batch_size_per_problem

            def epoch_sampler():
                for _ in range(n_batches):
                    if N <= bs:
                        idx = np.arange(N)
                    else:
                        idx = np.random.choice(N, size=bs, replace=False)

                    obs_b = prob_obs[idx]
                    pi_b = prob_pi[idx]
                    if self.policy_only:
                        yield (obs_b, pi_b)
                    else:
                        z_b = prob_z[idx]
                        yield (obs_b, pi_b, z_b)

            batch_iters.append(epoch_sampler())

        # Combine into aligned per-problem “feed_dict”
        combined = zip(*batch_iters)
        for combined_batch in combined:
            yield_val = []
            have_any = False

            for problem, batch in zip(self.problems, combined_batch):
                yield_val.append(batch)
                have_any = True

            assert have_any, "No epoch data at all for any problem — exploration produced nothing."
            yield yield_val

    def _get_replay_sizes(self):
        """Get the sizes of replay buffers for each problem."""
        # to avoid circular imports
        rv = []
        for problem in self.problems:
            rv.append(
                # to_local(problem.problem_service.get_replay_size())
                problem.get_replay_size()
            )
        return rv

    def calculate_balanced_succ_rate(self, worker_outs):
        if not self.balanced_success_rate:
            return 0, 0, 0

        # Use a dictionary to store counts and totals simultaneously
        # Structure: {difficulty: [sum_of_hits, total_count]}
        stats = {
            InstanceDifficulty.EASY: [0, 0],
            InstanceDifficulty.MEDIUM: [0, 0],
            InstanceDifficulty.HARD: [0, 0]
        }

        # Single pass: O(n) complexity
        for o in worker_outs:
            if o.instance_diff in stats:
                stats[o.instance_diff][0] += o.hit_goal_mean
                stats[o.instance_diff][1] += 1

        # Calculate rates with zero-division protection
        # Using a list comprehension for a clean return
        rates = [
            (val[0] / val[1]) if val[1] > 0 else 0.0
            for val in stats.values()
        ]

        return tuple(rates)

class ManualLoss:
    def __init__(self,
                 problems,
                 weight_manager,
                 summary_writer,
                 l1_reg_coeff,
                 l2_reg_coeff,
                 l1_l2_reg_coeff,
                 mse_coeff=1,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None,
                 strategy=SupervisedObjective.ANY_GOOD_ACTION):
        self.problems = problems
        self.weight_manager = weight_manager
        self.summary_writer = summary_writer
        self.l1_reg_coeff = l1_reg_coeff
        self.l2_reg_coeff = l2_reg_coeff
        self.l1_l2_reg_coeff = l1_l2_reg_coeff
        self.mse_coeff = mse_coeff
        self.strategy = strategy

    def __call__(self, act_dist_pred: List[tf.Tensor], act_dist: List[tf.Tensor], target_values=None, pred_values=None) \
            -> float:
        assert len(self.problems) == len(act_dist_pred), \
            "inconsistent input data size with num. problems"
        assert len(act_dist) == len(act_dist_pred), \
            "inconsistent output data sizes"
        losses = []
        batch_sizes = []
        loss_parts = None
        for i, problem in enumerate(self.problems):
            with tf.name_scope(f'Problem-{i}'):
                act_dist_pred_prob_i, act_dist_prob_i = act_dist_pred[i], act_dist[i]
                if target_values is not None and pred_values is not None:
                    z, v = target_values[i], pred_values[i]
                    this_loss, this_loss_parts = self._set_up_losses(
                        problem, act_dist_pred_prob_i, act_dist_prob_i,
                        target_values=z, pred_values=v
                    )
                else:
                    this_loss, this_loss_parts = self._set_up_losses(problem, act_dist_pred_prob_i, act_dist_prob_i)

                this_batch_size = tf.shape(input=act_dist_pred_prob_i)[0]
                losses.append(this_loss)
                batch_sizes.append(tf.cast(this_batch_size, tf.float32))
                if loss_parts is None:
                    loss_parts = this_loss_parts
                else:
                    # we care about these parts because we want to display them to
                    # the user (e.g. how much of my loss is L2 regularisation
                    # loss?)
                    assert len(loss_parts) == len(this_loss_parts), \
                        'diff. loss breakdown for diff. probs. (%s vs %s)' \
                        % (loss_parts, this_loss_parts)
                    # sum up all the parts
                    new_loss_parts = []
                    for old_part, new_part in zip(loss_parts, this_loss_parts):
                        assert old_part[0] == new_part[0], \
                            "names (%s vs. %s) don't match" % (old_part[0],
                                                               new_part[0])
                        to_add = new_part[1] * tf.cast(this_batch_size, tf.float32)
                        new_loss_parts.append((old_part[0], old_part[1] + to_add))
                    loss_parts = new_loss_parts
        with tf.name_scope('combine_all_losses'):
            op_loss \
                = sum(l * s for l, s in zip(losses, batch_sizes)) \
                  / sum(batch_sizes)

        # this is actually a list of (name, symbolic representation) pairs for
        # components of the loss
        assert loss_parts is not None

        for part_loss_name, part_loss in loss_parts:
            # tf.summary.scalar('loss-%s' % part_loss_name, part_loss)
            tf_and_log('loss-%s' % part_loss_name, part_loss)
        return op_loss

    @can_profile
    def _set_up_losses(self, problem, act_dist_pred, act_dist, target_values=0, pred_values=0):
        loss_parts = []
        # now the loss ops
        with tf.name_scope('loss'):
            if self.strategy == SupervisedObjective.ANY_GOOD_ACTION \
                    or self.strategy == SupervisedObjective.THERE_CAN_ONLY_BE_ONE:
                best_qv = tf.reduce_min(
                    input_tensor=act_dist, axis=-1, keepdims=True)
                # TODO: is 0.01 threshold too big? Hmm.
                act_labels = tf.cast(
                    tf.less(tf.abs(act_dist - best_qv), 0.01), 'float32')
                label_sum = tf.reduce_sum(
                    input_tensor=act_labels, axis=-1, keepdims=True)
                act_label_dist = act_labels / tf.math.maximum(label_sum, 1.0)

                # zero out disabled or dead-end actions!
                problem_service = problem.problem_service
                dead_end_value = to_local(
                    problem_service.get_ssipp_dead_end_value())
                act_label_dist *= tf.cast(act_labels < dead_end_value,
                                          'float32')
                # this tf.cond() call ensures that this still works when batch
                # size is 0 (in which case it returns a loss of 0)
                xent = tf.cond(pred=tf.size(input=act_label_dist) > 0,
                               true_fn=lambda: tf.reduce_mean(
                                   input_tensor=cross_entropy(act_dist_pred, act_label_dist),
                                   name='xent_reduce'),
                               false_fn=lambda: tf.constant(
                                   0.0, dtype=tf.float32, name='xent_ph'),
                               name='xent_cond')
                loss_parts.append(('xent', xent))
            elif self.strategy == SupervisedObjective.MAX_ADVANTAGE:
                state_values = tf.reduce_min(input_tensor=act_dist, axis=-1)
                exp_q = act_dist_pred * act_dist
                exp_vs = tf.reduce_sum(input_tensor=exp_q, axis=-1)
                # state value is irrelevant to objective, but is included
                # because it ensures that zero loss = optimal policy
                q_loss = tf.reduce_mean(input_tensor=exp_vs - state_values)
                loss_parts.append(('qloss', q_loss))
            elif self.strategy == SupervisedObjective.MCTS_POLICY_DIST:
                pi_targets = tf.convert_to_tensor(act_dist, dtype=tf.float32)
                # pi_targets = tf.maximum(pi_targets, 0.0)

                row_sums = tf.reduce_sum(pi_targets, axis=-1, keepdims=True)
                row_sums = tf.where(row_sums > 0.0, row_sums, tf.ones_like(row_sums))
                pi_targets = tf.math.divide_no_nan(pi_targets, row_sums)

                act_dist_pred_entropy = -tf.reduce_sum(act_dist_pred * tf.math.log(act_dist_pred + 1e-8), axis=-1)
                mean_act_dist_pred_entropy = tf.reduce_mean(act_dist_pred_entropy)
                tf_and_log("act_dist_entropy", mean_act_dist_pred_entropy)
                pi_targets_entropy = -tf.reduce_sum(pi_targets * tf.math.log(pi_targets + 1e-8), axis=-1)
                mean_pi_targets_entropy = tf.reduce_mean(pi_targets_entropy)
                tf_and_log("pi_entropy", mean_pi_targets_entropy)

                xent = tf.cond(
                    tf.size(pi_targets) > 0,
                    true_fn=lambda: tf.reduce_mean(cross_entropy(act_dist_pred, pi_targets)),
                    false_fn=lambda: tf.constant(0.0, dtype=tf.float32)
                )
                loss_parts.append(('xent', xent))

                mse = tf.reduce_mean(mean_squared_error(pred_values, target_values))
                mse *= self.mse_coeff
                loss_parts.append(('mse', mse))

            else:
                raise ValueError("Unknown strategy %s" % self.strategy)

            # regularisation---we need this because the
            # logisitic-regression-like optimisation problem we're solving
            # generally has no minimum point otherwise
            weights = self.weight_manager.all_weights
            weights_no_bias = [w for w in weights if len(w.shape) > 1]
            weights_all_bias = [w for w in weights if len(w.shape) <= 1]
            # downweight regulariser penalty on biases (for most DL work
            # they're un-penalised, but here I think it pays to have *some*
            # penalty given that there are some problems that we can solve
            # perfectly)
            bias_coeff = 0.05
            if self.l2_reg_coeff:
                def do_l2_reg(lst):
                    return sum(map(tf.nn.l2_loss, lst))

                l2_reg = self.l2_reg_coeff * do_l2_reg(weights_no_bias) \
                         + bias_coeff * self.l2_reg_coeff \
                         * do_l2_reg(weights_all_bias)
                loss_parts.append(('l2reg', l2_reg))

            if self.l1_reg_coeff:
                def do_l1_reg(lst):
                    return sum(tf.linalg.norm(tensor=w, ord=1) for w in lst)

                l1_reg = self.l1_reg_coeff * do_l1_reg(weights_no_bias) \
                         + bias_coeff * self.l1_reg_coeff \
                         * do_l1_reg(weights_all_bias)
                loss_parts.append(('l1reg', l1_reg))

            if self.l1_l2_reg_coeff:
                all_weights_ap = []
                # act_weights[:-1] omits the last layer (which we don't want to
                # apply group sparsity penalty to)
                all_weights_ap.extend(self.weight_manager.act_weights[:-1])
                all_weights_ap.extend(self.weight_manager.prop_weights)
                l1_l2_reg_accum = 0.0
                for weight_dict in all_weights_ap:
                    for trans_mat, bias in weight_dict.values():
                        bias_size, = bias.shape.as_list()
                        tm_shape = trans_mat.shape.as_list()
                        # tm_shape[0] is always 1, tm_shape[1] is size of
                        # input, and tm_shape[2] is network channel count
                        assert len(tm_shape) == 3 and tm_shape[0] == 1 \
                               and tm_shape[2] == bias_size, "tm_shape %s does " \
                                                             "not match bias size %s" % (tm_shape, bias_size)
                        trans_square = tf.reduce_sum(
                            input_tensor=tf.square(trans_mat), axis=[0, 1])
                        bias_square = tf.square(bias)
                        norms = tf.sqrt(trans_square + bias_square)
                        l1_l2_reg_accum += tf.reduce_sum(input_tensor=norms)
                l1_l2_reg = self.l1_l2_reg_coeff * l1_l2_reg_accum
                loss_parts.append(('l1l2reg', l1_l2_reg))

            with tf.name_scope('combine_parts'):
                loss = sum(p[1] for p in loss_parts)
                # loss = 0

        return loss, loss_parts
