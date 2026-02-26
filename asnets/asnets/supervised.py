import multiprocessing
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
# import tqdm
import tqdm.auto as tqdm
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set
import cProfile
import datetime
from asnets.heur_inputs import ActionCountDataGenerator, \
    HeuristicDataGenerator, LMCutDataGenerator, RelaxedDeadendDetector, \
    NumericLandmarkGenerator
from asnets.models import PropNetworkWeights, PropNetwork
from asnets.utils.generator_utils import InstanceDifficulty, ProgressionLevel, get_problem_names
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
    strip_parens, weak_ref_to, weighted_batch_iter
from asnets.utils.tf_utils import cross_entropy, empty_feed_value, \
    escape_name_tf, mean_squared_error
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
            domain = None,
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
            estimator_value_conversion_lambda: float = 0.1,
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
        self.domain=domain
        self.difficulty=difficulty
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
        self.estimator_value_conversion_lambda = estimator_value_conversion_lambda
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
                 domain,
                 # init_problem_name: str,
                 # domain_file_path_str: str,
                 domain_type: DomainType,
                 *,
                 dg_ssipp_heuristic_name: str = None,
                 dg_use_lm_cuts: bool = False,
                 dg_use_numeric_landmarks: bool = False,
                 dg_use_contributions: bool = False,
                 dg_use_act_history: bool = False,
                 difficulty: InstanceDifficulty = InstanceDifficulty.EASY,
                 fixed_instance: bool = False,
                 seed: int = None):
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
        # self.pddl_files = pddl_files
        self.pddl_files = [pddl_files[0]]
        self.domain_type = domain_type
        # LOGGER.info('Parsing %d PDDL files for domain type %s',
        #             len(self.pddl_files), domain_type.name)
        self.difficulty = difficulty
        self.seed = seed

        import mdpsim  # noqa: F811
        import ssipp  # noqa: F811

        # self.domain = Domain.from_pddl_name(extract_domain_name_from_file(domain_file_path_str))

        # self.domain.generate_instances(
        #     difficulty=self.difficulty,
        # )
        self.domain = domain
        if not fixed_instance:
            generated_problem_pddl_path = self.domain.get_realtime_instance(self.difficulty, self.seed)
        else:
            generated_problem_pddl_path = pddl_files[1]
        self.generated_problem_name = get_problem_names([generated_problem_pddl_path])[0]

        self.pddl_files += [str(generated_problem_pddl_path)]
        LOGGER.info(f'Starting to parse mdpsim problem: {self.generated_problem_name}')
        # MDPSim stuff
        self.mdpsim: ModuleType = mdpsim
        # self.mdpsim_problem = parse_problem_args(self.mdpsim, self.pddl_files,
        #                                          init_problem_name)
        self.mdpsim_problem = parse_problem_args(self.mdpsim, self.pddl_files, self.generated_problem_name)
        self.problem_name: str = self.mdpsim_problem.name.strip()

        LOGGER.debug(f'Finished parsing mdpsim problem: {self.problem_name}')

        # Maps to PyGroundAction object in MDPSim. Cannot use type hint.
        self.act_ident_to_mdpsim_act: Dict[str, Any] = {
            strip_parens(a.identifier): a
            for a in self.mdpsim_problem.ground_actions
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
            # problem_file = get_problem_file(self.pddl_files, self.problem_name)

            LOGGER.debug(f"Process {os.getpid()} Starting JVM...")
            start_jvm()

            LOGGER.debug("Creating J_PDDLDomain...")
            self.j_domain = J_PDDLDomain(domain_file)

            LOGGER.debug("Creating J_PDDLProblem...")
            # self.j_problem = J_PDDLProblem(problem_file, self.j_domain)
            self.j_problem = J_PDDLProblem(str(generated_problem_pddl_path), self.j_domain)
            LOGGER.debug("Calling prepareForSearch...")
            # self.j_problem.prepareForSearch(True, False)

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
                dg_use_contributions)
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

    def next_instance(self):
        generated_problem_pddl_path = self.domain.get_instance(self.difficulty)
        self.generated_problem_name = get_problem_names([generated_problem_pddl_path])[0]

        self.pddl_files = [self.pddl_files[0], str(generated_problem_pddl_path)]

        LOGGER.info(f'Starting to parse mdpsim problem...')
        LOGGER.info(f"Current problem name: {self.generated_problem_name}")
        # MDPSim stuff
        # import mdpsim  # noqa: F811
        # LOGGER.info('imported mdpsim properly')
        # self.mdpsim: ModuleType = mdpsim
        # self.mdpsim_problem = parse_problem_args(self.mdpsim, self.pddl_files,
        #                                          init_problem_name)
        self.mdpsim_problem = parse_problem_args(self.mdpsim, self.pddl_files, self.generated_problem_name)
        LOGGER.info('parsed problem properly')
        self.problem_name: str = self.mdpsim_problem.name.strip()

        LOGGER.info(f'Finished parsing mdpsim problem: {self.problem_name}')

        # Maps to PyGroundAction object in MDPSim. Cannot use type hint.
        self.act_ident_to_mdpsim_act: Dict[str, Any] = {
            strip_parens(a.identifier): a
            for a in self.mdpsim_problem.ground_actions
        }

        LOGGER.debug(f'Python-side extra data')
        # Python-side extra data
        self.domain_meta = get_domain_meta(self.mdpsim_problem.domain)
        self.problem_meta = get_problem_meta(self.mdpsim_problem,
                                             self.domain_meta)
        if self.domain_type == DomainType.NUMERIC:
            domain_file = get_domain_file(self.pddl_files)
            # problem_file = get_problem_file(self.pddl_files, self.problem_name)

            LOGGER.debug("Creating J_PDDLDomain...")
            self.j_domain = J_PDDLDomain(domain_file)

            LOGGER.debug("Creating J_PDDLProblem...")
            # self.j_problem = J_PDDLProblem(problem_file, self.j_domain)
            self.j_problem = J_PDDLProblem(str(generated_problem_pddl_path), self.j_domain)
            LOGGER.debug("Calling prepareForSearch...")
            # self.j_problem.prepareForSearch(True, False)

            self.j_problem.prepareForSearch(
                True,  # enable AIBR preprocessing
                False  # stop after grounding
            )
            LOGGER.debug("JPDDL init done.")


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
    LOGGER.info(f"[VALUE_PRED_LOG - across problems] mean: {tf.reduce_mean(combined_tensor)}, min: {tf.reduce_min(combined_tensor)}, max: {tf.reduce_max(combined_tensor)}")

def log_grad_norms(grads_and_vars):
    policy_grads = []
    value_grads = []

    for grad, var in grads_and_vars:
        if "final_act" in var.name.lower(): #policy head
            if grad is not None:
                policy_grads.append(tf.norm(grad))
        if "value_out" in var.name.lower(): #value head
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
    tf_logger.info(f"[TF_GRAD_NORMS_LOG] {base_name + '/' if base_name is not None else ''}policy_grad_norm : {policy_grad_norm}")
    tf_logger.info(f"[TF_GRAD_NORMS_LOG] {base_name + '/' if base_name is not None else ''}value_grad_norm : {value_grad_norm}")

def log_policy_target(pi_target_batch, problem: 'SingleProblem'):
    sampled_pi_targets_ind = np.random.choice(pi_target_batch.shape[0], size=3, replace=False)
    for pi_target_ind in sampled_pi_targets_ind:
        pi_target = pi_target_batch[pi_target_ind]
        pi_target_first_ten_entries = pi_target[:10]
        pi_target_sum = np.sum(pi_target)
        pi_target_argmax = np.argmax(pi_target)
        pi_target_argmax_name = problem.prob_meta.bound_acts_ordered[pi_target_argmax].__str__()
        LOGGER.info(f"[POLICY_TARGET_LOG - {problem.name}] pi_target (first 10 entries): {pi_target_first_ten_entries}, sum: {pi_target_sum} argmax: {pi_target_argmax}|{pi_target_argmax_name}")

def tf_and_log(name: str, value):
    tf.summary.scalar(name, value)
    base_name = tf.get_current_name_scope()
    # print(f"[TF_SUMMARY_SCALAR_LOG] {base_name + '/' if base_name is not None else ''}{name} : {value}")
    tf_logger.info(f"[TF_SUMMARY_SCALAR_LOG] {base_name + '/' if base_name is not None else ''}{name} : {value}")

def make_problem_service(config, set_proc_title=False):
    """Construct Service class for a particular problem. Note that we must
    construct classes, not instances (unfortunately), as there is no way of
    passing arguments to the service's initialisation code (AFAICT).

    The extra set_proc_title arg can be set to True if you want the
    ProblemService to figure out a descriptive name for the current process in
    top/htop/etc. It's mostly useful when you're starting a single subprocess
    per environment, and you want to know which subprocess corresponds to which
    environment."""
    assert isinstance(config, ProblemServiceConfig)
    #to avoid circular imports
    class ProblemService(rpyc.Service):
        """Spools up a new Python interpreter and uses it to sandbox SSiPP and
        MDPSim. Can interact with this to train a Q-network."""

        def exposed_collect_trajectory(self, weights) -> bool:
            """Collect a single trajectory using the given policy (represented
            as a function from flattened observation vectors to action
            numbers)."""
            self.internal_set_weights(weights)
            return self.internal_collect_trajectory(self.network)
        
        def exposed_explore_from_trajectories(self, weights):
            self.internal_set_weights(weights)
            self.internal_explore_from_trajectories(self.network)
        
        def exposed_explore_from_random_state(self, weights):
            self.internal_set_weights(weights)
            self.internal_explore_from_random_state(self.network)

        def exposed_explore_from_init_state(self, weights) -> bool:
            self.internal_set_weights(weights)
            return self.internal_explore_from_init_state(self.network)

        def exposed_dataset_is_empty(self):
            return len(self.replay) == 0

        def exposed_weighted_dataset(self):
            """Return weighted dataset.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: The dataset.
                The first element is tensor of observations (cstates as
                network inputs). The second element is tensor of Q-values at
                each cstate, ordered in the same way as bound_acts_ordered. The
                third element is the weight of each cstate, which is really just
                a count of how many times we saw that cstate.
            """
            rich_obs_qvs_zs, counts = self.replay.get_full_dataset()
            assert len(rich_obs_qvs_zs) > 0, "Empty replay %s" % (self.replay, )
            counts = np.asarray(counts, dtype='float32')
            # obs_tensor, pi_tensor = self.flatten_obs_qvs(rich_obs_qvs_zs)
            obs_tensor, pi_tensor, z_tensor = self.flatten_obs_pi_z(rich_obs_qvs_zs)
            return obs_tensor, pi_tensor, z_tensor, counts

        def exposed_env_reset(self):
            self.id_hash_to_state.clear()
            init_state_id, init_state_hash = self.internal_get_state_identifiers(self.internal_get_init_state())
            return init_state_id, init_state_hash

        def exposed_action_name(self, action_num):
            action_num = to_local(action_num)
            return get_action_name(self.p, action_num)

        # def exposed_env_step(self, action_num):
        #     action_num = to_local(action_num)
        #     next_cstate, step_cost \
        #         = sample_next_state(self.current_state, action_num, self.p)
        #     self.current_state = next_cstate
        #     current_state_id, current_state_hash = self.internal_get_state_identifiers(self.current_state)
        #     return (current_state_id, current_state_hash, step_cost, self.current_state.is_goal,
        #             self.current_state.is_terminal, self.internal_to_network_input(self.current_state),
        #             self.internal_get_applicable_action_mask(self.current_state))

        def exposed_env_simulate_step(self, cstate_to_simulate_from, action_num):
            """Perform an environment step without actually changing the state"""
            local_cstate_copy = to_local(cstate_to_simulate_from)
            following_cstate, step_cost = sample_next_state(local_cstate_copy, action_num, self.p)
            following_cstate_id, following_cstate_hash = self.internal_get_state_identifiers(following_cstate)
            return (following_cstate_id, following_cstate_hash, step_cost,
                    following_cstate.is_goal, following_cstate.is_terminal,
                    self.internal_to_network_input(following_cstate),
                    self.internal_get_applicable_action_mask(following_cstate))

        def exposed_env_simulate_batch_steps(self, cstate_id, cstate_hash, action_nums):
            """
            Perform multiple environment steps from the same parent state in one RPC call.
            Returns a list of (next_state, step_cost, is_goal, is_terminal) tuples.
            """
            try:
                cstate = self.id_hash_to_state.get((cstate_id,cstate_hash), None)
                if cstate is None:
                    raise KeyError(
                        f"Unknown cstate (id={cstate_id}, hash={cstate_hash}). "
                        f"Known states={len(self.id_hash_to_state)}. "
                        f"Did the client reuse state IDs across epochs / reconnect?"
                    )
                results = []
                for action_num in action_nums:
                    next_state, step_cost = sample_next_state(cstate, action_num, self.p)
                    next_state_id, next_state_hash = self.internal_get_state_identifiers(next_state)
                    results.append((
                        action_num,
                        next_state_id,
                        next_state_hash,
                        step_cost,
                        next_state.is_goal,
                        next_state.is_terminal,
                        self.internal_to_network_input(next_state), self.internal_get_applicable_action_mask(next_state),
                    ))
                return results
            except Exception as e:
                log_path = "/tmp/problemservice_exceptions.log"
                with open(log_path, "a") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"[PID {os.getpid()}] Exception in env_simulate_batch_steps:\n")
                    traceback.print_exc(file=f)
                    f.write("=" * 80 + "\n")

                # also print to stderr for live feedback
                print(f"[SERVER ERROR] Exception in env_simulate_batch_steps: {e}", file=sys.stderr)
                traceback.print_exc()

                # re-raise to ensure RPyC propagates the error correctly
                raise

        # note to self: RPyC doesn't support @property

        def exposed_get_ssipp_dead_end_value(self):
            return self.p.ssipp_dead_end_value

        def exposed_get_meta(self):
            """Get name, ProblemMeta and DomainMeta for the current problem."""
            return self.p.problem_meta, self.p.domain_meta

        def exposed_get_replay_size(self):
            return len(self.replay)
        
        def exposed_trim_replay(self):
            LOGGER.info(f'[{self.p.problem_name}] trimming replay buffer')
            self.replay.remove_oldest()

        def exposed_get_obs_dim(self):
            return self.internal_get_obs_dim()

        def internal_get_obs_dim(self):
            if not hasattr(self, '_cached_obs_dim'):
                self._cached_obs_dim = compute_observation_dim(self.p)
            return self._cached_obs_dim

        def exposed_get_act_dim(self):
            return self.internal_get_act_dim()

        def internal_get_act_dim(self):
            if not hasattr(self, '_cached_act_dim'):
                self._cached_act_dim = compute_action_dim(self.p)
            return self._cached_act_dim

        def exposed_get_dg_extra_dim(self):
            # TODO: factor this logic out into another function, since it's
            # used in several places (grep for '\.extra_dim for' or something)
            data_gens = self.p.data_gens
            return sum([g.extra_dim for g in data_gens])

        def exposed_get_max_len(self):
            return self.max_len

        def exposed_get_problem_names(self):
            # fetch a list of all problems loaded by MDPSim
            print('Service retrieving problem names')
            return sorted(self.p.mdpsim.get_problems().keys())

        def exposed_get_current_problem_name(self):
            return self.p.problem_name
        
        def exposed_get_num_traj_states(self):
            return len(self.traj_states)
        
        def exposed_get_num_new_pairs(self):
            # return len(self.expl_states)
            return self.expl_triplets
        
        def exposed_finish_explore(self, log=False):
            info_text = f"[{self.p.problem_name}] generated {self.expl_triplets} actual (exploration only) new triplets, and {len(self.expl_states)} total (exploration + HER + bootstrapping) triplets"
            LOGGER.info(info_text)
            self.replay.update(self.expl_states)
            self.traj_states.clear()
            self.model_cache = {}
            z_sum = sum([z for (_,(_,z)) in self.expl_states])
            LOGGER.info(f'[Z_SUM] The sum of all currently placed triplets\' z value in self.expl_states is {z_sum}')
            if log:
                last_states_mean = np.mean(self.last_states_value_cache) if len(self.last_states_value_cache) > 0 else "None"
                last_states_min = np.min(self.last_states_value_cache) if len(self.last_states_value_cache) > 0 else "None"
                last_states_max = np.max(self.last_states_value_cache) if len(self.last_states_value_cache) > 0 else "None"
                LOGGER.info(f"[LAST_STATES_LOG] '5-last-states' in the latest exploration period information: mean: {last_states_mean}, min: {last_states_min}, max: {last_states_max} ")
                sampled_indices = np.random.choice(len(self.expl_states), replace=False, size=min(5,len(self.expl_states)))\
                    if len(self.last_states_value_cache) > 0 else []
                sampled_triplets = [self.expl_states[i] for i in sampled_indices]
                for state, pi_val in sampled_triplets:
                    pi, val = pi_val
                    pi_pred, val_pred = self.network(state.to_network_input())
                    LOGGER.info(f"[COSINE_SIMILARITY] For the sampled state, the cosine similarity is {cosine_similarity(np.array(pi), np.array(pi_pred).T)}")
            self.expl_states.clear()
            self.last_states_value_cache.clear()
            self.expl_triplets = 0

        def exposed_initialise(self):
            assert not self.initialised, "Can't double-init"
            print("ProblemService started initialisation.")

            self.p = PlannerExtensions(
                config.pddl_files,
                # config.init_problem_name,
                config.domain,
                config.domain_type,
                dg_ssipp_heuristic_name=config.ssipp_dg_heuristic,
                dg_use_lm_cuts=config.use_lm_cuts,
                dg_use_numeric_landmarks=config.use_numeric_landmarks,
                dg_use_contributions=config.use_contributions,
                dg_use_act_history=config.use_act_history,
                difficulty=config.difficulty,
                seed=config.random_seed,
            )
            self.only_one_good_action = config.only_one_good_action
            self.use_teacher_envelope = config.use_teacher_envelope

            self.traj_states = RandomPopContainer()
            self.model_cache = {}
            self.expl_states = []
            self.expl_triplets = 0

            # a list of planner trajectories with the outcome (z=0 for non-terminal trajectory, 1 for successful trajectory)
            self.planner_trajectories: List[tuple[List[tuple[CanonicalState,tuple[np.ndarray,int]]],int]] = []
            self.state_id_to_value_cache = {}
            self.last_states_value_cache = []
            self.estimator_value_conversion_lambda = config.estimator_value_conversion_lambda #default is 0.1

            self.id_hash_to_state: dict[tuple[int,int],CanonicalState] = {}
            self.curr_state_id = 0
            self.her_k = config.her_k
            self.mcts_her_strategy = config.mcts_her_strategy
            self.planner_bootstrapping = config.planner_bootstrapping
            self.planner_bootstrapping_her = config.planner_bootstrapping_her
            self.heuristic_bootstrapping = config.heuristic_bootstrapping
            self.bootstrap_k = config.bootstrap_k #default is 3
            CanonicalState.network_input_config(use_fluents=config.use_fluents, use_comparisons=config.use_comps)

            if config.teacher_planner == 'fd':
                # TODO: consider passing in teacher heuristic here, too; that
                # should give me more control over how the FD teacher works
                # (and let me do inadm. vs. adm. comparisons, among other
                # things)
                self.teacher = FDTeacher(
                    self.p,
                    heuristic=config.fd_heuristic,
                    timeout_s=config.teacher_timeout_s)
            elif config.teacher_planner == 'ssipp':
                self.teacher = SSiPPTeacher(
                    self.p,
                    'lrtdp',
                    config.ssipp_teacher_heuristic,
                    timeout_s=config.teacher_timeout_s)
            elif config.teacher_planner == 'domain-specific':
                self.teacher = DomainSpecificTeacher(self.p)
            elif config.teacher_planner == 'enhsp':
                self.teacher = ENHSPTeacher(
                    self.p,
                    config.teacher_timeout_s,
                    enhsp_config=config.enhsp_config)
            elif config.teacher_planner == 'metricff':
                self.teacher = MetricFFTeacher(
                    self.p,
                    timeout_s=config.teacher_timeout_s)
            # maximum length of a trace to gather
            self.max_len = config.max_len
            # will hold (state, action) pairs to train on
            self.replay = WeightedReplayBuffer()
            self.cached_init_state = self.internal_get_init_state()
            # hack to decide whether to get one or many rollouts (XXX)
            self.first_rollout = True

            if set_proc_title:
                # SPT_NOENV truncates the new title to avoid clobbering
                # /proc/PID/environ
                os.environ['SPT_NOENV'] = '1'
                old_title = setproctitle.getproctitle()
                new_title = '[%s] %s' % (self.p.problem_meta.name, old_title)
                setproctitle.setproctitle(new_title)

            self.stochastic = True

            self._tf_queue = queue.Queue()
            self._tf_thread = threading.Thread(
                target=self.internal_tf_worker,
                daemon=True
            )
            self._tf_thread.start()
            self.network_initialised = False
            self.initialised = True
            print("ProblemService finished initialisation.")

            self._profiler = cProfile.Profile()
            self._profiler.enable()
            return self.p.generated_problem_name

        def exposed_get_problem_name(self):
            assert self.initialised, "Problem was no initialised"
            return self.p.generated_problem_name

        def internal_tf_worker(self):
            while True:
                fn, args, kwargs, result_q = self._tf_queue.get()
                try:
                    print("[TF WORKER] before forward", flush=True)
                    res = fn(*args, **kwargs)
                    print("[TF WORKER] after forward", flush=True)
                    result_q.put((True, res))
                except Exception as e:
                    result_q.put((False, e))

        def internal_run_tf(self, fn, *args, **kwargs):
            result_q = queue.Queue()
            self._tf_queue.put((fn, args, kwargs, result_q))
            ok, res = result_q.get()
            if ok:
                return res
            raise res

        def exposed_flush_profiler(self):
            self._profiler.disable()
            self._profiler.dump_stats(f"/IdeaProjects/numeric-asnets/asnets/worker_{os.getpid()}.prof")

        def on_disconnect(self, conn):
            print(f"[DEBUG] Connection {conn} closed", file=sys.stderr, flush=True)

        def on_connect(self, conn):
            # we let the initialiser run later, so that it can execute
            # asynchronously (starting up PlannerExtensions & Planner is
            # expensive because it requires grounding the relevant problem)
            self.initialised = False
            self.estimator_initialised = False
            print(f"[DEBUG] Connection {conn} opened", file=sys.stderr, flush=True)

        def exposed_set_policy_only(self, value):
            self.policy_only = value

        # FIXME: don't cache at this level; it's inefficient when using
        # history-level features, b/c it will lead to lots and lots of
        # near-identical cstates being thrown into the cache
        @lru_cache(None)
        def opt_pol_experience(self, cstate: CanonicalState) \
                -> List[Tuple[CanonicalState, List[Tuple[BoundAction, float]]]]:
            """Get optimal policy from given state.

            Args:
                cstate (CanonicalState): Canonical state to start from.

            Returns:
                List[Tuple[CanonicalState, List[Tuple[BoundAction, float]]]]:
                Experience from the optimal policy, as a list of (state,
                [(action, q-value), ...]) tuples.
            """
            return planner_trace(self.teacher, self.p, cstate,
                                 self.only_one_good_action,
                                 self.use_teacher_envelope)

        def internal_collect_trajectory(self,
                                        model: Callable) -> bool:
            """Collect a single trajectory using the given policy. Add the
            trajectory to the internal trajectory collection.
            
            Args:
                mode (Callable): The policy to use.
                max_len (int): The maximum length of the trajectory.

            Returns:
                bool: Whether the trajectory was successful.
            """
            prob_meta = self.p.problem_meta
            path = []
            hit_goal = False
            cstate = self.internal_get_init_state()
            
            for _ in range(self.max_len):
                obs = to_local(cstate.to_network_input())
                obs_bytes = obs.tostring()
                if obs_bytes not in self.model_cache:
                    if self.policy_only:
                        act_dist = model(obs[None], training=False)
                    else:
                        act_dist, _ = model(obs[None], training=False)
                    
                    act_dist = tf.reshape(
                        to_local(act_dist),
                        [
                            -1,
                        ],
                    ).numpy()
                    if not self.stochastic:
                        chosen = int(np.argmax(act_dist))
                    else:
                        act_dist = act_dist / np.sum(act_dist)
                        chosen = int(
                            np.random.choice(np.arange(act_dist.shape[0]), p=act_dist)
                        )
                    # this cache update is actually thread-safe too thanks to
                    # Python's GIL
                    self.model_cache[obs_bytes] = chosen

                action = self.model_cache[obs_bytes]

                path.append((cstate, prob_meta.bound_acts_ordered[action]))

                cstate, _ = sample_next_state(cstate, action, self.p)
                if cstate.is_terminal:
                    if cstate.is_goal:
                        hit_goal = True
                    break
                
            for cstate, _ in path:
                self.traj_states.add(cstate)

            return hit_goal
        
        def internal_explore_from_trajectories(self, network: Callable) -> None:
            """Explore from the trajectory states."""
            while len(self.traj_states) > 0:
                self.internal_explore_from_random_state(network)


        def internal_get_state_identifiers(self, cstate: CanonicalState):
            state_hash = hash(cstate)
            self.curr_state_id += 1
            state_id = self.curr_state_id
            self.id_hash_to_state[state_id, state_hash] = cstate
            return state_id, state_hash

        def internal_get_state_from_identifiers(self, cstate_id: int, cstate_hash: int) -> CanonicalState:
            return self.id_hash_to_state.get((cstate_id, cstate_hash), None)

        def internal_get_state_value(self, state_id: int, state: CanonicalState) -> float:
            state_v = self.state_id_to_value_cache.get(state_id, None)
            if not state_v:
                state_h = self.internal_get_state_h(state)
                state_v = np.exp(-1 * self.estimator_value_conversion_lambda * state_h)
                self.state_id_to_value_cache[state_id] = state_v
            return state_v


        # def internal_explore_from_random_state(self) -> None:
        #     """Explore from a random state."""
        #     cstate = self.traj_states.pop_random()
        #
        #     try:
        #         teacher_experience = self.opt_pol_experience(cstate)
        #     except TeacherException as ex:
        #         LOGGER.warning(f'Teacher error on problem \
        #             {self.p.problem_name} ({ex})')
        #         return
        #
        #     filtered_envelope = []
        #
        #     for env_cstate, act in teacher_experience:
        #         nactions = sum(p[1] for p in env_cstate.acts_enabled)
        #
        #         if nactions <= 1:
        #             # skip states
        #             continue
        #         filtered_envelope.append((env_cstate, act))
        #
        #     self.expl_states.update(filtered_envelope)

        def internal_explore_from_random_state(self, network: Callable) -> None:
            """Self-play exploration for AlphaZero-style data generation beginning with a random state from an existing trajectory."""
            try:
                cstate = self.traj_states.pop_random()
            except ValueError as e:
                print(e)
                print("[WATCHDOG] Attempted popping from an empty traj_states,"
                      " probably restarted service, collecting a single trajectory to continue.")
                self.internal_collect_trajectory(network)
                cstate = self.traj_states.pop_random()
                # if this fails, it *should* break the whole process
            self.internal_explore_from_given_state(network, cstate)

        def internal_explore_from_init_state(self, network: Callable) -> bool:
            """Self-play exploration for AlphaZero-style data generation beginning with the start state of 'this' problem"""
            cstate = self.internal_get_init_state()
            print('Starting exploration from the initial state')
            return self.internal_explore_from_given_state(network, cstate)

        def internal_sample_k_future_states(self, curr_t, state_list) -> List[CanonicalState]:
            future_states = state_list[curr_t:]
            if len(future_states) < self.her_k:
                return future_states
            else:
                return np.random.choice(future_states, size=self.her_k, replace=False)

        def internal_sample_k_states_from_tree(self, mcts_tree: TrainingMCTS) -> List[tuple[CanonicalState,np.ndarray]]:
            all_nodes = mcts_tree.state_to_node.values()
            all_nodes = np.array(list(all_nodes))
            sampled_goals = np.random.choice(all_nodes, size=self.her_k, replace=False)
            good_nodes: Set[MCTSNode] = set()
            for goal in sampled_goals:
                curr_node: MCTSNode = goal
                while curr_node.parent:
                    good_nodes.add(curr_node)
                    curr_node = curr_node.parent

            output_states_and_pi: List[tuple[CanonicalState, np.ndarray]] = []
            act_dim = self.exposed_get_act_dim()
            for node in good_nodes:
                pi = np.zeros(act_dim, dtype=np.float32)
                if not node.children:
                    mask = mcts_tree.get_applicable_action_mask(node)
                    valid = np.where(mask)[0]
                    if len(valid) > 0:
                        pi[valid] = 1.0 / len(valid)
                    else:
                        pi[:] = 1.0 / act_dim
                    continue
                for action, child in node.children.items():
                    pi[action] = mcts_tree.N.get(child,0)
                if pi.sum() > 0:
                    pi /= pi.sum()
                else:
                    mask = mcts_tree.get_applicable_action_mask(node)
                    valid = np.where(mask)[0]
                    if len(valid) > 0:
                        pi[valid] = 1.0 / len(valid)
                    else:
                        pi[:] = 1.0 / act_dim

                output_states_and_pi.append((self.id_hash_to_state[node.state_id,hash(node)], pi))

            return output_states_and_pi

        def internal_explore_from_given_state(self, network: Callable, cstate: CanonicalState) -> bool:
            mcts_tree = TrainingMCTS(
                network=network,
                problem_service=self,
                iterations=config.training_mcts_iterations,
                # iterations=1,
                # TODO: implement curriculum training - don't use high iterations at the beginning
                #  as the network is quite random, and increase towards late phases
                expansion_k=config.expansion_k,
                exploration_weight=1,
                # TODO: optimise hyper-parameter 'exploration_weight' ('c' in puct formula)
            )
            cstate_id, cstate_hash = mcts_tree.initialise_tree(cstate)

            trajectory: List[tuple[CanonicalState,tuple[np.ndarray,float]]] = []  # will store (cstate, pi) along the episode
            id_hash_traj: List[tuple[int,int]] = []

            #default pi and z values
            act_dim = self.internal_get_act_dim()
            assert act_dim>0, f"Somehow the dimension of all actions is {act_dim}, which is illegal"
            pi = np.full(act_dim, 1/act_dim)
            z = 0

            # simulate one full episode
            for i in range(self.max_len):
                if cstate.is_terminal:
                    if cstate.is_goal:
                        LOGGER.info(f'[GOAL_ANNOUNCER] Reached goal after {i+1} steps on problem {self.p.problem_meta.name}')
                    else:
                        LOGGER.info(f'[GOAL_ANNOUNCER] Reached non-goal after {i+1} steps on problem {self.p.problem_meta.name}')
                    #if pi or z are not assigned that means the initial state was terminal
                    # which is 100% dumb so exception should be raised
                    trajectory.append((cstate, (pi, z)))
                    id_hash_traj.append((cstate_id, cstate_hash))
                    break

                # 1. Run MCTS from current state to get action distribution pi
                pi, z = mcts_tree.run_search()  # np.array [num_actions]

                # 2. Store current state and pi
                trajectory.append((cstate, (pi, z)))
                id_hash_traj.append((cstate_id, cstate_hash))

                # 3. Sample action from masked pi and re-root tree
                mask = mcts_tree.get_children_mask(act_dim=self.exposed_get_act_dim())
                # Zero out masked-out elements
                masked_pi = pi * mask
                # Renormalize to sum to 1
                masked_pi /= masked_pi.sum()

                action_index = np.random.choice(np.arange(len(pi)), p=masked_pi)

                cstate_id, cstate_hash = mcts_tree.step_forward(action_index)
                cstate = self.internal_get_state_from_identifiers(cstate_id, cstate_hash)

            states_only: np.ndarray[CanonicalState] = np.array([s[0] for s in trajectory])
            T = len(trajectory)
            self.expl_triplets += T
            for cstate, (pi, z) in trajectory:
                pi_key = tuple(np.ravel(pi)) if isinstance(pi, np.ndarray) else pi
                self.expl_states.append((cstate, (pi_key, z)))

            if self.planner_bootstrapping:
                if len(states_only) >= self.bootstrap_k:
                    sampled_traj_indices = np.random.choice(states_only, size=self.bootstrap_k, replace=False)
                else:
                    sampled_traj_indices = states_only
                print('[PLANNER_BOOTSTRAPPING] Gathering teacher experience')
                for state in sampled_traj_indices:
                    teacher_experience = []
                    state_pi_key_tuple_list = []
                    try:
                        teacher_experience = self.opt_pol_experience(state)
                        state_pi_key_tuple_list = self.internal_extract_pi_key(teacher_experience)
                    except TeacherException as ex:
                        LOGGER.warning(f'Teacher error on:\n'
                                       f'State: {state} \n'
                                       f'Problem: {self.p.problem_name} ({ex})')
                    filtered_envelope = []
                    if self.planner_bootstrapping_her or any(state.is_goal for state in [state for state,_ in teacher_experience]):
                        z = 1
                    else:
                        z = 0
                    for state, pi_key in state_pi_key_tuple_list:
                        nactions = sum(p[1] for p in state.acts_enabled)

                        # pi_key = tuple()

                        if nactions <= 1:
                            # skip states
                            continue
                        filtered_envelope.append((state,(pi_key,z)))

                    self.expl_states.extend(filtered_envelope)
                    self.planner_trajectories.append((filtered_envelope, z))
                print('[PLANNER_BOOTSTRAPPING] Finished gathering teacher experience')

            if self.heuristic_bootstrapping:
                if T >= self.bootstrap_k:
                    sampled_traj_indices: List[int] = np.random.choice(T, size=self.bootstrap_k, replace=False)
                else:
                    sampled_traj_indices: List[int] = [i for i in range(T)]
                print('[HEURISTIC_BOOTSTRAPPING] Acquiring sampled states heurstics.')
                for ind in sampled_traj_indices:
                    state_id = id_hash_traj[ind][0]
                    sampled_state = states_only[ind]
                    sampled_state_v = self.internal_get_state_value(state_id, sampled_state)
                    children = mcts_tree.get_children_states(state_id)
                    children_values = []
                    for action_id, child in children.items():
                        child_state_id = child.state_id
                        child_state = self.id_hash_to_state[child_state_id, hash(child)]
                        child_h = self.internal_get_state_h(child_state)
                        children_values.append((action_id, child_h))
                    logits = np.full(self.internal_get_act_dim(), -np.inf, dtype=np.float32)
                    for act, h_act in children_values:
                        logits[act] = -1 * self.estimator_value_conversion_lambda * h_act

                    # subtract max for stability (this handles -inf too)
                    shifted = logits - np.max(logits)

                    exp_vals = np.exp(shifted)
                    sampled_state_softmax = exp_vals / np.sum(exp_vals)
                    sampled_state_softmax = tuple(sampled_state_softmax)

                    self.expl_states.append((sampled_state, (sampled_state_softmax, sampled_state_v)))

            if not hasattr(self, "policy_only") or self.policy_only:
                # # 4. Determine game outcome z
                # if cstate.is_goal:
                #     print('[HER_DEBUG] Reached goal!')
                #     z_true = 1.0
                # elif cstate.is_terminal:
                #     z_true = -1.0
                # else:
                #     z_true = 0.0 # reached max_len without being terminal
                #
                # # 5. Add all states from trajectory with same outcome z
                # for cstate, pi in trajectory:
                #     pi_key = tuple(np.ravel(pi)) if isinstance(pi, np.ndarray) else pi
                #     self.expl_states.append((cstate, (pi_key, z_true)))

                # ------ Last states value cache ----------
                last_states = states_only[-5:]
                last_states_id = [state_id for state_id, state_hash in id_hash_traj[-5:]]
                last_states_values = [self.internal_get_state_value(state_id, state) for state_id, state in zip(last_states_id,last_states)]
                self.last_states_value_cache.extend(last_states_values)

                # ------ Hindsight Experience Replay ----------
                if self.her_k:
                    if self.mcts_her_strategy:
                        sampled_states_and_pi = self.internal_sample_k_states_from_tree(mcts_tree)
                        z_her = 1
                        for state, pi in sampled_states_and_pi:
                            pi_key = tuple(np.ravel(pi)) if isinstance(pi, np.ndarray) else pi
                            self.expl_states.append((state,(pi_key, z_her)))
                    else:
                        for t in range(T-1):
                            s_t, pi_t = trajectory[t]
                            s_tp1, _ = trajectory[t+1]

                            sampled_traj_indices = self.internal_sample_k_future_states(curr_t=t,state_list=states_only)

                            for s_k in sampled_traj_indices:
                                # HER success if the current state equals the chosen HER goal
                                if s_tp1 == s_k:
                                    z_her = 1.0
                                    # print('[HER_DEBUG] Found a state that is z_her=1')
                                else:
                                    z_her = 0.0
                                # Use same policy target
                                pi_key = tuple(np.ravel(pi_t)) if isinstance(pi_t, np.ndarray) else pi_t
                                # Add HER transition
                                self.expl_states.append((s_t, (pi_key, z_her)))
            return states_only[-1].is_goal, len(states_only)


        def flatten_obs_qvs(self, rich_obs_qvs):
            cstates, rich_qvs = zip(*rich_obs_qvs)
            obs_tensor = np.stack(
                [s.to_network_input() for s in cstates], axis=0)
            qv_lists = []
            for qv_pairs in rich_qvs:
                qv_dict = dict(qv_pairs)
                qv_list = [
                    qv_dict[ba] for ba in self.p.problem_meta.bound_acts_ordered
                ]
                qv_lists.append(qv_list)
            qv_tensor = np.array(qv_lists, dtype=float)
            return obs_tensor, qv_tensor

        def flatten_obs_pi_z(self, rich_obs_pi_z):
            cstates, rich_pi_z = zip(*rich_obs_pi_z)  # each entry is (cstate, (pi, z))
            obs_tensor = np.stack([s.to_network_input() for s in cstates], axis=0)

            pi_list = []
            z_list = []
            for pi, z in rich_pi_z:
                pi_list.append(pi)  # already a distribution over actions
                z_list.append(z)  # scalar outcome
            pi_tensor = np.array(pi_list, dtype=float)
            z_tensor = np.array(z_list, dtype=float).reshape(-1, 1)

            return obs_tensor, pi_tensor, z_tensor

        def exposed_initialise_estimator(self, enhsp_config: str):
            # to avoid circular imports
            from post_training.enhspwrapper import ENHSPEstimator
            assert self.initialised, "Can't init estimator before full object"
            assert not self.estimator_initialised, "Can't double-init"
            LOGGER.debug("ProblemService started estimator initialisation.")
            self.estimator = ENHSPEstimator(self.p, enhsp_config)
            self.estimator_initialised = True
            LOGGER.debug("ProblemService finished estimator initialisation.")

        def internal_get_state_h(self, cstate) -> float:
            assert self.estimator_initialised, "Can't get state h value without estimator initialised"
            return self.estimator.get_cstate_h(cstate)

        def exposed_get_state_h(self, cstate_id: int, cstate_hash: int=0) -> float:
            if cstate_hash == 0:
                assert isinstance(cstate_id, tuple)
                cstate_hash = cstate_id[1]
                cstate_id = cstate_id[0]
            return self.internal_get_state_h(self.id_hash_to_state.get((cstate_id, cstate_hash), None))

        def internal_get_init_state(self) -> CanonicalState:
            return get_init_cstate(self.p)

        def internal_to_network_input(self, cstate):
            if cstate is None:
                return self.current_state.to_network_input() #TODO: there is no 'self.current_state'
            return cstate.to_network_input()

        def exposed_to_network_input(self, cstate_id : int, cstate_hash : int):
            try:
                cstate = self.id_hash_to_state.get((cstate_id,cstate_hash),None)
                return self.internal_to_network_input(cstate)
            except Exception as e:
                log_path = "/tmp/problemservice_exceptions.log"
                with open(log_path, "a") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"[PID {os.getpid()}] Exception in to_network_input:\n")
                    traceback.print_exc(file=f)
                    f.write("=" * 80 + "\n")

                # also print to stderr for live feedback
                print(f"[SERVER ERROR] Exception in to_network_input: {e}", file=sys.stderr)
                traceback.print_exc()

                # re-raise to ensure RPyC propagates the error correctly
                raise

        def exposed_get_applicable_action_mask(self, cstate_id, cstate_hash):
            try:
                cstate = self.id_hash_to_state.get((cstate_id,cstate_hash), None)
                if cstate is None:
                    return None
                return self.internal_get_applicable_action_mask(cstate)
            except Exception as e:
                log_path = "/tmp/problemservice_exceptions.log"
                with open(log_path, "a") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"[PID {os.getpid()}] Exception in get_applicable_action_mask:\n")
                    traceback.print_exc(file=f)
                    f.write("=" * 80 + "\n")

                # also print to stderr for live feedback
                print(f"[SERVER ERROR] Exception in get_applicable_action_mask: {e}", file=sys.stderr)
                traceback.print_exc()

                # re-raise to ensure RPyC propagates the error correctly
                raise

        def internal_get_applicable_action_mask(self, cstate: CanonicalState):
            return [activated for _, activated in cstate.acts_enabled]

        # def exposed_make_network(self, weights_manager: PropNetworkWeights, prob_meta, dropout, debug, policy_network_only):
        def exposed_make_network(self, weights_np, prob_meta, dropout, debug, policy_network_only):
            assert self.initialised
            assert self.estimator_initialised
            weights_np = to_local(weights_np)
            prob_meta = to_local(prob_meta)
            dropout = to_local(dropout)
            debug = to_local(debug)
            policy_network_only = to_local(policy_network_only)
            print("starting to create network")
            weights_manager = PropNetworkWeights.from_numpy(prob_meta, weights_np)
            self.network = PropNetwork(weights_manager, prob_meta, dropout=dropout, debug=debug,
                                       policy_network_only=policy_network_only, trainable=False)
            hits = find_netrefs(self.network)
            assert not hits, f"Netrefs leaked into local model: {hits}"
            print("network created")
            init_cstate_as_network_input = to_local(self.internal_get_init_state().to_network_input())
            print("initial state achieved properly")
            # Run a dummy forward pass with the initial cstate to initialize TensorFlow weight shapes
            print("[TF WORKER] before forward", flush=True)
            print("num_props:", prob_meta.num_props, "len(bound_props_ordered):", len(prob_meta.bound_props_ordered),
                  flush=True)
            print("num_flnts:", getattr(prob_meta, "num_flnts", None), "len(bound_flnts_ordered):",
                  len(getattr(prob_meta, "bound_flnts_ordered", [])), flush=True)
            print("num_comps:", getattr(prob_meta, "num_comps", None), "len(bound_comps_ordered):",
                  len(getattr(prob_meta, "bound_comps_ordered", [])), flush=True)
            print("num_acts:", getattr(prob_meta, "num_acts", None), "len(bound_acts_ordered):",
                  len(getattr(prob_meta, "bound_acts_ordered", [])), flush=True)
            print("len(goal_props):", len(getattr(prob_meta, "goal_props", None)), flush=True)
            print("len(goal_flnts):", len(getattr(prob_meta, "goal_flnts", None)), flush=True)

            assert prob_meta.num_props == len(prob_meta.bound_props_ordered), (prob_meta.num_props,
                                                                               len(prob_meta.bound_props_ordered))
            if hasattr(prob_meta, "use_fluents") or hasattr(prob_meta, "num_flnts"):
                assert prob_meta.num_flnts == len(prob_meta.bound_flnts_ordered)
            if hasattr(prob_meta, "num_comps"):
                assert prob_meta.num_comps == len(prob_meta.bound_comps_ordered)

            self.network(init_cstate_as_network_input[None], training=False)
            print("[TF WORKER] after forward", flush=True)
            # self.internal_run_tf(
            #     self.network,
            #     init_cstate_as_network_input[None],
            #     training=False
            # )
            print(f"Remote weights for problem {prob_meta.name}: {len(self.network.get_weights())}")

            self.network_initialised = True

            # this is important to send back as it will be used to pass forward in the local (i.e. not on the service)
            # network before the beginning of the training/inference
            return init_cstate_as_network_input

        # def exposed_make_network(self, *args):
        #     print("make_network called")
        #     self.network_initialised = True
        #     return None

        def internal_set_weights(self, weights):
            assert self.network_initialised
            weights = to_local(weights)
            self.network.set_weights(weights)

        def exposed_turn_off_planner_bootstrapping(self):
            self.planner_bootstrapping = False

        def internal_extract_pi_key(self, teacher_experience):
            states_qv_tuple_list = [(key_value, tuple(item[1] for item in act_qv_tuple)) for key_value, act_qv_tuple in teacher_experience]
            states_pi_keys = []
            for state_qv_tuple in states_qv_tuple_list:
                state, qv = state_qv_tuple
                eps = 1e-6
                pi_key = np.array([1/(elem+eps) for elem in qv])
                pi_sum = np.sum(pi_key)
                pi_norm = pi_key/pi_sum
                pi_tuple = tuple(pi_norm)
                states_pi_keys.append((state, pi_tuple))
            return states_pi_keys

        def exposed_log_planner_trajectories(self):
            planner_call_count = len(self.planner_trajectories)
            planner_success_count = sum(z for _,z in self.planner_trajectories)
            planner_success_rate = planner_success_count/planner_call_count if planner_call_count > 0 else 0
            LOGGER.info(f"[PLANNER_BOOTSTRAPPING_LOG - {self.p.problem_name}] planner_call_count: {planner_call_count} trajectories, planner_success_count: {planner_success_count}, planner_success_rate: {planner_success_rate}")
            self.planner_trajectories.clear()

        def exposed_get_problem_data(self):
            return self.internal_get_obs_dim(), self.internal_get_act_dim(), self.p.domain_meta, self.p.problem_meta, self.exposed_get_dg_extra_dim()

        def exposed_update_difficulty(self, difficulty: int):
            """
            Update instance difficulty, 0 being easy, 1 being medium, 2 being hard.
            """
            self.p.update_difficulty(difficulty)


    return ProblemService

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
                 ):
        # gets incremented to deal with TF
        self.batches_seen = 0
        # self.problems = problems
        self.policy_only = policy_only
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
                "one of the LR step epoch nums (%s) was not an int" % (k, )
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

        # self.planner_bootstrapping = self.explorer.planner_bootstrapping
        # Quick sanity checks
        # assert hasattr(self, "network")
        # assert self.network is not None
        # assert self.network.trainable_weights is self.weight_manager.all_weights
        # # --- CHECK 10: single source of truth for weights ---
        # net_refs = {v.ref() for v in self.network.trainable_weights}
        # wm_refs = {v.ref() for v in self.weight_manager.all_weights}
        #
        # assert net_refs == wm_refs, (
        #     "Network trainable weights do not match weight_manager weights.\n"
        #     "This means gradients will NOT update the intended variables."
        # )

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
        # assert len(self.optimiser.variables) > 1, 'optimiser build wasn\'t succesful'
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
        iter_num = 0
        time_since_best = 0
        solve_thresh = 0.999

        tr = tqdm.trange(max_epochs, desc='epoch', leave=True)

        epoch = tf.Variable(0, dtype=tf.int64)
        self.summary_writer.set_as_default(step=epoch)

        for epoch_num in tr:
            epoch.assign(epoch_num)

            # --------------------------------------------------
            # 1. EXPLORE (spawn workers, compute grads there)
            # --------------------------------------------------
            t_explore=time()
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
            tf_and_log('states', n_states)
            tf_and_log('lr', self.optimiser.lr)

            iter_num += 1

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
                    or (self.save_every and iter_num % self.save_every == 0)
                    or iter_num == 1
            )

            if should_save:
                best_rate = total_succ_rate
                snapshot_path = os.path.join(
                    self.snapshot_dir,
                    f'snapshot_{iter_num}_{total_succ_rate:.4f}.pkl'
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
                    and time_since_best >= self.early_stop
                    and best_rate >= solve_thresh
            ):
                LOGGER.info('Terminating early (early stop condition met)')
                break

        return best_rate, elapsed_time, iter_num

    def apply_worker_grads(self, worker_outs):
        params = self.weight_manager.all_weights
        if not worker_outs:
            raise RuntimeError("No worker outputs.")

        # init accumulators
        import numpy as np
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
        # super().__init__(reduction, name)
        # #TODO: check if Loss class that was previously a superclass did anything in its init method
        self.problems = problems
        self.weight_manager = weight_manager
        self.summary_writer = summary_writer
        self.l1_reg_coeff = l1_reg_coeff
        self.l2_reg_coeff = l2_reg_coeff
        self.l1_l2_reg_coeff = l1_l2_reg_coeff
        self.mse_coeff=mse_coeff
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
                        target_values=z,pred_values=v
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
