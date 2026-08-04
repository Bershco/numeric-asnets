from abc import abstractmethod, ABC
from enum import Enum
from functools import lru_cache
from itertools import repeat

import joblib
import logging
import numpy as np
import os
import tensorflow as tf
from time import time
import tqdm.auto as tqdm
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import datetime

from asnets.checkpointing import save_checkpoint_dir, resolve_optimizer_path
from asnets.heur_inputs import ActionCountDataGenerator, \
    HeuristicDataGenerator, LMCutDataGenerator, RelaxedDeadendDetector, \
    NumericLandmarkGenerator
from asnets.utils.generator_utils import InstanceDifficulty, get_problem_names
from asnets.utils.mdpsim_utils import parse_problem_args
from asnets.prob_dom_meta import BoundAction, DomainType, get_domain_meta, \
    get_problem_meta
from asnets.interfaces.jpddl_interface import start_jvm
from asnets.interfaces.ssipp_interface import set_up_ssipp
from asnets.state_reprs import CanonicalState
from asnets.teacher import Teacher
from asnets.utils.prof_utils import can_profile
from asnets.utils.pddl_utils import get_domain_file
from asnets.utils.py_utils import TimerContext, strip_parens, weak_ref_to, weighted_batch_iter
from asnets.utils.tf_utils import cross_entropy, mean_squared_error, empty_feed_value
from asnets.models import PropNetwork, PropNetworkWeights
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
                 dg_use_act_history: bool = False, ):
        """Initialise a PlannerExtensions object.

        Args:
            pddl_files (List[str]): The PDDL files to load.
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
        self.pddl_files = pddl_files
        self.domain_type = domain_type

        import mdpsim  # noqa: F811
        import ssipp  # noqa: F811
        current_problem = pddl_files[1]
        self.current_problem_name = get_problem_names([current_problem])[0]

        print(f'Starting to parse mdpsim problem: {self.current_problem_name}')
        # MDPSim stuff
        self.mdpsim: ModuleType = mdpsim
        self.mdpsim_problem = parse_problem_args(self.mdpsim, self.pddl_files, self.current_problem_name)
        self.problem_name: str = self.mdpsim_problem.name.strip()

        print(f'Finished parsing mdpsim problem: {self.problem_name}')

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
        self.act_ident_to_ind: Dict[str, int] = {
            f"({a})": idx
            for a, idx in self.problem_meta._unique_id_to_index.items()
        }

        self.ind_to_act_ident: Dict[int, str] = {
            idx: f"({a})"
            for a, idx in self.problem_meta._unique_id_to_index.items()
        }
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
        if not jpype.isJVMStarted():
            print(f"[DEBUG GC] JVM not running at del | PID={os.getpid()}")

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


def tf_and_log(name: str, value):
    tf.summary.scalar(name, value)
    base_name = tf.get_current_name_scope()
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
    # Use MCTS visit distribution instead of a teacher altogether
    MCTS_VISIT_DIST = 3


class BaseTrainer(ABC):

    def __init__(self, weight_manager, summary_writer, explorer, validator, lr, l1_reg_coeff, l2_reg_coeff,
                 l1_l2_reg_coeff, lr_steps
                 ):
        self._weight_manager = weight_manager
        # may be None if no summaries tuple()should be written
        self.summary_writer = summary_writer
        self.explorer = explorer
        self.validator = validator
        self.lr = lr
        self.l1_reg_coeff = l1_reg_coeff
        self.l2_reg_coeff = l2_reg_coeff
        self.l1_l2_reg_coeff = l1_l2_reg_coeff
        self.tf_init_done = False
        lr_steps = [(0, lr)] + sorted(lr_steps)
        for k, lr in lr_steps:
            assert k >= 0, "one of the steps was negative (?)"
            assert isinstance(k, int), \
                "one of the LR step epoch nums (%s) was not an int" % (k,)
            assert lr > 0, \
                "one of the given learning rates was not positive (?)"
        self.lr_steps = lr_steps
        self.lr_steps_remaining = list(lr_steps)

    @abstractmethod
    def train(self, max_epochs):
        pass

    @abstractmethod
    def _init_tf(self):
        pass


class SupervisedTrainer(BaseTrainer):
    @can_profile
    def __init__(self,
                 # problems,
                 weight_manager,
                 summary_writer,
                 explorer,
                 validator,
                 start_time,
                 snapshot_dir,
                 *,
                 lr=0.001,
                 lr_steps=[],
                 l1_reg_coeff,
                 l2_reg_coeff,
                 l1_l2_reg_coeff,
                 mse_coeff,
                 batch_size,
                 train_steps_per_epoch=1,
                 policy_anchor_kl_coeff=0.0,
                 main_road_fraction=0.75,
                 tree_policy_weight=0.5,
                 grad_clip_norm=5.0,
                 hide_progress=False,
                 time_out=40,
                 early_stop=20,
                 save_every=20,
                 balanced_success_rate=True,
                 resume_from=None,
                 ):
        super().__init__(weight_manager, summary_writer, explorer, validator, lr, l1_reg_coeff, l2_reg_coeff,
                         l1_l2_reg_coeff, lr_steps)
        # gets incremented to deal with TF
        self.balanced_success_rate = balanced_success_rate
        self.hide_progress = hide_progress
        self.mse_coeff = mse_coeff
        self.batch_size = batch_size
        self.train_steps_per_epoch = train_steps_per_epoch
        self.policy_anchor_kl_coeff = float(policy_anchor_kl_coeff)
        if self.policy_anchor_kl_coeff < 0:
            raise ValueError("policy_anchor_kl_coeff must be non-negative")
        self.main_road_fraction = main_road_fraction
        self.tree_policy_weight = tree_policy_weight
        self.grad_clip_norm = grad_clip_norm
        self.timer = TimerContext()
        self.start_time = start_time
        self.timeout = time_out
        self.early_stop = early_stop
        self.save_every = save_every
        self.snapshot_dir = snapshot_dir
        self._policy_anchor_networks = {}
        self._policy_anchor_weights_np = None
        if self.policy_anchor_kl_coeff > 0:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            anchor_path = os.path.join(
                self.snapshot_dir, "policy_anchor_weights.joblib")
            if os.path.exists(anchor_path):
                self._policy_anchor_weights_np = joblib.load(anchor_path)
                anchor_source = "restored persisted stage-1 anchor"
            elif (
                    resume_from is not None
                    and not resume_from.endswith(".pkl")
                    and os.path.exists(os.path.join(
                        os.path.dirname(resume_from),
                        "policy_anchor_weights.joblib",
                    ))
            ):
                resume_anchor_path = os.path.join(
                    os.path.dirname(resume_from),
                    "policy_anchor_weights.joblib",
                )
                self._policy_anchor_weights_np = joblib.load(
                    resume_anchor_path)
                temp_path = f"{anchor_path}.{os.getpid()}.tmp"
                joblib.dump(
                    self._policy_anchor_weights_np,
                    temp_path,
                    compress=True,
                )
                os.replace(temp_path, anchor_path)
                anchor_source = (
                    "restored stage-1 anchor from resumed experiment "
                    f"{resume_anchor_path}"
                )
            else:
                self._policy_anchor_weights_np = \
                    self._weight_manager.export_numpy()
                temp_path = f"{anchor_path}.{os.getpid()}.tmp"
                joblib.dump(
                    self._policy_anchor_weights_np, temp_path, compress=True)
                os.replace(temp_path, anchor_path)
                anchor_source = "created from initial stage-1 weights"
            print(
                "[POLICY ANCHOR] enabled "
                f"coeff={self.policy_anchor_kl_coeff}; {anchor_source}; "
                f"path={anchor_path}"
            )
        self._init_tf()
        if resume_from is not None and not resume_from.endswith(".pkl"):
            opt_path = os.path.join(resume_from, "optimizer.joblib")
            if os.path.exists(opt_path):
                trainable_vars = self._weight_manager.all_weights
                assert len(trainable_vars) > 0
                # Force Adam slot variable creation
                self.optimizer.apply_gradients([
                    (tf.zeros_like(v), v)
                    for v in trainable_vars
                ])
                print(
                    "[Optimizer BEFORE restore]",
                    self.optimizer.iterations.numpy(),
                )
                saved_opt = joblib.load(opt_path)

                # ---------------------------------------
                # New format (dict keyed by variable name)
                # ---------------------------------------
                if isinstance(saved_opt, dict):

                    restored = 0
                    skipped = 0

                    for var in self.optimizer.variables():

                        entry = saved_opt.get(var.name)

                        if entry is None:
                            skipped += 1
                            continue

                        if tuple(var.shape) != tuple(entry["shape"]):
                            print(
                                f"[OPT SHAPE MISMATCH] "
                                f"{var.name} "
                                f"current={tuple(var.shape)} "
                                f"saved={entry['shape']}"
                            )
                            skipped += 1
                            continue

                        var.assign(entry["value"])
                        restored += 1

                    print(
                        f"[Optimizer] restored={restored} "
                        f"skipped={skipped}"
                    )

                # ---------------------------------------
                # Old format (list)
                # ---------------------------------------
                else:

                    opt_vals = saved_opt

                    if (
                            len(opt_vals) == len(self.optimizer.variables())
                            and all(
                        tuple(var.shape) == tuple(val.shape)
                        for var, val in zip(
                            self.optimizer.variables(),
                            opt_vals
                        )
                    )
                    ):
                        for var, val in zip(
                                self.optimizer.variables(),
                                opt_vals
                        ):
                            var.assign(val)

                        print("[Optimizer] restored from legacy checkpoint")

                    else:
                        print(
                            "[Optimizer] legacy checkpoint incompatible, "
                            "starting with fresh optimizer state"
                        )

                print(
                    "optimizer step:",
                    self.optimizer.iterations.numpy(),
                )

    @can_profile
    def _init_tf(self):
        """Do setup necessary for network (e.g. initialising weights)."""
        assert not self.tf_init_done, \
            "this class is not designed to be initialised twice"

        if len(self.lr_steps) > 1:
            # using a scheduler to control the learning rate
            boundaries = [i[0] for i in self.lr_steps[1:]]
            values = [i[1] for i in self.lr_steps]
            lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries, values)
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_scheduler)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self._log_ops = {}
        self.tf_init_done = True

    def train(self, max_epochs):
        last_rate = None
        time_since_best = 0

        patience_counter = 0
        cooldown_counter = 0

        best_valid_rate = None
        best_valid_average_plan_length = None

        PATIENCE = 2  # consecutive validations
        COOLDOWN_EPOCHS = 10
        VALIDATE_EVERY = 1
        THRESHOLD_EASY = 0.85

        tr = tqdm.trange(max_epochs, desc='epoch', leave=True)
        epoch = tf.Variable(0, dtype=tf.int64)
        self.summary_writer.set_as_default(step=epoch)

        for epoch_num in tr:
            epoch.assign(epoch_num)

            # --------------------------------------------------
            # 1. EXPLORE (spawn workers, compute grads there)
            # --------------------------------------------------
            t_explore = time()
            worker_outs = self.explorer.explore(self._weight_manager.export_numpy())
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
            # 2. INGEST REPLAY (central optimization follows in a later stage)
            # --------------------------------------------------
            replay_stats = self.explorer.add_worker_outputs_to_main_road_replay(
                worker_outs,
            )
            n_states = replay_stats["collected"]
            W0 = [w.numpy().copy() for w in self._weight_manager.all_weights]
            train_stats = self.train_from_replay()
            total_succ_rate = float(np.mean([out.hit_goal_mean for out in worker_outs]))
            succ_rate_easy, succ_rate_medium, succ_rate_hard = self.calculate_balanced_succ_rate(worker_outs)
            tf_and_log("train/updates", train_stats["updates"])
            if train_stats["updates"] > 0:
                deltas = [
                    np.mean(np.abs(w.numpy() - w0))
                    for w0, w in zip(W0, self._weight_manager.all_weights)
                ]
                tf_and_log("weight-delta/mean", np.mean(deltas))
                tf_and_log("weight-delta/max", np.max(deltas))
                tf_and_log("train-loss", train_stats["total_loss"])
                tf_and_log("train/policy_loss", train_stats["policy_loss"])
                tf_and_log("train/value_loss", train_stats["value_loss"])
                tf_and_log(
                    "train/policy_anchor_kl_loss",
                    train_stats["policy_anchor_kl_loss"],
                )
                tf_and_log("train/reg_loss", train_stats["reg_loss"])
                tf_and_log("grad/global_norm_unclipped", train_stats["grad_norm"])
                tf_and_log("grad/global_norm_clipped", train_stats["clipped_grad_norm"])
                tf_and_log("grad/was_clipped", train_stats["was_clipped"])
                tf_and_log("grad/none_grad_count", train_stats["none_grad_count"])
            else:
                LOGGER.warning(
                    "Replay training skipped: main-road=%d tree=%d",
                    replay_stats["main_road_size"],
                    replay_stats["tree_size"],
                )
            tf_and_log('replay/main_road_added', replay_stats["main_road_added"])
            tf_and_log('replay/tree_added', replay_stats["tree_added"])
            tf_and_log(
                'replay/tree_nodes_examined',
                replay_stats["tree_nodes_examined"],
            )
            tf_and_log('replay/tree_eligible', replay_stats["tree_eligible"])
            tf_and_log('replay/tree_emitted', replay_stats["tree_emitted"])
            tf_and_log(
                'replay/tree_unique_added',
                replay_stats["tree_unique_added"],
            )
            tf_and_log(
                'replay/tree_duplicates_merged',
                replay_stats["tree_duplicates_merged"],
            )
            tf_and_log('replay/tree_trimmed', replay_stats["tree_trimmed"])
            tf_and_log('replay/main_road_size', replay_stats["main_road_size"])
            tf_and_log('replay/tree_size', replay_stats["tree_size"])
            tf_and_log(
                'replay/compatibility_bucket_count',
                replay_stats["compatibility_bucket_count"],
            )
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
            tf_and_log('lr', self.optimizer.lr)

            if active_rates:
                total_succ_rate = balanced_rate  # if we want to balance rates, this is the real deal
            tr.set_postfix(
                succ_rate=total_succ_rate,
                states=n_states,
                main_road_replay=replay_stats["main_road_size"],
                tree_replay=replay_stats["tree_size"],
                train_loss=train_stats["total_loss"],
                lr=self.optimizer.lr,
                refresh=False,
            )

            last_rate = total_succ_rate
            snapshot_name = f"snapshot_{epoch_num}_{total_succ_rate:.4f}"


            # --------------------------------------------------
            # 2.1 validation + decay + progression
            # --------------------------------------------------
            if epoch_num % VALIDATE_EVERY == 0:
                success_rates, overall_succ_rate, validation_outs = \
                    self.validator.evaluate(self._weight_manager.export_numpy())
                solved_outs = [o for o in validation_outs if o.hit_goal]

                avg_plan_len = (
                    sum(len(o.plan) for o in solved_outs) / len(solved_outs)
                    if solved_outs else float("inf")
                )
                tf_and_log("validation/success_rate", overall_succ_rate)
                tf_and_log("validation/avg_plan_length", avg_plan_len)
                is_better = (
                        best_valid_rate is None
                        or overall_succ_rate > best_valid_rate
                        or (
                                overall_succ_rate == best_valid_rate
                                and avg_plan_len < best_valid_average_plan_length
                        )
                )
                if is_better:
                    best_valid_rate = overall_succ_rate
                    best_valid_average_plan_length = avg_plan_len

                    print(
                        f"[VALIDATION] New best! "
                        f"succ={best_valid_rate:.3f} "
                        f"avg_len={best_valid_average_plan_length:.2f} "
                        f"iter_num={epoch_num} "
                        f"snapshot_name={snapshot_name}"
                    )
                print(f"[VALIDATION] Current network validation success rate: {overall_succ_rate}")
                # -------------------------
                # 1. Estimator decay
                # -------------------------
                if cooldown_counter == 0:
                    if success_rates.get(InstanceDifficulty.EASY, 0.0) >= THRESHOLD_EASY:
                        patience_counter += 1
                    else:
                        patience_counter = 0

                    if patience_counter >= PATIENCE:
                        self.explorer.decay_estimator_coefficient()
                        print("[VALIDATION] Estimator coefficient decayed")

                        patience_counter = 0
                        cooldown_counter = COOLDOWN_EPOCHS
                else:
                    cooldown_counter -= VALIDATE_EVERY

                # -------------------------
                # 2. Instance Progression
                # -------------------------
                if self.can_progress(success_rates):
                    if self.explorer.advance_progression_level():
                        # this progresses the progression level and returns true if advanced, if current progression level is max - returns false
                        cooldown_counter = max(cooldown_counter, int(COOLDOWN_EPOCHS / 2))

            save_checkpoint_dir(
                snapshot_dir=self.snapshot_dir,
                snapshot_name=snapshot_name,
                weight_manager=self._weight_manager,
                optimizer=self.optimizer,
                trainer_state={
                    "epoch_num": epoch_num,
                    "best_rate": last_rate,
                    "time_since_best": time_since_best,
                },
            )
            tf.summary.flush()
            elapsed_time = time() - self.start_time
            if self.timeout and elapsed_time > self.timeout * 0.95:
                LOGGER.info('[TIMING_TERMINATION] Timeout reached')
                tr.refresh()  # this guarantees tqdm display of last iteration
                break
        return last_rate, elapsed_time, int(epoch)

    def can_progress(self, success_rates):
        thresholds = {
            InstanceDifficulty.EASY: 0.90,
            InstanceDifficulty.MEDIUM: 0.65,
            InstanceDifficulty.HARD: 0.50,
        }

        active = self.explorer.progression_level.get_active_difficulties()

        for d in active:
            if d not in success_rates:
                return False
            if success_rates[d] < thresholds[d]:
                return False

        return True

    def _sample_from_replay(self, problem, replay, batch_size):
        dataset = problem.weighted_dataset(replay)
        if problem.network.value_head_enabled:
            obs, pi, z, counts = dataset
            return next(weighted_batch_iter((obs, pi, z), counts, batch_size, 1))
        obs, pi, counts = dataset
        sampled_obs, sampled_pi = next(weighted_batch_iter((obs, pi), counts, batch_size, 1))
        return sampled_obs, sampled_pi, None

    def _sample_mixed_replay_batch(self, problem):
        main_road_available = len(problem.replay) > 0
        tree_available = len(problem.sampled_states_replay) > 0
        if not main_road_available and not tree_available:
            return None
        if main_road_available and tree_available:
            main_road_size = int(round(self.batch_size * self.main_road_fraction))
            tree_size = self.batch_size - main_road_size
        elif main_road_available:
            main_road_size, tree_size = self.batch_size, 0
        else:
            main_road_size, tree_size = 0, self.batch_size

        batches, policy_weights = [], []
        if main_road_size:
            batches.append(self._sample_from_replay(problem, problem.replay, main_road_size))
            policy_weights.append(np.ones(main_road_size, dtype=np.float32))
        if tree_size:
            batches.append(self._sample_from_replay(problem, problem.sampled_states_replay, tree_size))
            policy_weights.append(np.full(
                tree_size,
                self.tree_policy_weight,
                dtype=np.float32,
            ))

        obs = np.concatenate([batch[0] for batch in batches], axis=0)
        pi = np.concatenate([batch[1] for batch in batches], axis=0)
        z = (
            np.concatenate([batch[2] for batch in batches], axis=0)
            if problem.network.value_head_enabled else None
        )
        return obs, pi, z, np.concatenate(policy_weights, axis=0)

    def _replay_reg_loss(self, params, dtype):
        reg_loss = tf.constant(0.0, dtype=dtype)
        if self.l2_reg_coeff:
            reg_loss += tf.cast(self.l2_reg_coeff, dtype) * tf.add_n([
                tf.reduce_sum(tf.square(param)) for param in params
            ])
        if self.l1_reg_coeff:
            reg_loss += tf.cast(self.l1_reg_coeff, dtype) * tf.add_n([
                tf.reduce_sum(tf.abs(param)) for param in params
            ])
        if self.l1_l2_reg_coeff:
            reg_loss += tf.cast(self.l1_l2_reg_coeff, dtype) * tf.add_n([
                tf.reduce_sum(tf.abs(param)) + tf.reduce_sum(tf.square(param))
                for param in params
            ])
        return reg_loss

    def _anchor_policy(self, problem, obs_tf):
        key = id(problem)
        anchor_network = self._policy_anchor_networks.get(key)
        if anchor_network is None:
            anchor_weights = PropNetworkWeights.from_numpy(
                problem.problem_meta, self._policy_anchor_weights_np)
            anchor_network = PropNetwork(
                anchor_weights,
                problem.problem_meta,
                dropout=0.0,
                trainable=False,
            )
            self._policy_anchor_networks[key] = anchor_network
        anchor_out = anchor_network(obs_tf, training=False)
        anchor_policy = anchor_out[0] if isinstance(anchor_out, tuple) \
            else anchor_out
        return tf.stop_gradient(anchor_policy)

    def _train_replay_step(self):
        params = self._weight_manager.all_weights
        sampled_batches = [
            (problem, self._sample_mixed_replay_batch(problem))
            for problem in self.explorer.problems
        ]
        sampled_batches = [(problem, batch) for problem, batch in sampled_batches if batch is not None]
        if not sampled_batches:
            return None

        with tf.GradientTape() as tape:
            policy_losses, value_losses, anchor_kl_losses = [], [], []
            for problem, (obs, pi_tgt, z_tgt, policy_weights) in sampled_batches:
                obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
                pi_tgt_tf = tf.convert_to_tensor(pi_tgt, dtype=tf.float32)
                policy_weights_tf = tf.convert_to_tensor(policy_weights, dtype=tf.float32)
                if problem.network.value_head_enabled:
                    pi_pred, value_pred = problem.network(obs_tf, training=True)
                else:
                    pi_pred, value_pred = problem.network(obs_tf, training=True), None

                xent_per_example = -tf.reduce_sum(
                    pi_tgt_tf * tf.math.log(tf.clip_by_value(pi_pred, 1e-8, 1.0)),
                    axis=1,
                )
                policy_losses.append(tf.math.divide_no_nan(
                    tf.reduce_sum(xent_per_example * policy_weights_tf),
                    tf.reduce_sum(policy_weights_tf),
                ))
                if self.policy_anchor_kl_coeff > 0:
                    anchor_pi = self._anchor_policy(problem, obs_tf)
                    anchor_kl_per_example = tf.reduce_sum(
                        anchor_pi * (
                            tf.math.log(tf.clip_by_value(
                                anchor_pi, 1e-8, 1.0))
                            - tf.math.log(tf.clip_by_value(
                                pi_pred, 1e-8, 1.0))
                        ),
                        axis=1,
                    )
                    anchor_kl_losses.append(
                        tf.reduce_mean(anchor_kl_per_example))
                if value_pred is not None:
                    value_pred = tf.squeeze(value_pred, axis=-1)
                    z_tgt_tf = tf.convert_to_tensor(z_tgt, dtype=value_pred.dtype)
                    value_losses.append(tf.reduce_mean(tf.square(value_pred - z_tgt_tf)))

            policy_loss = tf.reduce_mean(policy_losses)
            value_loss = tf.reduce_mean(value_losses) if value_losses else tf.constant(0.0, policy_loss.dtype)
            policy_anchor_kl_loss = (
                tf.reduce_mean(anchor_kl_losses)
                if anchor_kl_losses
                else tf.constant(0.0, policy_loss.dtype)
            )
            reg_loss = self._replay_reg_loss(params, policy_loss.dtype)
            total_loss = (
                policy_loss
                + tf.cast(self.mse_coeff, policy_loss.dtype) * value_loss
                + tf.cast(
                    self.policy_anchor_kl_coeff, policy_loss.dtype
                ) * policy_anchor_kl_loss
                + reg_loss
            )

        raw_grads = tape.gradient(total_loss, params)
        none_grad_count = sum(grad is None for grad in raw_grads)
        grads = [tf.zeros_like(param) if grad is None else grad for param, grad in zip(params, raw_grads)]
        for grad in grads:
            tf.debugging.assert_all_finite(grad, "Non-finite gradient detected during replay training")

        grad_norm = tf.linalg.global_norm(grads)
        if self.grad_clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        clipped_grad_norm = tf.linalg.global_norm(grads)
        self.optimizer.apply_gradients(zip(grads, params))

        return {
            "total_loss": float(total_loss.numpy()),
            "policy_loss": float(policy_loss.numpy()),
            "value_loss": float(value_loss.numpy()),
            "policy_anchor_kl_loss": float(
                policy_anchor_kl_loss.numpy()),
            "reg_loss": float(reg_loss.numpy()),
            "grad_norm": float(grad_norm.numpy()),
            "clipped_grad_norm": float(clipped_grad_norm.numpy()),
            "was_clipped": float(self.grad_clip_norm is not None and grad_norm.numpy() > self.grad_clip_norm),
            "none_grad_count": float(none_grad_count),
        }

    def train_from_replay(self):
        step_stats = [self._train_replay_step() for _ in range(self.train_steps_per_epoch)]
        step_stats = [stats for stats in step_stats if stats is not None]
        if not step_stats:
            return {
                "updates": 0, "total_loss": 0.0, "policy_loss": 0.0,
                "value_loss": 0.0, "policy_anchor_kl_loss": 0.0,
                "reg_loss": 0.0, "grad_norm": 0.0,
                "clipped_grad_norm": 0.0, "was_clipped": 0.0,
                "none_grad_count": 0.0,
            }
        return {
            "updates": len(step_stats),
            **{key: float(np.mean([stats[key] for stats in step_stats])) for key in step_stats[0]},
        }

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


class OriginalSupervisedTrainer(BaseTrainer):
    def __init__(self,
                 problems,
                 weight_manager,
                 summary_writer,
                 explorer,
                 validator,
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
                 save_training_set=None,
                 use_saved_training_set=None,
                 resume_from=None,
                 hide_progress=False,
                 time_out=1000,
                 early_stop=20,
                 save_every=20,
                 ):
        super().__init__(weight_manager, summary_writer, explorer, validator, lr, l1_reg_coeff, l2_reg_coeff,
                         l1_l2_reg_coeff, lr_steps)
        # gets incremented to deal with TF
        self.batches_seen = 0
        self.problems = problems
        self.batch_size = batch_size
        self.batch_size_per_problem = max(batch_size // max(len(problems), 1), 1)
        self.opt_batches_per_epoch = opt_batches_per_epoch
        self.hide_progress = hide_progress
        self.timer = TimerContext()
        self.save_training_set = save_training_set
        self.use_saved_training_set = use_saved_training_set
        if use_saved_training_set:
            LOGGER.info("Loading saved training set from '%s'",
                        use_saved_training_set)
            self.loaded_training_set = joblib.load(use_saved_training_set)
        self.start_time = start_time
        self.timeout = time_out
        self.early_stop = early_stop
        self.save_every = save_every
        self.scratch_dir = scratch_dir
        self.snapshot_dir = snapshot_dir
        self.resume_from = resume_from
        self._init_tf()

    @property
    def value_head_enabled(self):
        return self._weight_manager.value_head_enabled

    def _get_replay_sizes(self):
        """Get the sizes of replay buffers for each problem."""
        rv = []
        for problem in self.problems:
            rv.append(len(problem.replay))
        return rv

    def train(self, max_epochs):
        best_train_rate = None
        best_valid_rate = None
        best_valid_average_plan_length = None
        keep_going = True
        iter_num = 0
        time_since_best = 0
        # fraction of rollouts that have to reach goal in order for problem
        # to be considered "solved"
        solve_thresh = 0.999
        tr = tqdm.trange(max_epochs, desc='epoch', leave=True)
        mean_loss = None

        # set up tensorboard logging
        epoch = tf.Variable(0, dtype=tf.int64)
        self.summary_writer.set_as_default(step=epoch)

        validate_every = 1
        consecutive_validations_best = 0
        consecutive_validation_patience = 50

        for epoch_num in tr:
            # update the epoch variable
            epoch.assign(epoch_num)
            elapsed_time = time() - self.start_time

            # only extend replay by a bit each time
            succs_probs = self.explorer.extend_replay(weights_np=self._weight_manager.export_numpy(),
                                                      epoch_num=epoch_num)
            train_succ_rate = np.mean([s for _, s in succs_probs])
            replay_sizes = self._get_replay_sizes()
            replay_size = sum(replay_sizes)

            tf.summary.scalar('lr', self.optimizer.lr)
            # update output
            tr.set_postfix(
                succ_rate=train_succ_rate,
                net_loss=mean_loss,
                states=replay_size,
                lr=self.optimizer.lr,
                refresh=False,
            )
            tf.summary.scalar('succ-rate/mean', train_succ_rate)

            for prob, prob_succ_rate in succs_probs:
                tf.summary.scalar('succ-rate/%s' % prob.name, prob_succ_rate)

            tf.summary.scalar('replay-size', replay_size)
            mean_loss = self._optimize(self.opt_batches_per_epoch)
            iter_num += 1
            # update output again
            tr.set_postfix(
                succ_rate=train_succ_rate,
                net_loss=mean_loss,
                states=replay_size,
                lr=self.optimizer.lr,
                refresh=False,
            )
            snapshot_name = f"snapshot_{iter_num}_{train_succ_rate:.3f}"
            if epoch_num % validate_every == 0:
                _, validation_succ_rate, validation_outs = \
                    self.validator.evaluate(self._weight_manager.export_numpy())
                solved_outs = [out for out in validation_outs if out.hit_goal]
                num_validation_instances_success = len(solved_outs)

                average_plan_length = (
                        sum(len(out.plan) for out in solved_outs)
                        / num_validation_instances_success
                ) if num_validation_instances_success > 0 else float("inf")
                print(f"[VALIDATION] Current network validation success rate: {validation_succ_rate:.3f} with an average plan length of {average_plan_length:.3f}")
                if best_valid_rate is None or validation_succ_rate > best_valid_rate or (validation_succ_rate == best_valid_rate and average_plan_length < best_valid_average_plan_length):
                    best_valid_rate = validation_succ_rate
                    best_valid_average_plan_length = average_plan_length
                    consecutive_validations_best = 0
                    print(f"[VALIDATION] New best reached! [success rate: {best_valid_rate} | average plan length: {best_valid_average_plan_length} | iteration {iter_num} | snapshot name: {snapshot_name}]")
                else:
                    consecutive_validations_best += 1
                if consecutive_validations_best >= consecutive_validation_patience:
                    keep_going = False
            # save checkout for every epoch, it's cheap.
            best_train_rate = train_succ_rate
            # snapshot!
            save_checkpoint_dir(
                snapshot_dir=self.snapshot_dir,
                snapshot_name=snapshot_name,
                weight_manager=self._weight_manager,
                optimizer=self.optimizer,
                trainer_state={
                    "epoch_num": epoch_num,
                    "iter_num": iter_num,
                    "best_rate": best_train_rate,
                    "time_since_best": time_since_best,
                },
            )  # also, always save timing data
            with open(os.path.join(self.scratch_dir, 'timing.json'), 'w') as fp:
                fp.write(self.timer.to_json())

            tf.summary.flush()

            if self.timeout:
                keep_going = keep_going and elapsed_time <= self.timeout

            if not keep_going:
                LOGGER.info('Terminating early')
                tr.refresh()  # this guarantees tqdm display of last iteration
                break

        return best_train_rate, elapsed_time, iter_num

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
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_scheduler)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.loss_fn = ManualLoss(
            problems=self.problems,
            weight_manager=self._weight_manager,
            summary_writer=self.summary_writer,
            l1_reg_coeff=self.l1_reg_coeff,
            l2_reg_coeff=self.l2_reg_coeff,
            l1_l2_reg_coeff=self.l1_l2_reg_coeff,
            name="loss_fn",
            strategy=SupervisedObjective.ANY_GOOD_ACTION
        )

        trainable_vars = list(self._weight_manager.all_weights)
        self.optimizer.build(trainable_vars)

        self._maybe_restore_optimizer()
        # tensorboard ops
        self._log_ops = {}
        self.tf_init_done = True

    def _maybe_restore_optimizer(self):
        """Restore optimizer state if resuming training."""
        if not hasattr(self, "resume_from") or not self.resume_from:
            return

        opt_path = resolve_optimizer_path(self.resume_from)

        if opt_path is None:
            print("[RESUME] No optimizer state found")
            return

        print(f"[RESUME] Restoring optimizer from {opt_path}")

        opt_weights = joblib.load(opt_path)
        opt_vars = self.optimizer.variables()

        assert len(opt_vars) == len(opt_weights), \
            f"Optimizer variable mismatch: {len(opt_vars)} vs {len(opt_weights)}"
        try:
            for var, val in zip(opt_vars, opt_weights):
                var.assign(val)
            print("[RESUME] Optimizer restored successfully")
        except Exception as e:
            print("[RESUME WARNING] Failed to restore optimizer:", e)

    def _make_batches(self, n_batches: int):
        """A generator yielding batches of data for training.

        Args:
            n_batches: Number of batches to yield.

        Yields:
            A batch of data as a list, where each element is a batch of data for
            a single problem of the form (obs_tensor, qvs_tensor). The batches
            are order in the same order as the problems in self.problems.
        """
        if not self.problems:
            raise RuntimeError("No compatibility replay buckets were collected")

        self.batch_size_per_problem = max(
            self.batch_size // len(self.problems),
            1,
        )
        batch_iters = []

        if self.save_training_set:
            to_save = {}
        cached_shapes = self.explorer.get_cached_shapes_per_problem()
        for problem in self.problems:
            if self.use_saved_training_set:
                assert not self.save_training_set, \
                    "saving training set & using a saved set are mutually " \
                    "exclusive options (doesn't make sense to write same " \
                    "dataset back out to disk!)"
                dataset = self.loaded_training_set[problem.name]
                if self.value_head_enabled:
                    prob_obs_tensor, prob_policy_target_tensor, prob_value_target_tensor, prob_counts = dataset
                    it = weighted_batch_iter(
                        (prob_obs_tensor, prob_policy_target_tensor, prob_value_target_tensor),
                        prob_counts,
                        self.batch_size_per_problem,
                        n_batches,
                    )
                else:
                    prob_obs_tensor, prob_policy_target_tensor, prob_counts = dataset
                    it = weighted_batch_iter(
                        (prob_obs_tensor, prob_policy_target_tensor),
                        prob_counts,
                        self.batch_size_per_problem,
                        n_batches,
                    )
                batch_iters.append(it)
                continue
            if len(problem.replay) == 0:
                LOGGER.warning("No data for problem '%s' yet (teacher time-out?)",
                               problem.name)
                batch_iters.append(repeat(None))
                if self.save_training_set:
                    to_save[problem.name] = None
            else:
                dataset = problem.weighted_dataset()
                if self.value_head_enabled:
                    prob_obs_tensor, prob_policy_target_tensor, prob_value_target_tensor, prob_counts = dataset
                    it = weighted_batch_iter(
                        (prob_obs_tensor, prob_policy_target_tensor, prob_value_target_tensor),
                        prob_counts,
                        self.batch_size_per_problem,
                        n_batches,
                    )
                else:
                    prob_obs_tensor, prob_policy_target_tensor, prob_counts = dataset
                    it = weighted_batch_iter(
                        (prob_obs_tensor, prob_policy_target_tensor),
                        prob_counts,
                        self.batch_size_per_problem,
                        n_batches,
                    )
                batch_iters.append(it)
                if self.save_training_set:
                    if self.value_head_enabled:
                        to_save[problem.name] = (
                            prob_obs_tensor,
                            prob_policy_target_tensor,
                            prob_value_target_tensor,
                            prob_counts,
                        )
                    else:
                        to_save[problem.name] = (
                            prob_obs_tensor,
                            prob_policy_target_tensor,
                            prob_counts,
                        )
        if self.save_training_set:
            LOGGER.info("Saving training set to disk'%s'",
                        self.save_training_set)
            dirname = os.path.dirname(self.save_training_set)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            joblib.dump(to_save, self.save_training_set)
        combined = zip(*batch_iters)
        # yield a complete feed dict
        for combined_batch in combined:
            assert len(combined_batch) == len(self.problems)
            yield_val = []
            have_batch = False
            for problem, batch in zip(self.problems, combined_batch):
                if batch is None:
                    yield_val.append(empty_feed_value(
                        *cached_shapes[problem.name]))
                else:
                    yield_val.append(batch)
                    have_batch = True
            assert have_batch, \
                "don't have any batches at all for training problems"
            yield yield_val

    def _optimize(self, n_batches):
        params = self._weight_manager.all_weights

        param_set = set(map(lambda v: v.ref(), params))
        tf_param_set = set(map(
            lambda v: v.ref(),
            self.problems[0].network.trainable_weights))

        assert param_set == tf_param_set, \
            "network has weird variables---debug this"

        all_batches_iter = self._make_batches(n_batches)
        tr = tqdm.tqdm(all_batches_iter, desc='batch', total=n_batches)

        start_time = time()
        losses = []
        for feed_dict in tr:
            # Each feed_dict is a list of batched data sets for each problem.
            # Each data set is a tuple of obs_tensor and q-value tensor.
            #
            # The obs_tensor has shape [batch_size, obs_dim]
            # The q-value tensor has shape [batch_size, num_actions]
            #
            # Second axis of he q-values are ordered in the same order as action
            # in bound_acts_ordered for the ProblemMeta.

            with tf.name_scope('grads_opt'):
                with tf.GradientTape() as tape:
                    dataset = list(zip(*feed_dict))
                    policy_pred_by_prob = []
                    value_pred_by_prob = []
                    if self.value_head_enabled:
                        obs_by_prob, policy_target_by_prob, value_target_by_prob = dataset
                        for i, problem in enumerate(self.problems):
                            policy_pred, value_pred = problem.network(obs_by_prob[i])
                            policy_pred_by_prob.append(policy_pred)
                            value_pred_by_prob.append(value_pred)
                        loss, loss_parts = self.loss_fn(policy_pred_by_prob, policy_target_by_prob,
                                                        value_pred_by_prob, value_target_by_prob)
                    else:
                        obs_by_prob, policy_target_by_prob = dataset
                        for i, problem in enumerate(self.problems):
                            policy_pred_by_prob.append(problem.network(obs_by_prob[i]))
                        loss, loss_parts = self.loss_fn(policy_pred_by_prob, policy_target_by_prob)
                    grads = tape.gradient(loss, params)
                self.optimizer.apply_gradients(
                    grads_and_vars=zip(grads, params))
            postfix_dict = {
                "loss": float(loss),
                **{name: float(val) for name, val in loss_parts},
            }
            tr.set_postfix(postfix_dict, refresh=False)
            losses.append(float(loss))

            if (self.batches_seen % 10) == 0:
                tf.summary.scalar('train-loss', loss)

            self.batches_seen += 1

        self.explorer.update_learning_time(time() - start_time)
        return np.mean(losses)

    def can_progress(self, success_rates): #progression is not needed in  the original trainer - it works on the original training set
        return False


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

    def __call__(self, act_dist_pred: List[tf.Tensor], act_dist: List[tf.Tensor], pred_values=None, target_values=None) \
            -> tuple[float, list]:
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
                    loss_parts = [
                        (name, val * tf.cast(this_batch_size, tf.float32))
                        for name, val in this_loss_parts
                    ]
                    # loss_parts = this_loss_parts
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

        # for part_loss_name, part_loss in loss_parts:
        # tf.summary.scalar('loss-%s' % part_loss_name, part_loss)
        # tf_and_log('loss-%s' % part_loss_name, part_loss)
        total_batch = sum(batch_sizes)

        loss_parts = [
            (name, val / total_batch)
            for name, val in loss_parts
        ]
        return op_loss, loss_parts

    @can_profile
    def _set_up_losses(self, problem, act_dist_pred, act_dist, target_values=None, pred_values=None):
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
                dead_end_value = problem.ssipp_dead_end_value
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
                if target_values is not None and pred_values is not None:
                    mse = tf.reduce_mean(mean_squared_error(pred_values, target_values))
                    mse *= self.mse_coeff
                    loss_parts.append(('mse', mse))
            elif self.strategy == SupervisedObjective.MAX_ADVANTAGE:
                state_values = tf.reduce_min(input_tensor=act_dist, axis=-1)
                exp_q = act_dist_pred * act_dist
                exp_vs = tf.reduce_sum(input_tensor=exp_q, axis=-1)
                # state value is irrelevant to objective, but is included
                # because it ensures that zero loss = optimal policy
                q_loss = tf.reduce_mean(input_tensor=exp_vs - state_values)
                loss_parts.append(('qloss', q_loss))
            elif self.strategy == SupervisedObjective.MCTS_VISIT_DIST:
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
