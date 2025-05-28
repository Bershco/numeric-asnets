import logging
import os
import subprocess
import time
from typing import Optional, List
import tempfile

from enhsp_wrapper.enhsp import ENHSP, PlanningResult, PlanningStatus

from asnets.multiprob import to_local
from asnets.state_reprs import CanonicalState
from asnets.interfaces.enhsp_interface import ENHSP_CONFIGS, ENHSPCache, BLACKLIST_OUTCOMES
from asnets.teacher_cache import TeacherException
from asnets.utils.pddl_utils import replace_init_state, hlist_to_sexprs

LOGGER = logging.getLogger(__name__)


class ENHSPWrapper(ENHSP):

    def __init__(self, params: str, horizon: int = 100):
        super().__init__(params)

    def plan(self, domain_path: str, problem_path: str) -> PlanningResult:
        """Plan with ENHSP.

        Args:
            domain_path (str): path to domain file
            problem_path (str): path to problem file

        Returns:
            PlanningResult: the planning result
        """
        cmd = [
            "java",
            "-jar",
            str(self.path),
            "-o",
            domain_path,
            "-f",
            problem_path,
            *self.params.split(),
        ]
        # We respect ENHSP's timeout, so we don't need to set a timeout here.
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start

        status = PlanningStatus.UNKNOWN
        plan = None

        for i, line in enumerate(result.stdout.split("\n")):
            if line in ["Problem Detected as Unsolvable", "Problem unsolvable"]:
                status = PlanningStatus.UNSOLVABLE
                break

            if line == "Found Plan:":
                status = PlanningStatus.SUCCESS
                plan = []

                for line in result.stdout.split("\n")[i + 1:]:
                    if line == "":
                        break
                    plan.append(line.split(": ")[1])

        if "Timeout" in result.stderr:
            status = PlanningStatus.TIMEOUT
        elif "Memory" in result.stderr:
            status = PlanningStatus.MEMEOUT  # not tested
        elif result.returncode != 0 and status == PlanningStatus.UNKNOWN:
            status = PlanningStatus.ERROR

        return PlanningResult(status, plan, duration, result.stdout,
                              result.stderr)


    def plan_from_string(self, domain_text: str, problem_text: str) -> PlanningResult:
        """Plan with ENHSP.

        Args:
            domain_text (str): the text of the domain (PDDL)
            problem_text (str): the text of the problem (PDDL)

        Returns:
            PlanningResult: the planning result
        """
        with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".pddl", dir=os.getcwd()
        ) as domain_file, tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".pddl", dir=os.getcwd()
        ) as problem_file:
            domain_file.write(domain_text)
            problem_file.write(problem_text)
            domain_file.flush()
            problem_file.flush()
            result = self.plan(domain_file.name, problem_file.name)

        os.remove(domain_file.name)
        os.remove(problem_file.name)
        return result

class ENHSPEstimator(ENHSPCache):
    DEFAULT_HORIZON = 100
    DEFAULT_ENHSP_CONFIG = 'hadd-gbfs'

    def __init__(self, planner_exts, enhsp_config:str = 'hadd-gbfs', horizon: int=100):
        super().__init__(planner_exts=planner_exts, timeout_s=-1)
        self.set_estimator_params(enhsp_config, horizon)
        self.worst_h = horizon * 2 # this would probably benefit from being a real value
        self.computed_states = {}

    def _get_plan(self, tup_state: CanonicalState.TupState) \
            -> Optional[List[str]]:
        """Get a plan from the teacher.

        Raises:
            TeacherException: If the given state is blacklisted or the teacher
            fails due to one of the BLACKLIST_OUTCOMES

        Args:
            tup_state (CanonicalState.TupState): The state to get a plan for.

        Returns:
            Optional[List[str]]: The plan, or None if the teacher determined
            a negative outcome, meaning it likely believes there is no plan.
        """

        problem_hlist = replace_init_state(self._problem_hlist, tup_state)
        problem_source = hlist_to_sexprs(problem_hlist)

        planner = ENHSPWrapper(self.estimator_params)
        result = planner.plan_from_string(self._domain_source, problem_source)

        # Handle the possible outcomes
        if result.status == PlanningStatus.UNKNOWN:
            LOGGER.warning(
                f'ENHSP returned UNKNOWN status with problem instance {problem_source}, assuming unsolvable.')
            return None
        if result.status == PlanningStatus.UNSOLVABLE or result.status == PlanningStatus.TIMEOUT:
            return None
        if result.status in BLACKLIST_OUTCOMES:
            self._blacklist.add(tup_state)
            # these shouldn't happen too often with right training problems
            # be verbose about it
            err_msg = 'ENHSP failed: status {}'.format(str(result.status))
            # err_msg += f', problem source {problem_source}'
            raise TeacherException(err_msg)

        assert result.status == PlanningStatus.SUCCESS

        plan = [
            a.strip('()')
            for a in result.plan]
        return plan

    def get_state_h_value(self, cstate: CanonicalState):
        cstate = to_local(cstate)
        if cstate in self.computed_states:
            return self.computed_states[cstate]
        plan = self._get_plan(cstate.to_tup_state())
        if plan is None:
            # didn't find a plan, whether this is unsolvable by the planner or
            # just couldn't in the horizon length, it's the same.
            # also I couldn't care to save this, to conserve memory rather than time
            return self.worst_h
        self.computed_states[cstate] = len(plan)
        return self.computed_states[cstate]

    def set_estimator_params(self, enhsp_config = DEFAULT_ENHSP_CONFIG, horizon = DEFAULT_HORIZON):
        self.estimator_params = ENHSP_CONFIGS[enhsp_config] \
                                + f' -dl {float(horizon)} -timeout 1'
