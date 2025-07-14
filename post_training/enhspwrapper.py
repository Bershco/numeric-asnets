import logging
import os
import subprocess
import time
import tempfile

from asnets.multiprob import to_local
from asnets.state_reprs import CanonicalState
from asnets.interfaces.enhsp_interface import ENHSP_CONFIGS, ENHSPCache, BLACKLIST_OUTCOMES
from asnets.utils.pddl_utils import replace_init_state, hlist_to_sexprs

logger = logging.getLogger(__name__)
JARPATH = f"{os.path.dirname(__file__)}/jpddlplus.jar"
temp_domain_file = None

class ENHSPEstimator(ENHSPCache):
    DEFAULT_ENHSP_CONFIG = 'hadd-gbfs'

    def __init__(self, planner_exts, enhsp_config:str = 'hadd-gbfs'):
        super().__init__(planner_exts=planner_exts, timeout_s=-1, enhsp_config=enhsp_config)
        self.computed_states = {}

    def get_state_h(self, cstate: CanonicalState):
        cstate = to_local(cstate)
        if cstate in self.computed_states:
            return self.computed_states[cstate]
        problem_hlist = replace_init_state(self._problem_hlist, cstate.to_tup_state())
        problem_source = hlist_to_sexprs(problem_hlist)

        return self.get_heuristic(self._domain_source ,problem_source)

    # the problem should already contain the current state as the 'initial' state in order to get its heuristic
    def get_heuristic(self, domain_text, problem_text):
        with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".pddl", dir=os.getcwd()
        ) as domain_file, tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".pddl", dir=os.getcwd()
        ) as problem_file:
            domain_file.write(domain_text)
            problem_file.write(problem_text)
            domain_file.flush()
            problem_file.flush()
            cmd = [
                "java",
                "-jar",
                str(JARPATH),
                "-o",
                domain_file.name,
                "-f",
                problem_file.name,
                #TODO: add support of more flags
            ]
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start
            if result.returncode != 0:
                logger.critical("Heuristic estimation through Java failed:\n" + result.stderr)
            logger.info(f"Getting heuristic estimation of state took {duration} ms")
            heuristic_value = float("inf")
            for i, line in enumerate(result.stdout.split("\n")):
                if line.startswith("Heuristic Value:"):
                    heuristic_value = float(line[16:].strip())
                    logger.info(f"Heuristic value received is {heuristic_value}")
                    break
            if heuristic_value == float("inf"):
                logger.debug("No heuristic value found through ENHSP heuristic, given infinity instead.")

            os.remove(problem_file.name)
            return heuristic_value

