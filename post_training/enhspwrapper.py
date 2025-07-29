import logging
import os
import subprocess
import tempfile

from asnets.multiprob import to_local
from asnets.state_reprs import CanonicalState
from asnets.interfaces.enhsp_interface import ENHSPCache, ENHSP_CONFIGS
from asnets.utils.pddl_utils import replace_init_state, hlist_to_sexprs

logger = logging.getLogger(__name__)
JARPATH = f"{os.path.dirname(__file__)}/ENHSPHeuristicServer.jar"

class ENHSPEstimator(ENHSPCache):
    DEFAULT_ENHSP_CONFIG = 'hadd-gbfs'

    def __init__(self, planner_exts, enhsp_config:str = 'hadd-gbfs'):
        super().__init__(planner_exts=planner_exts, timeout_s=-1, enhsp_config=enhsp_config)
        self.enhsp_config = enhsp_config #The above line directly converts the enhsp text into parameters with timeout
        # and we later need the parameters without the timeout
        self.computed_states = {}
        self.heuristic_client = None
        self.heuristic_client_initialised = False

    def get_state_h(self, cstate: CanonicalState):
        cstate = to_local(cstate)
        if cstate in self.computed_states:
            return self.computed_states[cstate]
        problem_hlist = replace_init_state(self._problem_hlist, cstate.to_tup_state())
        problem_source = hlist_to_sexprs(problem_hlist)

        return self.get_heuristic(problem_source)

    def initialise_heuristic_server(self, init_instance_oneline: str):
        self.heuristic_client = HeuristicClient(
            jar_path=JARPATH,
            domain_text=self._domain_source,
            init_instance_text=init_instance_oneline,
            enhsp_config=ENHSP_CONFIGS.get(self.enhsp_config, self.DEFAULT_ENHSP_CONFIG))
        logger.info(f"Starting the heuristic server with config: {self.enhsp_config}")
        self.heuristic_client_initialised = True

    # the problem should already contain the current state as the 'initial' state in order to get its heuristic
    def get_heuristic(self, problem_text):
        if not self.heuristic_client_initialised:
            self.initialise_heuristic_server(problem_text)
        heuristic_value = self.heuristic_client.get_heuristic(problem_text)
        if heuristic_value == float("inf"):
            logger.debug("No heuristic value found through ENHSP heuristic, given infinity instead.")
        return heuristic_value


class HeuristicClient:
    def __init__(self, jar_path: str, domain_text: str, init_instance_text: str, enhsp_config: str):
        # Create and name the domain temp file
        self._domain_temp = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix=".pddl", prefix="domain_heuristic_"
        )
        self._domain_temp.write(domain_text.strip())
        self._domain_temp.close()
        domain_path = self._domain_temp.name
        logger.info(f"Created temporary domain file: {domain_path}")

        # Create and name the instance temp file
        self._instance_temp = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix=".pddl", prefix="instance_heuristic_"
        )
        self._instance_temp.write(init_instance_text.strip())
        self._instance_temp.close()
        instance_path = self._instance_temp.name
        logger.info(f"Created temporary instance file: {instance_path}")
        self.enhsp_config = enhsp_config

        # Launch the ENHSP server
        self.proc = subprocess.Popen(
            ['java', '-jar', jar_path, '-o', domain_path, '-f', instance_path, *enhsp_config.split()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Wait until the ENHSP heuristic server prints "READY"
        while True:
            line = self._read_line()
            # logger.info(f"Received line: {line} from Java server.")
            if not line:
                raise RuntimeError("Java server failed to initialize (no READY).")
            if line.strip() == "READY":
                break

    def _send_line(self, line: str):
        self.proc.stdin.write(line.strip() + "\n")
        self.proc.stdin.flush()

    def _read_line(self) -> str:
        return self.proc.stdout.readline()

    def get_heuristic(self, problem_pddl_oneline: str) -> float:
        self._send_line(problem_pddl_oneline)
        h = float("inf")
        while True:
            line = self._read_line().strip()
            # logger.info(f"Received the following line from the Java server: {line}")
            if not line:
                logger.warning("No heuristic found through ENHSP heuristic.")
                break
            if line.strip().startswith("Heuristic Value:"):
                h = float(line.strip().split()[-1])
                break
        return h

    def close(self):
        try:
            self._send_line("EXIT")
        except Exception:
            pass
        self.proc.terminate()
        self.proc.wait()

        # Clean up temp files
        for f in [self._domain_temp.name, self._instance_temp.name]:
            try:
                os.unlink(f)
                logger.info(f"Deleted temp file: {f}")
            except OSError as e:
                logger.warning(f"Failed to delete temp file {f}: {e}")

