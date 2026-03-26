import logging
import os
import re
import subprocess
import tempfile

import numpy as np

from asnets.interfaces.enhsp_interface import ENHSPCache, ENHSP_CONFIGS
from asnets.utils.pddl_utils import replace_init_state, hlist_to_sexprs

logger = logging.getLogger(__name__)
JARPATH = f"{os.path.dirname(__file__)}/ENHSPHeuristicServer.jar"

class ENHSPEstimator(ENHSPCache):
    DEFAULT_ENHSP_CONFIG = 'hadd-gbfs'

    def __init__(self, planner_exts, enhsp_config: str = 'hadd-gbfs'):
        super().__init__(planner_exts=planner_exts, timeout_s=-1, enhsp_config=enhsp_config)
        self.enhsp_config = enhsp_config
        self.state_key_cache = {}
        self.heuristic_client = None
        self.heuristic_client_initialised = False
        self.act_to_ind = planner_exts.act_ident_to_ind

    def get_cstate_h_and_pi(self, cstate) -> tuple[float,np.ndarray]:
        key = cstate.state_key
        if key in self.state_key_cache:
            return self.state_key_cache[key]
        problem_hlist = replace_init_state(self._problem_hlist, cstate.to_tup_state())
        problem_pddl_oneliner = hlist_to_sexprs(problem_hlist)
        res = self.get_heuristic_and_pi(problem_pddl_oneliner)
        self.state_key_cache[key] = res
        return res

    def initialise_heuristic_server(self, init_instance_oneline: str):
        self.heuristic_client = HeuristicClient(
            jar_path=JARPATH,
            domain_text=self._domain_source,
            init_instance_text=init_instance_oneline,
            enhsp_config=ENHSP_CONFIGS.get(self.enhsp_config, self.DEFAULT_ENHSP_CONFIG),
            act_to_ind=self.act_to_ind,
        )
        logger.info(f"Starting the heuristic server with config: {self.enhsp_config if ENHSP_CONFIGS.__contains__(self.enhsp_config) else self.DEFAULT_ENHSP_CONFIG}")
        self.heuristic_client_initialised = True

    # the problem should already contain the current state as the 'initial' state in order to get its heuristic
    def get_heuristic_and_pi(self, problem_pddl_oneliner: str) -> tuple[float,np.ndarray]:
        if not self.heuristic_client_initialised:
            self.initialise_heuristic_server(problem_pddl_oneliner)
        return self.heuristic_client.get_heuristic_and_pi(problem_pddl_oneliner)

    def get_heuristic_and_pi_batched(self, problem_pddl_oneliner_list: list[str]) -> list[tuple[float,np.ndarray]]:
        if not self.heuristic_client_initialised:
            self.initialise_heuristic_server(problem_pddl_oneliner_list[0]) # this is just for initialization
        return self.heuristic_client.get_heuristic_and_pi_batched(problem_pddl_oneliner_list)

class HeuristicClient:
    def __init__(self, jar_path: str, domain_text: str, init_instance_text: str, enhsp_config: str, act_to_ind: dict[str,int],):
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
        self.act_to_ind = act_to_ind
        self.act_dim = len(act_to_ind.keys())
        self._one_hot_buffer = np.empty(self.act_dim, dtype=np.float32)

        # Wait until the ENHSP heuristic server prints "READY"
        while True:
            line = self._read_line()
            if not line:
                raise RuntimeError("Java server failed to initialize (no READY).")
            if line.strip() == "READY":
                break

    def _send_line(self, line: str):
        self.proc.stdin.write(line.strip() + "\n")
        self.proc.stdin.flush()

    def _read_line(self) -> str:
        return self.proc.stdout.readline()

    def get_heuristic_and_pi(
            self,
            problem_pddl_oneliner: str
    ) -> tuple[float, np.ndarray]:
        self._send_line(problem_pddl_oneliner)
        h = float("inf")
        # reuse buffer
        one_hot = self._one_hot_buffer
        one_hot.fill(1.0 / self.act_dim)
        while True:
            line = self._read_line()
            if not line:
                logger.warning("Empty response from heuristic server.")
                break
            # fast prefix checks without strip()
            if line.startswith("Heuristic Value:"):
                h = float(line[17:])
                continue
            if line.startswith("Best Action:"):
                best_action = line[13:].rstrip("\n")
                idx = self.act_to_ind.get(best_action)
                if idx is not None:
                    one_hot.fill(0.0)
                    one_hot[idx] = 1.0
                continue
            if line.startswith("END_RESULT"):
                break
        # return copy (caller must not mutate shared buffer)
        return h, one_hot.copy()

    def get_heuristic_and_pi_batched(
            self,
            problem_pddl_oneliner_list: list[str]
    ) -> list[tuple[float, np.ndarray]]:
        n = len(problem_pddl_oneliner_list)
        if n == 0:
            return []
        self._send_line(f"BATCH {n}")
        for p in problem_pddl_oneliner_list:
            self._send_line(p)
        results = []
        one_hot = self._one_hot_buffer
        while True:
            line = self._read_line()
            if line.startswith("BATCH_DONE"):
                break
            if not line.startswith("BEGIN_RESULT"):
                continue
            h = float("inf")
            one_hot.fill(1.0 / self.act_dim)
            while True:
                line = self._read_line()
                if line.startswith("Heuristic Value:"):
                    h = float(line[17:])
                    continue
                if line.startswith("Best Action:"):
                    best_action = line[13:].rstrip("\n")
                    idx = self.act_to_ind.get(best_action)
                    if idx is not None:
                        one_hot.fill(0.0)
                        one_hot[idx] = 1.0

                    continue
                if line.startswith("END_RESULT"):
                    results.append((h, one_hot.copy()))
                    break
        assert len(results) == n, (
            f"Expected {n} heuristic results but received {len(results)}"
        )
        return results

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

