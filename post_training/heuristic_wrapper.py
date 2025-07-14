import tempfile
import os
import time
import subprocess
import logging

JARPATH = f"{os.path.dirname(__file__)}/jpddlplus.jar"
logger = logging.getLogger(__name__)
temp_domain_file = None

# the problem should already contain the current state as the 'initial' state in order to get its heuristic
def get_heuristic(domain_text, problem_text):
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
