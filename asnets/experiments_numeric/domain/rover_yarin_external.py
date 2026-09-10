"""Frozen external Rover test set generated with Yarin's rovergen wrapper."""

import os


PDDL_DIR = os.environ["YARIN_ROVER_PDDL_DIR"]
COMMON_PDDLS = ["domain.pddl"]
TRAIN_PDDLS = [f"train/pfile{i}.pddl" for i in range(1, 5)]
TRAIN_NAMES = None
TEST_RUNS = [([f"external/pfile{i}.pddl"], None) for i in range(20)]

