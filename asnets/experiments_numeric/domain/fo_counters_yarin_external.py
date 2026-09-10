"""Frozen external FO Counters test set generated with Yarin's generator."""

import os


PDDL_DIR = os.environ["YARIN_FO_COUNTERS_PDDL_DIR"]
COMMON_PDDLS = ["domain.pddl"]
TRAIN_PDDLS = [
    "train/instance_2.pddl",
    "train/instance_3.pddl",
    "train/instance_4.pddl",
]
TRAIN_NAMES = None
TEST_RUNS = [([f"external/pfile{i}.pddl"], None) for i in range(20)]

