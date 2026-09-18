"""Frozen twenty-instance Rover state-capture set for value-head V1."""

from .rover import *  # noqa: F401,F403

TEST_RUNS = [
    ([f"valid_{tier}/pfile{index}.pddl"], None)
    for tier, stop in (("easy", 7), ("medium", 7), ("hard", 6))
    for index in range(stop)
]
