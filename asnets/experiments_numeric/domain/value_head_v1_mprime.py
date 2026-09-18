"""Frozen twenty-instance IPC-scale MPrime state-capture set for value-head V1."""

from .mprime_validation_ipc_scale_v1 import *  # noqa: F401,F403

TEST_RUNS = [
    ([f"validation_ipc_scale_v1/{tier}/pfile{index}.pddl"], None)
    for tier, start, stop in (
        ("easy", 0, 7), ("medium", 10, 17), ("hard", 20, 26)
    )
    for index in range(start, stop)
]
