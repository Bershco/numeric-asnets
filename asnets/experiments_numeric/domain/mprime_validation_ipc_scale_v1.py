"""Frozen IPC-scale MPrime validation set exposed as policy-evaluation runs."""

from experiments_numeric.domain.mprime import *  # noqa: F401,F403


TEST_RUNS = [
    ([f"validation_ipc_scale_v1/{tier}/pfile{index}.pddl"], None)
    for tier, start in (("easy", 0), ("medium", 10), ("hard", 20))
    for index in range(start, start + 10)
]

# Use the same frozen, versioned split for validation-led training.  Keeping
# this override in a separate domain module preserves the original MPrime
# module and its historically flawed validation set for provenance.
VALIDATION_PDDLS = {
    tier: [
        f"validation_ipc_scale_v1/{tier}/pfile{index}.pddl"
        for index in range(start, start + 10)
    ]
    for tier, start in (("easy", 0), ("medium", 10), ("hard", 20))
}
