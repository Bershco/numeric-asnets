# asnets/spawn_context.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from asnets.state_reprs import sample_next_state, get_init_cstate


# import whatever you already use
# CanonicalState, PlannerExtensions, sample_next_state, get_init_cstate, etc.

@dataclass
class LocalExploreContext:
    """
    Replacement for ProblemService inside a spawned worker.
    This holds the env, estimator, and the state cache that MCTS expects.
    """
    planner_exts: "PlannerExtensions"
    estimator: object  # ENHSPEstimator or whatever
    prob_meta: object  # ProblemMeta (already exists in PlannerExtensions)
    act_dim: int

    # runtime caches
    id_hash_to_state: dict[tuple[int, int], "CanonicalState"] = None
    curr_state_id: int = 0

    def __post_init__(self):
        if self.id_hash_to_state is None:
            self.id_hash_to_state = {}

    # ----- state id bookkeeping -----
    def get_state_identifiers(self, cstate: "CanonicalState") -> tuple[int, int]:
        state_hash = hash(cstate)
        self.curr_state_id += 1
        sid = self.curr_state_id
        self.id_hash_to_state[(sid, state_hash)] = cstate
        return sid, state_hash

    def get_state_from_identifiers(self, sid: int, shash: int) -> "CanonicalState":
        return self.id_hash_to_state[(sid, shash)]

    # ----- API that TrainingMCTS currently calls on problem_service -----
    def get_act_dim(self) -> int:
        return self.act_dim

    def get_state_h(self, sid: int, shash: int) -> float:
        cstate = self.get_state_from_identifiers(sid, shash)
        return self.estimator.get_cstate_h(cstate)

    def to_network_input(self, sid: int, shash: int):
        cstate = self.get_state_from_identifiers(sid, shash)
        return cstate.to_network_input()

    def get_applicable_action_mask(self, sid: int, shash: int):
        cstate = self.get_state_from_identifiers(sid, shash)
        # your canonical state uses acts_enabled list: [(act, enabled), ...]
        return np.array([enabled for _, enabled in cstate.acts_enabled], dtype=bool)

    def env_simulate_batch_steps(self, sid: int, shash: int, action_nums: list[int]):
        """
        Local replacement for exposed_env_simulate_batch_steps.
        Returns the same tuple structure your MCTS expects.
        """
        cstate = self.get_state_from_identifiers(sid, shash)

        results = []
        for action_id in action_nums:
            next_state, step_cost = sample_next_state(cstate, action_id, self.planner_exts)

            next_sid, next_shash = self.get_state_identifiers(next_state)

            results.append((
                action_id,
                next_sid,
                next_shash,
                step_cost,
                next_state.is_goal,
                next_state.is_terminal,
                next_state.to_network_input(),
                np.array([enabled for _, enabled in next_state.acts_enabled], dtype=bool),
            ))
        return results

    # ----- init state -----
    def get_init_state(self) -> "CanonicalState":
        return get_init_cstate(self.planner_exts)
