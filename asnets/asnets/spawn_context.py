# asnets/spawn_context.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List

from asnets.state_reprs import sample_next_state, get_init_cstate, CanonicalState
from post_training.enhspwrapper import ENHSPEstimator


# import whatever you already use
# CanonicalState, PlannerExtensions, sample_next_state, get_init_cstate, etc.

@dataclass
class LocalExploreContext:
    """
    Replacement for ProblemService inside a spawned worker.
    This holds the env, estimator, and the state cache that MCTS expects.
    """
    planner_exts: "PlannerExtensions"
    estimator: ENHSPEstimator
    estimator_h_to_v_coeff: float = 1

    # ----- API that TrainingMCTS currently calls on problem_service -----
    def get_act_dim(self) -> int:
        return self.planner_exts.problem_meta.num_acts

    def to_network_input(self, state: CanonicalState):
        return state.to_network_input()

    def get_applicabe_action_mask(self, state: CanonicalState):
        return np.array([enabled for _, enabled in state.acts_enabled], dtype=bool)

    def env_simulate_batch_steps(self,
                                 cstate: CanonicalState,
                                 action_nums: list[int]):
        """
        Local replacement for exposed_env_simulate_batch_steps.
        Returns the same tuple structure your MCTS expects.
        """
        results = []
        mdpsim_state = cstate.to_mdpsim(self.planner_exts)
        for action_id in action_nums:
            next_state, step_cost = sample_next_state(cstate, int(action_id), self.planner_exts, mdpsim_state=mdpsim_state)

            results.append((
                action_id,
                next_state,
                step_cost,
                next_state.is_goal,
                next_state.is_terminal,
                next_state.to_network_input(),
                next_state.get_applicable_action_mask(),
            ))
        return results

    def env_simulate_step(self,
                          cstate: CanonicalState,
                          action_id: int):
        next_state, _ = sample_next_state(cstate, int(action_id), self.planner_exts, mdpsim_state=cstate.to_mdpsim(self.planner_exts))
        return next_state

    # ----- init state -----
    def get_init_state(self) -> "CanonicalState":
        return get_init_cstate(self.planner_exts)

    def get_state_v_pi_one_hot_est(self, cstate):
        state_h, state_pi = self.estimator.get_cstate_h_and_pi(cstate)
        return float(np.exp(-1 * self.estimator_h_to_v_coeff * state_h)), state_pi

    def get_state_pi_est(self, cstate_children):
        logits = np.full(self.get_act_dim(), -np.inf, dtype=np.float32)
        for act, child_state in cstate_children:
            state_h, _ = self.estimator.get_cstate_h_and_pi(child_state) # get_state_h_and_pi returns a one-hot
            logits[act] = -1 * self.estimator_h_to_v_coeff * state_h #ln(child.state_v)

        # subtract max for stability (this handles -inf too)
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted)
        state_softmax = exp_vals / np.sum(exp_vals)
        return state_softmax
