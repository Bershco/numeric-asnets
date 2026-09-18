import numpy as np


# ============================================================
# Base Policy Interface
# ============================================================

class ActionSelectionPolicy:

    def __init__(self, worker_tag="WORKER", selection_trace_enabled=False,
                 **kwargs):
        self.worker_tag = worker_tag
        self.selection_trace_enabled = bool(selection_trace_enabled)
        self._selection_trace = None
        if "epoch" not in kwargs or kwargs["epoch"] is None or kwargs["epoch"] == 0:
            print(f"{self.worker_tag} ACTION POLICY INITIALIZED")
            print(f"{self.worker_tag} policy_class = {self.__class__.__name__}")

            for attr in [
                "distance_threshold",
                "epsilon",
                "temperature",
                "decay_rate",
                "duplicate_penalty",
                "root_visit_tie_break",
            ]:
                if hasattr(self, attr):
                    print(f"{self.worker_tag} {attr} = {getattr(self, attr)}")

    def select_action(
            self, mcts, pi: np.ndarray, *, remaining_horizon=None) -> int:
        raise NotImplementedError

    def begin_selection_trace(self):
        """Begin one opt-in trace without affecting selector semantics."""
        self._selection_trace = [] if self.selection_trace_enabled else None

    def _trace_stage(self, stage, **details):
        if self._selection_trace is None:
            return None
        entry = {"stage": stage, **details}
        self._selection_trace.append(entry)
        return entry

    def consume_selection_trace(self):
        trace = self._selection_trace
        self._selection_trace = None
        return [] if trace is None else trace


# ============================================================
# Base Policies
# ============================================================

class ArgmaxPolicy(ActionSelectionPolicy):

    _Q_TIE_ATOL = 1e-8

    def __init__(self, root_visit_tie_break="action_id", **kwargs):
        if root_visit_tie_break not in {"action_id", "q", "policy"}:
            raise ValueError(
                "root_visit_tie_break must be action_id, q, or policy")
        self.root_visit_tie_break = root_visit_tie_break
        super().__init__(**kwargs)

    def select_action(self, mcts, pi, *, remaining_horizon=None):
        current = int(np.argmax(pi))
        if (self.root_visit_tie_break == "action_id"
                and self._selection_trace is None):
            return current
        max_visit_share = float(pi[current])
        tied_actions = np.flatnonzero(pi == max_visit_share)
        trace = self._trace_stage(
            "root_visit_argmax",
            input_argmax=current,
            selector_input_distribution=np.asarray(
                pi, dtype=np.float64).tolist(),
            max_visit_share=max_visit_share,
            tied_actions=[int(action) for action in tied_actions],
            tie_break=self.root_visit_tie_break,
        )
        if self.root_visit_tie_break == "action_id":
            if trace is not None:
                trace["selected_action"] = current
            return current

        if len(tied_actions) <= 1 or max_visit_share <= 0:
            if trace is not None:
                trace["selected_action"] = current
                trace["tie_break_applied"] = False
            return current

        root = mcts.curr_tree_root
        if self.root_visit_tie_break == "policy":
            priors = np.asarray(root.act_dist)[tied_actions]
            selected = int(tied_actions[int(np.argmax(priors))])
            if trace is not None:
                trace["tie_break_applied"] = True
                trace["tie_break_scores"] = [float(value) for value in priors]
                trace["selected_action"] = selected
            return selected

        q_by_action = {
            int(action): float(child.Q_value)
            for action, child in root.children.items()
            if child is not None
        }
        if any(int(action) not in q_by_action for action in tied_actions):
            if trace is not None:
                trace["selected_action"] = current
                trace["tie_break_applied"] = False
                trace["fallback"] = "missing_child_q"
            return current
        sign = getattr(mcts, "sign", None)
        if sign is None:
            sign = -1 if getattr(mcts, "minimization", False) else 1
        signed_q = np.asarray([
            float(sign) * q_by_action[int(action)]
            for action in tied_actions
        ])
        best = float(np.max(signed_q))
        effectively_best = tied_actions[np.isclose(
            signed_q, best, rtol=0.0, atol=self._Q_TIE_ATOL)]
        selected = int(effectively_best[0])
        if trace is not None:
            trace["tie_break_applied"] = True
            trace["tie_break_scores"] = [float(value) for value in signed_q]
            trace["effectively_best"] = [
                int(action) for action in effectively_best]
            trace["selected_action"] = selected
        return selected


class SamplePolicy(ActionSelectionPolicy):

    def select_action(self, mcts, pi, *, remaining_horizon=None):
        selected = int(np.random.choice(len(pi), p=pi))
        self._trace_stage(
            "sample", selected_action=selected,
            selector_input_distribution=np.asarray(
                pi, dtype=np.float64).tolist())
        return selected


class VisitProportionalPolicy(ActionSelectionPolicy):

    def select_action(self, mcts, pi, *, remaining_horizon=None):

        root = mcts.curr_tree_root
        act_dim = len(pi)

        visits = np.zeros(act_dim, dtype=np.float32)

        for action, child in root.children.items():
            if child is not None:
                visits[action] = child.visit_count

        s = visits.sum()

        if s > 0:
            visits /= s
            selected = int(np.random.choice(act_dim, p=visits))
            self._trace_stage(
                "visit_proportional", selected_action=selected,
                selector_input_distribution=np.asarray(
                    visits, dtype=np.float64).tolist())
            return selected

        selected = int(np.argmax(pi))
        self._trace_stage(
            "visit_proportional_fallback", selected_action=selected,
            selector_input_distribution=np.asarray(
                pi, dtype=np.float64).tolist())
        return selected


# ============================================================
# Mixins
# ============================================================

class GoalChaseMixin:

    def __init__(self, distance_threshold=np.inf, **kwargs):
        self.distance_threshold = distance_threshold
        super().__init__(**kwargs)

    def select_action(self, mcts, pi, *, remaining_horizon=None):

        root = mcts.curr_tree_root

        goal_is_within_remaining_horizon = (
            remaining_horizon is None
            or root.known_distance_to_goal <= remaining_horizon
        )
        if (
                root.known_distance_to_goal < np.inf
                and root.known_distance_to_goal <= self.distance_threshold
                and goal_is_within_remaining_horizon
                and root.best_goal_child is not None
        ):
            for action, child in root.children.items():
                if child is root.best_goal_child:
                    self._trace_stage(
                        "goal_chase",
                        applied=True,
                        selected_action=int(action),
                        selector_input_distribution=np.asarray(
                            pi, dtype=np.float64).tolist(),
                        known_distance_to_goal=int(
                            root.known_distance_to_goal),
                        remaining_horizon=remaining_horizon,
                    )
                    return action

        self._trace_stage(
            "goal_chase",
            applied=False,
            known_distance_to_goal=(
                None if root.known_distance_to_goal == np.inf
                else int(root.known_distance_to_goal)),
            remaining_horizon=remaining_horizon,
        )
        return super().select_action(
            mcts, pi, remaining_horizon=remaining_horizon)


class TemperatureMixin:

    def __init__(self, temperature=1.0, **kwargs):
        self.temperature = temperature
        super().__init__(**kwargs)

    def select_action(self, mcts, pi, *, remaining_horizon=None):

        if self.temperature != 1.0:
            pi = np.power(pi, 1.0 / self.temperature)

            s = pi.sum()
            if s > 0:
                pi /= s

        self._trace_stage(
            "temperature", applied=self.temperature != 1.0,
            temperature=float(self.temperature))

        return super().select_action(
            mcts, pi, remaining_horizon=remaining_horizon)


class EpsilonGreedyMixin:

    def __init__(self, epsilon=0.1, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def select_action(self, mcts, pi, *, remaining_horizon=None):
        random_draw = float(np.random.rand())
        if random_draw < self.epsilon:
            pi_sum = np.sum(pi)
            if pi_sum > 0:
                pi_norm = pi / pi_sum
            else:
                pi_norm = np.ones_like(pi) / len(pi)
            selected = int(np.random.choice(len(pi), p=pi_norm))
            self._trace_stage(
                "epsilon_greedy", applied=True,
                epsilon=float(self.epsilon), random_draw=random_draw,
                selected_action=selected,
                selector_input_distribution=np.asarray(
                    pi_norm, dtype=np.float64).tolist())
            return selected

        self._trace_stage(
            "epsilon_greedy", applied=False,
            epsilon=float(self.epsilon), random_draw=random_draw)
        return super().select_action(
            mcts, pi, remaining_horizon=remaining_horizon)


class ExplorationDecayMixin:

    def __init__(self, decay_rate=0.999, **kwargs):
        self.decay_rate = decay_rate
        super().__init__(**kwargs)

    def step_decay(self):

        if hasattr(self, "epsilon"):
            self.epsilon *= self.decay_rate

        if hasattr(self, "temperature"):
            self.temperature *= self.decay_rate


class PathDuplicatePenaltyMixin:

    def __init__(self, duplicate_penalty=0.0, **kwargs):
        self.duplicate_penalty = duplicate_penalty
        # the name penalty is reversed, the value 0.0 means the highest penalty - a ban.
        super().__init__(**kwargs)

    def select_action(self, mcts, pi, *, remaining_horizon=None):
        root = mcts.curr_tree_root
        if root.children is None or root.children.is_empty():
            self._trace_stage(
                "duplicate_penalty", applied=False,
                reason="no_root_children")
            return super().select_action(
                mcts, pi, remaining_horizon=remaining_horizon)
        traj_mask = root.get_child_on_trajectory_mask()
        if traj_mask.size == 0:
            self._trace_stage(
                "duplicate_penalty", applied=False,
                reason="empty_trajectory_mask")
            return super().select_action(
                mcts, pi, remaining_horizon=remaining_horizon)
        orig_pi = pi
        pi = pi.copy()
        child_actions = root.children.actions_np
        penalties = np.where(
            traj_mask > 0,
            self.duplicate_penalty,
            1.0,
        )
        pi[child_actions] *= penalties
        s = pi.sum()
        if s > 0:
            pi /= s
        else:
            pi = orig_pi
        self._trace_stage(
            "duplicate_penalty",
            applied=bool(np.any(traj_mask > 0)),
            duplicate_penalty=float(self.duplicate_penalty),
            duplicate_actions=[
                int(action) for action, duplicate in zip(
                    child_actions, traj_mask > 0) if duplicate],
            fallback_to_input=bool(s <= 0),
        )
        return super().select_action(
            mcts, pi, remaining_horizon=remaining_horizon)


class TerminalSafeMixin:
    """Keep external MCTS execution away from known non-goal terminals.

    This is deliberately an evaluation/action-selection guard. It does not
    change MCTS backup values or training targets. Duplicate avoidance is
    applied inside this mixin so a safe duplicate is restored before a known
    terminal action can become eligible.
    """

    _Q_TOLERANCE = 1e-6

    def __init__(self, duplicate_penalty=0.0, **kwargs):
        self.duplicate_penalty = (
            0.0 if duplicate_penalty is None else duplicate_penalty)
        self.terminal_actions_excluded = 0
        self.safe_duplicate_fallbacks = 0
        self.no_safe_child_events = 0
        super().__init__(**kwargs)

    @staticmethod
    def _normalise_or_fallback(vector, fallback, eligible):
        result = np.asarray(vector, dtype=np.float64).copy()
        total = float(result.sum())
        if total > 0.0:
            result /= total
            return result
        result[:] = 0.0
        fallback = np.asarray(fallback, dtype=np.float64)
        result[eligible] = fallback[eligible]
        total = float(result.sum())
        if total > 0.0:
            result /= total
        elif np.any(eligible):
            result[eligible] = 1.0 / int(np.count_nonzero(eligible))
        return result

    def _validate_root_q_values(self, mcts, root):
        if root.children is None or root.children.is_empty():
            return
        q_values = np.asarray(
            [float(child.Q_value) for child in root.children.values()],
            dtype=np.float64,
        )
        finite = bool(np.all(np.isfinite(q_values)))
        if getattr(mcts, "minimization", False):
            valid = finite
        else:
            tol = self._Q_TOLERANCE
            valid = finite and bool(np.all(q_values >= -tol)) and bool(
                np.all(q_values <= 1.0 + tol))
        if not valid:
            print(
                f"[MCTS SAFE INVARIANT] invalid_root_q_values="
                f"{q_values.tolist()} minimization="
                f"{getattr(mcts, 'minimization', False)}"
            )
            raise ValueError(
                "MCTS-SAFE root Q-values violate the declared value convention")

    def select_action(self, mcts, pi, *, remaining_horizon=None):
        if hasattr(mcts, "ensure_safe_root_child"):
            mcts.ensure_safe_root_child()

        root = mcts.curr_tree_root
        raw_pi = np.asarray(pi, dtype=np.float64).copy()
        self._validate_root_q_values(mcts, root)

        act_dim = len(raw_pi)
        safe_mask = np.zeros(act_dim, dtype=bool)
        terminal_mask = np.zeros(act_dim, dtype=bool)
        duplicate_mask = np.zeros(act_dim, dtype=bool)
        priors = np.zeros(act_dim, dtype=np.float64)
        visits = np.zeros(act_dim, dtype=np.int64)

        if root.children is not None and not root.children.is_empty():
            trajectory = root.get_child_on_trajectory_mask()
            for index, (action, child) in enumerate(root.children.items()):
                action = int(action)
                is_goal = bool(child is not None and child.goal_state)
                is_terminal = bool(child is not None and child.terminal_state)
                is_duplicate = bool(index < len(trajectory) and trajectory[index] > 0)
                is_safe = bool(child is not None and (is_goal or not is_terminal))
                safe_mask[action] = is_safe
                terminal_mask[action] = bool(is_terminal and not is_goal)
                duplicate_mask[action] = is_duplicate
                priors[action] = float(root.children.priors[index])
                visits[action] = int(root.children.visits[index])

        post_terminal = raw_pi.copy()
        excluded_count = 0
        if np.any(safe_mask):
            excluded = terminal_mask & (raw_pi > 0.0)
            excluded_count = int(np.count_nonzero(excluded))
            self.terminal_actions_excluded += excluded_count
            post_terminal[~safe_mask] = 0.0
            post_terminal = self._normalise_or_fallback(
                post_terminal, priors, safe_mask)
        else:
            self.no_safe_child_events += 1

        post_duplicate = post_terminal.copy()
        duplicate_fallback = False
        eligible_safe = safe_mask if np.any(safe_mask) else (raw_pi > 0.0)
        if np.any(duplicate_mask & eligible_safe):
            post_duplicate[duplicate_mask] *= self.duplicate_penalty
            if float(post_duplicate.sum()) <= 0.0 and np.any(eligible_safe):
                post_duplicate = post_terminal.copy()
                self.safe_duplicate_fallbacks += 1
                duplicate_fallback = True
            else:
                post_duplicate = self._normalise_or_fallback(
                    post_duplicate, post_terminal, eligible_safe)

        trace = self._trace_stage(
            "terminal_safe",
            applied=bool(excluded_count or np.any(duplicate_mask)),
            terminal_excluded_actions=[
                int(action) for action in np.flatnonzero(
                    terminal_mask & (raw_pi > 0.0))],
            duplicate_actions=[
                int(action) for action in np.flatnonzero(duplicate_mask)],
            duplicate_fallback=bool(duplicate_fallback),
            no_safe_child=bool(not np.any(safe_mask)),
            post_terminal_policy=post_terminal.tolist(),
            post_duplicate_policy=post_duplicate.tolist(),
        )
        action = super().select_action(
            mcts, post_duplicate, remaining_horizon=remaining_horizon)
        if trace is not None:
            trace["selected_action"] = int(action)
        # Bulk campaigns log only safety events. Full root vectors are emitted
        # above solely when an invariant fails; printing them on every action
        # can turn a normal evaluation into a multi-gigabyte debug log.
        if excluded_count or duplicate_fallback or not np.any(safe_mask):
            print(
                "[MCTS SAFE EVENT] "
                f"selected={int(action)} selected_N={int(visits[int(action)])} "
                f"terminal_excluded={excluded_count} "
                f"duplicate_fallback={int(duplicate_fallback)} "
                f"no_safe_child={int(not np.any(safe_mask))} "
                f"excluded_total={self.terminal_actions_excluded} "
                f"duplicate_fallbacks_total={self.safe_duplicate_fallbacks} "
                f"no_safe_total={self.no_safe_child_events}"
            )
        return int(action)


# ============================================================
# Policy Builder
# ============================================================

BASE_POLICIES = {
    "argmax": ArgmaxPolicy,
    "sample": SamplePolicy,
    "visit": VisitProportionalPolicy,
}


def build_action_policy(
        base_policy: str,
        worker_tag="WORKER",
        distance_threshold=None,
        epsilon=None,
        temperature=None,
        decay_rate=None,
        epoch=None,
        duplicate_penalty=None,
        terminal_safe=False,
        root_visit_tie_break="action_id",
        selection_trace_enabled=False,
):
    base = BASE_POLICIES[base_policy]

    mixins = []

    if distance_threshold is not None:
        mixins.append(GoalChaseMixin)

    if decay_rate is not None and decay_rate != 0.0:
        mixins.append(ExplorationDecayMixin)

    if epsilon is not None and epsilon != 0.0:
        mixins.append(EpsilonGreedyMixin)

    if temperature is not None and temperature != 0.0:
        mixins.append(TemperatureMixin)

    if terminal_safe:
        mixins.append(TerminalSafeMixin)
    elif duplicate_penalty is not None:
        mixins.append(PathDuplicatePenaltyMixin)

    bases = tuple(mixins + [base])

    # Create readable class name
    name_parts = [base.__name__.replace("Policy", "")]

    for m in mixins:
        name_parts.append(m.__name__.replace("Mixin", ""))

    class_name = "".join(name_parts) + "Policy"

    cls = type(class_name, bases, {})

    return cls(
        worker_tag=worker_tag,
        distance_threshold=distance_threshold,
        epsilon=epsilon,
        temperature=temperature,
        decay_rate=decay_rate,
        epoch=epoch,
        duplicate_penalty=duplicate_penalty,
        root_visit_tie_break=root_visit_tie_break,
        selection_trace_enabled=selection_trace_enabled,
    )
