import numpy as np


# ============================================================
# Base Policy Interface
# ============================================================

class ActionSelectionPolicy:

    def __init__(self, worker_tag="WORKER", **kwargs):
        self.worker_tag = worker_tag
        if "epoch" not in kwargs or kwargs["epoch"] is None or kwargs["epoch"] == 0:
            print(f"{self.worker_tag} ACTION POLICY INITIALIZED")
            print(f"{self.worker_tag} policy_class = {self.__class__.__name__}")

            for attr in [
                "distance_threshold",
                "epsilon",
                "temperature",
                "decay_rate",
                "duplicate_penalty",
            ]:
                if hasattr(self, attr):
                    print(f"{self.worker_tag} {attr} = {getattr(self, attr)}")

    def select_action(
            self, mcts, pi: np.ndarray, *, remaining_horizon=None) -> int:
        raise NotImplementedError


# ============================================================
# Base Policies
# ============================================================

class ArgmaxPolicy(ActionSelectionPolicy):

    def select_action(self, mcts, pi, *, remaining_horizon=None):
        return int(np.argmax(pi))


class SamplePolicy(ActionSelectionPolicy):

    def select_action(self, mcts, pi, *, remaining_horizon=None):
        return int(np.random.choice(len(pi), p=pi))


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
            return int(np.random.choice(act_dim, p=visits))

        return int(np.argmax(pi))


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
                    return action

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

        return super().select_action(
            mcts, pi, remaining_horizon=remaining_horizon)


class EpsilonGreedyMixin:

    def __init__(self, epsilon=0.1, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def select_action(self, mcts, pi, *, remaining_horizon=None):
        if np.random.rand() < self.epsilon:
            pi_sum = np.sum(pi)
            if pi_sum > 0:
                pi_norm = pi / pi_sum
            else:
                pi_norm = np.ones_like(pi) / len(pi)
            return int(np.random.choice(len(pi), p=pi_norm))

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
            return super().select_action(
                mcts, pi, remaining_horizon=remaining_horizon)
        traj_mask = root.get_child_on_trajectory_mask()
        if traj_mask.size == 0:
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

        action = super().select_action(
            mcts, post_duplicate, remaining_horizon=remaining_horizon)
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
    )
