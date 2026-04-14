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
            ]:
                if hasattr(self, attr):
                    print(f"{self.worker_tag} {attr} = {getattr(self, attr)}")

    def select_action(self, mcts, pi: np.ndarray) -> int:
        raise NotImplementedError


# ============================================================
# Base Policies
# ============================================================

class ArgmaxPolicy(ActionSelectionPolicy):

    def select_action(self, mcts, pi):
        return int(np.argmax(pi))


class SamplePolicy(ActionSelectionPolicy):

    def select_action(self, mcts, pi):
        return int(np.random.choice(len(pi), p=pi))


class VisitProportionalPolicy(ActionSelectionPolicy):

    def select_action(self, mcts, pi):

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

    def select_action(self, mcts, pi):

        root = mcts.curr_tree_root

        if (
                root.known_distance_to_goal < np.inf
                and root.known_distance_to_goal <= self.distance_threshold
                and root.best_goal_child is not None
        ):
            for action, child in root.children.items():
                if child is root.best_goal_child:
                    return action

        return super().select_action(mcts, pi)


class TemperatureMixin:

    def __init__(self, temperature=1.0, **kwargs):
        self.temperature = temperature
        super().__init__(**kwargs)

    def select_action(self, mcts, pi):

        if self.temperature != 1.0:
            pi = np.power(pi, 1.0 / self.temperature)

            s = pi.sum()
            if s > 0:
                pi /= s

        return super().select_action(mcts, pi)


class EpsilonGreedyMixin:

    def __init__(self, epsilon=0.1, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def select_action(self, mcts, pi):
        if np.random.rand() < self.epsilon:
            # random ***valid*** action using masked pi (by using p=pi all inapplicable actions have 0 probs)
            return int(np.random.choice(len(pi), p=pi))

        return super().select_action(mcts, pi)

class ExplorationDecayMixin:

    def __init__(self, decay_rate=0.999, **kwargs):
        self.decay_rate = decay_rate
        super().__init__(**kwargs)

    def step_decay(self):

        if hasattr(self, "epsilon"):
            self.epsilon *= self.decay_rate

        if hasattr(self, "temperature"):
            self.temperature *= self.decay_rate


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
):
    base = BASE_POLICIES[base_policy]

    mixins = []

    if decay_rate is not None:
        mixins.append(ExplorationDecayMixin)

    if epsilon is not None:
        mixins.append(EpsilonGreedyMixin)

    if temperature is not None:
        mixins.append(TemperatureMixin)

    if distance_threshold is not None:
        mixins.append(GoalChaseMixin)

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
    )
# ============================================================
# Example
# ============================================================

# worker_tag = "W3531548953|45341"

# policy = build_action_policy(
#     base_policy="visit",
#     worker_tag=worker_tag,
#     epsilon=0.1,
#     decay_rate=0.999,
#     distance_threshold=3,
# )

# action = policy.select_action(mcts, pi)

# if hasattr(policy, "step_decay"):
#     policy.step_decay()
