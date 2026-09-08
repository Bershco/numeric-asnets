"""Policy-anchor coefficient controllers for Stage-2 replay training."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Optional


POLICY_ANCHOR_CONTROLLER_VERSION = 1


@dataclass
class PolicyAnchorKLController:
    """Keep a constant KL coefficient or adapt it toward a target KL.

    The adaptive rule is the multiplicative rule described for PPO's adaptive
    KL penalty: double the coefficient above ``tolerance * target`` and halve
    it below ``target / tolerance``.  ``observe`` is intentionally independent
    of TensorFlow so its behavior and checkpoint state are easy to test.
    """

    mode: str
    coefficient: float
    target: Optional[float] = None
    tolerance: float = 1.5
    factor: float = 2.0
    min_coefficient: float = 1e-6
    max_coefficient: float = 1e4
    observations: int = 0
    adjustments: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"constant", "adaptive_target"}:
            raise ValueError(
                "policy anchor KL mode must be constant or adaptive_target")
        self.coefficient = float(self.coefficient)
        if not isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("policy anchor KL coefficient must be finite and non-negative")
        if not isfinite(self.tolerance) or self.tolerance <= 1:
            raise ValueError("policy anchor KL tolerance must be finite and greater than 1")
        if not isfinite(self.factor) or self.factor <= 1:
            raise ValueError("policy anchor KL adaptation factor must be finite and greater than 1")
        if (
                not isfinite(self.min_coefficient)
                or not isfinite(self.max_coefficient)
                or self.min_coefficient < 0
                or self.max_coefficient < self.min_coefficient
        ):
            raise ValueError("invalid policy anchor KL coefficient bounds")
        if self.mode == "adaptive_target":
            if self.coefficient <= 0:
                raise ValueError("adaptive target-KL mode requires a positive initial coefficient")
            if self.target is None or not isfinite(self.target) or self.target <= 0:
                raise ValueError("adaptive target-KL mode requires a positive finite target")
        elif self.target is not None:
            self.target = float(self.target)

    def observe(self, measured_kl: float) -> dict[str, Any]:
        """Observe post-update KL and return an auditable adjustment record."""
        measured_kl = float(measured_kl)
        if not isfinite(measured_kl) or measured_kl < 0:
            raise ValueError("measured policy anchor KL must be finite and non-negative")

        before = self.coefficient
        action = "constant"
        self.observations += 1
        if self.mode == "adaptive_target":
            if measured_kl > self.target * self.tolerance:
                self.coefficient = min(
                    self.max_coefficient, self.coefficient * self.factor)
                action = "increase"
            elif measured_kl < self.target / self.tolerance:
                self.coefficient = max(
                    self.min_coefficient, self.coefficient / self.factor)
                action = "decrease"
            else:
                action = "hold"
            if self.coefficient != before:
                self.adjustments += 1

        return {
            "coefficient_before": before,
            "coefficient_after": self.coefficient,
            "measured_kl": measured_kl,
            "action": action,
            "adjusted": float(self.coefficient != before),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": POLICY_ANCHOR_CONTROLLER_VERSION,
            "mode": self.mode,
            "coefficient": self.coefficient,
            "target": self.target,
            "tolerance": self.tolerance,
            "factor": self.factor,
            "min_coefficient": self.min_coefficient,
            "max_coefficient": self.max_coefficient,
            "observations": self.observations,
            "adjustments": self.adjustments,
        }

    def restore(self, state: Optional[Mapping[str, Any]]) -> bool:
        """Restore a continuation exactly; reject incompatible controllers."""
        if state is None:
            return False
        if state.get("version") != POLICY_ANCHOR_CONTROLLER_VERSION:
            raise ValueError("unsupported policy anchor KL controller state version")
        for key in (
                "mode", "target", "tolerance", "factor",
                "min_coefficient", "max_coefficient",
        ):
            if state.get(key) != getattr(self, key):
                raise ValueError(
                    f"policy anchor KL continuation mismatch for {key}: "
                    f"saved={state.get(key)!r}, requested={getattr(self, key)!r}")
        self.coefficient = float(state["coefficient"])
        self.observations = int(state.get("observations", 0))
        self.adjustments = int(state.get("adjustments", 0))
        self.__post_init__()
        return True
