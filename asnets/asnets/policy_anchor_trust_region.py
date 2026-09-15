"""Hard trust-region utilities for policy-anchor replay training.

The controller is intentionally small and framework agnostic.  The trainer is
responsible for snapshotting/restoring TensorFlow variables and for retrying
the same replay batch; this module freezes the calibrated decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from statistics import fmean
from typing import Iterable


@dataclass(frozen=True)
class PolicyAnchorTrustRegionDecision:
    mean_kl: float
    p99_kl: float
    excessive: bool
    finite: bool


class PolicyAnchorTrustRegion:
    """Stable-calibrated per-step KL guard.

    A proposed update is excessive when either the mean or p99 deterministic
    current-policy KL exceeds its frozen limit. ``max_retries`` counts retries
    after the initial attempt.
    """

    def __init__(
            self,
            mean_kl_limit: float,
            p99_kl_limit: float,
            max_retries: int = 2,
            learning_rate_factor: float = 0.5,
    ) -> None:
        self.mean_kl_limit = float(mean_kl_limit)
        self.p99_kl_limit = float(p99_kl_limit)
        self.max_retries = int(max_retries)
        self.learning_rate_factor = float(learning_rate_factor)
        for name, value in (
                ("mean_kl_limit", self.mean_kl_limit),
                ("p99_kl_limit", self.p99_kl_limit)):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if (
                not isfinite(self.learning_rate_factor)
                or not 0.0 < self.learning_rate_factor < 1.0
        ):
            raise ValueError("learning_rate_factor must be finite and in (0, 1)")

    def retry_learning_rate(
            self, base_learning_rate: float, retry_number: int) -> float:
        """Return the LR for a one-based backtracking retry number."""
        base_learning_rate = float(base_learning_rate)
        retry_number = int(retry_number)
        if not isfinite(base_learning_rate) or base_learning_rate <= 0:
            raise ValueError("base_learning_rate must be finite and positive")
        if retry_number < 1 or retry_number > self.max_retries:
            raise ValueError("retry_number outside configured retry range")
        return base_learning_rate * self.learning_rate_factor ** retry_number

    def decide(self, per_example_kl: Iterable[float]) \
            -> PolicyAnchorTrustRegionDecision:
        values = tuple(float(value) for value in per_example_kl)
        if not values:
            raise ValueError("per_example_kl must be a non-empty vector")
        if not all(isfinite(value) for value in values):
            return PolicyAnchorTrustRegionDecision(
                mean_kl=float("inf"), p99_kl=float("inf"),
                excessive=True, finite=False)
        # Numerical roundoff may make an analytical KL microscopically
        # negative.  Preserve the mathematically valid non-negative range.
        values = tuple(sorted(max(value, 0.0) for value in values))
        mean_kl = fmean(values)
        # Match NumPy's default linear quantile interpolation without making
        # this small decision-rule module depend on NumPy.
        position = (len(values) - 1) * 0.99
        lower = floor(position)
        fraction = position - lower
        upper = min(lower + 1, len(values) - 1)
        p99_kl = values[lower] + fraction * (values[upper] - values[lower])
        return PolicyAnchorTrustRegionDecision(
            mean_kl=mean_kl,
            p99_kl=p99_kl,
            excessive=(
                mean_kl > self.mean_kl_limit
                or p99_kl > self.p99_kl_limit
            ),
            finite=True,
        )

    def to_dict(self) -> dict:
        return {
            "mean_kl_limit": self.mean_kl_limit,
            "p99_kl_limit": self.p99_kl_limit,
            "max_retries": self.max_retries,
            "learning_rate_factor": self.learning_rate_factor,
        }
