"""Persisted validation-selection state for resumable training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

VALIDATION_STATE_VERSION = 1
TRAINER_STATE_FILE = "trainer_state.joblib"


def _pddl_identity(path: str) -> dict[str, str]:
    """Return a location-independent identity when the PDDL is readable."""
    try:
        contents = Path(path).read_bytes()
    except OSError:
        return {"path": os.path.normcase(os.path.abspath(path))}
    return {
        "name": Path(path).name,
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def validation_set_fingerprint(specs: Iterable[Any]) -> str:
    """Fingerprint the ordered validation instances and difficulty labels."""
    payload = [
        {
            "difficulty": getattr(
                getattr(spec, "difficulty", None),
                "name",
                str(getattr(spec, "difficulty", None)),
            ),
            "pddls": [_pddl_identity(path) for path in spec.pddls],
        }
        for spec in specs
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_trainer_state(resume_from: Optional[str]) -> Optional[dict[str, Any]]:
    """Load state only from an explicitly supplied checkpoint directory."""
    if not resume_from or not os.path.isdir(resume_from):
        return None
    path = os.path.join(resume_from, TRAINER_STATE_FILE)
    if not os.path.isfile(path):
        return None
    import joblib
    state = joblib.load(path)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid trainer state in {path}: expected a mapping")
    return state


@dataclass
class ValidationState:
    fingerprint: str
    trainer_kind: str
    version: int = VALIDATION_STATE_VERSION
    best_rate: Optional[float] = None
    best_average_plan_length: Optional[float] = None
    non_improving_count: int = 0
    best_checkpoint: Optional[str] = None
    best_cumulative_epoch: Optional[int] = None

    def observe(
        self,
        rate: float,
        average_plan_length: float,
        checkpoint: str,
        cumulative_epoch: int,
    ) -> bool:
        is_better = (
            self.best_rate is None
            or rate > self.best_rate
            or (
                rate == self.best_rate
                and average_plan_length < self.best_average_plan_length
            )
        )
        if is_better:
            self.best_rate = float(rate)
            self.best_average_plan_length = float(average_plan_length)
            self.non_improving_count = 0
            self.best_checkpoint = checkpoint
            self.best_cumulative_epoch = int(cumulative_epoch)
        else:
            self.non_improving_count += 1
        return is_better

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def restore(
        cls,
        *,
        expected_fingerprint: str,
        expected_trainer_kind: str,
        trainer_state: Optional[dict[str, Any]],
        is_resume: bool,
    ) -> tuple["ValidationState", bool]:
        """Return state and whether a legacy resume needs baseline validation."""
        if trainer_state is None:
            return cls(
                fingerprint=expected_fingerprint,
                trainer_kind=expected_trainer_kind,
            ), is_resume

        saved = trainer_state.get("validation_state")
        if saved is None:
            legacy_kind = (
                "stage1_imitation" if "iter_num" in trainer_state
                else "stage2_mcts"
            )
            phase_changed = legacy_kind != expected_trainer_kind
            return cls(
                fingerprint=expected_fingerprint,
                trainer_kind=expected_trainer_kind,
            ), not phase_changed
        if not isinstance(saved, dict):
            raise ValueError("Invalid validation_state: expected a mapping")
        if saved.get("trainer_kind") != expected_trainer_kind:
            return cls(
                fingerprint=expected_fingerprint,
                trainer_kind=expected_trainer_kind,
            ), False
        if saved.get("version") != VALIDATION_STATE_VERSION:
            raise ValueError(
                "Unsupported validation-state version: "
                f"{saved.get('version')!r}")
        if saved.get("fingerprint") != expected_fingerprint:
            raise ValueError(
                "Cannot resume validation early stopping: the validation set "
                "does not match the checkpoint's validation set")
        return cls(**saved), False


def cumulative_epoch_offset(trainer_state: Optional[dict[str, Any]]) -> int:
    """Find the first cumulative epoch of the continuation."""
    if trainer_state is None:
        return 0
    previous = trainer_state.get(
        "cumulative_epoch", trainer_state.get("epoch_num"))
    return int(previous) + 1 if previous is not None else 0
