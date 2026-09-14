"""Strict loading of optimizer batches captured by the first-update audit.

This module deliberately contains no sampling code.  A frozen replay run must
consume exactly the numbered arrays captured by the source run, in exactly the
same order, or fail before applying an optimizer update.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FrozenReplayStep:
    index: int
    path: Path
    sha256: str
    batches: tuple[tuple[str, tuple[np.ndarray, ...]], ...]


class FrozenReplaySchedule:
    """Read and validate a contiguous directory of captured replay batches."""

    def __init__(self, directory: str | Path, expected_steps: int):
        self.directory = Path(directory)
        self.expected_steps = int(expected_steps)
        if self.expected_steps <= 0:
            raise ValueError("expected_steps must be positive")
        if not self.directory.is_dir():
            raise ValueError(
                f"Frozen replay directory does not exist: {self.directory}")
        expected = [
            self.directory / f"optimizer_step_{index:03d}.npz"
            for index in range(self.expected_steps)
        ]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise ValueError(
                "Frozen replay schedule is incomplete; missing: "
                + ", ".join(missing[:5]))
        extra = sorted(self.directory.glob("optimizer_step_*.npz"))
        if len(extra) != self.expected_steps:
            raise ValueError(
                "Frozen replay schedule must contain exactly "
                f"{self.expected_steps} numbered files; found {len(extra)}")
        self._paths = tuple(expected)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_step(
            self,
            index: int,
            problems_by_signature: Mapping[str, Any],
    ) -> FrozenReplayStep:
        if index < 0 or index >= self.expected_steps:
            raise IndexError(
                f"Frozen replay step {index} outside 0.."
                f"{self.expected_steps - 1}")
        path = self._paths[index]
        batches = []
        with np.load(path, allow_pickle=False) as payload:
            batch_indices = sorted(
                int(key.removeprefix("obs_"))
                for key in payload.files if key.startswith("obs_"))
            if not batch_indices:
                raise ValueError(f"Frozen replay step has no batches: {path}")
            if batch_indices != list(range(len(batch_indices))):
                raise ValueError(
                    f"Frozen replay batch indices are not contiguous: {path}")
            for batch_index in batch_indices:
                suffix = str(batch_index)
                required = {
                    f"obs_{suffix}", f"pi_tgt_{suffix}",
                    f"policy_weights_{suffix}",
                    f"problem_signature_{suffix}",
                }
                missing = sorted(required.difference(payload.files))
                if missing:
                    raise ValueError(
                        f"Frozen replay batch {batch_index} is missing "
                        f"{missing}: {path}")
                signature_arr = payload[f"problem_signature_{suffix}"]
                if signature_arr.ndim != 0:
                    raise ValueError(
                        f"Problem signature must be scalar in {path}")
                obs = np.asarray(payload[f"obs_{suffix}"])
                pi_tgt = np.asarray(payload[f"pi_tgt_{suffix}"])
                policy_weights = np.asarray(
                    payload[f"policy_weights_{suffix}"])
                signature = str(signature_arr.item())
                problem = problems_by_signature.get(signature)
                if problem is None and not signature:
                    # Phase A was captured before SingleProblem persisted its
                    # compatibility signature. Resolve only when the archived
                    # tensor widths identify exactly one initialized bucket.
                    candidates = [
                        (candidate_signature, candidate)
                        for candidate_signature, candidate
                        in problems_by_signature.items()
                        if obs.ndim == 2 and pi_tgt.ndim == 2
                        and int(candidate.obs_dim) == obs.shape[1]
                        and int(candidate.act_dim) == pi_tgt.shape[1]
                    ]
                    if len(candidates) == 1:
                        signature, problem = candidates[0]
                if problem is None:
                    raise ValueError(
                        "Frozen replay problem signature has no initialized "
                        f"bucket: {signature!r}")
                z_key = f"z_tgt_{suffix}"
                z_tgt = np.asarray(payload[z_key]) if z_key in payload else None
                if obs.ndim != 2 or pi_tgt.ndim != 2:
                    raise ValueError(
                        f"Frozen observations/targets must be rank two: {path}")
                if obs.shape[0] != pi_tgt.shape[0]:
                    raise ValueError(
                        f"Frozen observation/target row mismatch: {path}")
                if policy_weights.shape != (obs.shape[0],):
                    raise ValueError(
                        f"Frozen policy-weight shape mismatch: {path}")
                if z_tgt is not None and z_tgt.shape[0] != obs.shape[0]:
                    raise ValueError(
                        f"Frozen value-target shape mismatch: {path}")
                if obs.shape[1] != int(problem.obs_dim):
                    raise ValueError(
                        f"Frozen observation width mismatch for {signature}")
                if pi_tgt.shape[1] != int(problem.act_dim):
                    raise ValueError(
                        f"Frozen policy width mismatch for {signature}")
                for name, array in (
                        ("obs", obs), ("pi_tgt", pi_tgt),
                        ("policy_weights", policy_weights)):
                    if not np.all(np.isfinite(array)):
                        raise ValueError(
                            f"Frozen {name} contains non-finite values: {path}")
                if z_tgt is not None and not np.all(np.isfinite(z_tgt)):
                    raise ValueError(
                        f"Frozen z_tgt contains non-finite values: {path}")
                batches.append((
                    signature,
                    (obs, pi_tgt, z_tgt, policy_weights),
                ))
        return FrozenReplayStep(
            index=index,
            path=path,
            sha256=self._sha256(path),
            batches=tuple(batches),
        )
