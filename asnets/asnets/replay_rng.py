"""Replay-step RNG capture and verification helpers.

The trust-region diagnostic retries one frozen replay batch after restoring the
network and Adam state.  For an exact learning-rate/backtracking comparison it
must also restore every RNG that can affect the training-mode forward pass.
This module keeps the TensorFlow-specific introspection narrow and provides a
raw-gradient digest that lets the scientific job fail closed if an unobserved
random source still changes the retry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np


def _iter_network_objects(networks: Iterable[Any]):
    """Yield each network/layer once without assuming a Keras minor version."""
    seen = set()
    stack = list(networks)
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        for attr in ("submodules", "layers"):
            children = getattr(obj, attr, ())
            try:
                stack.extend(list(children))
            except TypeError:
                pass


def _state_variable(candidate: Any):
    """Return an assignable RNG state variable, if the object exposes one."""
    if candidate is None:
        return None
    state = getattr(candidate, "state", None)
    if state is not None and hasattr(state, "numpy") and hasattr(state, "assign"):
        return state
    return None


def tensorflow_rng_variables(tf_module: Any, networks: Iterable[Any]):
    """Find global and Keras-layer generator state variables.

    TensorFlow/Keras has used both public global generators and private
    per-layer ``RandomGenerator`` wrappers.  We record both, deduplicating the
    underlying variables.  Legacy stateful-op counters are additionally reset
    through ``tf.random.set_seed`` when a snapshot is restored.
    """
    candidates = []
    random_module = getattr(tf_module, "random", None)
    for owner in (random_module, getattr(random_module, "experimental", None)):
        getter = getattr(owner, "get_global_generator", None)
        if getter is not None:
            try:
                candidates.append(getter())
            except (RuntimeError, ValueError):
                pass

    for obj in _iter_network_objects(networks):
        for attr in ("_random_generator", "random_generator", "seed_generator"):
            wrapper = getattr(obj, attr, None)
            if wrapper is None:
                continue
            candidates.append(wrapper)
            candidates.append(getattr(wrapper, "_generator", None))
            candidates.append(getattr(wrapper, "generator", None))

    variables = []
    seen = set()
    for candidate in candidates:
        variable = _state_variable(candidate)
        if variable is None or id(variable) in seen:
            continue
        seen.add(id(variable))
        variables.append(variable)
    return variables


@dataclass
class ReplayRNGSnapshot:
    """Restorable Python, NumPy, TensorFlow, and Keras RNG state."""

    step_seed: int
    python_state: object
    numpy_state: tuple
    tensorflow_states: List[Tuple[Any, np.ndarray]]

    @classmethod
    def capture(cls, tf_module: Any, networks: Iterable[Any], step_seed: int):
        # Reset legacy TensorFlow stateful-op counters to a step-specific seed.
        # The same call on restore is required for exact retry reproduction.
        tf_module.random.set_seed(int(step_seed))
        variables = tensorflow_rng_variables(tf_module, networks)
        return cls(
            step_seed=int(step_seed),
            python_state=random.getstate(),
            numpy_state=np.random.get_state(),
            tensorflow_states=[
                (variable, np.asarray(variable.numpy()).copy())
                for variable in variables
            ],
        )

    def restore(self, tf_module: Any):
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        tf_module.random.set_seed(self.step_seed)
        for variable, value in self.tensorflow_states:
            variable.assign(value)

    @property
    def tensorflow_generator_count(self) -> int:
        return len(self.tensorflow_states)


def replay_step_seed(base_seed: int, optimizer_step: int) -> int:
    """Derive a stable positive TensorFlow seed for one optimizer step."""
    if base_seed < 0 or optimizer_step < 0:
        raise ValueError("Replay RNG seeds and optimizer steps must be non-negative")
    # Use a prime stride and stay inside the signed 31-bit range accepted by
    # every supported TensorFlow release.
    return int((base_seed + 1_000_003 * optimizer_step) % 2_147_483_647)


def gradient_sha256(gradients: Sequence[Any]) -> str:
    """Hash gradient dtype, shape, and exact bytes in parameter order."""
    digest = hashlib.sha256()
    for gradient in gradients:
        value = gradient.numpy() if hasattr(gradient, "numpy") else gradient
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
