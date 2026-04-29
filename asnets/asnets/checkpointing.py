import os
import joblib
import shutil
from typing import Optional, Any
import tensorflow as tf
import numpy as np
import joblib
import os

WEIGHTS_FILE = "weights.joblib"
OPTIMIZER_FILE = "optimizer.joblib"
TRAINER_STATE_FILE = "trainer_state.joblib"


def resolve_weights_path(resume_from: str) -> str:
    """
    Supports both:
      1. old direct file: snapshot_x.pkl
      2. new directory: snapshot_x/weights.joblib
    """
    if os.path.isdir(resume_from):
        return os.path.join(resume_from, WEIGHTS_FILE)
    return resume_from


def resolve_optimizer_path(resume_from: str) -> Optional[str]:
    if not os.path.isdir(resume_from):
        return None

    path = os.path.join(resume_from, OPTIMIZER_FILE)
    return path if os.path.exists(path) else None


def save_checkpoint_dir(
    *,
    snapshot_dir: str,
    snapshot_name: str,
    weight_manager,
    optimizer,
    trainer_state: Optional[dict[str, Any]] = None,
) -> str:

    snapshot_path = os.path.join(snapshot_dir, snapshot_name)
    os.makedirs(snapshot_path, exist_ok=True)

    # --------------------------------------------------
    # Save weights
    # --------------------------------------------------
    weight_manager.save(os.path.join(snapshot_path, WEIGHTS_FILE))

    # --------------------------------------------------
    # Save optimizer variables
    # --------------------------------------------------
    opt_vars = optimizer.variables()

    if opt_vars:
        opt_numpy = [v.numpy() for v in opt_vars]

        joblib.dump(
            opt_numpy,
            os.path.join(snapshot_path, OPTIMIZER_FILE),
            compress=True,
        )
    else:
        opt_numpy = []

    # --------------------------------------------------
    # Save trainer state
    # --------------------------------------------------
    if trainer_state is not None:
        joblib.dump(
            trainer_state,
            os.path.join(snapshot_path, TRAINER_STATE_FILE),
            compress=True,
        )

    # --------------------------------------------------
    # Save debug fingerprints (for verification)
    # --------------------------------------------------
    weight_fp = [
        (
            float(tf.reduce_mean(w)),
            float(tf.math.reduce_std(w)),
            float(tf.linalg.norm(w)),
        )
        for w in weight_manager.all_weights
    ]

    optimizer_fp = []

    for v in optimizer.variables():
        mean_val = float(tf.reduce_mean(v))

        if v.dtype.is_floating:
            std_val = float(tf.math.reduce_std(v))
            norm_val = float(tf.linalg.norm(v))
        else:
            std_val = 0.0
            norm_val = float(mean_val)

        optimizer_fp.append(
            (
                v.name,
                mean_val,
                std_val,
                norm_val,
            )
        )

    optimizer_iteration = (
        int(optimizer.variables()[0])
        if optimizer.variables()
        else None
    )

    debug_data = {
        "weight_fp": weight_fp,
        "optimizer_fp": optimizer_fp,
        "optimizer_iteration": optimizer_iteration,
    }

    joblib.dump(
        debug_data,
        os.path.join(snapshot_path, "checkpoint_debug.joblib"),
        compress=True,
    )

    # --------------------------------------------------
    # Optional console logging (compact)
    # --------------------------------------------------
    print(
        f"[CHECKPOINT SAVED] {snapshot_name} | "
        f"W0 mean/std/norm="
        f"{weight_fp[0][0]:.6f}/"
        f"{weight_fp[0][1]:.6f}/"
        f"{weight_fp[0][2]:.6f} | "
        f"opt_iter={optimizer_iteration}"
    )

    return snapshot_path

def weight_fingerprint(weight_manager):
    import numpy as np
    import tensorflow as tf

    stats = []

    for w in weight_manager.all_weights:
        stats.append((
            float(tf.reduce_mean(w)),
            float(tf.math.reduce_std(w)),
            float(tf.linalg.norm(w)),
        ))

    return stats

def optimizer_fingerprint(optimizer):
    import tensorflow as tf

    vars = optimizer.variables()

    stats = []

    for v in vars:
        stats.append((
            v.name,
            float(tf.reduce_mean(v)),
            float(tf.math.reduce_std(v)),
            float(tf.linalg.norm(v)),
        ))

    return stats
