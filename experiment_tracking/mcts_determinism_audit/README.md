# MCTS-DETERMINISM-AUDIT

This bounded audit localizes repeat-run divergence before any large rerun is
considered. It uses one exact Drone checkpoint, one historically variable
instance, one worker, width 20, 70 simulations, estimator coefficient 0.5,
PUCT 0.1, and the SAFE-1 terminal action selector.

Two arms receive three exact repeats each:

- `ordinary`: the current evaluation environment;
- `deterministic_cpu`: deterministic TensorFlow operations plus one thread for
  TensorFlow, OpenMP, MKL and OpenBLAS, and a fixed `PYTHONHASHSEED`.

At every committed external action, the opt-in `--action-debug` logger records
compact digests for the physical state, action-count feature, applicable-action
mask, raw network policy, child statistics, estimator value and selected
action. The first differing record separates state/action ordering, network,
estimator, PUCT/statistics and timeout-only divergence.

The audit requests six jobs of 2 CPUs and 20 GiB for at most two hours each.
They depend on a 1-CPU/4-GiB/10-minute container preflight. Existing production
files and running jobs are not modified: the instrumented worker is bound from
an experiment-local overlay into the isolated SAFE-CONTEXT checkout.

Authoritative inputs and Slurm identities are in `manifest.csv`; stdout paths
are recorded in `submissions.tsv` after submission.

