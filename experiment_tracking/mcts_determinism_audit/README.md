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

## Result — 31 August 2026

All six runs solved `problem_5_2_2` in exactly 105 actions.  Successful runtime
ranged from 137.51 to 151.03 seconds, so coverage is `1/1` at 30 minutes,
2 hours and 6 hours for every arm.

- The three ordinary repeats on `cs-cpu-07` were checksum-identical at every
  one of the 105 committed decisions.
- Deterministic-CPU repeat 1 also ran on `cs-cpu-07` and was identical to the
  ordinary runs.  The deterministic environment itself therefore did not
  alter this trajectory on matched hardware.
- Deterministic-CPU repeats 2 and 3 ran on `ise-cpu-intl-07` and were identical
  to each other.  Relative to `cs-cpu-07`, they differed in the raw network
  policy checksum at 78/105 steps and child-statistics checksum at 45/105
  steps, but physical-state, action-history and selected-action digests
  differed at 0/105 steps.

The audit therefore localizes real numerical variation to CPU/node type.  It
does not support random process timing as the cause, and it does not show an
action or coverage change on this instance.  A stronger follow-up would pin
matched repeats to one node type and separately compare node types on an
instance known to diverge in selected action.
