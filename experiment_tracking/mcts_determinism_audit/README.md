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

## Follow-up candidate gate

`followup_candidates.csv` ranks fresh same-commit Horizon pairs that changed
coverage. The primary candidate is Drone VH-on seed `1239739722`: unaware job
`20684991` scored 9/20 and aware job `20684992` scored 6/20 while recording zero
horizon cutoffs. The per-instance join is complete. The aware run lost
`problem_3_3_4`, `problem_4_2_5`, and `problem_8_1_4`; it gained no instance.
`problem_3_3_4` is the bounded follow-up target because the unaware run solved
it in 404.85 seconds and the aware run terminated unsolved in 379.24 seconds.
This makes it cheap enough for matched, node-pinned checksum repeats. The
archived logs do not contain the new per-root checksum records, so they
establish divergent outcomes but cannot identify the first differing action
without this rerun.

The bounded follow-up was submitted after cluster-side shell syntax checks.
Preflight job `20788665` gates six one-instance jobs `20788666`--`20788671`.
Three repeats are pinned to `cs-cpu-07` and three to `ise-cpu-intl-07`; each
requests two CPUs, 20 GiB and at most two hours. The target is VH-on seed
`1239739722`, checkpoint epoch 23, `problem_3_3_4`, with one worker and the
same fixed width-20/70-simulation search configuration.

## Follow-up result — 1 September 2026

Preflight `20788665` completed in 1m29s.  Jobs `20788666`--`20788671`
all completed in 6m28s--7m42s and produced 165 committed-decision records.
Every repeat followed the same 165-action trajectory and failed the target
instance (`0/1`).

- Within each node family, all three repeats were exactly reproducible.
- Across `cs-cpu-07` and `ise-cpu-intl-07`, physical-state, action-history and
  selected-action sequences remained identical.
- The raw network-policy sequence and child-statistics sequence differed by
  node family, reproducibly, despite the identical selected actions.

The two audits therefore establish a precise result: node-family numerical
variation changes network predictions and derived tree statistics, but it did
not change a selected action, trajectory, or outcome on either tested
instance. Random worker timing is not supported as the explanation.

The follow-up does **not** directly explain the historical Horizon
aware/unaware coverage divergence. Its six jobs all ran the unaware arm; the
script did not pass `--eval-mcts-enforce-remaining-horizon`. It also used one
worker and the current diagnostic/SAFE overlay, whereas the historical pair
used the original multiprocess evaluation path. The follow-up therefore
establishes hardware repeatability of one current baseline trajectory, not an
aware-versus-unaware replay. Moreover, all six current runs failed after 165
actions even though the historical unaware run solved the same instance. At
least one non-horizon execution-path difference remains.

The next causal audit, if prioritized, must run fresh aware and unaware arms
from the same current commit on the same node family and worker configuration,
then record the first differing root state, network output, estimator value,
per-child N/Q/U/prior vector, goal-chase decision and selected action. A second
block may reproduce the historical three-worker setup. Until that experiment
exists, hardware variation, random process timing and the Horizon flag itself
have not been shown to cause the historical three-instance coverage change.

Exact compact hashes and source-log pointers are in `followup_results.csv`.
