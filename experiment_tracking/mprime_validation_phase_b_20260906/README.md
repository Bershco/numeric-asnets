# MPrime validation adequacy Phase B — active, 6 September 2026

The earlier preparation-only status is superseded. There are 240 deterministic
candidates, two independent seed pools and three structural tiers. The generator
does not inject a pleasure at each goal food. Goals are initially absent and at
least 2/3/4 directed graph edges from every initial pleasure position. This
prevents the previous universal direct overcome/succumb witness. It does not
claim to perfectly reproduce IPC difficulty: adequacy is the question being tested.

Candidate selection is frozen before network scoring: ascending seed order,
first ten planner-successful and VAL-valid candidates in each tier, with minimum
planner plan lengths 4/6/8. The planner is hmrp-ha-gbfs with a 60-second limit.
Timeout candidates are rejected by that rule; plan length is a witness length,
not an optimal-length assertion. No test or network score selects candidates.

## Submitted workflow

- Planner array `21039224`, six tasks, each 2 CPUs / 8 GiB / 90 minutes.
- Finalizer `21039342`, afterok-dependent on the whole planner array, 1 CPU /
  1 GiB / 10 minutes. It refuses to freeze unless all six ten-instance groups
  exist, all 60 file hashes match, and the two sets contain no identical files.
- It then submits a one-checkpoint, two-replicate container/VAL preflight.
- Only after that preflight succeeds does its 60-lineage rescore array start,
  at most 12 concurrent tasks, each 3 CPUs / 20 GiB / 24 hours.
- Every completed checkpoint/replicate retains a log, VAL CSV and a completion
  marker tied to checkpoint identity and the frozen manifest SHA-256. Partial
  inference is never accepted as a complete validation result.

The inventory is 290 Stage-1 plus 840 Stage-2 checkpoints: 1,130 distinct
checkpoints, **2,260 checkpoint-replicate evaluations** across the two sets.
Earlier prose stating 1,130 evaluations for both sets undercounted the workload.
Training/evaluation job IDs and original log paths are in `checkpoints.csv`.

The remote work directory is
`/home/hersco/training_new_domains/2026-09-06/mprime_phase_b/`.
It contains `planner_task_N.csv`, planner plans and VAL logs,
`frozen_validation_manifest.csv`, `rescore_submission.json`, and per-lineage
`rescore/` outputs. The two new domain modules are added to the isolated
safe-context checkout; production domain definitions are untouched.

## Interpretation and remaining scientific work

The candidate set is not declared adequate merely because the planners solve it.
After rescoring, assess saturation, within-lineage rank agreement between the
two replicates, bootstrap stability and selected epochs. The current campaign
tests Stage-1/Stage-2 checkpoint selection. Anchor-rank stability additionally
requires rescoring the saved 28 anchor-tuning lineages on the same frozen sets;
that follow-up is documented but has not been submitted by this controller.
No MPrime MCTS evaluation is submitted or authorized by this workflow.

Local candidate hashes, parameters and seeds are in `candidates.csv`; generated
PDDL files are in `candidates/`. Planner evidence is copied locally as compact
CSV files. The exact generator and controller scripts are versioned with this
directory, so the candidate construction can be reproduced.
