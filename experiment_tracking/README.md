# Thesis experiment registry

This directory is the canonical index for thesis experiments. Reports should
read these files instead of repeatedly rediscovering completed results from the
cluster filesystem.

## Files

- `experiment_registry.csv`: one row per named experiment, including its scientific
  role, status, configuration boundary, and reason when held.
- `live_jobs.csv`: replaceable queue snapshot. This contains only jobs that are
  running, pending, completing, or deliberately held in Slurm.
- `experiment_results.csv`: canonical completed-result ledger. Each row identifies the
  checkpoint/configuration, checkpoint epoch (parsed directly from the checkpoint
  path), score, VAL evidence, terminal condition, and source
  logs. A scheduler timeout or OOM is a terminal fixed-budget outcome; its
  already printed and VAL-valid plans remain part of the conservative score.
- `policy_endpoint_results.csv`: complete seed-level policy endpoint evidence
  for MAIN-VAL, MAIN-TERM, and PRESERVE-4, with literal source-log paths.
- `stage1_mcts_results.csv`: exact per-seed Stage-1 MCTS outcomes. Timeout
  instance IDs and unfinished instances after scheduler/OOM termination are
  separate fields.
- `experiment_statistics.csv`: complete endpoint and paired comparisons,
  confidence intervals,
  exact paired sign-flip p-values, and Holm-adjusted p-values.
- `policy_paired_seed_results.csv`: the seed-level joins underlying every
  paired policy statistic.
- `mcts_paired_seed_results.csv`: the seed-level policy/MCTS joins, including
  Slurm terminal state, exact search configuration, timeout counts, and
  unfinished-instance counts.
- `slurm_accounting_20260826_1608.psv`: immutable accounting export captured
  during the 2026-08-26 home-quota incident.
- `failed_jobs_20260826.csv`: every failed top-level job in that accounting
  export. Exit `0:53` identifies the quota/environment-retrieval cascade;
  exit `1:0` remains marked for individual log diagnosis rather than being
  automatically attributed to quota.
- `quota_recovery_20260826/failed_anchor_training_checkpoint_audit.psv`:
  immutable bounded audit of all 45 quota-interrupted anchor-training jobs,
  including the exact surviving snapshot root, terminal epoch, and log path.
- `quota_recovery_20260826/failed_anchor_training_continuations.csv`: exact
  checkpoint-continuation manifest for those 45 jobs.  These are resumptions,
  not fresh restarts; optimizer and persisted validation state are required.
- Cluster-side `quota_recovery/storage_audit_*`: read-only ranked storage audit
  outputs.  They are candidates for review, never automatic deletion lists.
- per-experiment directories: immutable launch manifests and any compact
  derived summaries specific to that experiment.
- `evaluation_coverage_plan.csv`: authoritative coverage contract for every
  training experiment. It records the required learning-curve policy work,
  required endpoint MCTS comparisons, materialized entry counts, and deliberate
  exclusions. Refresh it whenever training creates new checkpoints.

## Provenance rules

1. `source_training_log` points to the training or continuation log that
   establishes the lineage, seed, selected epoch, final epoch, and training
   configuration.
2. `source_evaluation_log` points to the evaluation log that establishes the
   evaluator outcome and printed plans.
3. `source_validation_log` points to the VAL evidence. It may equal the
   evaluation log when VAL ran inline. Interrupted evaluations may instead
   point to a post-hoc VAL audit log while retaining the interrupted original
   evaluation log separately.
4. Never delete or silently replace an original result with a retry. Store both
   paths and mark which row is authoritative.
5. A completed Slurm job and a fully classified evaluation are different
   concepts. `unprocessed_after_job_end` is nonzero only when an OOM or
   scheduler termination prevented the evaluator from reaching those
   instances. Per-instance six-hour timeouts are classified failures.
6. Completed rows are static. Only the live-job snapshot is repeatedly queried.

## 2026-08-26 quota recovery

The home quota incident caused hundreds of scheduler/environment failures and
45 interrupted Stage-2 anchor-training jobs.  The reinstallable pip cache was
purged (about 3.3 GB), followed by the separately approved JetBrains cache
(4.33 GB).  No experiment logs, checkpoints, manifests, source trees, or
container images were deleted.

Policy-evaluation failures were recovered through exact manifests.  The 45
training failures were audited individually and found to contain intact
weights, optimizer state, and `trainer_state.joblib` at epochs 35--98.  They
are continued for only their remaining epochs through controller `20595724`;
the failed portability-check controller `20595666` submitted no experiments.

## Report convention

Every status report must list:

- live experiments;
- completed experiments;
- held experiments and their reasons;
- current running/pending/held workload;
- results sourced from `experiment_results.csv` and
  `experiment_statistics.csv`.

`experiments.csv` is the legacy seed used to construct `experiment_registry.csv`.
The obsolete `completed_results.csv`, `statistics.csv`, and `held_work.csv`
were removed after reconciliation; they must not be recreated or cited.

If the live cluster cannot be reached, report the snapshot timestamp explicitly
instead of presenting it as current.
