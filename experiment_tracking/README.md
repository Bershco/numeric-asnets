# Thesis experiment registry

This directory is the canonical index for thesis experiments. Reports should
read these files instead of repeatedly rediscovering completed results from the
cluster filesystem.

## Files

- `../docs/cluster_access_and_ssh.md`: canonical cluster-access procedure,
  including the recurring wrong-Windows-profile versus stale-VPN-route failure
  and its bounded retry sequence.
- `experiment_registry.csv`: one row per named experiment, including its scientific
  role, status, configuration boundary, and reason when held.
- `live_jobs.csv`: replaceable queue snapshot. This contains only jobs that are
  running, pending, completing, or deliberately held in Slurm.
- `live_experiment_status.csv`: replaceable experiment-level roll-up of the
  live queue and latest immutable result-ledger counts. It separates running,
  dependency/resource-pending, and deliberately held work.
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
- `mcts_policy_regression_audit.csv`: all 100 matched Stage-1 policy/MCTS seed
  pairs with an explicit relation and first-pass cause class. Every
  `policy_gt_mcts` row requires an instance-level log audit; aggregate means
  must never be used to assume that MCTS dominates policy seed by seed.
- `policy_mcts_instance_audit.csv`: all available matched Stage-1
  policy/MCTS instances, including policy and MCTS outcomes, steps, runtimes,
  timeout/action-limit classification, and literal source-log paths. This is
  the authoritative source for diagnosing seed-level gains and regressions.
- `block_grouping_policy_mcts_instance_audit.csv`: the focused Block Grouping
  extract used to explain the width-5/20 result without rerunning inference.
- `../docs/mcts_drone_dead_end_audit.md`: exact persisted-trajectory diagnosis
  of the four Drone policy-only successes and the required terminal-safety
  selection correction.
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
- `mcts_progressive_widening_pilot/results.csv`: terminal PW/fixed-top-5
  results joined to their matched policy and fixed-top-20 baselines. Running
  rows remain in `live_jobs.csv` until they become static.
- `mcts_progressive_widening_pilot/summary.csv`: completed ten-seed matched
  coverage, runtime, paired inference, and weighted node/root branching
  summaries for fixed top-5 and both progressive-widening variants.
- `mcts_counters_width_sensitivity/main_term_stage2_narrow_held_manifest.csv`
  and its submission ledger: the twenty held MAIN-TERM Stage-2 Counters
  width-5/20 replacements. The obsolete held width-20/70 jobs were cancelled
  only after all replacements existed and were explicitly held.
- `large_log_compaction_20260827.csv`: immutable verification/deletion ledger
  for the 17 debug-spam logs compacted on 2026-08-27. It records source and
  compact checksums and exact reclaimed bytes.
- Cluster job `20652215` is the read-only post-compaction audit. It inventories
  every file above 25 MiB, directory usage to depth four, modification times,
  extensions, and experiment-output domains. Its partial inventory already
  identifies 743 files above 25 MiB and multiple remaining Counters logs of
  0.7--1.0 GB that fell below the previous unique-file threshold.
- `four_domain_preservation/zenotravel_anchor_evidence.csv`: the 28-lineage
  replay that reproduces the frozen Zenotravel coefficients. Delivery and TPP
  use the same validated finalizer after their remaining lineages terminate.
- `evaluation_coverage_plan.csv`: authoritative coverage contract for every
  training experiment. It records the required learning-curve policy work,
  required endpoint MCTS comparisons, materialized entry counts, and deliberate
  exclusions. Refresh it whenever training creates new checkpoints.
- `mcts_progressive_widening_sensitivity/manifest.csv`: predeclared two-seed
  SAFE+PW sensitivity gate with sixteen one-factor-at-a-time jobs.
- `mcts_progressive_widening_sensitivity/submissions.tsv`: active corrected
  submission ledger. `cancelled_verbose_submissions.tsv` preserves stopped
  logging-preflight IDs and must not be interpreted as result rows.
- `mcts_progressive_widening_sensitivity/results.csv`: static VAL-confirmed
  terminal results accumulated from the active two-seed sensitivity campaign.
- `mcts_horizon_pilot/results.csv` and `posthoc_val_20666467.csv`: complete
  cutoff-only horizon results and their separate post-hoc VAL evidence. This is
  distinct from the held, fully horizon-indexed MCTS-SAFE-2 design.
- `drone_mcts_release_20260828.csv`: exact twenty-seven-job release ledger for
  terminal-led Stage-2, validation-led Stage-2 and long-training Drone MCTS
  endpoints. FO Counters and Rover remained held.
- `mcts_counters_width_sensitivity/stage1_narrow_terminal_results.csv` and
  `stage2_narrow_terminal_results.csv`: static narrow width-5/20 terminal
  outcomes. OOM rows retain classified-instance counts and conservative
  VAL-confirmed lower bounds instead of being dropped.
- `mcts_safe_drone/targeted_results_20260828.csv`: frozen outcomes for the four
  targeted SAFE cases. The broader jobs were cancelled after those targets
  completed because the older skip-list transport expanded their scope.
- `mcts_counters_width_sensitivity/stage2_narrow_release.csv`: the twenty and
  only twenty Stage-2 Counters width-5/20 jobs released from the legacy hold on
  2026-08-28.

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

## Large-log compaction

Completed experiment text logs larger than 1 GiB may be replaced by a verified
compact evidence record when their size is dominated by repeated debug output.
Exact duplicate groups are also eligible when each copy is at least 500 MiB;
the same evidence and safety checks apply.
The compact record must retain the original path, size, modification time and
SHA-256; job/commit/checkpoint/seed/configuration and command evidence; all
epoch, loss, coverage, selection and termination markers; per-instance
outcomes, runtimes and complete printed plans; VAL evidence; and the original
head and tail. It must also have its own checksum and manifest row.

Deletion is allowed only after verification, only for logs at least 24 hours
old, and never while the corresponding job is active. A compact file must not
reuse the original filename or pretend to be the raw source. Primary unique
logs that cannot pass the evidence checks remain untouched. The implementation
is `../scripts/compact_large_experiment_logs.py`.

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

Comparison tables must contain actual comparable numeric observations in every
score column. Never present a dash as though it were a score. A seed lacking a
matched policy or baseline belongs in a separate progress/provenance table until
that value is recovered.

`experiments.csv` is the legacy seed used to construct `experiment_registry.csv`.
The obsolete `completed_results.csv`, `statistics.csv`, and `held_work.csv`
were removed after reconciliation; they must not be recreated or cited.

If the live cluster cannot be reached, report the snapshot timestamp explicitly
instead of presenting it as current.
