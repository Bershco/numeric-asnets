# Thesis experiment registry

This directory is the canonical index for thesis experiments. Reports should
read these files instead of repeatedly rediscovering completed results from the
cluster filesystem.

The current advisor-facing snapshot is
`advisor_audit_20260830/current_status.md`, with the machine-readable equivalent
in `advisor_audit_20260830/experiment_status.csv`. Cluster access must use the
single `uni-cluster` Windows-profile alias documented in
`cluster_access_and_ssh.md`.

## Files

- `cluster_access_and_ssh.md`: canonical cluster-access procedure,
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
- `mcts_runtime_cutoff_jobs_20260831.csv` and
  `mcts_runtime_cutoff_instances_20260831.csv`: deterministic recensoring of
  static MCTS results at 30-minute, two-hour, and six-hour per-instance limits.
  These are post-hoc score counterfactuals over recorded instance runtimes;
  they do not model extra scheduler throughput that a fresh hard-capped run
  would create. Active-job rows are explicitly provisional lower bounds.
- `evaluation_stochasticity_audit.md`: evaluation-path audit separating
  inactive sampling code from active numerical, estimator, ordering, timing,
  and build sources that can cause repeated MCTS runs to diverge.
- `mcts_progressive_widening_sensitivity/manifest.csv`: predeclared two-seed
  SAFE+PW sensitivity gate with sixteen one-factor-at-a-time jobs.
- `mcts_progressive_widening_sensitivity/submissions.tsv`: active corrected
  submission ledger. `cancelled_verbose_submissions.tsv` preserves stopped
  logging-preflight IDs and must not be interpreted as result rows.
- `mcts_progressive_widening_sensitivity/results.csv`: static VAL-confirmed
  terminal results accumulated from the active two-seed sensitivity campaign.
- `mcts_progressive_widening_sensitivity/kmin3_extension_manifest.csv` and
  `kmin3_extension_submissions.tsv`: four additional matched seeds in both VH
  modes for the promoted Kmin=3 arm (eight jobs, `20684748`--`20684755`).
- `mcts_horizon_pilot/results.csv` and `posthoc_val_20666467.csv`: complete
  cutoff-only horizon results and their separate post-hoc VAL evidence. This is
  distinct from the held, fully horizon-indexed MCTS-SAFE-2 design.
- `mcts_horizon_pilot/difference_audit.csv`: exact changed successful-instance
  membership for the two horizon reruns that exceeded historical coverage. Zero
  cutoffs and zero infeasible-goal decisions classify the differences as
  run-to-run/build variation rather than a horizon effect.
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
- `mcts_counters_width_sensitivity/stage2_narrow_pair_audit.csv`: all 354
  instances from the six terminal VH-off Stage-2 policy/narrow pairs, including
  outcome class, termination cause, runtime and literal source logs.
- `mcts_counters_width_sensitivity/stage2_policy_only_trajectory_audit.csv`:
  policy length, persisted MCTS length, common prefix, decrement count and
  multiset overrun for all twelve Stage-2 policy-only successes.
- `mcts_counters_width_sensitivity/stage2_policy_only_failure_summary.csv`:
  cause roll-up for those twelve losses. Eleven are exact 10,000-action
  divergences and zero are six-hour instance timeouts; one instance was not
  processed to a terminal record before the job ended and remains a
  conservative unclassified failure.
- `mprime_validation_ipc_scale_v1/validation_test_checkpoint_audit.csv` and
  `validation_test_agreement_summary.csv`: the complete 290-checkpoint corrected
  MPrime validation/test join and its 20 lineage plus two pooled correlations.
- `mprime_validation_ipc_scale_v1/corrected_learning_curve_aggregate.csv` and
  `corrected_learning_curves.{png,svg}`: the auditable every-five corrected
  validation/test curve, with changing run counts retained explicitly.
- `mprime_validation_ipc_scale_v1/anchor_tuning_manifest.csv` and
  `anchor_tuning_submissions.tsv`: all 28 corrected MPrime Stage-2 tuning jobs.
- `four_domain_preservation/tpp_stage2_policy_ready.csv` and its submission
  ledger: idempotent every-five plus selected/final Stage-2 policy work.
  Controllers 20687998, 20688711 and 20689471 handled successive terminal
  waves; controller 20718336 was submitted after the terminal count reached
  thirteen of sixteen held-out lineages.
- `four_domain_preservation/delivery_stage2_policy_ready.csv` and its submission
  ledger: controller 20717925 materialized six terminal lineages and submitted
  126 every-five plus endpoint policy evaluations.
- `mprime_validation_ipc_scale_v1/materialize_anchor_policy.py` and
  `submit_anchor_policy.py`: separate idempotent tuning-policy pipeline. After
  restoring the authoritative 28-row training ledger and separating it from
  the corrected-Stage-1 policy ledger, controller 20718118 submitted 105 policy
  evaluations for five terminal MPrime anchor lineages.
- `mcts_safe_context/corrected_diagnostic_instance_results.csv`: all twelve
  requested instance outcomes from corrected diagnostics 20688113--20688116,
  including context multipliers and original log pointers. The twenty matched
  full jobs were released after the bounded gate completed without overflow.
- `mcts_safe_context/context_revisit_summary.csv`: reproducible instance and
  aggregate accounting of physical-state revisits, different-action-history
  revisits, true full-context reuses, and contextual node multiplication. It
  is derived by `../scripts/summarize_safe_context_revisits.py` from the
  immutable twelve-instance diagnostic ledger.
- `main_val_stage2_drone_mcts_manifest.csv`: the sixteen MAIN-VAL Drone
  Stage-2 selected-checkpoint MCTS evaluations proven absent by a complete
  accounting search. Controller `20726009` submitted them as
  `20726030`--`20726045`; the four pre-existing matched results remain in
  `main_val_stage2_drone_mcts_gap.csv` and were not duplicated.
- `mcts_progressive_widening_cross_domain/pilot_manifest.csv`: the frozen
  twenty-job Kmin=3 cross-domain extension submitted as `20726010`--`20726029`.
  It covers two seeds and both VH modes for Block Grouping, FO Counters,
  Rover, and Counters Stage 1, plus Counters Stage 2. Every analysis must show
  the full six-hour result and the deterministic post-hoc 30-minute result;
  Block Grouping/Counters use 20 simulations to match narrow search, while
  FO Counters/Rover use 70 to match normal search.
- `mcts_progressive_widening_cross_domain/README.md`: the explicit
  budget-matched interpretation and predeclared cell-promotion rule. The Drone
  PW campaigns use 70 simulations; a 20-simulation Block Grouping/Counters
  result must never be described as the standard 70-simulation PW result.
- `mcts_counters_width_sensitivity/validate_terminal_partial.sbatch`:
  lightweight YYMAXDEPTH=12000 post-hoc VAL for interrupted narrow-search
  logs. Jobs 20719777--20719786 cover the ten newly terminal OOM lineages that
  were not yet present in the static result CSVs; inference is never rerun.
- `mcts_safe_context/stale_*_submissions_20260829.tsv`: immutable provenance for
  the one-instance diagnostic jobs and twenty stale held full jobs replaced
  after the restriction/validator wrapper repairs.
- `drone_endpoint_mcts_results.csv`: static reconciliation of all 27 completed
  Drone endpoint MCTS jobs with literal training/evaluation/VAL provenance.
- `drone_endpoint_mcts_paired_results.csv` and
  `drone_endpoint_mcts_summary.csv`: matched policy/MCTS joins and seed-level
  paired summaries for those Drone endpoint campaigns.
- `four_domain_preservation/zenotravel_stage2_lineages.tsv`: the complete
  20-lineage Zenotravel confirmation roster, including the four reused tuning
  winners. `zenotravel_stage2_missing_policy_submissions.tsv` records the 72
  recovery evaluations that fill their missing curves/endpoints.
- `mcts_progressive_widening_sensitivity/kmin3_runtime_*.csv`: immutable
  whole-job and successful-instance runtime evidence for policy, fixed top-20
  and PW Kmin3. Historical policy logs lacked per-instance elapsed records;
  jobs 20720780--20720787 recapture timing on the exact checkpoints without
  replacing the original scores.

The 2026-08-30 policy refresh found a log-format compatibility bug in selected
checkpoint materialization. Older logs wrote `New best reached ... iteration
... snapshot name`, while current logs write `[VALIDATION] New best! ...
iter_num=... snapshot_name=...`. Every-five and final endpoints were already
present; only off-grid selected endpoints could be absent. The parser now
accepts both forms, and idempotent controllers 20720613--20720616 recovered
Delivery, TPP, MPrime and Zenotravel selected endpoints without duplicating
existing evaluations.

This was a regular-expression compatibility bug, not missing training data.
The older form is emitted as `New best reached! ... iteration <epoch> ...
snapshot name: <path>`; the current form is `[VALIDATION] New best! ...
iter_num=<epoch> ... snapshot_name=<path>`. Source checkpoints for the four
reused Zenotravel tuning winners and every off-grid selected endpoint were
found. The 72 required Zenotravel recovery evaluations were submitted; no
training lineage or weight file had to be recreated.

The static Zenotravel reconciliation then exposed a second, lineage-level
case: two VH-off tuning winners continued in new experiment directories, and
their continuation-local validation baseline did not identify the global best
from the original phase. Comparing success and average validation-plan length
across both phases selects original-job 20553987 epoch 13 and original-job
20553995 epoch 20. Their policy evaluations are jobs 20724194 and 20724195;
`four_domain_preservation/zenotravel_stage2_global_selected_policy_submissions.tsv`
records the exact checkpoints. Continuation-local epochs must never silently
replace these global lineage-best endpoints.

Future policy-only evaluations request ten CPUs for ten evaluator workers.
Older six-CPU submissions remain valid evidence and are not repeated merely
for resource symmetry. Generic and preservation/MPrime policy controllers now
upgrade older manifest rows to at least one requested CPU per worker at
submission time.

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

Every result table—not only Counters—must be comparative. Search experiments
include the matched policy-only result and the declared search baseline whenever
those observations exist; training experiments include the relevant Stage-1
and Stage-2 endpoints. Every progressive-widening row therefore includes
matched policy-only and fixed top-20 scores. Every Counters narrow Stage-1 row
includes matched policy and normal width-20/70; Stage-2 narrow includes policy
only because Stage-2 normal MCTS is intentionally outside that comparison.

`experiments.csv` is the legacy seed used to construct `experiment_registry.csv`.
The obsolete `completed_results.csv`, `statistics.csv`, and `held_work.csv`
were removed after reconciliation; they must not be recreated or cited.

If the live cluster cannot be reached, report the snapshot timestamp explicitly
instead of presenting it as current.

## Advisor audit package

The durable 30 August 2026 whole-project audit is under
`experiment_tracking/advisor_audit_20260830/`. It contains RQ-aligned paired
statistics, the exact grouped Slurm resource snapshot, the story-hole register,
one scored improvement for every registered experiment, the meeting brief, and
the advisor-ready PowerPoint. Use these files instead of reconstructing static
results from raw logs during each status request.
