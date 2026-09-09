# Targeted experiment and figure update — 9 September 2026, 16:40 IDT

This update is intentionally limited to the user-requested live items: MPrime
validation adequacy Phase B, the FO Counters PW70 missing-instance recovery,
adaptive-KL training and its epoch-0 policy checks, and the revised advisor
figures. It does not supersede the complete experiment registry.

## MPrime Phase B

- 2,157 / 2,260 checkpoint-replicate validations are complete (95.4%).
- 39 / 60 lineages are complete; every lineage has produced results.
- 103 checkpoint-replicate validations remain.
- Seventeen original idempotent rescore tasks and retry task 40 are running.
  Retry task 14 completed after submission.
- The original tasks have approximately 6.6 hours left before their 24-hour
  allocation limit. Retry 40 has a fresh allocation but skips every completed
  point.
- No Phase-B checkpoint or anchor conclusion is frozen yet. The next analysis
  compares both independent validation replicates for saturation, cross-replicate
  rank agreement, selected-epoch stability and anchor-ranking stability.

Canonical progress ledger:
`experiment_tracking/mprime_validation_phase_b_progress_latest.csv`.
Remote provenance root:
`/home/hersco/training_new_domains/2026-09-06/mprime_phase_b`.

## FO Counters PW70 missing instance

The prior seed-2082152039 job classified 19 / 20 instances and left only test
position 15 (`instance_15.pddl`) unclassified. Treating the campaign as n=9 or
silently counting that instance as a permanent failure would both be weaker than
finishing the declared matched seed.

Job 21154686 therefore evaluates only position 15 with the exact original
checkpoint and PW70 configuration. It uses one worker because workers only
parallelize independent test instances; one worker avoids recreating the
three-worker memory failure without changing the per-instance search algorithm.
The job requests 2 CPUs, 40 GiB and an eight-hour allocation with the original
six-hour per-instance limit. When it finishes, the declared seed score is either
7/20 or 8/20, and the ten-seed aggregate will include the supplemental log as
provenance rather than rerunning or replacing the other 19 outcomes.

Manifest and source paths:
`experiment_tracking/mcts_progressive_widening_cross_domain/fo_off_2082152039_single_instance_recovery_20260909.csv`.

## Adaptive KL

The implemented adaptive controller has not yet changed behavior relative to the
constant-KL baseline. Every observed realized KL remains below the target band,
the coefficient is already at its declared minimum of 3, and both jobs report
zero coefficient adjustments. The current evidence therefore tests the
instrumented controller path, but does not yet constitute an adaptive-vs-constant
intervention.

- Stable control job 21144389 has 74 saved Stage-2 checkpoints and remains at
  30/30 validation.
- Catastrophic-seed job 21144388 was automatically requeued. Its earlier output
  directory with 58 saved checkpoints remains intact. The requeued process
  restarted from the Stage-1 source and currently has six checkpoints in a new
  directory. The earlier scientific outputs were not lost; some compute is being
  repeated because the batch wrapper does not automatically resume Stage-2 when
  an explicit Stage-1 source is present.
- The requeued run is being left intact because a naive resume from a Stage-2
  snapshot risks redefining the KL teacher/anchor at the continuation boundary.
  Avoiding that semantic change is more important than recovering the repeated
  compute blindly.
- Two policy-only epoch-0 tests were submitted to answer the essential question:
  did the first adaptive update preserve the original 20/20 test policy? The
  initial array 21154746 failed at CLI parsing and was replaced by 21154752.
  Both replacement tasks then hit native evaluator exit -4 on
  `ise-cpu-intl-14`/`ise-cpu-intl-24`; the outlier-only retry 21154755 did the
  same on `ise-cpu-intl-24`. Final replacement 21154765 uses a fresh
  five-worker identity and is pinned to the known-good `cs-cpu-07` node. Every
  failed attempt ended before producing a test result and is retained in the
  companion ledger.

Companion ledger:
`experiment_tracking/anchor_kl_epoch0_policy_eval_20260909.csv`.
Training progress:
`experiment_tracking/anchor_kl_control_progress_latest.csv`.

## Revised advisor figures

The first four candidates have been replaced rather than merely restyled:

1. `a_all_domain_endpoint_evidence` reports absolute Stage-1 and Stage-2 means,
   denominators, matched n, normalized percentage-point change, 95% CI and the
   applicable p-value. Counters can no longer look dominant merely because it has
   59 test instances.
2. `b_stage1_to_stage2_learning_dynamics` has an explicit dotted S1/S2 boundary.
   The left marker is the matched Stage-1 endpoint; the right side is the Stage-2
   trajectory and full observed min–max envelope. The connecting dotted segment
   is explicitly not presented as an inferred Stage-1 curve.
3. `c_preserve3_seed_robustness` adds readable domain/VH labels, a change-axis
   title, a legend, cell means, and labels for every regression of at least five
   plans.
4. `d_mprime_validation_selection_cost` directly plots validation-selected test
   performance against the retrospective observed-best checkpoint. It shows mean
   selection regret of 2.5–3.1 plans and the total Stage-2 saturation that caused
   it, rather than an abstract method diagnostic.

All SVGs, rendered PNGs, companion CSVs and reconstruction notes are under:
`experiment_tracking/learning_curves/advisor_v2_20260909/`.
