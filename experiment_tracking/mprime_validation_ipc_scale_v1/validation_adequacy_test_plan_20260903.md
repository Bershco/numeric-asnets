# MPrime validation-adequacy test

Current status (6 September): Phase B is running. The preparation-only and
35-percent readiness statements below are historical. See
`../mprime_validation_phase_b_20260906/README.md` for the frozen candidate
protocol, planner array21039224, finalizer21039342 and gated rescore chain.

## Current evidence

The concern is already empirically supported; the remaining question is how to
replace the validation design without selecting it on the IPC test results.

- Across the 290 corrected Stage-1 checkpoint evaluations, pooled Spearman
  validation/test rank correlation is only 0.255 for VH-off and 0.252 for
  VH-on. Only 5/10 VH-off and 7/10 VH-on lineages have positive within-lineage
  rank correlation.
- The validation-selected Stage-1 checkpoint loses a mean 3.1 test problems for
  VH-off and 2.5 for VH-on relative to the retrospectively observed test-best
  checkpoint. This regret is descriptive only; test-best must never be used as
  a training selection rule.
- All twenty corrected validation-led Stage-2 lineages select epoch 0. Their
  ten-seed test means change only 15.0 -> 15.2 (VH-off) and 14.6 -> 15.2
  (VH-on), while individual seed changes range in both directions. Immediate
  validation saturation supplies no temporal checkpoint resolution.

Sources: `validation_test_checkpoint_audit.csv`,
`validation_test_agreement_summary.csv`, and
`validation_led_stage2_selected_results_20260903.csv`.

## Phase A: exhaustive retrospective audit, no new inference

Join every already evaluated checkpoint within each lineage and report:

1. Spearman and Kendall validation/test rank agreement per lineage and pooled
   only as a secondary summary.
2. Validation-score saturation: number of unique scores, fraction at maximum,
   first maximum epoch, and longest maximum-score plateau.
3. Selection regret relative to final and retrospectively observed test-best
   checkpoints. The oracle test-best comparison diagnoses the selector but is
   never used to choose a checkpoint.
4. Stability under bootstrap resampling of the existing validation instances
   and leave-one-difficulty-band-out resampling.
5. Separate Stage-1 early-stopping adequacy from Stage-2 checkpoint-selection
   adequacy. A set can be adequate for one and inadequate for the other.

Phase A uses the existing 290 Stage-1 and 840 Stage-2 policy/checkpoint records;
it requires no retraining and no new policy test evaluation.

## Phase B: independent frozen validation replicate

Generate two deterministic candidate validation replicates before inspecting
any network score. Each replicate must be stratified across IPC-scale structural
features and include a harder upper tail so that current policies do not
saturate immediately. Freeze generator commit, parameters, seeds, file list and
SHA-256 checksums.

### Phase-B readiness correction — 4 September 2026

The existing `validation_ipc_scale_v1` generator cannot simply be rerun with
larger object ranges. Its goal constructor deliberately adds a direct
two-action witness (`overcome`, then `succumb`) for every generated goal. This
guarantees solvability, but it can make structurally large instances
behaviorally easy and is a plausible cause of complete Stage-2 saturation.

Phase B therefore requires a planner-side difficulty gate before any network
checkpoint is opened:

1. generate a deterministic candidate pool from an IPC-style goal process
   without inserting a direct witness for every goal;
2. solve/filter candidates using a frozen planner configuration;
3. stratify the two disjoint replicates by structural features and nontrivial
   planner plan-length bands;
4. freeze the generator commit, selection algorithm, seeds, selected file list,
   planner outcome/plan length and SHA-256 checksums;
5. run a one-checkpoint container preflight, then release the 60 resumable
   lineage jobs only if both sets remain disjoint, solvable and nontrivial.

The runnable Phase-B campaign is not yet submitted. The detailed readiness
gate is recorded in `phase_b_readiness_20260904.csv`. Submission confidence is
35%, below the user-approved 85% threshold, because neither replicate nor its
checksum manifest, 60-lineage manifest or parameterized wrapper currently
exists.

Evaluate all saved Stage-1 and Stage-2 checkpoints on both candidate replicates.
This is validation-only rescoring: existing networks are reused. For the full
current MPrime evidence the upper-bound scope is 1,130 checkpoint evaluations
(290 Stage-1 plus 420 validation-led Stage-2 plus 420 terminal-led Stage-2),
grouped by 60 lineages rather than one Slurm job per checkpoint.

Choose between candidate validation designs using only validation-side
criteria:

- low saturation and useful score resolution;
- positive rank agreement between the two independent replicates;
- stable selected epochs under bootstrap and difficulty-band removal;
- stable anchor ranking across independent replicates.

Only after the design is frozen should IPC test coverage be opened for a final
honest diagnostic. The candidate is adequate only if it improves rank agreement
and reduces selection regret on held-out test evidence without having been
chosen from that evidence.

## Retraining gate

No Stage-1 retraining is needed to test validation adequacy. Retraining becomes
necessary only if the new frozen validation design passes Phase B and the old
early-stopping rule is shown to have terminated lineages before useful later
checkpoints could exist. In that case, first run a two-seed-per-VH fixed-budget
continuation pilot; expand to ten seeds only if later checkpoint selection is
stable on both frozen validation replicates.

## Status

Phase A Stage-1 analysis is complete and stored in
`validation_adequacy_phase_a_stage1_seeds_20260903.csv` plus its summary:

| VH | Lineages/checkpoints | Mean unique validation scores | Mean fraction at validation maximum | Positive Spearman lineages | Mean within-lineage Spearman / Kendall | Mean selected-test regret |
|---|---:|---:|---:|---:|---:|---:|
| off | 10 / 153 | 7.3 | .124 | 5/10 | .097 / .070 | 3.1 plans |
| on | 10 / 137 | 7.7 | .105 | 7/10 | .232 / .173 | 2.5 plans |

These results independently confirm weak checkpoint-ranking resolution. Phase A
Stage-2 consolidation is complete over all 840 policy checkpoint evaluations;
it requires no new inference. Phase B remains an active top-priority
preparation task, not a submitted Slurm campaign. Existing MPrime results remain
reported with an explicit validation-adequacy warning.
