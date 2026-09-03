# MPrime validation-adequacy test

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

Phase A is active analysis. Phase B is held-design: instance generation and
checksums must be reviewed before any validation-only Slurm rescore is
submitted. Existing MPrime results remain reported with an explicit validation
adequacy warning.
