# MPrime validation-design audit

Status: validation-set redesign required before MPrime validation-selected
results are used as confirmatory evidence.

## Diagnosis

All twenty replicated MPrime Stage-1 lineages (ten value-head off and ten
value-head on) selected epoch 1 as their validation-best checkpoint. This is
not evidence about Numeric ASNets or the published paper. It is evidence that
our generated validation distribution is a poor proxy for the IPC test
distribution.

The current validation problems are substantially smaller than the test
problems:

| Structural property | Current validation | IPC test set |
|---|---:|---:|
| Food objects | mean about 6.8; maximum 10 | mean 12.95; maximum 22 |
| Pain objects | mean about 4.7 | mean 16.05; maximum 46 |
| Graph edges | mean about 14.4 | mean 33.05; maximum 64 |

The existing easy/medium/hard generator settings therefore reward very early
checkpoints and do not track generalization to the substantially larger test
instances.

The first frozen replacement sanity run exposed a more fundamental generator
bug: the generator sampled every goal `craves` fact directly from facts already
placed in `:init`.  ENHSP therefore returned a valid zero-action plan for all
30 replacement instances.  No checkpoint rescoring was launched from that
invalid set.  The generator was corrected so every goal is absent initially
and has a guaranteed `overcome` + `succumb` witness.  The replacement set is
regenerated and frozen only after the corrected planner sanity run reports
strictly positive plan lengths.

## Scientific treatment

- Label the existing validation-selected MPrime result as a validation-design
  failure; do not present it as a finding about the original paper.
- Preserve the original validation instances, manifests and results unchanged
  for provenance.
- Do not use test coverage or test plans to choose a new checkpoint. Structural
  size statistics may be used to define a better validation distribution, but
  checkpoint selection must remain independent of test outcomes.
- Continue reporting the terminal-checkpoint experiment separately because it
  does not depend on this validation selector.

## Corrective protocol

1. Add a deterministic generator seed and write a manifest containing every
   generator argument, output filename and checksum.
2. Generate a new versioned MPrime validation set whose food, pain and graph
   size ranges overlap the IPC test distribution. Keep it in a new directory;
   never overwrite the current validation set.
3. Audit the generated structural distribution before examining any policy
   result. Freeze the instances and manifest once accepted.
4. Run a planner-only sanity check to reject malformed or unsolvable instances
   using a predeclared rule, not network performance.
5. Re-evaluate all existing MPrime Stage-1 checkpoints on the frozen replacement
   validation set. This is selection-only work; retraining is not initially
   required.
6. Select checkpoints with the same declared validation rule used elsewhere,
   then run the already fixed test evaluation exactly once per selected
   checkpoint.
7. Use the replacement validation set for subsequent MPrime Stage-2 tuning and
   held-out confirmation. Keep the old and corrected results visibly separate.

No replacement MPrime validation result is authoritative until these steps are
complete.

## Corrected checkpoint-ranking audit (2026-08-29)

The corrected campaign is fully evaluated: 290/290 checkpoint policy
evaluations completed and agreed with VAL. The immutable joined evidence is in
`experiment_tracking/mprime_validation_ipc_scale_v1/validation_test_checkpoint_audit.csv`;
the lineage and pooled summaries are in `validation_test_agreement_summary.csv`.
Every row points to its checkpoint, training log, evaluation job and evaluation
log.

| VH | Pooled checkpoints | Pearson(validation,test) | Spearman(validation,test) |
|---|---:|---:|---:|
| off | 153 | 0.208 | 0.255 |
| on | 137 | 0.278 | 0.252 |

Per-lineage Spearman correlations range from -0.341 to 0.728. The selected
checkpoint trails the best observed test checkpoint by a mean 3.1 plans/off
and 2.5 plans/on. Those regrets are diagnostic only: test performance must not
be used to select a checkpoint.

The corrected set no longer has the old universal epoch-1 failure, but its
ranking agreement is weak. MPrime therefore moves out of the historically
perfect preservation group and into the imperfect mainstream extension (six
imperfect domains, three stable perfect domains). The failed 18/20 preservation
gate remains a recorded outcome. The corrected validation-selected Stage-1
checkpoints still receive the same two-seed, seven-coefficient Stage-2 anchor
tuning used elsewhere.

Controller 20688286 completed successfully and submitted all 28 tuning jobs,
using seeds 1963100312 and 2011206605 in both VH modes. At the 2026-08-29
15:28 snapshot, two jobs were running and 26 were resource-pending. The frozen
manifest and submission ledger are retained beside the corrected Stage-1 data.

### Learning-curve interpretation

The every-five aggregate is stored in
`corrected_learning_curve_aggregate.csv` and rendered as
`corrected_learning_curves.{png,svg}`. It excludes off-grid selected/final
endpoints rather than silently mixing them into regular epochs.

The corrected validation signal is useful at the coarse scale but weak at
fine-grained checkpoint ranking. Validation and held-out test coverage both
generally decline during later training, and validation selection improves
mean test coverage relative to the final checkpoint (15.0 versus 13.5/off;
14.6 versus 13.2/on). Selected epochs are materially earlier than terminal
epochs: medians 18.5 versus 68/off and 13.5 versus 63/on. However, the wide
seed envelope and pooled Spearman correlations near 0.25 show that validation
does not reliably order nearby checkpoints.

The declared treatment is therefore: retain the corrected validation set for
selection and report its weakness; do not call MPrime a stable historically
perfect domain; and run MPrime through the same imperfect-domain Stage-2
anchor/confirmation route as the other five imperfect domains. Test coverage
remains an audit outcome and is never used to select a checkpoint.

## Frozen replacement implementation

The replacement is generated by
`scripts/generate_mprime_validation_ipc_scale_v1.py` into the versioned,
non-overwriting directory
`problems/numeric/mprime/validation_ipc_scale_v1/`.  It contains ten easy,
ten medium and ten hard problems.  The declared food/pain ranges overlap the
IPC test scale, while generation remains independent of network or test
coverage.  `manifest.csv` stores every parameter, realized structural count,
and file checksum; `metadata.json` records the fixed seed and independence
rule.

The generator now accepts an explicit deterministic seed.  Existing callers
that omit it retain the old non-deterministic behavior.

The frozen set must pass planner/parse sanity checks before checkpoint scoring.
After that, all existing Stage-1 MPrime checkpoints are evaluated through
`experiments_numeric.domain.mprime_validation_ipc_scale_v1`.  Whether epoch 1
remains best is an outcome, not a criterion for regenerating the set.  If the
corrected selector chooses a later checkpoint, that fixed choice feeds Stage 2;
if not, the validation design must be reported as still inadequate rather than
tuned against test coverage.
