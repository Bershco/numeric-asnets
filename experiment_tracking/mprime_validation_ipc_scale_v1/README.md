# MPrime corrected-validation experiment

The original generated validation set selected epoch 1 for all twenty MPrime
Stage-1 lineages and was structurally much smaller than the IPC test set.  The
replacement protocol is fixed before checkpoint scoring and does not use test
coverage for generation or selection.

## Frozen design

- 30 deterministic instances: ten easy, ten medium, ten hard.
- Food and pain ranges overlap the IPC test scale.
- Fixed generator seeds and SHA-256 checksums are recorded in the generated
  set's `manifest.csv` and `metadata.json`.
- Goals must be absent initially and every generated instance must pass a
  planner sanity check with a strictly positive plan length.
- `lineage_manifest.tsv` points to the twenty existing Stage-1 lineages whose
  checkpoints will be rescored after the sanity gate passes.

## Current state

The first replacement set exposed a generator bug: all goals were already true
and all 30 planner plans had length zero.  That set was rejected before any
selection decision.  The generator was corrected, the 30-instance set was
regenerated, and every file passed the planner sanity gate with a non-zero plan.

The corrected checkpoint rescore then exposed two deployment-only issues:

1. the production validator did not yet support `--summary-csv`; and
2. a temporary validator copied outside the repository inferred the wrong
   domain-module root from its own location.

The validator was placed under the production repository's `asnets/tools`
directory and the corrected MPrime module was changed to an absolute package
import.  The fixed rescore campaign is resumable: it reuses any complete policy
evaluation log and performs only the missing VAL summary.

The 20/20 launch gate is now satisfied without consulting test coverage. In
all twenty old lineages, a later checkpoint beats epoch 1 on the corrected
validation set. The last two lineages (seed 1073581256) score 6/30 and 5/30 at
epoch 1, while later checkpoints reach 26/30 and 23/30. This directly refutes
the old validation set's universal epoch-1 selection.

The fresh 20-lineage Stage-1 run (ten seeds, VH off/on) using this exact frozen
validation module was submitted by controller `20618715`, which completed with
exit `0:0`. Training jobs `20618716`--`20618735` were all running at the first
post-submission check. Their downstream contract is every available epoch
divisible by five plus validation-selected/final policy endpoints and both
validation-selected/final Stage-1 MCTS endpoints.

## Stage-2 mainstream branches

MPrime now follows both mainstream checkpoint-selection branches:

- `MAIN-EXT6-MPRIME`: Stage 2 begins from the corrected validation-selected
  Stage-1 checkpoint.
- `MAIN-TERM-EXT6-MPRIME`: Stage 2 begins from the corrected final Stage-1
  checkpoint.

The current 28-lineage anchor grid is validation-led and selects one coefficient
per VH mode.  The terminal-led branch deliberately reuses those frozen
coefficients rather than repeating the full tuning grid.  Its twenty corrected
final checkpoints are resolved in `terminal_stage2_planned_manifest.csv`, but
the rows remain `blocked_anchor_freeze` until tuning completes.  Both branches
require every-five Stage-2 policy curves, validation-selected/final policy
endpoints, and validation-selected Stage-2 MCTS; Stage-2-final MCTS remains
excluded.

At the 2026-08-31 00:13 IDT snapshot, 27/28 anchor-tuning lineages are
terminal.  The last lineage, job `20688852`, is at epoch 87/100 with an internal
ETA of roughly 2h36m.  The old all-coefficient policy controller `20743983` and
73 policy jobs that had not started were cancelled: external test-policy curves
are not inputs to coefficient selection, so evaluating every checkpoint for
every losing coefficient was redundant.  Thirty already-running policy jobs
were left undisturbed and remain useful evidence.

Coefficient selection uses validation measurements already written by the 28
tuning-training logs: mean validation AUC across the two tuning seeds, then mean
peak and final validation coverage as tie-breakers. It does not wait for
test-policy evaluations and never uses test coverage to choose a coefficient.
The policy-curve controller is an analysis/provenance pipeline only.  After
anchor selection, full policy learning curves and endpoints remain required for
the winning tuning lineages and both mainstream confirmation branches.

Automatic finalizer job `20755820` is dependency-gated on the final tuning
lineage.  It verifies all 28 tuning logs, freezes one coefficient per VH mode by
validation AUC/peak/final, reuses the four winning tuning lineages, submits the
sixteen held-out validation-led lineages, and submits twenty separate
terminal-led lineages.  Its submission ledger is idempotent, so a partial
failure can be retried without duplicating already recorded jobs.  It does not
use test-policy results for tuning.

## Invalid first Stage-2 freeze and corrected rescore — 31 August 2026

Finalizer `20755820` did complete and initially froze coefficient **0** for both
VH modes, but that freeze is invalid.  All seven coefficients tied exactly at
validation AUC, peak and final coverage because every one of the 28 training
commands referenced the obsolete `valid_easy`, `valid_medium` and `valid_hard`
paths.  Those files were the earlier goals-already-true set: the tuning logs
report validation coverage 1.0 and average plan length 0.0 from iteration zero.
Coefficient zero won only through the predefined smallest-coefficient tie-break.

No training rerun is needed to correct coefficient selection.  The 28 tuning
lineages already saved every-five and final checkpoints.  The corrected
campaign evaluates 21 checkpoints per lineage—588 total—on the frozen
`mprime_validation_ipc_scale_v1` set, then applies the same two-seed mean AUC,
peak and final tie-break contract used for the other domains:

- preflight array task: `20768884_[0]` (one checkpoint only);
- full resumable 28-task array: `20768902_[0-27]`, dependent on the preflight;
- batch implementation: `scripts/mprime_anchor_corrected_validation_rescore.sbatch`.

### Rescore harness defect found at first execution

Preflight `20768884` was a false pass.  The lifecycle refactor changed the
worker entry point to send results over a one-way Pipe, while the legacy
difficulty-wave evaluator still supplied a multiprocessing Queue.  Every
worker therefore raised `AttributeError: 'Queue' object has no attribute
'send'`.  The validator accepted the resulting zero-plan log and the wrapper
wrote an invalid `.done` marker.  Full-array task `20768902_0` reproduced the
same defect; no score from either job is evidence.

The local repair supports both Pipe `send()` and legacy Queue `put()`, adds a
regression test, and makes the rescore wrapper reject worker-crash markers and
require a nonempty summary before writing `.done`.  The array must be recreated
from that repaired checkout and the false marker removed.  The remaining saved
checkpoints are intact; no Stage-2 retraining is implied by this harness bug.

The originally materialized Stage-2 jobs remain preserved but explicitly held:

- validation-led held-out jobs `20760292--20760307`;
- terminal-led jobs `20760308--20760327`;
- policy controllers `20760692` and `20760693`.

They will be released only after corrected validation freezes one coefficient
per VH.  The terminal-led branch will reuse that corrected validation-led
coefficient; it does not need another 28-lineage tuning grid.
