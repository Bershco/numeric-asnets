# MPrime anchor ranking on frozen Phase-B replicate A

## Purpose

Phase C selected **Phase-B replicate A** as the final MPrime checkpoint
validator. The existing coefficient-10 decision was made on the superseded
IPC-scale validator, so it cannot be carried forward without checking whether
the seven anchor coefficients retain the same ordering.

This is a validation-only rescore. It reuses the 28 saved anchor-tuning
lineages and performs no training:

- two Stage-1 source seeds (`1963100312`, `2011206605`);
- two value-head modes;
- seven coefficients (`0`, `.03`, `.3`, `1`, `3`, `10`, `30`);
- 21 saved checkpoints per lineage (every fifth epoch plus final);
- 30 frozen Phase-B-A validation problems per checkpoint.

The total is **28 tasks, 588 checkpoint evaluations, and 17,640 individual
problem evaluations**. Test-set scores are never used to choose a coefficient.
The declared ranking is mean validation AUC, then mean peak, then mean final,
then the smaller coefficient only for an exact remaining tie.

## Safety and provenance

`manifest.csv` joins every array task to its original Stage-2 training job and
training log. The frozen validator checksum is stored in every row. The exact
60-row Phase-B manifest and replicate-A domain module were copied from the
completed cluster campaign; the replicate-A subset contains 10 easy, 10 medium
and 10 hard problems with 30 distinct PDDL hashes.

The evaluator writes a done marker only after a full 30-instance final summary,
worker-failure scan, post-hoc VAL pass and nonempty summary CSV. A done marker
is tied to the checkpoint, training job, validator module and frozen-manifest
hash, so a changed identity is rejected rather than silently skipped.

No coefficient finalizer launches training. It emits evidence and a proposed
per-VH freeze, with a mandatory manual curve review before any new MPrime
Stage-2 lineage is submitted.

## Prepared execution

- Compute smoke: one checkpoint from one lineage, **3 CPUs, 20 GiB, 2 hours**.
- Full array: 28 independent tasks, **3 CPUs, 20 GiB and 24 hours each**.
- Maximum simultaneous request without a throttle: **84 CPUs and 560 GiB**.
- Full scientific work: 588 saved-checkpoint evaluations; no retraining.

The smoke is intentionally one real checkpoint over all 30 frozen instances:
it tests the container, compiled TensorFlow operator, checkpoint restoration,
domain module, worker lifecycle, final-summary guard and VAL integration. Once
it succeeds, the full array can be submitted without an artificial concurrency
throttle; Slurm will start tasks as resources permit.

After the full array succeeds, an analysis-only 1-CPU/2-GiB finalizer writes
the 14 coefficient rows and two proposed per-VH winners into the work directory.
It cannot submit training. `submit_mprime_anchor_phase_b_a_20260912.py` records
the smoke, full-array and finalizer job IDs in an idempotent JSON ledger and
gates each step with `afterok`.

## Current execution

Implementation commit `14d095d1` is pushed to `codex/mcts-safe-context` and
deployed file-selectively into the dirty isolated cluster checkout so unrelated
operational artifacts were not overwritten. Local and cluster-side regression
tests passed 3/3; all 28 source logs and all 588 checkpoint directories exist.

The first dependency chain failed safely before evaluation: smoke `21221744`
found that the isolated checkout lacked the 240 tracked frozen-validator PDDL
files. Array `21221745` and finalizer `21221746` were cancelled automatically
by the failed dependency. The missing files were deployed from the pushed,
checksum-controlled branch and counted before replacement submission.

The first corrected chain also failed safely before evaluation: smoke
`21222348` found that the file-selective deployment lacked the frozen validator
Python module. Array `21222349` and finalizer `21222350` were automatically
cancelled. The module was then deployed and its SHA-256 verified against the
local branch.

The corrected compute smoke `21223398` passed and released full array
`21223399[0-27]`. Some task attempts then completed while others failed or were
held, so an exact paired-marker audit created six resumable recovery tasks as
`21233926[0-3,15,24]`. The bounded idempotent recheck controller is `21233927`;
it remains dependency-pending and will submit only still-incomplete lineage
indices before launching an analysis-only finalizer.

At the 14 September 12:24 IDT snapshot, 519/588 checkpoint evaluations were
complete. All original task attempts had left the queue. Five resumable recovery
tasks were running, requesting 20 CPUs and 100 GiB, with about 24 minutes to
their 24-hour hard bounds. Recheck controller `21233927` remained
dependency-pending and will audit and resubmit only identities still missing
after those tasks.
No Stage-2 training can be launched by this chain: its proposed coefficient
winners require manual curve review first.

At 12:48 IDT those five allocations had left the queue and the durable count
was 520/588. This was only a short controller transition, not an unhandled
failure or a Slurm requeue. Controller `21233927` audited the paired done
markers and submitted successor array `21254779` for the 17 lineage indices
that were still incomplete; analysis-only controller `21254780` is held by its
declared dependency until that array terminates. By the 14 September 15:01 IDT
audit, 15 successor tasks were visible as running and the durable count had
advanced to 534/588. The successor is idempotent: each task skips every valid
checkpoint result already present and evaluates only missing identities.

The Stage-2 reuse question is separate from the Stage-1 MCTS reuse audit. An
exact epoch join shows that 16/20 old validation-led Stage-2 lineages definitely
start from a different Stage-1 checkpoint than the final Phase-B-A selection.
Four initially appeared to have the same selected epoch: off/1239739722,
off/1472491096, on/1963100312 and on/2082152039. That epoch match alone does
not establish reusable Stage-2 training: reuse additionally requires the exact
Stage-1 checkpoint hash, the eventual frozen coefficient, code and full
training configuration to match. The final reuse decision is therefore made
only after the coefficient curves are complete and reviewed. Stage-1 MCTS is
a separate result identity: none of its historical runs used the final
Phase-B-A-selected checkpoint, so all 20 exact Stage-1 MCTS evaluations were
required even though some old Stage-2 training lineages may still prove
reusable.

At 14 September 23:19 IDT, 587/588 checkpoint evaluations were durable. Only
task `21254779_7` remained running, requesting 4 CPUs and 20 GiB with about
14 hours left on its 24-hour hard allocation; analysis-only controller
`21254780` remained dependency-pending. One point is too little to justify an
early coefficient freeze: the complete curves still require manual review
before any Stage-2 reuse or retraining decision.

At 23:30 IDT the final point completed and controller `21254780` finalized the
full 588/588 evidence set. The declared rule selects **anchor 30 for VH-off**
and **anchor 10 for VH-on**. Manual review found no completeness or parsing
anomaly. VH-off anchor 30 narrowly exceeds anchor 10 on AUC in both tuning
seeds; VH-on anchor 10 has the best aggregate AUC, peak and final score,
although the two individual seeds do not prefer it uniformly over anchor 30.
The coefficients are therefore frozen under the predeclared selection rule,
not because of an exact tie or a smallest-coefficient fallback. Local copies
with remote provenance are `anchor_evidence_final.csv` and
`proposed_anchor_freeze_final.csv`.
