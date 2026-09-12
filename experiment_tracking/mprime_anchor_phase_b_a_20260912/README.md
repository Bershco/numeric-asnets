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

Corrected dependency chain:

- compute smoke `21222348`;
- full array `21222349[0-27]`, released only after smoke success;
- analysis-only finalizer `21222350`, released only after all 28 tasks succeed.

At the 12 September 18:53 IDT snapshot, the smoke was pending resources and the
other jobs were dependency-pending. No Stage-2 training is launched by this
chain.
