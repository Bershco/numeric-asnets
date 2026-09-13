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

At the 14 September 00:21 IDT snapshot, 469/588 checkpoint evaluations were
complete. All original task attempts had left the queue. Six resumable recovery
tasks were running, requesting 24 CPUs and 120 GiB, with about 12h26m to their
24-hour hard bounds. Recheck controller `21233927` remained dependency-pending
and will audit and resubmit only identities still missing after those six tasks.
No Stage-2 training can be launched by this chain: its proposed coefficient
winners require manual curve review first.

The Stage-2 reuse question is separate from the Stage-1 MCTS reuse audit. An
exact epoch join shows that 16/20 old validation-led Stage-2 lineages definitely
start from a different Stage-1 checkpoint than the final Phase-B-A selection.
Four are only candidates for reuse: off/1239739722, off/1472491096,
on/1963100312 and on/2082152039. They remain reusable only if checkpoint hash,
the eventual frozen coefficient, code and full training configuration also
match. Consequently the defensible fresh-training count is currently 16--20,
not a proven 20.
