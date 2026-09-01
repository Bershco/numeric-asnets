# Cluster outage recovery plan

## Purpose

This is the recovery procedure for the possible BGU power-loss event beginning
on 1 September 2026. It is intentionally conservative: the first restored
connection is used only to inventory Slurm and filesystem state. No job is
released, cancelled, or submitted until the evidence has been classified.

The last verified pre-outage scheduler snapshot is **2026-09-01 10:37:46 IDT**:

| State | Jobs | CPUs | RAM |
|---|---:|---:|---:|
| Running | 61 | 366 | 4,512 GiB |
| Dependency-pending controllers | 5 | 10 | 5 GiB |
| Operationally held/requeued | 1 | 10 | 20 GiB |
| Deliberately held | 58 | 340 | 4,130 GiB |
| Ordinary pending | 0 | 0 | 0 GiB |

The exact local baseline is
`pre_outage_job_inventory_20260901_1037.csv`. It contains 162 records because
it also includes already-terminal comparison rows that must be checked for
artifact integrity, especially the submitted Drone rows and Rover retry log.

The inventory reconciles exactly to the scheduler roll-up: the 22 individually
known running tails request 132 CPUs and 2,640 GiB; adding the 39 running
PRESERVE-3-TERM jobs (234 CPUs, 1,872 GiB) gives the frozen total of 61 jobs,
366 CPUs, and 4,512 GiB. Its 58 deliberate holds total 340 CPUs and 4,130 GiB.
All 162 job IDs are unique. The only uncertainty intentionally retained is
which individual PRESERVE-3-TERM IDs made up its 21-terminal/39-running split;
the post-outage accounting and logs resolve that rather than inferring it from
submission order.

## What was live at the last snapshot

| Experiment | Pre-outage position | Recovery concern |
|---|---|---|
| PRESERVE-3-TERM Stage-2 training | 21/60 terminal; 39 running | Preserve every latest checkpoint and determine which jobs completed before the outage versus were interrupted. |
| PRESERVE-3-TERM policy/controllers | Five dependency controllers; one Zenotravel/off policy job operationally held | Reconstruct dependency state before releasing or rerunning anything. |
| MCTS-PW70-CROSS-DOMAIN | 12/12 running | Preserve completion JSONL and printed plans; resume only unfinished instances. |
| MCTS-HORIZON-COUNTERS | 8/8 running | Preserve aware/unaware pairing and do not mix arms across code/configuration. |
| MAIN-VAL-S2-MCTS Drone | 19/20 complete overall; job 20726041 was the final running submitted row | Inspect whether it finished; otherwise resume that one row only. |
| MCTS-LEGACY-ROVER | job 20559649 running | Resume only unfinished instances. Job 20559583 separately needs deduplicating post-hoc VAL, not inference repetition. |
| MPrime Stage-2 branches | 36 training jobs plus two controllers deliberately held | They remain held until corrected coefficient selection is complete. |
| Legacy FO Counters MCTS | 20 deliberately held | It remains held unless explicitly reprioritized. |

## First restored connection: read-only sequence

1. Read `cluster_access_and_ssh.md` before connecting. Use the documented
   normal Windows profile and `uni-cluster` alias exclusively. If a connection
   fails, wait at least one minute, reread the access instructions during that
   minute, and check that the profile and alias are correct before retrying.
2. Confirm the login node clock, controller availability, partitions, and node
   health. A successful SSH login alone does not prove Slurm or shared storage
   is healthy.
3. Capture fresh `squeue`, `sacct`, `sinfo`, filesystem, and Git/checkpoint
   evidence before any mutation.
4. Run the read-only audit from the synchronized repository checkout:

   ```text
   python scripts/post_outage_cluster_audit.py
   ```

   It queries every job in the frozen inventory, reads only bounded head/tail
   slices of stdout, checks checkpoint directories and completion JSONL, and
   writes timestamped CSV and Markdown reports under
   `experiment_tracking/post_outage_audit/`.
5. Review its classifications manually. The script has no submission,
   cancellation, release, requeue, or file-edit operation.

## Classification rules

| Observed evidence | Classification | Default recovery |
|---|---|---|
| `COMPLETED` plus final result/checkpoint | Complete | Preserve; trigger only missing downstream work. |
| Training interrupted by `NODE_FAIL`, `BOOT_FAIL`, `REVOKED`, `PREEMPTED`, outage-timed `FAILED`/`CANCELLED`, with a checkpoint | Resumable training | Continue from the latest verified checkpoint. |
| MCTS interrupted with completion JSONL or printed plans | Resumable evaluation | Resume unfinished instances; post-hoc VAL all printed plans. |
| Policy evaluation interrupted | Cheap idempotent retry | Resubmit only the missing endpoint/curve point. |
| Controller interrupted or dependency lost | Idempotent controller retry | Rebuild dependency from terminal training evidence, then rerun controller. |
| `TIMEOUT` or `OUT_OF_MEMORY` with partial MCTS evidence | Fixed-budget terminal result | Keep its conservative score and VAL evidence; resource sensitivity is a separate experiment. |
| `TIMEOUT` training with a valid checkpoint | Declared terminal endpoint | Preserve the last checkpoint under the existing protocol. |
| Deliberately held job | Scientific hold | Keep held during recovery. |
| Missing accounting and missing artifacts | Unknown | Do not guess; inspect filesystem timestamps and scheduler records before a clean rerun. |

An infrastructure outage and an algorithmic/resource failure are not
interchangeable. A job that was already OOM or timed out before the outage stays
a declared terminal result; it is not silently reclassified as outage damage.

## Remaining-time calculation

For every interrupted job, the audit reports:

`remaining = max(0, original Slurm time limit - recorded elapsed time)`

The initial continuation recommendation is `remaining + 1 hour` for teardown
and startup overhead, capped at the original allocation. This is an operational
starting point, not an automatic submission value:

- training continuations additionally report the latest and target epoch;
- resumable MCTS retains the original per-instance timeout and skips completed
  instances from its completion ledger;
- if Slurm accounting was lost, no numeric remaining-time claim is made until
  log timestamps and progress markers reconstruct it;
- a continuation never overwrites the pre-outage log or checkpoint directory.

## Recovery batches after review

1. **Artifact/provenance repairs:** Rover retry-plan VAL and inspection of the
   operational Zenotravel policy tail. These are cheap and do not repeat
   inference.
2. **Controllers and policy evaluations:** rerun only controllers whose
   dependency chain disappeared; materialize missing every-five and endpoint
   policy rows idempotently.
3. **Training continuations:** submit per domain/VH cell from the latest valid
   snapshots, preserving original seeds/configurations and assigning new job
   IDs/log paths.
4. **MCTS continuations:** resume only unfinished instances for PW70, Counters
   Horizon, final Drone, and final Rover jobs; preserve printed plans and run
   post-hoc VAL.
5. **Previously approved new work:** only after the recovered workload and
   filesystem are stable, consider the three-task MPrime validation continuation
   and the fresh causal Horizon determinism audit.
6. **Scientific holds:** MPrime Stage-2 and FO Counters remain held. Recovery
   does not broaden authorization to release them.

## Integrity checks before any continuation

- Verify shared filesystems are mounted read/write and old logs/checkpoints are
  readable.
- Verify the Apptainer image, VAL binaries, PDDL manifests, completion JSONL,
  and selected checkpoints still exist.
- Record the Git commit of each active checkout. Do not make running and resumed
  arms silently use different commits.
- Check whether Slurm accounting times overlap the outage window; do not treat a
  coincidental pre-outage OOM or application failure as infrastructure damage.
- Preserve original stdout paths and checksums. New continuations get new job
  IDs and new logs while linking back to the interrupted job.
- After submission, compare the first resumed instance/checkpoint against its
  pre-outage record before releasing a large continuation batch.

## Locally prepared but not dispatched

The approved recovery ledger is
`approved_recovery_actions_20260901.csv`. At the outage boundary these actions
were implemented/tested locally but not partially submitted: Zenotravel policy
release, TPP timeout-checkpoint fallback, three-task MPrime corrected-validation
continuation, Rover retry-deduplicated VAL, and the causal Horizon determinism
audit. Their dispatch still requires a healthy cluster and current-state check.
