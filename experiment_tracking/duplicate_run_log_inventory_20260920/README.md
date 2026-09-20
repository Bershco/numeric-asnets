# Duplicate launcher-log inventory campaign

This campaign measures how much cluster storage could be reclaimed by removing
the duplicate `runs/<md5(command)>` side of successful ASNet launcher runs while
retaining the scientifically consumed `<output>/run-info` copy. It is strictly
inventory-only: none of its scripts contain a deletion operation.

## Scope and safety contract

The scan root is the complete cluster `asnets/experiment-results` tree. The
campaign covers outputs produced through `run_experiment.py`, `run_learning.py`
and `run_planning.py` by enumerating every `run-info` directory and the
immediate `runs/<digest>` children of every experiment-prefix `runs` directory.

A run directory is counted as reclaimable only when all of the following hold:

1. the MD5 of the exact `run-info/cmdline` bytes names the corresponding
   `runs/<digest>` directory;
2. both trees contain the same relative entries and entry types;
3. file sizes match;
4. every file is stream-read and SHA-256 checked byte-for-byte, including at
   least `cmdline`, `stdout` and `stderr`;
5. when `stdout` exposes `Unique prefix:`, that path resolves to an exactly
   matched `run-info` candidate;
6. no symlink, special file, read failure or scan exception weakens the proof.

The reported allocated-byte total includes filesystem block allocation for the
whole duplicate run tree. Missing counterparts, incomplete/failed launcher
runs, content mismatches and orphan run directories are retained in the
exception inventory and contribute zero reclaimable bytes.

Even a listed candidate is only a snapshot. Any future deletion campaign must
be separately approved and must repeat the same validation immediately before
removal.

## Scalable execution

The preparation job performs one GNU `find` traversal and writes stable JSONL
shards. It does not use recursive Python globbing over NFS. Each array task
processes only its shard and appends one fsynced result per digest. A restarted
task skips durable results, and the summarizer deduplicates by digest.

Recommended initial shape:

| Stage | Slurm shape | Resources | Purpose |
|---|---|---|---|
| Prepare | 1 job | 1 CPU, 2 GiB, 4h | NUL-safe discovery and 32 shard manifests |
| Compare | 32-task array, at most 4 concurrent | 1 CPU, 4 GiB, 12h/task | Full-tree streaming comparisons without an NFS metadata storm |
| Summarize | 1 job after the array | 1 CPU, 2 GiB, 1h | Exact totals, candidate CSV and exception JSONL |

The four-task concurrency limit is an intentional shared-filesystem I/O safety
measure, not a compute-capacity restriction.

## Prepared submission plan

No cluster job was submitted while developing this campaign. After an exact
production checkout and campaign directory are chosen, the dependency chain is:

```bash
REPO_ROOT=/home/hersco/bershco-nu-asnets/numeric-asnets
SCAN_ROOT="$REPO_ROOT/asnets/experiment-results"
CAMPAIGN_DIR="$HOME/quota_recovery/duplicate_run_log_inventory_20260920"
EXCLUDE='ise-cpu-intl-[01,05,15,18,24,26]'

prep=$(sbatch --parsable --exclude="$EXCLUDE" \
  --export=ALL,REPO_ROOT="$REPO_ROOT",SCAN_ROOT="$SCAN_ROOT",CAMPAIGN_DIR="$CAMPAIGN_DIR",SHARD_COUNT=32 \
  "$REPO_ROOT/scripts/duplicate_run_log_inventory_prepare.sbatch")

scan=$(sbatch --parsable --exclude="$EXCLUDE" --dependency="afterok:$prep" \
  --array=0-31%4 \
  --export=ALL,REPO_ROOT="$REPO_ROOT",CAMPAIGN_DIR="$CAMPAIGN_DIR" \
  "$REPO_ROOT/scripts/duplicate_run_log_inventory_scan.sbatch")

summary=$(sbatch --parsable --exclude="$EXCLUDE" --dependency="afterany:$scan" \
  --export=ALL,REPO_ROOT="$REPO_ROOT",CAMPAIGN_DIR="$CAMPAIGN_DIR" \
  "$REPO_ROOT/scripts/duplicate_run_log_inventory_summarize.sbatch")
```

The final artifacts are `summary.md`, `summary.json`,
`deletion_candidates.csv`, `exceptions.jsonl`, the immutable shard manifests
and resumable per-shard result ledgers.

## Submitted inventory

The inventory-only chain was submitted on 20 September 2026:

- preparation `21479642`;
- comparison array `21479643_[0-31]%4`;
- summary `21479644`.

The preparation allocation experienced a Slurm user-environment hold and was
released after its exclusions were verified. Its no-requeue replacement
`21480090` then failed closed because the first attempt had already made the
campaign directory nonempty; no source tree was changed. Attempt 2 exposed a
submission-wrapper bug: placing the Slurm log directory inside the fail-closed
campaign root made the root nonempty before preparation began. The canonical
attempt therefore keeps scheduler logs in a sibling directory and scientific
inventory state under
`/home/hersco/quota_recovery/duplicate_run_log_inventory_20260920_attempt3`.
This campaign contains no deletion operation. Any later removal still requires
a separate exact-candidate review and fresh approval.

The canonical inventory-only chain is attempt 3:

- preparation `21480550`;
- 32-shard comparison array `21480552_[0-31]%4`;
- summary `21480554`.

Its scientific root is
`/home/hersco/quota_recovery/duplicate_run_log_inventory_20260920_attempt3`;
scheduler logs live in a sibling directory so they cannot violate the
fail-closed empty-root precondition.  Earlier attempts are noncanonical and
cannot contribute candidates or totals.
