# Cluster home-quota recovery — 19 September 2026

The cluster stopped accepting new job environments because the user's home
quota was exhausted. This was an operational storage failure, not an
experimental failure. Cleanup was limited to material that was either copied
and verified locally first, byte-identical to a retained remote copy, or a
provably unused duplicate software environment.

## Reclaimed space

| Class | Safety proof | Reclaimed bytes |
|---|---|---:|
| Eight KL first-update raw batch directories | copied to the local thesis archive; SCP completed; 21,262-file count and 149,438,359 logical bytes matched | 716,770,304 |
| Duplicate unused `venv-asnets` | no queued/running job or process referenced it; the production environment remains | 5,830,058,496 |
| Seven duplicate `run-info/stdout` files | every deleted file passed `cmp -s` against its retained `runs/<uuid>/stdout` copy | 7,966,117,691 |
| **Total** | only verified-safe targets | **14,512,946,491 bytes (13.516 GiB)** |

The unique KL payload is retained losslessly at
`C:/Users/roeeh/Desktop/School/Meitar/Thesis/cluster_storage_archive/2026-09-19/kl_first_update_batches`.
Its aggregate transfer record is:

- files: 21,262;
- logical bytes: 149,438,359;
- aggregate archive digest recorded at transfer:
  `999434bee1acd71bba2d03cf1ff8fd212805056d808bdb56cf1dcebe26dcbeeb`.

The deleted stdout copies and their retained counterparts are listed in
`deleted_items.csv`. The original storage census remains in
`experiment_tracking/storage_audit_20260826/storage_audit_summary_20595700.txt`.

## Restart and verification

After cleanup:

- all eight interrupted endogenous-KL training arms resumed from their latest
  durable checkpoints with 24-hour allocations;
- the incremental policy-evaluation watcher was repaired so unfinished
  identities use the deployed evaluator commit while historical completed
  results keep their original provenance;
- the value-head V1 source recoveries resumed;
- all 590 Counters Action-ID classifications were reconciled and the approved
  172-task full-root trace campaign was released with at most 12 tasks running
  concurrently;
- the final unresolved MCTS-divergence identity was submitted as an exact
  recovery.

Fresh logs were scanned after restart. No new `Disk quota exceeded`, `No space
left`, Slurm user-environment retrieval, traceback, OOM, or killed-task
signature was present. The queue contained 208 running/pending/dependency
tasks, safely below the 2,000-task ceiling.

## Deliberately retained

Checkpoints, compact gradient summaries, hashes, manifests, completion
ledgers, policy results, unique logs, and scientific result files remain on
the cluster. No ambiguous candidate was removed. Additional large unique
artifacts must be archived and verified locally before any future deletion.
