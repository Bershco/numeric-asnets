# Cluster storage audit — 2026-08-26

Authoritative source job: `20595700` (completed in 34m42s).
Fine-grained follow-up: `20598529` (completed in 46m02s).

The original audit artifacts in this directory were copied verbatim from:

`/home/hersco/training_new_domains/2026-08-26/quota_recovery/`

## Completed cleanup

- Deleted only `/home/hersco/.local/lib/python3.9/site-packages` after resolving
  the exact path, confirming it was a directory, confirming no user process
  referenced Python 3.9 or that path, and confirming the production Python 3.10
  virtual environment had user-site loading disabled.
- Recreated the now-empty directory.
- Reclaimed approximately 6.8 GiB. The removed packages are recoverable only by
  reinstalling them.
- Deleted `/home/hersco/original-nu-asnets` after confirming that no live Slurm
  job or user process referenced it.  It occupied 10,243,448,832 bytes and can
  be recovered by cloning `ryanxwang/numeric-asnets` at commit
  `b7d015396f8258adc953657c1773b7c7ee2f8044`.
  This is the exact historical checkout that was deleted; it must not be
  described as the current upstream `main` HEAD unless that is independently
  verified at restoration time.
- Earlier quota recovery also removed approximately 3.4 GB of pip cache and
  2.8 GB of JetBrains cache.  Total reclaimed across these four approved
  cleanups is approximately 23.7 GB (decimal; about 22.1 GiB).

## Confirmed candidates — not deleted
- Giant Counters logs: several 1.3–2.8 GB wrapper logs have similarly sized
  internal `runs/.../stdout` copies. Preserve the wrapper log and its VAL result;
  do not remove the internal copy until its exact relationship and provenance
  pointer are recorded.
- Reproducibility clones/staging: retain at least one complete tested bundle and
  its LFS object; duplicated clones and staging archives require a checksum-based
  consolidation plan.

## Giant-log diagnosis

Bounded head/tail evidence is saved in `large_counters_bounded_audit.txt`.
The June Counters Stage-2/MCTS-training logs and recent narrow-MCTS logs ran with
verbose MCTS debugging. They repeatedly printed root child/parent consistency
tables and incoming-parent details. The fine-grained audit counts those repeated
blocks without modifying the logs.

No experiment output, checkpoint, problem-generator artifact, or
reproducibility bundle has been deleted by this audit.  The only historical
repository removed was the explicitly approved, reproducible upstream clone
recorded under Completed cleanup.

## Fine-grained findings

The production repository occupies about 265.8 GB, of which about 245.7 GB is
under `asnets/experiment-results`:

- snapshots: 179.1 GB across about 42,394 directories;
- run directories: 40.6 GB across about 5,783 directories;
- run-info directories: 19.9 GB across about 36,394 directories; and
- TensorBoard data: 4.3 GB across about 4,069 directories.

The largest domain aggregates are Counters (76.5 GB), Rover (58.0 GB), Drone
(53.0 GB), FO Counters (19.0 GB), Delivery (15.9 GB), and Block Grouping
(11.1 GB).  These figures are directory-storage measurements, not experimental
scores.

The problem generator occupies almost exactly 12.0 GB; 11.986 GB is under its
Counters subtree and 11.986 GB is generated instance material. These are the
realtime-generated training artifacts from the abandoned dynamic-instance
experiment. The user explicitly approved deleting the generated Counters
instances without preserving their seeds, parameters, or checksums; the
generator source itself must remain untouched. The approved exact target is
`/home/hersco/bershco-nu-asnets/numeric-asnets/problem_generator/counters/instances`.

The largest Counters logs are pathological debug outputs.  For example, job
20278848 contains 405,894 repeated root-consistency blocks and 9,459,748
incoming-parent lines.  Its 2,810,086,794-byte internal stdout appears twice
with identical size and timestamp, while the Slurm wrapper is a third near-copy.
Other giant logs contain roughly 253,000--384,000 root blocks and 4.8--10.8
million incoming-parent lines each.

Do not regex-edit those raw logs in place. There are two valid, mutually
exclusive retention policies:

1. **Raw-preservation policy:** keep one compressed raw copy and delete only
   checksum-proven duplicates. A full extracted replacement is unnecessary;
   only the retained path/checksum and duplicate mapping are required.
2. **Compact-evidence policy:** extract configuration, checkpoint,
   per-instance outcomes/runtimes, plans, and VAL evidence, verify the compact
   record, and then delete every raw copy.

For selected scientific results, use raw preservation. For redundant debug
campaigns that are not selected results, compact evidence may be used. Never
mix the policies by keeping a raw copy *and* requiring a full extraction merely
to justify duplicate removal.

The recent giant logs are not an active default behavior. Jobs `20278848`
(2.81 GB, last modified 2026-08-19) and `20278849` (2.02 GB, last modified
2026-08-20) were Counters VH-on narrow MCTS evaluations with width 5 and 20/70
simulations respectively. They explicitly ran with `puct_debug=True` and
`action_debug=True`, causing hundreds of thousands of full root-consistency
tables and millions of incoming-parent lines. Current production manifests
must keep both debug flags disabled; compact profiling added for progressive
widening is separate and does not print these tables.

When the compact-evidence policy is selected, the procedure is:

1. extract configuration, checkpoint, per-instance outcome/runtime, plan, and
   VAL status into compact immutable records;
2. record paths and SHA-256 checksums for every original copy;
3. prove which internal stdout files are exact duplicates;
4. retain one provenance-bearing raw copy, preferably compressed; and
5. remove only verified duplicates after the compact record is audited.

This should reclaim tens of gigabytes without destroying experimental evidence.

## Quota accounting limitation

The cluster quota client currently fails to contact its RPC quota service, so
the exact per-user quota ceiling and exact bytes remaining are not available.
`df` reports 170 TB free on the shared `/home` filesystem, but that is not the
user quota and must not be presented as personal free space.  The bounded audit
placed the pre-cleanup tracked footprint around 337 GB and the post-cleanup
tracked footprint around 313--314 GB; these are audit estimates, not an
authoritative quota reading.
