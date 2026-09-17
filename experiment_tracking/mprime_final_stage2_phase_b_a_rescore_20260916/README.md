# Final MPrime Stage-2 Phase-B-A endpoint rescore

The twenty final Stage-2 training lineages are valid and complete, but their
ordinary internal validation set saturated. It is now a legacy,
non-authoritative provenance field only. **Phase-B-A is the sole canonical
MPrime validation set** for selecting checkpoints and reporting validation
results. The earlier endpoint materializer selected epoch 0 from the wrong
selector and is excluded from all final MPrime RQs, tables and plots.

The validator error affected selection, not optimization: validation scores
were not used in the Stage-2 loss and did not change network weights. Therefore
the defensible repair is to rescore every saved checkpoint once, not retrain
twenty lineages. Future MPrime training must evaluate Phase-B-A inline so this
420-evaluation bridge is never needed again.

This repair evaluates all 420 saved checkpoints (twenty lineages x twenty-one
epochs: 0, 5, ..., 95, 99) on the frozen thirty-instance Phase-B replicate A.
It does not retrain any network and does not use test scores for selection.
Each lineage selects maximum Phase-B-A coverage with the earliest epoch as the
predeclared tie-break.

Safety/provenance gates:

- exact twenty-lineage/420-checkpoint ready-manifest checksum;
- frozen validator-manifest and validator-module checksums;
- checksum of every validator PDDL and checkpoint weights file;
- exact evaluator checkout commit;
- exact expected epoch set;
- idempotent per-checkpoint completion records;
- finalizer refuses fewer or more than twenty-one results per lineage.

Slurm route:

- failed filename-only preflight `21390248` (no evaluation occurred) and
  repaired compute preflight `21390403`;
- original lineage arrays `21390404`/`21392547` produced 174 exact
  summary+completion-identity pairs, then were cancelled before acceleration;
- missing-key manifest
  `missing_checkpoint_manifest_20260916_164528.csv` freezes the remaining 246
  unique lineage/epoch identities; SHA-256
  `f2732949116b14625707966f46c29e91e9ca6782f9f754957d60554d46f55d42`;
- one-key smoke `21411412_0` validates the new runner;
- fine array `21411413_[0-245%246]` depends on the smoke and runs one output
  identity per task, 3 requested CPUs and 20 GiB, eight-hour hard limit, with
  `ise-cpu128-03` excluded;
- its 246-task cap requests 4,920 GiB and left about 964 GiB headroom under the
  6-TiB user envelope at submission;
- original task `21411413_100` failed after 1m51s on `ise-cpu-intl-01` when its
  native evaluator exited with code -4 before producing a score. Exact retry
  `21412365_100` repeatedly entered an environment-retrieval hold before the
  evaluator started. It was superseded without scientific output by
  environment-independent exact retry `21414946_100`, which reruns only that
  frozen manifest identity using `--export=NIL` and excludes the failed node
  families;
- superseded finalizer `21412366` was released after `21414946`, then failed on
  the incorrect assumption that every VAL summary must contain thirty data
  rows.  Its downstream `21412367` was cancelled and produced no scientific
  work.  Corrected finalization is recorded below;
- pending jobs `21411414` and `21411415` were cancelled before execution
  because their original `afterok:21411413` path could never release after
  task 100 failed.

The replaced arrays/controllers `21390404`, `21392547`, `21392548`,
`21392549`, `21412366` and `21412367`, plus obsolete infinite policy watcher
`21362500`, were cancelled or failed and superseded. Their evidence/logs were
retained. The canonical downstream controller joins selected
checkpoint hashes to existing policy logs and exact search identities. If a
selected endpoint lacks valid policy evidence, it submits only that exact
endpoint into a dedicated date-frozen recovery root, records a recovery
ledger, and schedules one dependent continuation. It fails closed if the
recovered evidence is still absent, so search cannot start from an unverified
endpoint or loop indefinitely.

The canonical replacement-chain submission ledger is
`parallel_replacement_20260916_164528.tsv`. The older `submissions.tsv` is
retained only as historical provenance for the superseded lineage-level chain.
The exact task-100 repair and replacement dependencies are recorded in
`task100_repair_chain_20260916.tsv`.

After finalization, existing policy logs are reused by exact training-job,
epoch and checkpoint-hash identity. Only selected endpoints without valid
policy evidence may be recovered. Fixed/PW search manifests must be rebuilt
from `phase_b_a_selected_endpoints.csv`; the old ready-manifest analysis roles
must never be used for final endpoint selection.

## Finalization correction and live handoff

All 420 checkpoint evaluations are complete.  A Phase-B-A VAL summary CSV has
one row per successful, VAL-valid plan rather than one row per validation
problem; the authoritative coverage is the unique terminal
`[EVAL FINAL] success=X/30` record.  The superseded finalizer incorrectly
required thirty CSV rows and failed despite valid evidence.  A diagnostic
missing-manifest produced under that false assumption is retained only as an
invalid operational artifact and is excluded from scientific provenance.

The corrected completion gate requires exactly one terminal `X/30` record,
exactly `X` VAL-valid CSV rows, the expected receipt/checkpoint identity and
the complete epoch set.  Corrected finalizer `21428590` and first downstream
controller `21428591` completed on 2026-09-17.  Twenty Phase-B-A-selected
endpoints are frozen in `phase_b_a_selected_endpoints.csv`.

Local immutable copies for rapid analysis are
`phase_b_a_selected_endpoints_20260917.csv` and
`selected_policy_recovery_manifest_20260917.csv`; both retain the remote
checkpoint and log paths used by the controller.

Exact-hash reconciliation found twelve already evaluated selected policy
endpoints and eight genuinely missing VH-on endpoints.  Jobs
`21428604`--`21428611` evaluate only those eight (selected epochs 10, 40, 30,
5, 15, 20, 99 and 10).  Controller `21428612` depends on all eight with
`afterany`, validates their outputs, then materializes the matched fixed 20/70
and PW70 search manifests.  At the 2026-09-17 10:35 IDT snapshot the eight
policy jobs were running; fixed and PW search were not yet running.

No validation score, completed policy endpoint or search identity is blindly
duplicated.  The corrected scripts interpret the actual successes-only VAL
format and fail closed on any remaining missing endpoint.
