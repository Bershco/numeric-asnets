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
  `21412365_100` reruns only that frozen manifest identity and excludes that
  node plus the already excluded `ise-cpu128-03`;
- replacement finalizer `21412366` waits for `afterany:21411413` and
  `afterok:21412365`. It still refuses any missing identity, mismatched receipt,
  non-binary VAL row, or anything other than twenty-one results per lineage;
- replacement downstream controller `21412367` depends only on
  `afterok:21412366`. Pending jobs `21411414` and `21411415` were cancelled
  before execution because their original `afterok:21411413` path could never
  release after task 100 failed.

The replaced arrays/controllers `21390404`, `21392547`, `21392548` and
`21392549`, plus obsolete infinite policy watcher `21362500`, were cancelled.
Their evidence/logs were retained. The downstream controller joins selected
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

Live snapshot 2026-09-16 17:36 IDT: 174/420 checkpoint evaluations were
identity-valid at the replacement freeze. Smoke `21411412_0` completed and
released the array. Of the 246 missing identities, 245 original tasks plus the
one exact task-100 retry are running concurrently. The finalizer and downstream
controller are dependency-pending. Historical service time averages about 29
minutes per checkpoint, but the parallel makespan is determined by the slowest
task; the hard scientific bound is eight hours per fine task once allocated.
