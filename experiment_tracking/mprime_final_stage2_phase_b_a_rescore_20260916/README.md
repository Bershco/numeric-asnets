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

- failed filename-only preflight `21390248` (no evaluation occurred);
- repaired compute preflight `21390403`, completed successfully;
- dependency-gated main array `21390404_[0-19]`, 3 CPUs and 20 GiB per task,
  twenty-four-hour hard limit;
- tasks 1--11 of that array failed before scientific scoring on
  `ise-cpu128-03` because node-local semaphore creation returned ENOSPC;
- exact recovery array `21392547_[1-11]`, with only that node excluded;
- replacement finalizer `21392548`, 1 CPU and 2 GiB, waiting `afterany` on the
  original array and `afterok` on the recovery;
- replacement downstream controller `21392549`, dependency-gated on the new
  finalizer. Obsolete pending jobs `21390429` and `21390981` were cancelled.
  The controller joins selected checkpoint hashes to existing policy logs,
  submits no duplicate endpoint evaluation, and submits only missing final
  fixed/PW identities. Premature provisional aggregates are not inputs.

After finalization, existing policy logs are reused by exact training-job,
epoch and checkpoint-hash identity. Only selected endpoints without valid
policy evidence may be recovered. Fixed/PW search manifests must be rebuilt
from `phase_b_a_selected_endpoints.csv`; the old ready-manifest analysis roles
must never be used for final endpoint selection.

Live snapshot 2026-09-16 15:24 IDT: 128/420 checkpoint evaluations have
durable `.done.json` and `.val.csv` pairs. Twenty workers are active. The
observed aggregate rate since 13:16 is about 41 evaluations/hour, implying an
approximately seven-hour remaining estimate if the rate persists; the
per-task scheduler limit remains the conservative hard bound. Finalizer
`21392548` and downstream controller `21392549` remain dependency-pending.
