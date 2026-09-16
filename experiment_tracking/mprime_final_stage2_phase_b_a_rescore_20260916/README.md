# Final MPrime Stage-2 Phase-B-A endpoint rescore

The twenty final Stage-2 training lineages are valid and complete, but their
ordinary internal validation set saturated. The earlier endpoint materializer
therefore selected epoch 0 from the wrong selector and must not define the final
MPrime Stage-2 RQs.

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
- repaired compute preflight `21390403`;
- dependency-gated main array `21390404_[0-19]`, 3 CPUs and 20 GiB per task,
  twenty-four-hour hard limit;
- dependency-gated finalizer `21390429`, 1 CPU and 2 GiB.
- fail-closed downstream controller `21390981`, dependency-gated on the
  finalizer. It joins selected checkpoint hashes to existing policy logs,
  submits no duplicate endpoint evaluation, audits the quarantined epoch-0
  searches for exact hash/configuration reuse, and submits only missing final
  fixed/PW identities.

After finalization, existing policy logs are reused by exact training-job,
epoch and checkpoint-hash identity. Only selected endpoints without valid
policy evidence may be recovered. Fixed/PW search manifests must be rebuilt
from `phase_b_a_selected_endpoints.csv`; the old ready-manifest analysis roles
must never be used for final endpoint selection.
