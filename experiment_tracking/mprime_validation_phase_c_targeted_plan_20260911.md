# MPrime validation Phase C — targeted submitted design

Status: **submitted behind a planner/VAL construction gate on 11 September 2026**.

Phase B proved two things: the original validation set was saturated, and two
harder independent replicates remove saturation but still rank Stage-2
checkpoints weakly. Their consensus selector is not consistently better than
either replicate, so simply drawing a third set from the same distribution and
rescoring all 1,130 checkpoints is not justified.

If MPrime remains in scope, Phase C should:

1. Freeze one structurally redesigned validation set before any network scoring.
2. Match the fixed test suite's support more closely without copying test files:
   goal distances, graph/object-size tails, initial numeric-value regimes and
   absence of trivial direct witnesses.
3. For each of the 40 primary lineages (Stage 1 and validation-led Stage 2,
   two VH modes, ten seeds), evaluate only the union of the five highest-ranked
   checkpoints from replicate A, five from replicate B, the final checkpoint
   and the original baseline. Deduplicate this union; hard cap 12 candidates.
4. Within the Phase-C set, select checkpoints using a rule frozen before seeing
   Phase-C or test scores. Test coverage cannot choose among checkpoints. It is
   used only afterward to compare whole validation designs; choosing the final
   design this way is explicitly test-informed and will be disclosed as such.

The frozen checkpoint-candidate union contains 347 actual evaluations across
40 lineage jobs (6–12 candidates per lineage), below the predeclared maximum
of 480. Each inference task requests 3 CPUs and 20 GiB, so all 40 can use at
most 120 requested CPUs and 800 GiB concurrently. This Slurm installation
rounds the running allocation to four CPUs on some nodes, so the observed
allocation ceiling may be 160 CPUs while the request remains three per task.
Each has a conservative 24-hour
allocation; there is deliberately no artificial array concurrency throttle.

The corrected three-way planner/VAL candidate screen is Slurm array `21183533`
(2 CPUs, 8 GiB and 4 hours per tier). Dependent finalizer `21183534` freezes ten
certified instances per easy/medium/hard tier, writes checksums, generates the
isolated experiment module, runs one compute-node preflight, and only then
submits the 40-lineage rescore array. The deployed checkpoint manifest contains
no test-policy-score column. Phase C is nevertheless **test-support-informed**:
its structural ranges were deliberately chosen from aggregate fixed-test-suite
support. It is therefore a transductive validation-design audit, not a pristine
independent validation experiment.

The initial screen `21183472[0-2]` failed before planner execution because the
candidate hashes described pre-write LF text while Windows had durably written
CRLF bytes. No scientific work was lost. The generator now hashes post-write
bytes. Screen `21183496[0-2]` started during the slow replacement transfer,
before the corrected manifest landed, and repeated the same pre-planning
failure. Remote checksums were verified before `21183533` was submitted.

All three corrected tier screens completed and certified 10/10. Finalizer
`21183534` froze the set. Compute preflight `21183541` completed successfully
in 1:21:45, releasing the 40-lineage array. At the targeted 11 September
18:48 IDT check, all forty tasks in `21183542[0-39]` were running.

Decision after Phase C: choose the most defensible available validation design
even if its rank agreement is imperfect, then audit anchor-coefficient ranking
and determine whether checkpoint re-evaluation or Stage-2 retraining is needed.
Because the existing test scores help compare validator quality, later results
must disclose this selection and cannot be described as untouched confirmatory
performance on an independent final test set.
