# Current experiment status

Updated: 2026-09-16T15:40:00+03:00 (focused reconciliation)

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State | Allocated / pending | CPU / RAM when allocated | Expected / hard remainder |
|---|---|---:|---:|---|
| MPrime Phase-B-A Stage-2 endpoint rescore | 132/420 evaluations durable; 9 original tasks plus exact recovery `21392547_[1-11]` running; finalizer `21392548` and downstream `21392549` dependency-pending | 20 / 2 | 80 CPU / 400 GiB | about 7 h at the observed aggregate rate; running-task hard remainder about 20 h 25 m |
| MPrime policy retry watcher | all 420 checkpoint-policy identities have already been submitted and no child remains active/retryable; `21362500` is an infinite five-minute watcher, not scientific compute | 1 / 0 | 2 CPU / 2 GiB | operationally obsolete; cancellation requires explicit approval |
| Counters strict policy-prior confirmation | 19/20 original Slurm tasks terminal, but scheduler-terminal is not equivalent to 59/59 scientific completion; original task 1 remains running | 1 / 0 | 6 CPU / 120 GiB | running-task hard remainder about 41 h 35 m |
| Counters task-9 exact recovery | exact three-identity recovery; source ledgers remain 56/59 action-ID and 57/59 policy-prior until reconciliation | 1 / 0 | 2 CPU / 120 GiB | hard remainder about 5 h 21 m |
| Counters full-root trace controller | dependency-pending | 0 / 1 | 0 | starts only after every action-ID ledger reconciles to 59/59 |
| TPP Phase-D KL/RNG isolation | smoke passed; the repaired one-epoch treatments completed and passed the corrected frozen-batch/RNG/pre-treatment audit; endpoint array `21409234_[0-1]` submitted | 0 / 2 | endpoints up to 10 CPU / 40 GiB | endpoint hard bound 2 h |

## Scientific status

- All twenty final MPrime Stage-2 training lineages are complete. Their training
  logs used the ordinary saturated validator; that field is legacy,
  non-authoritative provenance only. Phase-B-A is the sole canonical MPrime
  selector and reported validator. Validation was not part of the training
  loss, so weights remain valid; the required correction is endpoint rescoring.
- The corrected MPrime repair is 420 Phase-B-A evaluations: twenty lineage jobs
  x twenty-one saved checkpoints on the frozen thirty-instance replicate A.
  The runner verifies the evaluator checkout commit, validator manifest/module,
  every PDDL checksum, every checkpoint hash, and the exact epoch set. The
  finalizer uses maximum Phase-B-A score with earliest-epoch tie-breaking.
- 403/420 MPrime checkpoints already have valid policy logs; seventeen exact
  non-selected-under-the-old-validator gaps are recovering. Once Phase-B-A
  selects endpoints, existing logs are reused and only a truly selected missing
  identity is rerun.
- Premature MPrime search array `21388436` mislabeled old-validator epoch-0
  endpoints as Phase-B-A and was cancelled. Its provisional aggregate table
  was deleted. Raw logs are not RQ inputs; only an exact selected
  checkpoint-hash/configuration identity may be reused after selection.
- The Counters targeted causal pilot remains: on three VH-off policy-success
  instances, action-ID/Q solved 0/3 and policy-prior solved 3/3. The unrelated
  VH-on arm is no longer presented as a failure because its policies solved 0/3.
  The ten-seed confirmation is not yet terminal, so primary Counters RQ values
  remain unchanged.
- Recovery is identity-based. Success, action-limit failure, explicit six-hour
  timeout, and explicitly instance-attributed OOM are terminal. A job-level OOM
  does not classify identities with no attributable outcome. Task 7 explicitly
  printed evaluator identity 55's (`fz_instance_56.pddl`) 21,600-second
  timeout; controller `21362904` will add it to the durable ledger
  post-terminal without rerunning it.
- TPP exact-RNG closure is complete. Active and inactive catastrophic arms both
  scored 20/20; both stable-control arms also scored 20/20. This establishes no
  guard advantage inside that treatment, but it does **not** show that RNG alone
  avoided the collapse: both arms also changed the historical anchor KL from a
  dropout-current to deterministic-current forward. Phase D holds RNG fixed and
  restores the historical KL to isolate that confound. The broader three-arm,
  two-domain pilot remains held.

## Canonical sources

- Live scheduler snapshot: `experiment_tracking/live_experiment_status_latest.csv`
- MPrime rescore manifest: `experiment_tracking/mprime_final_stage2_phase_b_a_rescore_20260916/manifest.csv`
- Cancelled premature-search provenance: `experiment_tracking/mprime_final_stage2_search_20260916/README.md`
- Primary RQ report: `experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
