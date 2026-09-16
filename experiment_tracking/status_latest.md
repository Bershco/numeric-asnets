# Current experiment status

Updated: 2026-09-16T12:22:46+03:00 (focused reconciliation)

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State | Allocated / pending | CPU / RAM when allocated | Expected / hard remainder |
|---|---|---:|---:|---|
| MPrime Phase-B-A Stage-2 endpoint rescore | preflight complete; 9 original tasks plus exact recovery `21392547_[1-11]` running; finalizer `21392548` and downstream `21392549` pending | 20 / 2 | 80 CPU / 400 GiB | <=24 h hard per running task; downstream then <=1 h total |
| MPrime unresolved checkpoint policy curves | 11/17 terminal; five running and one completing | 6 / 0 | up to 12 CPU / 120 GiB | <=6 h 26 m hard |
| MPrime policy retry controller | allocated reconciliation controller | 1 / 0 | 2 CPU / 2 GiB | <=61 h hard |
| Counters strict policy-prior confirmation | 18/20 terminal; tasks 0 and 1 running | 2 / 0 | 12 CPU / 240 GiB | task 0 <=45 m hard; requeued task 1 <=44 h 45 m hard |
| Counters task-9 exact recovery | first recovered identity terminal; second active; third remains | 1 / 0 | 2 CPU / 120 GiB | <=9 h 15 m hard |
| Counters full-root trace controller | dependency-pending | 0 / 1 | 0 | starts only after every action-ID ledger reconciles to 59/59 |
| TPP exact-RNG guard closure | compute smoke `21394945` passed; all four active/inactive one-epoch arms running; four endpoints dependency-pending | 4 / 4 | training 24 CPU / 192 GiB; endpoints max 20 CPU / 80 GiB | <=1 h 53 m training hard, then <=2 h endpoints |

## Scientific status

- All twenty final MPrime Stage-2 training lineages are complete. Their training
  logs used the ordinary saturated validator; its all-epoch-0 choice is not the
  adopted Phase-B-A selection and has been removed from RQ1/RQ3.
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
- TPP selected-pair rollback/LR backtracking prevented the analogous first-update
  collapse (20/20 bad seed; 20/20 stable control), but dropout was resampled.
  The exact-RNG closure now compares active and inactive guards for both seeds
  under identical stochastic schedules: four one-epoch jobs and four endpoints.
  A broader three-arm, two-domain pilot stays held until this treatment is frozen.

## Canonical sources

- Live scheduler snapshot: `experiment_tracking/advisor_followup_20260910/live_workload_targeted_20260916_1128.csv`
- MPrime rescore manifest: `experiment_tracking/mprime_final_stage2_phase_b_a_rescore_20260916/manifest.csv`
- Cancelled premature-search provenance: `experiment_tracking/mprime_final_stage2_search_20260916/README.md`
- Primary RQ report: `experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
