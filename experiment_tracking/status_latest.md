# Current experiment status

Updated: 2026-09-16T11:28:41+03:00

Primary RQ evidence is validation-led only. Terminal-led campaigns remain
provenance archives and are excluded from primary tables and plots.

## Live workload

| Experiment | State | Allocated / pending | CPU / RAM when allocated | Expected / hard remainder |
|---|---|---:|---:|---|
| MPrime Phase-B-A Stage-2 endpoint rescore | repaired preflight `21390403` running; array `21390404`, finalizer `21390429`, and downstream controller `21390981` dependency-pending | 1 / 22 | preflight 4 CPU / 20 GiB; main 60 CPU / 400 GiB | preflight <=1 h 35 m hard; main <=24 h after it passes |
| MPrime unresolved checkpoint policy curves | 2/17 terminal; 15 exact identities running | 15 / 0 | 30 CPU / 300 GiB | <=7 h 12 m hard |
| Premature MPrime epoch-0 fixed/PW array | 40 tasks running; scientifically quarantined; explicit cancellation approval requested | 40 / 0 | 240 CPU / 4,800 GiB | do not treat completion as final RQ evidence |
| MPrime policy retry controller | allocated reconciliation controller | 1 / 0 | 2 CPU / 2 GiB | <=61 h hard |
| Counters strict policy-prior confirmation | 17/20 terminal; three original tasks running | 3 / 0 | 18 CPU / 360 GiB | two <=1 h 47 m hard; last <=45 h 31 m hard |
| Counters task-9 exact recovery | first recovered identity terminal; second active; third remains | 1 / 0 | 2 CPU / 120 GiB | <=9 h 15 m hard |
| Counters full-root trace controller | dependency-pending | 0 / 1 | 0 | starts only after every action-ID ledger reconciles to 59/59 |

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
  endpoints as Phase-B-A. At 11:07 it had 222/800 durable classifications. Its
  outputs are quarantined. After independent selection, exact checkpoint-hash
  and configuration matches may be reused; all other rows stay excluded.
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
  Four exact-RNG closure jobs remain the minimum causal wrap-up. A broader
  all-updates guard pilot stays held until that treatment is frozen.

## Canonical sources

- Live scheduler snapshot: `experiment_tracking/advisor_followup_20260910/live_workload_targeted_20260916_1128.csv`
- MPrime rescore manifest: `experiment_tracking/mprime_final_stage2_phase_b_a_rescore_20260916/manifest.csv`
- Quarantined search provenance: `experiment_tracking/mprime_final_stage2_search_20260916/README.md`
- Primary RQ report: `experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`
- Experiment registry: `experiment_tracking/experiment_registry.csv`
