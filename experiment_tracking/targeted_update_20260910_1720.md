# Targeted experiment update — 10 September 2026, 17:20 IDT

This refresh deliberately queried only work that was live or unresolved after
the previous snapshot.

## Current workload

One job is live: adaptive-KL outlier `21144388`, using 6 CPUs and 48 GiB. It
reached Stage-2 epoch 86. Validation is 22/30 at that checkpoint. Realized KL
remains below the target, so the adaptive coefficient remains 3 and the
controller has made zero adjustments. The stable control `21144389` completed
100 epochs. At the observed outlier pace, approximately six hours remain.

## MPrime Phase B

The queue is empty, but the audit is not complete: 2,241/2,260 exact
checkpoint-replicate CSVs exist. Fifty-eight of sixty lineages are complete.
The exact missing tail is:

- array index 19, `stage1-on-1972442430`: epochs 45, 50, 55, 60 and 64 on both
  replicates (10 evaluations);
- array index 24, `validation_led-off-1472491096`: replicate 1 at epoch 80 and
  both replicates at epochs 85, 90, 95 and 99 (9 evaluations).

An idempotent two-index continuation is sufficient. No MPrime training or MCTS
is required to complete Phase B itself.

## FO Counters PW70 recovery

Job `21157787` used the intended evaluator slot 14, confirmed as
`instance_15.pddl`, one worker, 2 CPUs and the full 120 GiB. It completed after
21,603.5 seconds with no plan. The seed remains 7/20 and the ten-seed FO/off
PW70 mean remains 8.40/20.

## FO Counters Stage-2 validation-led MCTS

The alleged five missing jobs do not exist. All twenty seed/VH identities ran.
Seventeen have complete records; the remaining three are real partial
allocations:

| VH | seed | job | scheduler state | durable successes | interpretation |
|---|---:|---:|---|---:|---|
| off | 2082152039 | 20943885 | failed after 65h28m | 5 | partial lower bound |
| off | 923500475 | 20945845 | failed after 54h12m | 7 | partial lower bound |
| on | 2011206605 | 20945846 | 72h timeout | 6 | partial lower bound |

Deduplicating all printed plans adds no hidden success beyond those ledgers.
Therefore no duplicate five-job submission is warranted.

## Counters policy-success losses

The earlier count of twelve was correct only for VH-off. The complete
validation-led Stage-2 join finds:

- VH-off: 12 policy-success/MCTS-failure instances (11 ordinary unsolved, one
  unclassified after interruption);
- VH-on: 13 such instances (all ordinary unsolved), despite an aggregate MCTS
  gain because MCTS adds more successes elsewhere;
- combined descriptive count: 25, not 12.

Every seed, policy job/log and MCTS job/ledger is recorded in
`advisor_followup_20260910/counters_policy_mcts_failure_global.csv`.
