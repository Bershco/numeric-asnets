# Progressive widening: complete experiment narrative

This report distinguishes PW20 and PW70. They are never pooled. `PW20` means
20 MCTS simulations per external action; `PW70` means 70 simulations. The
standard promoted schedule is `Kmin=3`, `c=0.6`, `alpha=0.5`, `Kmax=20`, with
SAFE external action selection.

## Why progressive widening was tested

Fixed top-k search creates every permitted child immediately. Progressive
widening begins with a small policy-ranked action set and admits additional
actions as visits accumulate. The intended trade-off was to preserve most or
all of fixed-search coverage while reducing generated states, retained nodes,
memory and runtime.

## Experiment sequence

1. **Drone pilot, 10 matched seeds per arm.** Compared fixed top-5, historical
   fixed top-20, PW `c=0.6`, and PW `c=1.0`, all with 70 simulations.
2. **Drone sensitivity.** Tested SAFE-PW baseline, `alpha=0.65`, 140
   simulations and `Kmin=3`. Kmin=3 was the only promising direction and was
   expanded to four matched seeds per VH mode.
3. **Cross-domain two-seed screen.** Block Grouping and Counters accidentally
   used PW20 because they inherited the narrow 5/20 comparator budget. FO
   Counters and Rover correctly used PW70. This incident is documented and the
   budgets remain explicitly separated.
4. **Corrected PW70 screen.** Reran Block Grouping and Counters with 70
   simulations.
5. **Five- then ten-seed PW70 confirmation.** Promoted FO Counters and Rover to
   ten matched seeds per VH mode. Counters Stage-2 received five-seed PW70
   confirmation but did not qualify for further expansion.
6. **Focused diagnostics.** Tested PW20/PW70 on three exact Counters snapshots
   where fixed narrow search lost policy successes, and all four search
   variants on the catastrophic TPP/off seed.

## Drone: efficiency gain, insufficient coverage

| Arm | n | Mean coverage | Mean whole-job runtime | Change versus fixed top-20 |
|---|---:|---:|---:|---:|
| Historical fixed top-20 | 10 | 8.6/20 | 11.83h | baseline |
| Fixed top-5 | 10 | 8.7/20 | 11.24h | +0.1 |
| PW70, c=0.6 | 10 | 6.7/20 | 8.12h | -1.9; 95% CI [-3.35,-0.45] |
| PW70, c=1.0 | 10 | 6.2/20 | 5.03h | -2.4; 95% CI [-3.92,-0.88] |

The original PW schedules were genuinely cheaper: retained nodes fell from
3,622,099 for fixed top-5 to 692,524 (`c=.6`) and 319,398 (`c=1.0`). The price
was a statistically significant coverage loss.

Kmin=3 improved the balance:

| Arm | n | Mean coverage | Mean whole-job runtime |
|---|---:|---:|---:|
| Policy | 8 | 7.0/20 | 0.066h |
| Fixed top-20 | 8 | 10.5/20 | 11.76h |
| PW70 Kmin=3 | 8 | 9.5/20 | 9.99h |

Successful-instance runtime distributions show where the efficiency gain lies:

| Arm | Successful instances | Median | Mean | P90 | Maximum |
|---|---:|---:|---:|---:|---:|
| Fixed top-20 | 84 | 130s | 395s | 893s | 6,127s |
| PW70 Kmin=3 | 76 | 94s | 147s | 318s | 594s |

Kmin=3 therefore shortened the successful-instance tail substantially, but
still solved eight fewer matched instances than fixed search. Drone does not
support replacing fixed MCTS with this PW schedule.

## Block Grouping and Counters screen

PW20 was useful as a budget-matched narrow-search diagnostic, not as evidence
about standard PW70. Block Grouping was unpromising: it did not improve
coverage consistently and often increased whole-job runtime. For example,
BG/off fixed 5/20 averaged about 7h39m versus 12h26m for PW20; BG/on averaged
8h38m versus 9h22m.

The time was not primarily Python selection. In the instrumented BG PW20 jobs,
successor generation used 55.2% of completed wall time, estimator/evaluation
32.4%, network inference 6.9%, selection 4.4%, and backpropagation 0.3%.

Counters produced isolated strong recoveries—most notably one Stage-2/off seed
where policy and PW20 scored 59/59 while fixed narrow scored 49/59, with PW20
finishing in 3h47m versus 18h05m fixed. Across five PW70 seeds, however:

| VH | Policy | Fixed narrow 30m / 2h / 6h | PW70 30m / 2h / 6h | Conclusion |
|---|---:|---:|---:|---|
| off | 37.8/59 | 34.6 / 36.4 / 36.4 | 29.4 / 33.6 / 36.4 | Fixed parity only at 6h; below policy |
| on | 32.4/59 | 27.4 / 34.2 / 35.6 | 22.0 / 28.4 / 31.4 | Below policy and fixed |

Thus, PW can repair particular narrow-search failures but was not reliably the
best Counters configuration.

## Ten-seed PW70 result: FO Counters succeeds, Rover reaches parity

| Cell | Policy | Fixed 30m / 2h / 6h | PW70 30m / 2h / 6h | PW-policy 6h [95% CI]; Holm p | Conclusion |
|---|---:|---:|---:|---|---|
| FO/off | 4.2 | 7.5 / 7.8 / 7.8 | 8.4 / 8.4 / 8.4 | +4.2 [3.26,5.14]; .008 | Significant gain; fixed parity |
| FO/on | 3.7 | 5.3 / 5.7 / 5.7 | 7.3 / 7.3 / 7.3 | +3.6 [2.70,4.50]; .008 | Significant gain; higher mean than fixed |
| Rover/off | 4.0 | 4.8 / 5.0 / 5.0 | 4.7 / 4.7 / 4.7 | +0.7 [.02,1.38]; .250 | Fixed parity; not significant |
| Rover/on | 3.8 | 4.4 / 4.4 / 4.4 | 4.5 / 4.6 / 4.6 | +0.8 [-.08,1.68]; .250 | Fixed parity; not significant |

Almost all successful FO/Rover PW70 plans were already found inside 30 minutes:
FO is identical at 30m, 2h and 6h; Rover/on adds only 0.1 plan after 30m. Whole
allocations nevertheless ranged from roughly 7.5h to 28h because unsuccessful
instances can consume long timeouts. This separates **time to find successful
plans** from **time to certify failures**.

FO/off currently includes the declared 7/20 OOM-partial seed with one
unclassified instance counted unsuccessful. Exact recovery of that instance is
running separately; a success would raise the ten-seed mean from 8.4 to 8.5.

## Focused negative diagnostics

- On three Counters snapshots where fixed narrow search lost severe policy
  coverage, neither PW20 nor PW70 restored every policy success. PW20 seed
  923500475 reached 48/59 and was the strongest partial recovery.
- On the catastrophic TPP/off Stage-2 seed, policy scored 9/20, fixed narrow
  4/20, PW20 10/20, fixed normal 4/20 and PW70 7/20. Search could not generally
  repair the collapsed policy.

## Defensible overall conclusion

Progressive widening is **not a universal replacement for fixed MCTS**.

- It is a clear efficiency/coverage failure under the original Drone schedule.
- Kmin=3 improves Drone efficiency but remains below fixed-search coverage.
- It is not consistently useful for Block Grouping or Counters.
- It is strongly supported for FO Counters: significant gains over policy,
  fixed-search parity or better means, and essentially all successes inside
  30 minutes.
- In Rover it is best described as fixed-search parity, not a demonstrated
  improvement.

Authoritative row-level sources are `pw70_ten_seed_results_latest.csv`,
`pw70_ten_seed_statistics_latest.csv`, `../mcts_progressive_widening_pilot/summary.csv`,
`../mcts_progressive_widening_sensitivity/kmin3_runtime_summary.csv`, and
`terminal_results_20260901.csv`.
