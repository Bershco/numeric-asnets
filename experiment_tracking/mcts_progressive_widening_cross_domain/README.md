# Cross-domain progressive-widening screen

This directory now contains two deliberately distinct experiments.  This
distinction was added after discovering that the first Block Grouping/Counters
cross-domain screen had inherited the narrow comparator's 20-simulation budget
rather than the normal 70-simulation PW budget.

1. `MCTS-PW-CROSS-DOMAIN` / `pw-kmin3-*`: the original **budget-matched PW20
   screen**.  It remains valid as a comparison against fixed narrow 5/20.
2. `MCTS-PW70-CROSS-DOMAIN` / `pw70-kmin3-*`: the separately submitted
   **standard-budget PW70 follow-up**.  It uses the same checkpoints and PW
   schedule but 70 simulations.  Its twelve jobs cover Block Grouping Stage 1,
   Counters Stage 1, and validation-led Counters Stage 2, both VH modes and two
   seeds.

The two arms must never be pooled or labelled interchangeably.  Final tables
must show policy, fixed 5/20, PW20, and PW70 in separate columns, plus fixed
20/70 wherever a matched result exists.

- Block Grouping Stage 1 and Counters Stage 1/Stage 2 use 20 simulations so
  Kmin=3 PW is matched to the scientifically relevant fixed-narrow 5/20 arm.
- FO Counters and Rover use 70 simulations so PW is matched to their available
  fixed-normal 20/70 arm.
- The completed Drone PW pilot and Kmin=3 extension use 70 simulations.

Therefore, a successful 20-simulation Counters row establishes that widening is
better than fixed width under the same small simulation budget. It does **not**
establish the result of PW with the standard 70-simulation budget. A promoted
Block Grouping/Counters cell must receive a separately labelled 70-simulation
confirmation if the thesis claim concerns standard-budget PW.

## Submitted PW70 follow-up

The twelve rows in `pw70_followup_manifest.csv` were submitted as jobs
`20755790`--`20755801`; exact job-to-row provenance is in
`pw70_followup_submissions.tsv`.  They are ordinary resource-pending jobs, not
held jobs.

The first terminal PW20 cell is Block Grouping Stage 1, VH-on.  For seeds
1963100312 and 2011206605, policy and fixed 5/20 both average 17/20, while PW20
averages 18/20.  Whole-job runtime averages about 8h38m for fixed 5/20 and
9h22m for PW20.  This is a coverage-positive candidate, but not yet a formal
promotion because it has not demonstrated an efficiency benefit; retained-node
evidence and the PW70 follow-up remain relevant.

## Promotion rule

A cell means one domain, training stage, and VH mode—not one favourable seed or
one test instance. The screen uses two matched seeds per cell. A cell is
promising when PW:

1. is no worse than policy on mean coverage;
2. is within five percentage points of its matched fixed-search mean;
3. reduces runtime, timeouts, OOMs, or retained nodes materially; and
4. does not introduce a systematic new policy-success regression.

To avoid choosing only a favourable VH result, if either VH cell for one
domain/stage is promoted, both VH modes are expanded to at least five matched
seeds. Fresh 30-minute confirmation remains a later, separately labelled arm.
