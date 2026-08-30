# Cross-domain progressive-widening screen

This is a **budget-matched screening experiment**, not one uniform 70-simulation
PW campaign.

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

