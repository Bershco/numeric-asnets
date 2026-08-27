#!/usr/bin/env bash
set -euo pipefail

work=/home/hersco/training_new_domains/2026-08-27/mcts_safe_drone_diagnostic
batch="$work/mcts_safe_drone_diagnostic.sbatch"

submit() {
  local name=$1 checkpoint=$2 seed=$3 vh=$4 skip=$5 label=$6
  sbatch --job-name="$name" --output="$work/%x_%j.out" \
    --export=ALL,CHECKPOINT="$checkpoint",SEED="$seed",VALUE_HEAD="$vh",SKIP_INSTANCE_NUMBERS="$skip",RUN_LABEL="$label" \
    "$batch"
}

submit SAFE3_D_OFF_1963 \
  '/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.drone-experiments_numeric.architecture_2.drone-2026-08-20T18:13:20.913251/P[domain,problem_1_1_4,problem_1_8_1,problem_8_1_...]-S[0.003,50,enhsp-hadd-astar]-MO[]-T[518400]-04905448-2fc32f3/snapshots/snapshot_157_0.375' \
  1963100312 off '1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20' safe_off_s1963100312_e157_archfix

submit SAFE3_D_OFF_5349 \
  '/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.drone-experiments_numeric.architecture_2.drone-2026-08-20T18:13:20.913938/P[domain,problem_1_1_4,problem_1_8_1,problem_8_1_...]-S[0.003,50,enhsp-hadd-astar]-MO[]-T[518400]-04905447-3c74d66/snapshots/snapshot_69_0.500' \
  534933607 off '1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20' safe_off_s534933607_e69_archfix

submit SAFE3_D_ON_2011 \
  '/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.drone-experiments_numeric.architecture_2.drone-2026-08-20T18:13:03.540334/P[domain,problem_1_1_4,problem_1_8_1,problem_8_1_...]-S[0.003,50,enhsp-hadd-astar]-MO[]-T[518400]-04905447-067843/snapshots/snapshot_19_0.500' \
  2011206605 on '1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20' safe_on_s2011206605_e19_archfix
