#!/usr/bin/env bash
set -euo pipefail

out=/home/hersco/training_new_domains/2026-09-01/mcts_horizon_determinism_causal
checkout=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
preflight="$checkout/scripts/mcts_horizon_determinism_preflight.sbatch"
batch="$checkout/scripts/mcts_horizon_determinism_causal.sbatch"
checkpoint='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.drone-experiments_numeric.architecture_2.drone-2026-08-20T18:13:54.901932/P[domain,problem_1_1_4,problem_1_8_1,problem_8_1_...]-S[0.003,50,enhsp-hadd-astar]-MO[]-T[518400]-04905448-7d9a779/snapshots/snapshot_23_0.375'
checkpoint_b64=$(printf '%s' "$checkpoint" | base64 -w0)

mkdir -p "$out"
preflight_id=$(sbatch --parsable \
  --job-name=MCTS_HDET_PREFLIGHT \
  --output="$out/preflight_%j.out" \
  "$preflight")

printf 'run_id\thorizon_arm\trepeat\tjob_id\tdependency_job_id\tstdout\n' > "$out/submissions.tsv"
for arm in unaware aware; do
  for repeat in 1 2 3; do
    run_id="hdet-${arm}-r${repeat}"
    job_id=$(sbatch --parsable \
      --dependency="afterok:${preflight_id}" \
      --nodelist=cs-cpu-07 \
      --job-name="MCTS_HDET_${arm}_${repeat}" \
      --output="$out/%x_%j.out" \
      --export="ALL,CHECKPOINT_B64=${checkpoint_b64},HORIZON_ARM=${arm},REPEAT=${repeat}" \
      "$batch")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$run_id" "$arm" "$repeat" "$job_id" "$preflight_id" \
      "$out/${job_id}_${arm}_r${repeat}.txt" >> "$out/submissions.tsv"
  done
done

echo "preflight_job_id=$preflight_id"
cat "$out/submissions.tsv"
