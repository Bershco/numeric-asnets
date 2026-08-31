#!/usr/bin/env bash
set -euo pipefail

out=/home/hersco/training_new_domains/2026-08-31/mcts_determinism_audit
overlay="$out/overlay"
preflight="$overlay/mcts_determinism_preflight.sbatch"
batch="$overlay/mcts_determinism_audit.sbatch"
checkpoint='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.drone-experiments_numeric.architecture_2.drone-2026-08-20T18:13:20.913938/P[domain,problem_1_1_4,problem_1_8_1,problem_8_1_...]-S[0.003,50,enhsp-hadd-astar]-MO[]-T[518400]-04905447-3c74d66/snapshots/snapshot_69_0.500'
checkpoint_b64=$(printf '%s' "$checkpoint" | base64 -w0)

mkdir -p "$out"
preflight_id=$(sbatch --parsable \
  --job-name=MCTS_DETERMINISM_PREFLIGHT \
  --output="$out/preflight_%j.out" \
  "$preflight")

printf 'run_id\tarm\trepeat\tjob_id\tdependency_job_id\tstdout\n' \
  > "$out/submissions.tsv"

for arm in ordinary deterministic_cpu; do
  for repeat in 1 2 3; do
    run_id="det-${arm}-r${repeat}"
    job_id=$(sbatch --parsable \
      --dependency="afterok:${preflight_id}" \
      --job-name="MCTS_DET_${arm}_${repeat}" \
      --output="$out/%x_%j.out" \
      --export="ALL,CHECKPOINT_B64=${checkpoint_b64},RUN_ARM=${arm},REPEAT=${repeat}" \
      "$batch")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$run_id" "$arm" "$repeat" "$job_id" "$preflight_id" \
      "$out/${job_id}_${arm}_r${repeat}.txt" \
      >> "$out/submissions.tsv"
  done
done

echo "preflight_job_id=$preflight_id"
cat "$out/submissions.tsv"

