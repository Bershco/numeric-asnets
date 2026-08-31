#!/usr/bin/env bash
set -euo pipefail

out=/home/hersco/training_new_domains/2026-08-31/mcts_determinism_followup
preflight=/home/hersco/training_new_domains/2026-08-31/mcts_determinism_audit/overlay/mcts_determinism_preflight.sbatch
batch="$out/overlay/mcts_determinism_followup.sbatch"
checkpoint='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.drone-experiments_numeric.architecture_2.drone-2026-08-20T18:13:54.901932/P[domain,problem_1_1_4,problem_1_8_1,problem_8_1_...]-S[0.003,50,enhsp-hadd-astar]-MO[]-T[518400]-04905448-7d9a779/snapshots/snapshot_23_0.375'
checkpoint_b64=$(printf '%s' "$checkpoint" | base64 -w0)

mkdir -p "$out"
preflight_id=$(sbatch --parsable \
  --job-name=MCTS_DET_FOLLOW_PREFLIGHT \
  --output="$out/preflight_%j.out" \
  "$preflight")

printf 'run_id\tnode_family\trepeat\tjob_id\tdependency_job_id\tstdout\n' > "$out/submissions.tsv"

for node_family in cs-cpu-07 ise-cpu-intl-07; do
  for repeat in 1 2 3; do
    run_id="det-follow-${node_family}-r${repeat}"
    job_id=$(sbatch --parsable \
      --dependency="afterok:${preflight_id}" \
      --nodelist="$node_family" \
      --job-name="MCTS_DET_F_${repeat}" \
      --output="$out/%x_%j.out" \
      --export="ALL,CHECKPOINT_B64=${checkpoint_b64},NODE_FAMILY=${node_family},REPEAT=${repeat}" \
      "$batch")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$run_id" "$node_family" "$repeat" "$job_id" "$preflight_id" \
      "$out/${job_id}_${node_family}_r${repeat}.txt" >> "$out/submissions.tsv"
  done
done

echo "preflight_job_id=$preflight_id"
cat "$out/submissions.tsv"
