#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-19
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-11,13,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_frozen_replay_crossover/train_%A_%a.txt
set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"
REPO=$CHECKOUT
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
SOURCE_ROOT=/home/hersco/training_new_domains/2026-09-16/imperfect_kl_semantics_screen/v2_exact_legacy_architecture
ROOT=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_frozen_replay_crossover
test "$(git -C "$REPO" rev-parse HEAD)" = "$CODE_COMMIT"
test -e "$REPO/asnets/asnets/ops/_asnet_ops_impl.so"

PAIR=$((SLURM_ARRAY_TASK_ID / 2)); MODE=$((SLURM_ARRAY_TASK_ID % 2))
case "$PAIR" in
  0) DOMAIN=block_grouping; SEED=42; TRAIN_MOD=experiments_numeric.architecture.actprop_2l_comparison_mcts_hadd_gbfs; HEUR=hadd-gbfs; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.block_grouping-experiments_numeric.architecture.actprop_2l_comparison_dynamic-2025-01-20T12:05:26.798164_41_hours_not_completed_properly/P/snapshots/snapshot_948_0.250000.pkl' ;;
  1) DOMAIN=block_grouping; SEED=2026; TRAIN_MOD=experiments_numeric.architecture.actprop_2l_comparison_mcts_hadd_gbfs; HEUR=hadd-gbfs; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.block_grouping-experiments_numeric.architecture.actprop_2l_comparison_dynamic-2025-01-20T12:05:26.798164_41_hours_not_completed_properly/P/snapshots/snapshot_948_0.250000.pkl' ;;
  2) DOMAIN=drone; SEED=2026; TRAIN_MOD=experiments_numeric.architecture.actprop_2l_all_mcts_hadd_astar; HEUR=hadd-astar; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.drone-experiments_numeric.architecture.actprop_2l_all_dynamic-2025-07-23T12:05:18.384799/P[domain,problem_1_1_4,problem_1_8_1,problem_8_1_...]-S[0.0003,50,enhsp-hadd-astar]-MO[]-T[603000]-04339159-1d27f5b/snapshots/snapshot_56_0.250000.pkl' ;;
  3) DOMAIN=drone; SEED=42; TRAIN_MOD=experiments_numeric.architecture.actprop_2l_all_mcts_hadd_astar; HEUR=hadd-astar; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.drone-experiments_numeric.architecture.actprop_2l_all_dynamic-2025-07-23T12:05:18.384799/P[domain,problem_1_1_4,problem_1_8_1,problem_8_1_...]-S[0.0003,50,enhsp-hadd-astar]-MO[]-T[603000]-04339159-1d27f5b/snapshots/snapshot_56_0.250000.pkl' ;;
  4) DOMAIN=fo_counters; SEED=2026; TRAIN_MOD=experiments_numeric.architecture.actprop_2l_fluent_mcts; HEUR=hmrmax-astar; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.fo_counters-experiments_numeric.architecture.actprop_2l_fluent-2025-07-22T16:32:02.325216/P[domain,instance_2,instance_3,instance_4]-S[0.0003,50,enhsp-hmrmax-astar]-MO[]-T[603000]-04337986-2745680/snapshots/snapshot_228_1.000000.pkl' ;;
  5) DOMAIN=fo_counters; SEED=42; TRAIN_MOD=experiments_numeric.architecture.actprop_2l_fluent_mcts; HEUR=hmrmax-astar; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.fo_counters-experiments_numeric.architecture.actprop_2l_fluent-2025-07-22T16:32:02.325216/P[domain,instance_2,instance_3,instance_4]-S[0.0003,50,enhsp-hmrmax-astar]-MO[]-T[603000]-04337986-2745680/snapshots/snapshot_228_1.000000.pkl' ;;
  6) DOMAIN=rover; SEED=42; TRAIN_MOD=experiments_numeric.architecture.actprop_2l_all_mcts_hmrp_ha_gbfs; HEUR=hmrp-ha-gbfs; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.rover-experiments_numeric.architecture.actprop_2l_all_dynamic-2025-07-23T15:45:28.566224/P[domain,pfile1,pfile2,pfile3,pfile4]-S[0.0003,50,enhsp-hmrp-ha-gbfs]-MO[]-T[603000]-04339379-6de1a75/snapshots/snapshot_145_0.750000.pkl' ;;
  7) DOMAIN=rover; SEED=2026; TRAIN_MOD=experiments_numeric.architecture.actprop_2l_all_mcts_hmrp_ha_gbfs; HEUR=hmrp-ha-gbfs; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.rover-experiments_numeric.architecture.actprop_2l_all_dynamic-2025-07-23T15:45:28.566224/P[domain,pfile1,pfile2,pfile3,pfile4]-S[0.0003,50,enhsp-hmrp-ha-gbfs]-MO[]-T[603000]-04339379-6de1a75/snapshots/snapshot_145_0.750000.pkl' ;;
  8) DOMAIN=counters; SEED=534933607; TRAIN_MOD=experiments_numeric.architecture_2.counters_mcts; HEUR=hmrmax-astar; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.counters-experiments_numeric.architecture_2.counters-2026-08-20T18:13:09.114960/P[domain,fz_instance_4,inv_instance_4,rnd_instanc...]-S[0.003,50,enhsp-hmrmax-astar]-MO[]-T[518400]-04905447-3c74d66/snapshots/snapshot_43_1.000' ;;
  9) DOMAIN=counters; SEED=2082152039; TRAIN_MOD=experiments_numeric.architecture_2.counters_mcts; HEUR=hmrmax-astar; SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.counters-experiments_numeric.architecture_2.counters-2026-08-20T18:13:09.109794/P[domain,fz_instance_4,inv_instance_4,rnd_instanc...]-S[0.003,50,enhsp-hmrmax-astar]-MO[]-T[518400]-04905447-faf7d6/snapshots/snapshot_12_1.000' ;;
  *) exit 2 ;;
esac

if (( MODE == 0 )); then
  TREATMENT=frozen_legacy
  KL_FLAG=()
  FORWARD=training
else
  TREATMENT=frozen_deterministic_current
  KL_FLAG=(--policy-anchor-kl-deterministic-current)
  FORWARD=deterministic
fi
RNG=$((880000 + PAIR))
FROZEN_BATCHES="${SOURCE_ROOT}/${DOMAIN}_${SEED}_now_legacy/first_update_audit.jsonl.batches"
OUT="${ROOT}/${DOMAIN}_${SEED}_${TREATMENT}"
mkdir -p "$OUT"
CHECKSUM_FILE="${ROOT}/frozen_input_checksums/pair_${PAIR}.sha256"
test -s "$CHECKSUM_FILE"
sha256sum --check --strict "$CHECKSUM_FILE"
if [[ -d "$SOURCE" ]]; then test -f "${SOURCE}/weights.joblib"; else test -f "$SOURCE"; fi
test "$(find "$FROZEN_BATCHES" -maxdepth 1 -type f -name 'optimizer_step_*.npz' | wc -l)" = 60
export ASN_FIRST_UPDATE_AUDIT_PATH="${OUT}/first_update_audit.jsonl"
export ASN_FIRST_UPDATE_WEIGHT_DIR="${OUT}/optimizer_step_weights"

apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
  set -euo pipefail; cd '$REPO/asnets'; source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
  export ASN_FIRST_UPDATE_AUDIT_PATH='$ASN_FIRST_UPDATE_AUDIT_PATH'
  export ASN_FIRST_UPDATE_WEIGHT_DIR='$ASN_FIRST_UPDATE_WEIGHT_DIR'
  ./run_experiment '$TRAIN_MOD' experiments_numeric.domain.'$DOMAIN' \
    --resume-from '$SOURCE' --resume-train --no-eval --original-training-set --disable-value-head \
    --random-seed '$SEED' --num-workers 3 --jpddl-max-heap 4g --override-mse-coeff 0.5 \
    --mcts-exploration-weight 0.1 --override-enhsp-config '$HEUR' --mcts-expansion-size 20 \
    --mcts-iterations 0 --sample-k-additional-states 0 --use-estimator 0.5 \
    --max-opt-epochs 1 --supervised-lr 0.0003 --policy-anchor-kl-coeff 3 \
    --policy-anchor-fixed-step-rng-seed '$RNG' --frozen-replay-batch-dir '$FROZEN_BATCHES' ${KL_FLAG[*]}
" 2>&1 | tee "${OUT}/training.stdout"

SNAPSHOT_DIR=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "${OUT}/training.stdout")
ENDPOINT=$(find "$SNAPSHOT_DIR" -maxdepth 1 -type d -name 'snapshot_0_*' | sort | tail -n 1)
test -f "${ENDPOINT}/weights.joblib"
printf '%s\n' "$ENDPOINT" > "${OUT}/endpoint_path.txt"
python3 - "$ASN_FIRST_UPDATE_AUDIT_PATH" "$FROZEN_BATCHES" "$RNG" "$FORWARD" <<'PY'
import json, pathlib, sys
p, source, base, forward = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).resolve(), int(sys.argv[3]), sys.argv[4]
rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
steps=[x for x in rows if x.get('record_type')=='optimizer_step']
assert len(steps)==60
assert any(x.get('record_type')=='frozen_replay_complete' for x in rows)
for i,x in enumerate(steps):
    assert x['kl_current_forward']==forward
    assert x['fixed_step_rng_base_seed']==base
    assert x['fixed_step_rng_step_seed']==(base+1000003*i)%2147483647
    assert pathlib.Path(x['source_frozen_batch_file']).parent.resolve()==source
    assert x['source_frozen_batch_sha256']
print('verified exact 60-step frozen replay, hashes, fixed RNG and KL treatment')
PY
