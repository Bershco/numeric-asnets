#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-11
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-11,13,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/tpp_kl_multirng_susceptibility/train_%A_%a.txt

set -euo pipefail
: "${CHECKOUT:?Set CHECKOUT to the isolated diagnostic checkout}"
: "${CODE_COMMIT:?Set CODE_COMMIT to the reviewed diagnostic commit}"
REPO=$CHECKOUT
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-16/tpp_kl_multirng_susceptibility
PHASE_A=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a
CHECKSUMS=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover/frozen_schedule_checksums.csv
test "$(git -C "$REPO" rev-parse HEAD)" = "$CODE_COMMIT"
test -e "$REPO/asnets/asnets/ops/_asnet_ops_impl.so"
python3 "${ROOT}/freeze_tpp_phase_a_batch_checksums.py" --verify --manifest "$CHECKSUMS"

RNGS=(271828 1618033 20260916)
RNG=${RNGS[$((SLURM_ARRAY_TASK_ID / 4))]}
COMBO=$((SLURM_ARRAY_TASK_ID % 4))
case "$COMBO" in
  0|1)
    ROLE=catastrophic_outlier; SEED=1972442430
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:58:34.413651/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911972-72aea46/snapshots/snapshot_15_1.000'
    SOURCE_SHA=62c01cac5cbacb776e0c2b0e3d421d0314bb1abe282c14925c51602e2bfaecec
    FROZEN_BATCHES="${PHASE_A}/catastrophic_outlier_1972442430/first_update_audit.jsonl.batches"
    ;;
  2|3)
    ROLE=stable_control; SEED=2082152039
    SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:23:43.293616/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911937-faf7d6/snapshots/snapshot_5_0.667'
    SOURCE_SHA=e97c179d214e0a84b824f584accdecf56373de6d92c353166ef5b2ed18b7b353
    FROZEN_BATCHES="${PHASE_A}/stable_control_2082152039/first_update_audit.jsonl.batches"
    ;;
esac
if (( COMBO % 2 == 0 )); then
  TREATMENT=legacy_dropout_current
  KL_FLAG=()
  EXPECTED_FORWARD=training
else
  TREATMENT=deterministic_current
  KL_FLAG=(--policy-anchor-kl-deterministic-current)
  EXPECTED_FORWARD=deterministic
fi
ARM="${ROLE}_${TREATMENT}_rng${RNG}"
OUT="${ROOT}/${ARM}"
mkdir -p "$OUT"
test "$(sha256sum "${SOURCE}/weights.joblib" | awk '{print $1}')" = "$SOURCE_SHA"
export ASN_FIRST_UPDATE_AUDIT_PATH="${OUT}/first_update_audit.jsonl"
export ASN_FIRST_UPDATE_WEIGHT_DIR="${OUT}/optimizer_step_weights"

apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif /bin/bash -lc "
    set -euo pipefail
    cd '$REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
    export ASN_FIRST_UPDATE_AUDIT_PATH='$ASN_FIRST_UPDATE_AUDIT_PATH'
    export ASN_FIRST_UPDATE_WEIGHT_DIR='$ASN_FIRST_UPDATE_WEIGHT_DIR'
    ./run_experiment experiments_numeric.architecture_2.tpp_mcts experiments_numeric.domain.tpp \
      --resume-from '$SOURCE' --resume-train --no-eval --original-training-set \
      --disable-value-head --random-seed '$SEED' --num-workers 3 --jpddl-max-heap 4g \
      --override-mse-coeff 0.5 --mcts-exploration-weight 0.1 \
      --override-enhsp-config hadd-astar --mcts-expansion-size 20 --mcts-iterations 0 \
      --sample-k-additional-states 0 --use-estimator 0.5 --max-opt-epochs 1 \
      --supervised-lr 0.0003 --policy-anchor-kl-coeff 3 \
      --policy-anchor-fixed-step-rng-seed '$RNG' ${KL_FLAG[*]} \
      --frozen-replay-batch-dir '$FROZEN_BATCHES'
  " 2>&1 | tee "${OUT}/training.stdout"

SNAPSHOT_DIR=$(awk -F': ' '/^Snapshot directory:/{print $2; exit}' "${OUT}/training.stdout")
ENDPOINT=$(find "$SNAPSHOT_DIR" -maxdepth 1 -type d -name 'snapshot_0_*' | sort | tail -n 1)
test -f "${ENDPOINT}/weights.joblib"
printf '%s\n' "$ENDPOINT" > "${OUT}/endpoint_path.txt"
python3 - "$ASN_FIRST_UPDATE_AUDIT_PATH" "$FROZEN_BATCHES" "$RNG" "$EXPECTED_FORWARD" <<'PY'
import json, pathlib, sys
path, batches, base, expected = sys.argv[1], pathlib.Path(sys.argv[2]).resolve(), int(sys.argv[3]), sys.argv[4]
steps = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines()
         if line.strip() and json.loads(line).get("record_type") == "optimizer_step"]
assert len(steps) == 60
for index, step in enumerate(steps):
    assert step["kl_current_forward"] == expected
    assert step["fixed_step_rng_base_seed"] == base
    assert step["fixed_step_rng_step_seed"] == (base + 1000003 * index) % 2147483647
    assert step["trust_region"] is None and step["trust_region_retries"] == 0
    assert pathlib.Path(step["source_frozen_batch_file"]).parent.resolve() == batches
print("verified 60 frozen replay and fixed-RNG optimizer steps")
PY
