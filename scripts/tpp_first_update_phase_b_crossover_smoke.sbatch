#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --exclude=ise-cpu-intl-13
#SBATCH --output=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover/smoke_%j.txt

set -euo pipefail
REPO=${CHECKOUT:-/home/hersco/bershco-nu-asnets/tpp-first-update-b-9732f3a5}
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover
PHASE_A=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a
SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:23:43.293616/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911937-faf7d6/snapshots/snapshot_5_0.667'
FROZEN_BATCHES="${PHASE_A}/stable_control_2082152039/first_update_audit.jsonl.batches"
OUT="${ROOT}/smoke_${SLURM_JOB_ID}"
mkdir -p "$OUT"

python3 "${ROOT}/freeze_tpp_phase_a_batch_checksums.py" \
  --verify --manifest "${ROOT}/frozen_schedule_checksums.csv"

apptainer exec \
  --bind /home/hersco:/home/hersco \
  --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
  /home/hersco/Docker/image.sif \
  /bin/bash -lc "
    set -euo pipefail
    cd '$REPO/asnets'
    source '$PRODUCTION_REPO/venv-asnets/bin/activate'
    export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
    python -m py_compile \
      asnets/frozen_replay.py asnets/supervised.py \
      asnets/spawn_train_worker.py asnets/explorer_spawn_grads.py \
      asnets/scripts/run_asnets.py asnets/scripts/run_experiment.py
    TEST_OUTPUT=\$(python -m unittest discover -s tests -p 'test_frozen_replay.py' -v 2>&1)
    printf '%s\n' "\$TEST_OUTPUT"
    grep -Eq '^Ran 5 tests?' <<<"\$TEST_OUTPUT"
    export ASN_FIRST_UPDATE_AUDIT_PATH='$OUT/audit.jsonl'
    export ASN_FIRST_UPDATE_WEIGHT_DIR='$OUT/optimizer_step_weights'
    ./run_experiment \
      experiments_numeric.architecture_2.tpp_mcts \
      experiments_numeric.domain.tpp \
      --resume-from '$SOURCE' --resume-train --no-eval \
      --original-training-set --disable-value-head --random-seed 2082152039 \
      --num-workers 3 --jpddl-max-heap 4g \
      --override-mse-coeff 0.5 --mcts-exploration-weight 0.1 \
      --override-enhsp-config hadd-astar --mcts-expansion-size 20 \
      --mcts-iterations 0 --sample-k-additional-states 0 --use-estimator 0.5 \
      --max-opt-epochs 1 --supervised-lr 0.0003 \
      --policy-anchor-kl-coeff 3 \
      --frozen-replay-batch-dir '$FROZEN_BATCHES' | tee '$OUT/training.stdout'
  "

python3 - "$OUT/audit.jsonl" "$FROZEN_BATCHES" <<'PY'
import json, pathlib, sys
records = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.strip()]
steps = [row for row in records if row.get("record_type") == "optimizer_step"]
weights = [row for row in records if row.get("record_type") == "optimizer_step_weights"]
assert len(steps) == 60 and len(weights) == 60
assert all(row.get("source_frozen_batch_sha256") for row in steps)
assert all(pathlib.Path(row["path"]).is_file() for row in weights)
assert any(row.get("record_type") == "frozen_replay_complete" for row in records)
assert not any(row.get("record_type") == "worker_outputs" for row in records)
print("[TPP-FIRST-UPDATE-B-SMOKE] real 60-step frozen-replay path passed")
PY
