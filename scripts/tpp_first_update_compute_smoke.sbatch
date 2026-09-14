#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=6
#SBATCH --exclude=ise-cpu256-[11-12],ise-cpu-intl-[01,08-15,18,24-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a/smoke_%j.txt

set -euo pipefail
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
OUT=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a/smoke_${SLURM_JOB_ID}
SOURCE='/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp-2026-08-25T06:23:43.293616/P[domain,pfile1,pfile2,pfile4]-S[0.003,50,enhsp-hadd-gbfs]-MO[]-T[518400]-04911937-faf7d6/snapshots/snapshot_5_0.667'
mkdir -p "$OUT"

apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
  set -euo pipefail
  cd '$REPO/asnets'
  source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  export PYTHONPATH='$REPO':\${PYTHONPATH:-}
  python -m py_compile asnets/supervised.py asnets/parllel_explore_spawn_grads.py asnets/spawn_train_worker.py
  python -m unittest discover -s tests -p 'test_policy_anchor.py' -v
  export ASN_FIRST_UPDATE_AUDIT_PATH='$OUT/audit.jsonl'
  ./run_experiment \
    experiments_numeric.architecture_2.tpp_mcts_first_update_smoke \
    experiments_numeric.domain.tpp_first_update_smoke \
    --resume-from '$SOURCE' --resume-train --no-eval \
    --original-training-set --disable-value-head --random-seed 2082152039 \
    --worker-logs --num-workers 1 --jpddl-max-heap 4g \
    --override-mse-coeff 0.5 --mcts-exploration-weight 0.1 \
    --override-enhsp-config hadd-astar --mcts-expansion-size 20 \
    --mcts-iterations 0 --sample-k-additional-states 0 --use-estimator 0.5 \
    --max-opt-epochs 1 --supervised-lr 0.0003 \
    --policy-anchor-kl-coeff 3 | tee '$OUT/training.stdout'
  SNAPSHOT_DIR=\$(awk -F': ' '/^Snapshot directory:/{print \$2; exit}' '$OUT/training.stdout')
  ENDPOINT=\$(find \"\$SNAPSHOT_DIR\" -maxdepth 1 -type d -name 'snapshot_0_*' | sort | tail -n 1)
  test -f \"\$ENDPOINT/weights.joblib\"
  export ASN_POLICY_PROBE_CAPTURE_DIR='$OUT/stage1_probe'
  ./run_experiment experiments_numeric.architecture_2.tpp_mcts \
    experiments_numeric.domain.tpp_first_update_smoke \
    --resume-from '$SOURCE' --disable-value-head --random-seed 2082152039 \
    --num-workers 1 --jpddl-max-heap 4g
  export ASN_POLICY_PROBE_INPUT_DIR='$OUT/stage1_probe'
  export ASN_POLICY_PROBE_OUTPUT_DIR='$OUT/endpoint_probe'
  unset ASN_POLICY_PROBE_CAPTURE_DIR
  ./run_experiment experiments_numeric.architecture_2.tpp_mcts \
    experiments_numeric.domain.tpp_first_update_smoke \
    --resume-from \"\$ENDPOINT\" --disable-value-head --random-seed 2082152039 \
    --num-workers 1 --jpddl-max-heap 4g
  python - '$OUT' <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
records = [json.loads(line) for line in (out / 'audit.jsonl').read_text().splitlines() if line.strip()]
steps = [r for r in records if r.get('record_type') == 'optimizer_step']
schedules = [r for r in records if r.get('record_type') == 'worker_schedule']
assert len(steps) == 1, len(steps)
assert len(schedules) == 1, len(schedules)
required = {'policy_gradient_l2', 'weighted_anchor_gradient_l2',
            'policy_anchor_gradient_cosine', 'step_kl_p99',
            'frozen_batch_file'}
assert required <= steps[0].keys(), required - steps[0].keys()
assert pathlib.Path(steps[0]['frozen_batch_file']).is_file()
assert list((out / 'stage1_probe').glob('*.npz'))
assert list((out / 'endpoint_probe').glob('*.json'))
print('[TPP-FIRST-UPDATE-SMOKE] real replay step and probe round-trip passed')
PY
"
