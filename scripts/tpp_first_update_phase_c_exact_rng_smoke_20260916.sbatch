#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --exclude=ise-cpu-intl-[01,09-11,13,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/tpp_first_update_phase_c_exact_rng/smoke_%j.txt

set -euo pipefail
: "${CHECKOUT:?Set CHECKOUT to the deployed exact-RNG checkout}"
: "${CODE_COMMIT:?Set CODE_COMMIT to the reviewed exact-RNG commit}"
REPO=$CHECKOUT
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-16/tpp_first_update_phase_c_exact_rng
PHASE_A=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_a
CHECKSUMS=/home/hersco/training_new_domains/2026-09-14/tpp_first_update_phase_b_crossover/frozen_schedule_checksums.csv
THRESHOLDS=/home/hersco/training_new_domains/2026-09-15/tpp_first_update_phase_c_prevention_lrbt/calibrated_thresholds.json

mkdir -p "$ROOT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$CODE_COMMIT"
test -f "$THRESHOLDS"
python3 "${ROOT}/freeze_tpp_phase_a_batch_checksums.py" \
  --verify --manifest "$CHECKSUMS"

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
      asnets/replay_rng.py asnets/policy_anchor_trust_region.py \
      asnets/frozen_replay.py asnets/supervised.py \
      asnets/scripts/run_asnets.py asnets/scripts/run_experiment.py
    TEST_OUTPUT=\$(python -m unittest \
      tests.test_replay_rng tests.test_policy_anchor_trust_region \
      tests.test_frozen_replay -v 2>&1)
    printf '%s\n' \"\$TEST_OUTPUT\"
    grep -Eq '^Ran 13 tests?' <<<\"\$TEST_OUTPUT\"
    ./run_experiment --help | \
      grep -q -- '--policy-anchor-trust-restore-rng-seed'
    python '$ROOT/smoke_tpp_exact_rng_retry.py'
  "

python3 - "$THRESHOLDS" <<'PY'
import json, pathlib, sys
row = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert row["calibration_role"] == "stable_control"
assert row["calibration_steps"] == 60
assert float(row["mean_kl_limit"]) > 0
assert float(row["p99_kl_limit"]) > 0
print("[TPP-EXACT-RNG-SMOKE] code, frozen inputs, and thresholds verified")
PY
