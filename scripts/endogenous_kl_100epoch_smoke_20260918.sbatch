#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-15,25,27-28]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot/smoke_%j.txt
set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"; : "${CAMPAIGN_ROOT:?}"
test "$(git -C "$CHECKOUT" rev-parse --short=8 HEAD)" = "${CODE_COMMIT:0:8}"
MANIFEST="$CAMPAIGN_ROOT/manifest.csv"
test "$(($(wc -l < "$MANIFEST") - 1))" = 8
python3 - "$MANIFEST" <<'PY'
import csv, hashlib, pathlib, sys
rows=list(csv.DictReader(open(sys.argv[1])))
assert [int(r['arm_index']) for r in rows] == list(range(8))
assert {(r['domain'],r['seed']) for r in rows} == {
 ('drone','534933607'),('drone','923500475'),
 ('fo_counters','534933607'),('fo_counters','923500475')}
for r in rows:
 p=pathlib.Path(r['source_checkpoint'])/'weights.joblib'
 assert p.is_file(), p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['source_checkpoint_sha256']
PY
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
test -e "$CHECKOUT/asnets/asnets/ops/_asnet_ops_impl.so"
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "
 set -euo pipefail; cd '$CHECKOUT/asnets'; source '$PRODUCTION_REPO/venv-asnets/bin/activate';
 export PYTHONPATH='$CHECKOUT/asnets':'$CHECKOUT':\${PYTHONPATH:-};
 python -m py_compile asnets/supervised.py asnets/scripts/run_asnets.py asnets/scripts/run_experiment.py;
 ./run_experiment --help | grep -q -- '--policy-anchor-kl-deterministic-current'
"
touch "$CAMPAIGN_ROOT/smoke_passed"
