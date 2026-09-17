#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --array=0-3
#SBATCH --exclude=ise-cpu128-03,ise-cpu-intl-[01,09-15,27]
#SBATCH --output=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_primary_lineage_crossover/old_source_%A_%a.txt
set -euo pipefail
: "${CHECKOUT:?}"
OLD=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_frozen_replay_crossover
ROOT=/home/hersco/training_new_domains/2026-09-17/imperfect_kl_primary_lineage_crossover
case "$SLURM_ARRAY_TASK_ID" in
 0) DOMAIN=block_grouping; EVAL=experiments_numeric.architecture.actprop_2l_comparison; SOURCE=$(python3 -c "import csv;print(list(csv.DictReader(open('$OLD/manifest.csv')))[0]['source_checkpoint'])") ;;
 1) DOMAIN=drone; EVAL=experiments_numeric.architecture.actprop_2l_all; SOURCE=$(python3 -c "import csv;print(list(csv.DictReader(open('$OLD/manifest.csv')))[2]['source_checkpoint'])") ;;
 2) DOMAIN=fo_counters; EVAL=experiments_numeric.architecture.actprop_2l_fluent; SOURCE=$(python3 -c "import csv;print(list(csv.DictReader(open('$OLD/manifest.csv')))[4]['source_checkpoint'])") ;;
 3) DOMAIN=rover; EVAL=experiments_numeric.architecture.actprop_2l_all; SOURCE=$(python3 -c "import csv;print(list(csv.DictReader(open('$OLD/manifest.csv')))[6]['source_checkpoint'])") ;;
esac
apptainer exec --bind /home/hersco:/home/hersco --bind /home/hersco/apptainer_fake_passwd:/etc/passwd /home/hersco/Docker/image.sif /bin/bash -lc "set -euo pipefail; cd '$CHECKOUT/asnets'; source /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/activate; export PYTHONPATH='$CHECKOUT/asnets':'$CHECKOUT':\${PYTHONPATH:-}; ./run_experiment '$EVAL' experiments_numeric.domain.'$DOMAIN' --resume-from '$SOURCE' --disable-value-head --random-seed 42 --num-workers 5 --jpddl-max-heap 4g"
