#!/bin/bash

set -uo pipefail

source /home/hersco/test_sbatches/inner_tests/resources.sh

########################################
# Small helpers
########################################

vh_label() { [[ -z "$1" ]] && echo "vh" || echo "novh"; }

compact_num_label() {
    local v="$1"

    # 0.5 -> .5
    v="${v#0}"

    # 1.0 -> 1, 10.0 -> 10, .0 -> ""
    v="${v%.0}"

    # Safety: 0.0 became empty
    [[ -z "$v" ]] && v="0"

    echo "$v"
}

est_label() {
    if [[ -n "$ESTIMATOR_DECAY_FLAG" ]]; then
        echo "ed"
    elif [[ -n "$ESTIMATOR_COEFF" && "$ESTIMATOR_COEFF" != "0.0" && "$ESTIMATOR_COEFF" != "0" ]]; then
        echo "e$(compact_num_label "$ESTIMATOR_COEFF")"
    fi
}

exploration_label() {
    echo "c$(compact_num_label "$1")"
}

seed_label() {
    echo "s$1"
}

tree_suffix() {
    [[ "$1" == *mcts* ]] && echo "_K$2"
}

run_mode_prefix() {
    if [[ -n "$RESUME_FROM_ARG" && -n "$RESUME_TRAIN_FLAG" ]]; then
        echo "Re-Tr"
    elif [[ -n "$RESUME_FROM_ARG" ]]; then
        echo "Ev"
    elif $EVAL_REQUESTED; then
        echo "Tr-Ev"
    else
        echo "Tr"
    fi
}

append_part() {
    local part="$1"
    [[ -n "$part" ]] && JOB_PARTS+=("$part")
}

build_job_name() {
    local domain="$1"
    local short_arch="$2"
    local orig="$3"
    local vh_flag="$4"
    local tree_sample="$5"
    local exploration_weight="$6"
    local seed="$7"

    local mode_part
    local hw_part
    local vh_part
    local est_part
    local seed_part
    local tree_part
    local exp_part
    local prefix

    prefix="$(run_mode_prefix)"
    hw_part=$([[ -z "$SBATCH_GPU_ARGS" ]] && echo "" || echo "gpu")
    vh_part="$(vh_label "$vh_flag")"
    est_part="$(est_label)"
    tree_part="$(tree_suffix "$short_arch" "$tree_sample")"
    exp_part="$(exploration_label "$exploration_weight")"
    seed_part="$(seed_label "$seed")"

    if [[ -z "$orig" ]]; then
        mode_part="real"
    else
        mode_part="orig"
    fi

    JOB_PARTS=()
    append_part "$prefix"
    append_part "$domain"
    append_part "$short_arch"
    append_part "$mode_part"
    append_part "$hw_part"
    append_part "$vh_part"
    append_part "$est_part"
    append_part "$exp_part"
    append_part "$seed_part"

    local joined
    joined="$(IFS=_; echo "${JOB_PARTS[*]}")"

    echo "${joined}${tree_part}${JOB_SUFFIX}"
}

########################################
# Defaults
########################################

ALL_DOMAINS=(block_grouping counters delivery drone fo_counters mprime rover tpp zenotravel)
DOMAIN_LIST=("${ALL_DOMAINS[@]}")

RUN_DATE=$(date +%Y-%m-%d)
OUTPUT_SUBDIR=""

########################################
# Experiment axes
########################################

TRAINING_TRIES=1

ARCHITECTURE_LIST=("actprop_2l_comparison_mcts")
ARCHITECTURE_OVERRIDDEN=false
DOMAIN_ARCHITECTURE_MODE=""
MSE_COEFF_LIST=("0.5")
TREE_SAMPLE_LIST=("0")

ORIG_TRAINING=("" "--original-training-set")
VALUE_HEAD_OPTIONS=("")

ESTIMATOR_COEFF_OPTIONS=("0.0")
ESTIMATOR_DECAY_OPTIONS=("")

EXPLORATION_WEIGHT_OPTIONS=("0.1")
SEED_OPTIONS=("42")

TREE_SAMPLE_FULL_GRID_VALUES=(
    "0"
    "5"
    "10"
    "20"
)

ESTIMATOR_FULL_GRID_VALUES=(
    "0.0"
    "0.5"
#    "1.0"
)

EXPLORATION_WEIGHT_FULL_GRID_VALUES=(
    "0.01"
    "0.1"
    "1.0"
    "10.0"
)

SEED_FULL_GRID_VALUES=(
    "2024"
    "2025"
    "2026"
    "42"
    "1111999"
    "6363"
    "100626"
    "12345"
    "13"
    "777"
)

########################################
# Runtime options
########################################

SBATCH_GPU_ARGS=""
SBATCH_TIME_ARGS="--time=3-00:00:00"
SBATCH_MEM_ARGS=""
SBATCH_CPU_ARGS=""
# Campaign-private safety override.  ise-cpu128-03 repeatedly exhausted its
# POSIX semaphore space during MPrime policy evaluation on 2026-09-15.
SBATCH_NODE_EXCLUDE_ARGS="${SBATCH_NODE_EXCLUDE_ARGS:---exclude=ise-cpu128-03}"

DRY_RUN=false
JOB_SUFFIX=""

RESUME_FROM_ARG=""
RESUME_TRAIN_FLAG=""

NO_EVAL_FLAG="--no-eval"
EVAL_WITH_MCTS_FLAG=""
EVAL_START_WAVE_ARG=""
EVAL_SCHEDULING_ARG=""
SKIP_INSTANCE_NUMBERS_ARG=""
EVAL_INSTANCE_TIMEOUT_ARG=""
ACTION_DEBUG_FLAG=""
PUCT_DEBUG_FLAG=""
EVAL_REQUESTED=false

NUM_WORKERS=10
JPDDL_MAX_HEAP="1g"
ENABLE_PROFILING=0
ESTIMATOR_MODE_COUNT=0

MCTS_EXPANSION_SIZE_ARG=""
MCTS_ITERATIONS_ARG=""
POLICY_ANCHOR_KL_COEFF="0.0"
OVERRIDE_MAX_EPOCHS="1000"
OVERRIDE_LR="0.003"

########################################
# Formatting
########################################

DOMAIN_WIDTH=14
ARCH_WIDTH=30
MSE_WIDTH=6
TREE_SAMPLE_WIDTH=4
MODE_WIDTH=8
VH_WIDTH=6
EST_WIDTH=8
EXP_WIDTH=8
JOBNAME_WIDTH=80

########################################
# Helpers
########################################

split_csv_into_array() {
    local IFS=','
    read -ra RESULT <<< "$1"
    echo "${RESULT[@]}"
}

join_by_comma() {
    local IFS=', '
    echo "$*"
}

count_array() {
    echo "$#"
}

########################################
# Help
########################################

print_help() {
cat <<EOF
Usage:
  ./submit_training.sh [options]

Domains:
  --dom-<domain>                    Run a single domain
  --override-domains d1,d2,...

Available domains:
  block_grouping,counters,delivery,drone,
  fo_counters,mprime,rover,tpp,zenotravel

Dataset: (relevant only to training)
  --random-only                     Random/realtime dataset only (default)
  --original-only                   Original training dataset only

Architecture:
  --override-arch A[,B,...]

    Short names only, e.g.
      comparison_mcts
      comparison
      fluent
      dynamic

  --domain-architecture MODE
      Use experiments_numeric.architecture_2.<domain> when MODE=policy,
      or experiments_numeric.architecture_2.<domain>_mcts when MODE=mcts.
      This cannot be combined with --override-arch.

Training:
  --tries N                         Repeat each configuration N times
  --override-mse V1,V2,...

Tree sampling: (relevant only to training)
  --override-tree-sampling K1,K2,...
  --tree-sample-full-grid           K={0,5,10,20}

Estimator (choose ONE mode):
  --use-estimator X                 Fixed coefficient
  --use-estimator-decay             Decaying coefficient
  --estimator-full-grid             coeff={0.0,0.5,1.0}

Exploration:
  --exploration-weight C
  --exploration-weight-full-grid    C={0.01,0.1,1.0,10.0}

Seeds:
  --seed S1,S2,...
  --seed-full-grid                  2024,2025,2026,42,
                                    1111999,6363,100626,
                                    12345,13,777

Value head:
  --vh-off                          Disable value head
  --vh-on-off                       Run both enabled and disabled

Resume / evaluation:
  --train-from PATH                 Resume training from checkpoint
  --eval-from PATH                  Evaluate/infer from checkpoint
  --eval-with-mcts                  Use MCTS for final test inference
  --eval-start-wave N               Resume final evaluation from wave N; ignored for training-only jobs
  --eval-scheduling wave|rolling    Evaluation worker scheduling mode
  --skip-instance-numbers 1,4,7     Skip completed one-based test positions
  --eval-instance-timeout SECONDS   Per-instance evaluation timeout
  --action-debug                    Log policy/MCTS action comparisons
  --puct-debug                      Log detailed root PUCT statistics
  --eval                            Run evaluation after training

MCTS:
  --mcts-expansion-size K
  --mcts-iterations N

Resources:
  --workers N                       Default: 10
  --jpddl-max-heap SIZE             JPDDL heap per worker; default: 1g
  --enable-profiling                Enable cProfile and worker profiles
  --gpu                             Request RTX 6000 GPU
  --time LIMIT                      Example: 3-00:00:00

Misc:
  --job-suffix TEXT
  --output-subdir NAME             Write logs to RUN_DATE/NAME
  --dry-run
  --help

Run modes:
  Tr       Fresh training (default)
  Tr-Ev    Training + evaluation
  Ev       Evaluation from checkpoint (--eval-from)
  Re-Tr    Resume training (--train-from)

Job-name fields:
  orig     Original training dataset
  real     Random/realtime dataset
  vh       Value head enabled
  novh     Value head disabled
  eX       Fixed estimator coefficient
  ed       Estimator decay
  cX       Exploration weight
  sX       Seed
  KX       Tree sample size (MCTS architectures only)
  succonly Successful runs only

Examples:
  ./submit_training.sh --dom-counters

  ./submit_training.sh \
      --dom-counters \
      --override-arch comparison_mcts \
      --vh-off \
      --use-estimator 0.5 \
      --exploration-weight 0.1

  ./submit_training.sh \
      --eval-from /path/to/snapshot \
      --workers 10

  ./submit_training.sh \
      --dom-counters \
      --output-subdir policy_eval

EOF
exit 0
}

########################################
# Argument parsing
########################################

while [[ $# -gt 0 ]]; do
case "$1" in

--help) print_help ;;

--original-only)
    ORIG_TRAINING=("--original-training-set")
    shift
;;

--random-only)
    ORIG_TRAINING=("")
    shift
;;

--override-domains)
    shift
    DOMAIN_LIST=($(split_csv_into_array "$1"))
    shift
;;

--dom-*)
    DOMAIN_LIST=("${1#--dom-}")
    shift
;;

--override-mse)
    shift
    MSE_COEFF_LIST=($(split_csv_into_array "$1"))
    shift
;;

--override-tree-sampling)
    shift
    TREE_SAMPLE_LIST=($(split_csv_into_array "$1"))
    shift
;;

--tree-sample-full-grid)
    TREE_SAMPLE_LIST=("${TREE_SAMPLE_FULL_GRID_VALUES[@]}")
    shift
;;

--override-arch)
    ARCHITECTURE_OVERRIDDEN=true
    shift
    raw_arches=($(split_csv_into_array "$1"))
    ARCHITECTURE_LIST=()
    for a in "${raw_arches[@]}"; do
        ARCHITECTURE_LIST+=("actprop_2l_${a}")
    done
    shift
;;

--domain-architecture)
    shift
    case "$1" in
        policy|mcts) DOMAIN_ARCHITECTURE_MODE="$1" ;;
        *) echo "ERROR: --domain-architecture must be policy or mcts"; exit 1 ;;
    esac
    shift
;;

--tries)
    shift
    TRAINING_TRIES="$1"
    shift
;;

--mcts-expansion-size)
    shift
    MCTS_EXPANSION_SIZE_ARG="--mcts-expansion-size $1"
    shift
;;

--mcts-iterations)
    shift
    MCTS_ITERATIONS_ARG="--mcts-iterations $1"
    shift
;;

--gpu)
    SBATCH_GPU_ARGS="--gpus=rtx_6000:1"
    shift
;;

--enable-profiling)
    ENABLE_PROFILING=1
    shift
;;

--output-subdir)
    shift
    OUTPUT_SUBDIR="$1"
    shift
;;

--vh-off)
    VALUE_HEAD_OPTIONS=("--disable-value-head")
    shift
;;

--vh-on-off)
    VALUE_HEAD_OPTIONS=("" "--disable-value-head")
    shift
;;

--use-estimator)
    shift
    ESTIMATOR_COEFF_OPTIONS=("$1")
    ((ESTIMATOR_MODE_COUNT++))
    shift
;;

--use-estimator-decay)
    ESTIMATOR_DECAY_OPTIONS=("--use-estimator-decay")
    ((ESTIMATOR_MODE_COUNT++))
    shift
;;

--estimator-full-grid)
    ESTIMATOR_COEFF_OPTIONS=("${ESTIMATOR_FULL_GRID_VALUES[@]}")
    ((ESTIMATOR_MODE_COUNT++))
    shift
;;

--exploration-weight)
    shift
    EXPLORATION_WEIGHT_OPTIONS=("$1")
    shift
;;

--exploration-weight-full-grid)
    EXPLORATION_WEIGHT_OPTIONS=("${EXPLORATION_WEIGHT_FULL_GRID_VALUES[@]}")
    shift
;;

--seed)
    shift
    SEED_OPTIONS=($(split_csv_into_array "$1"))
    shift
;;

--seed-full-grid)
    SEED_OPTIONS=("${SEED_FULL_GRID_VALUES[@]}")
    shift
;;

--workers)
    shift
    NUM_WORKERS="$1"
    shift
;;

--jpddl-max-heap)
    shift
    [[ "$1" =~ ^[1-9][0-9]*[kKmMgG]$ ]] || {
        echo "ERROR: --jpddl-max-heap must be a positive integer followed by k, m, or g"
        exit 1
    }
    JPDDL_MAX_HEAP="${1,,}"
    shift
;;

--train-from)
    shift
    RESUME_FROM_ARG="--resume-from $1"
    RESUME_TRAIN_FLAG="--resume-train"
    shift
;;

--eval-from)
    shift
    RESUME_FROM_ARG="--resume-from $1"
    RESUME_TRAIN_FLAG=""
    NO_EVAL_FLAG=""
    shift
;;

--eval-with-mcts)
    EVAL_WITH_MCTS_FLAG="--eval-with-mcts"
    shift
;;
--eval-start-wave)
    shift
    [[ "$1" =~ ^-?[0-9]+$ ]] || {
        echo "ERROR: --eval-start-wave must be an integer"
        exit 1
    }
    EVAL_START_WAVE_ARG="--eval-start-wave $1"
    shift
;;
--eval-scheduling)
    shift
    [[ "$1" == "wave" || "$1" == "rolling" ]] || {
        echo "ERROR: --eval-scheduling must be wave or rolling"
        exit 1
    }
    EVAL_SCHEDULING_ARG="--eval-scheduling $1"
    shift
;;
--skip-instance-numbers)
    shift
    [[ "$1" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || {
        echo "ERROR: expected one-based comma-separated instance numbers"
        exit 1
    }
    SKIP_INSTANCE_NUMBERS_ARG="--skip-instance-numbers $1"
    shift
;;
--eval-instance-timeout)
    shift
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: evaluation timeout must be positive integer seconds"
        exit 1
    }
    EVAL_INSTANCE_TIMEOUT_ARG="--eval-instance-timeout $1"
    shift
;;
--action-debug)
    ACTION_DEBUG_FLAG="--action-debug"
    shift
;;
--puct-debug)
    PUCT_DEBUG_FLAG="--puct-debug"
    shift
;;

--eval)
    NO_EVAL_FLAG=""
    EVAL_REQUESTED=true
    shift
;;

--job-suffix)
    shift
    JOB_SUFFIX="_$1"
    shift
;;

--dry-run)
    DRY_RUN=true
    shift
;;

--time)
    shift
    SBATCH_TIME_ARGS="--time=$1"
    shift
;;

--mem)
    shift
    SBATCH_MEM_ARGS="--mem=$1"
    shift
;;

--cpus)
    shift
    SBATCH_CPU_ARGS="--cpus-per-task=$1"
    shift
;;

--max-opt-epochs)
    shift
    OVERRIDE_MAX_EPOCHS="$1"
    shift
;;

--supervised-lr)
    shift
    OVERRIDE_LR="$1"
    shift
;;

--policy-anchor-kl-coeff)
    shift
    POLICY_ANCHOR_KL_COEFF="$1"
    shift
;;

*)
    echo "Unknown argument: $1"
    exit 1
;;

esac
done

if (( ESTIMATOR_MODE_COUNT > 1 )); then
    echo "ERROR: choose only one of:"
    echo "  --use-estimator X"
    echo "  --use-estimator-decay"
    echo "  --estimator-full-grid"
    exit 1
fi

if [[ -n "$DOMAIN_ARCHITECTURE_MODE" && "$ARCHITECTURE_OVERRIDDEN" == true ]]; then
    echo "ERROR: --domain-architecture cannot be combined with --override-arch"
    exit 1
fi

if [[ -n "$DOMAIN_ARCHITECTURE_MODE" ]]; then
    ARCHITECTURE_LIST=("domain_specific")
fi

########################################
# Summary
########################################

expected_jobs=$((
    ${#DOMAIN_LIST[@]} *
    ${#ARCHITECTURE_LIST[@]} *
    TRAINING_TRIES *
    ${#MSE_COEFF_LIST[@]} *
    ${#TREE_SAMPLE_LIST[@]} *
    ${#ORIG_TRAINING[@]} *
    ${#VALUE_HEAD_OPTIONS[@]} *
    ${#ESTIMATOR_COEFF_OPTIONS[@]} *
    ${#ESTIMATOR_DECAY_OPTIONS[@]} *
    ${#EXPLORATION_WEIGHT_OPTIONS[@]} *
    ${#SEED_OPTIONS[@]}
))

echo
echo "==== SLURM TRAINING CONFIG ===="
echo "Run date: $RUN_DATE"
echo "Run mode prefix: $(run_mode_prefix)"
echo "Output subdir: ${OUTPUT_SUBDIR:-<none>}"
echo "Domains: $(join_by_comma "${DOMAIN_LIST[@]}")"
echo "Architectures: $(join_by_comma "${ARCHITECTURE_LIST[@]}")"
echo "Domain architecture mode: ${DOMAIN_ARCHITECTURE_MODE:-off}"
echo "Tries: $TRAINING_TRIES"
echo "MSE coeffs: $(join_by_comma "${MSE_COEFF_LIST[@]}")"
echo "Tree sample K: $(join_by_comma "${TREE_SAMPLE_LIST[@]}")"
echo "Dataset modes: $(join_by_comma "${ORIG_TRAINING[@]:-(empty means random/realtime)}")"
echo "Value head options: $(join_by_comma "${VALUE_HEAD_OPTIONS[@]:-(on)}")"
echo "Estimator coeffs: $(join_by_comma "${ESTIMATOR_COEFF_OPTIONS[@]}")"
echo "Estimator decay options: $(join_by_comma "${ESTIMATOR_DECAY_OPTIONS[@]:-(off)}")"
echo "Exploration weights: $(join_by_comma "${EXPLORATION_WEIGHT_OPTIONS[@]}")"
echo "Seeds: $(join_by_comma "${SEED_OPTIONS[@]}")"
echo "Workers: $NUM_WORKERS"
echo "Eval enabled: $([[ -z "$NO_EVAL_FLAG" ]] && echo yes || echo no)"
echo "Dry run: $DRY_RUN"
echo "Expected jobs: $expected_jobs"
echo

########################################
# Counters
########################################

success_count=0
fail_count=0

########################################
# Submission
########################################

for domain in "${DOMAIN_LIST[@]}"; do
  for architecture_selector in "${ARCHITECTURE_LIST[@]}"; do

    if [[ "$DOMAIN_ARCHITECTURE_MODE" == "policy" ]]; then
        arch="$domain"
        export ARCHITECTURE_MODULE="experiments_numeric.architecture_2.${domain}"
    elif [[ "$DOMAIN_ARCHITECTURE_MODE" == "mcts" ]]; then
        arch="${domain}_mcts"
        export ARCHITECTURE_MODULE="experiments_numeric.architecture_2.${domain}_mcts"
    else
        arch="$architecture_selector"
        export ARCHITECTURE_MODULE="experiments_numeric.architecture.${arch}"
    fi

    export DOMAIN="$domain"
    export ARCHITECTURE="$arch"
    export OUTPUT_DIR="/home/hersco/training_new_domains/${RUN_DATE}"

    if [[ -n "$OUTPUT_SUBDIR" ]]; then
        OUTPUT_DIR="${OUTPUT_DIR}/${OUTPUT_SUBDIR}"
    fi

    mkdir -p "$OUTPUT_DIR"

    for ((i=1; i<=TRAINING_TRIES; i++)); do
      for mse_coeff in "${MSE_COEFF_LIST[@]}"; do
        for tree_sample in "${TREE_SAMPLE_LIST[@]}"; do
          for orig in "${ORIG_TRAINING[@]}"; do
            for vh_flag in "${VALUE_HEAD_OPTIONS[@]}"; do
              for estimator_coeff in "${ESTIMATOR_COEFF_OPTIONS[@]}"; do
                for estimator_decay_flag in "${ESTIMATOR_DECAY_OPTIONS[@]}"; do
                  for exploration_weight in "${EXPLORATION_WEIGHT_OPTIONS[@]}"; do
                    for seed in "${SEED_OPTIONS[@]}"; do

                        export ESTIMATOR_COEFF="$estimator_coeff"
                        export ESTIMATOR_DECAY_FLAG="$estimator_decay_flag"
                        export EXPLORATION_WEIGHT="$exploration_weight"
                        export SEED="$seed"

                        if [[ "$ESTIMATOR_COEFF" == "0.0" || "$ESTIMATOR_COEFF" == "0" ]]; then
                            export USE_ESTIMATOR_ARG=""
                        else
                            export USE_ESTIMATOR_ARG="--use-estimator $ESTIMATOR_COEFF"
                        fi

                        export USE_ESTIMATOR_DECAY="$ESTIMATOR_DECAY_FLAG"
                        export MSE_COEFF="$mse_coeff"
                        export TREE_SAMPLE="$tree_sample"
                        export ORIGINAL_TRAINING_SET="$orig"
                        export DISABLE_VALUE_HEAD="$vh_flag"
                        export RESUME_FROM_ARG
                        export RESUME_TRAIN_FLAG
                        export NO_EVAL_FLAG
                        export EVAL_WITH_MCTS_FLAG
                        export EVAL_START_WAVE_ARG
                        export EVAL_SCHEDULING_ARG
                        export SKIP_INSTANCE_NUMBERS_ARG
                        export EVAL_INSTANCE_TIMEOUT_ARG
                        export ACTION_DEBUG_FLAG
                        export PUCT_DEBUG_FLAG
                        export NUM_WORKERS
                        export JPDDL_MAX_HEAP
                        export MCTS_EXPANSION_SIZE_ARG
                        export MCTS_ITERATIONS_ARG
                        export POLICY_ANCHOR_KL_COEFF
                        export OVERRIDE_MAX_EPOCHS
                        export OVERRIDE_LR

                        SHORT_ARCH="${arch#actprop_2l_}"
                        SHORT_ARCH="${SHORT_ARCH/comparison/comp}"
                        SHORT_ARCH="${SHORT_ARCH/fluent/flnt}"
                        SHORT_ARCH="${SHORT_ARCH/dynamic/dyn}"

                        if [[ -z "$orig" ]]; then
                          MODE_LABEL="realtime"
                        else
                          MODE_LABEL="original"
                        fi

                        VH_LABEL="$(vh_label "$vh_flag")"
                        EST_LABEL="$(est_label)"
                        EXP_LABEL="$(exploration_label "$exploration_weight")"
                        SEED_LABEL="$(seed_label "$seed")"
                        RUN_PREFIX="$(run_mode_prefix)"

                        CURRENT_JOB_NAME="$(build_job_name \
                          "$domain" \
                          "$SHORT_ARCH" \
                          "$orig" \
                          "$vh_flag" \
                          "$tree_sample" \
                          "$exploration_weight" \
                          "$seed"
                        )"
                        export CURRENT_JOB_NAME

                        ########################################
                        # DRY RUN
                        ########################################

                        if $DRY_RUN; then
                          printf "[DRY] prefix=%-6s domain=%-${DOMAIN_WIDTH}s arch=%-${ARCH_WIDTH}s mse=%-${MSE_WIDTH}s K=%-${TREE_SAMPLE_WIDTH}s mode=%-${MODE_WIDTH}s vh=%-${VH_WIDTH}s est=%-${EST_WIDTH}s c=%-${EXP_WIDTH}s job=%-${JOBNAME_WIDTH}s\n" \
                            "$RUN_PREFIX" \
                            "$DOMAIN" \
                            "$SHORT_ARCH" \
                            "$MSE_COEFF" \
                            "$TREE_SAMPLE" \
                            "$MODE_LABEL" \
                            "$VH_LABEL" \
                            "${EST_LABEL:-none}" \
                            "$EXP_LABEL" \
                            "$CURRENT_JOB_NAME"

                          ((success_count++))
                          continue
                        fi

                        ########################################
                        # REAL SUBMISSION
                        ########################################

                        tmp_err=$(mktemp)
                        job_output=$(sbatch \
                          $SBATCH_GPU_ARGS \
                          $SBATCH_NODE_EXCLUDE_ARGS \
                          $SBATCH_TIME_ARGS \
                          $SBATCH_MEM_ARGS \
                          $SBATCH_CPU_ARGS \
                          --export=ALL,ENABLE_PROFILING="${ENABLE_PROFILING}" \
                          --job-name="$CURRENT_JOB_NAME" \
                          --output="${OUTPUT_DIR}/%j_${CURRENT_JOB_NAME}.txt" \
                          /home/hersco/training_new_domains/run_train_domain.sbatch \
                          2>"$tmp_err")

                        job_id=$(echo "$job_output" | awk '{print $4}')

                        if [[ "$job_id" =~ ^[0-9]+$ ]]; then
                          echo "$job_id" >> all_submitted_job_ids.txt
                          printf "[OK ] job=%-8s name=%s\n" "$job_id" "$CURRENT_JOB_NAME"
                          ((success_count++))
                        else
                          printf "[FAIL] name=%s\n" "$CURRENT_JOB_NAME"
                          cat "$tmp_err"
                          ((fail_count++))
                        fi

                        rm -f "$tmp_err"

                    done
                  done
                done
              done
            done
          done
        done
      done
    done

  done
done

########################################
# Final summary
########################################

echo
echo "==== SUBMISSION SUMMARY ===="

if $DRY_RUN; then
  echo "Total jobs (DRY RUN): $success_count"
else
  echo "Successful: $success_count"
  echo "Failed: $fail_count"
fi

echo "Expected jobs: $expected_jobs"
echo "============================"
echo
