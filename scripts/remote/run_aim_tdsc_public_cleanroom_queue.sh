#!/usr/bin/env bash
set -euo pipefail

# Clean-room public TDSC reproduction.  This queue reads only the materialized
# public split protocol and trains every checkpoint inside its own RUN_ROOT.
# It never copies or links a historical checkpoint or prediction file.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GPU_ID="${GPU_ID:-0}"
TRAIN_SEED="${TRAIN_SEED:-3407}"
CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lzj_env}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-outputs/protocol/aim_tdsc_public_exact_repeated_cv}"
RUN_ROOT="${RUN_ROOT:-outputs/aim_tdsc_public_cleanroom}"
TABLE_ROOT="${TABLE_ROOT:-outputs/tables/aim_tdsc_public_cleanroom}"

if [[ "${TRAIN_SEED}" != "3407" ]]; then
  echo "The locked public reproduction protocol requires TRAIN_SEED=3407." >&2
  exit 2
fi

cd "${PROJECT_DIR}"

active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "${active}" ]]; then
  echo "GPU ${GPU_ID} is busy (compute PIDs: ${active}); refusing to co-run." >&2
  exit 3
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda initialization script not found: ${CONDA_SH}" >&2
  exit 2
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

membership_manifest="${PROTOCOL_ROOT}/materialization_manifest.json"
if [[ ! -s "${membership_manifest}" ]]; then
  echo "Missing materialized public protocol: ${membership_manifest}" >&2
  echo "Run scripts/experiments/materialize_tdsc_public_exact_splits.py first." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${TABLE_ROOT}"
if source_commit="$(git rev-parse HEAD 2>/dev/null)"; then
  printf '%s\n' "${source_commit}" > "${TABLE_ROOT}/source_commit.txt"
else
  printf '%s\n' "unavailable (source archive without Git metadata)" \
    > "${TABLE_ROOT}/source_commit.txt"
fi
sha256sum \
  scripts/experiments/materialize_tdsc_public_exact_splits.py \
  scripts/analysis/summarize_r2plus1d_evidence.py \
  scripts/remote/run_aim_tdsc_public_cleanroom_queue.sh \
  "${membership_manifest}" \
  > "${TABLE_ROOT}/input_and_script_sha256.txt"
python scripts/analysis/capture_runtime_environment.py \
  --output "${TABLE_ROOT}/runtime_environment.json"

# One fold spans all 200 public cases across its three disjoint roles, so this
# shape/path audit covers the complete newly built cache before training.
python scripts/analysis/audit_cached_arrays.py \
  --csv "${PROTOCOL_ROOT}/splits/split_seed_3407/fold_1/train.csv" \
  --csv "${PROTOCOL_ROOT}/splits/split_seed_3407/fold_1/selection_val.csv" \
  --csv "${PROTOCOL_ROOT}/splits/split_seed_3407/fold_1/outer_test.csv" \
  --output "${TABLE_ROOT}/cache_audit.json"

run_cell() {
  local model="$1"
  local split_seed="$2"
  local fold="$3"
  local train_batch eval_batch aug_policy

  case "${model}" in
    r3d)
      train_batch=28
      eval_batch=32
      aug_policy=xy_flip_only
      ;;
    convnext_triplanar_25d_mil)
      train_batch=46
      eval_batch=64
      aug_policy=slice_xy
      ;;
    r2plus1d)
      train_batch=28
      eval_batch=32
      aug_policy=xy_flip_only
      ;;
    *)
      echo "Unsupported model: ${model}" >&2
      return 2
      ;;
  esac

  local split="${PROTOCOL_ROOT}/splits/split_seed_${split_seed}/fold_${fold}"
  local result_prefix="${RUN_ROOT}/results"
  local checkpoint_prefix="${RUN_ROOT}/checkpoints"
  local log_prefix="${RUN_ROOT}/logs"
  if [[ "${model}" == "r2plus1d" ]]; then
    # Match summarize_r2plus1d_evidence.py's dataset/model directory contract.
    result_prefix="${RUN_ROOT}/r2plus1d/results/tdsc"
    checkpoint_prefix="${RUN_ROOT}/r2plus1d/checkpoints/tdsc"
    log_prefix="${RUN_ROOT}/r2plus1d/logs/tdsc"
  fi
  local run="${result_prefix}/${model}/split_seed_${split_seed}/fold_${fold}/train_seed_${TRAIN_SEED}"
  local checkpoint="${checkpoint_prefix}/${model}/split_seed_${split_seed}/fold_${fold}/train_seed_${TRAIN_SEED}/best_model.pth"
  local log_dir="${log_prefix}/${model}/split_seed_${split_seed}/fold_${fold}/train_seed_${TRAIN_SEED}"

  for role in train selection_val outer_test; do
    if [[ ! -s "${split}/${role}.csv" ]]; then
      echo "Missing exact public split: ${split}/${role}.csv" >&2
      return 2
    fi
  done
  mkdir -p "$(dirname "${checkpoint}")" "${log_dir}" "${run}"

  # Resumption is confined to this clean-room output root.  A fresh RUN_ROOT
  # therefore trains all 45 checkpoints from scratch.
  if [[ ! -s "${checkpoint}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train.py \
      --train-csv "${split}/train.csv" \
      --val-csv "${split}/selection_val.csv" \
      --model "${model}" --seed "${TRAIN_SEED}" --epochs 80 \
      --batch-size "${train_batch}" --accum-steps 1 --num-workers 0 \
      --lr 1e-4 --backbone-lr-scale 0.05 --weight-decay 0.05 \
      --warmup-epochs 5 --patience 10 --freeze-backbone-epochs 5 \
      --monitor-start-epoch 1 --target-shape 128,128,64 \
      --spatial-aug-policy "${aug_policy}" \
      --disable-selection-threshold-search --score-weight-auc 1 \
      --score-weight-mcc 0 --score-weight-bias 0 --score-min-delta 0 \
      --min-sensitivity 0 --min-specificity 0 --max-bias-gap 1 \
      --save-path "${checkpoint}" --log-dir "${log_dir}" \
      --swanlab-mode disabled
  fi

  for role in selection_val outer_test; do
    local prediction="${run}/${role}_predictions.csv"
    if [[ ! -s "${prediction}" ]]; then
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/test.py \
        --test-csv "${split}/${role}.csv" --model "${model}" \
        --checkpoint "${checkpoint}" --batch-size "${eval_batch}" \
        --num-workers 0 --target-shape 128,128,64 \
        --export-pred-csv "${prediction}" --swanlab-mode disabled
    fi
  done
  echo "completed public model=${model} split=${split_seed} fold=${fold} train_seed=${TRAIN_SEED}"
}

for split_seed in 3407 42 1234; do
  for fold in 1 2 3 4 5; do
    for model in r3d convnext_triplanar_25d_mil r2plus1d; do
      run_cell "${model}" "${split_seed}" "${fold}"
    done
  done
done

# This summary covers the R3D, TriMIL, and fixed equal-logit fusion results.
# R(2+1)D prediction CSVs remain in the same matched folds for the paired
# comparison analysis.
python scripts/analysis/summarize_triplanar_tdsc_cv.py \
  --result-root "${RUN_ROOT}/results" \
  --output-dir "${TABLE_ROOT}/r3d_trimil_fusion" \
  --bootstrap-repeats 10000 --bootstrap-seed 20260914

python scripts/analysis/summarize_r2plus1d_evidence.py \
  --tdsc-paired "${TABLE_ROOT}/r3d_trimil_fusion/paired_oof_predictions.csv" \
  --r2-result-root "${RUN_ROOT}/r2plus1d/results" \
  --output-dir "${TABLE_ROOT}/r2plus1d" \
  --bootstrap-replicates 10000 --bootstrap-seed 20260914

echo "Public clean-room queue complete."
echo "Models/checkpoints: ${RUN_ROOT}"
echo "Summary tables: ${TABLE_ROOT}"
