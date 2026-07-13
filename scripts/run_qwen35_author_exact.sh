#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"

repo_dir="${RYS_REPO_DIR:-/workspace/RYS}"
result_dir="${RYS_RESULT_DIR:-/workspace/results}"
log_dir="${RYS_LOG_DIR:-/workspace/logs}"
mkdir -p "$result_dir" "$log_dir"
cd "$repo_dir"

log_file="$log_dir/qwen35_27b_author_exact.log"
exit_file="$log_dir/qwen35_27b_author_exact.exit"
rm -f "$exit_file"
exec >"$log_file" 2>&1
trap 'printf "%s\n" "$?" >"$exit_file"' EXIT

configs=(
  --config 'baseline=baseline'
  --config 'single_24_35=blocks:24,35'
  --config 'layer10_x3=repeat:10,2'
  --config 'beam_le20=blocks:43,45;28,34'
  --config 'beam_overall=blocks:39,45;24,35;9,20;43,46'
)

.venv/bin/python scripts/evaluate_exact_configs.py \
  --model-path Qwen/Qwen3.5-27B-FP8 \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --math-dataset-path datasets/math_16.json \
  --eq-dataset-path datasets/eq_16.json \
  --output "$result_dir/qwen35_27b_author_exact_small.json" \
  "${configs[@]}"
