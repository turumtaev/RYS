#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"

repo_dir="${RYS_REPO_DIR:-/workspace/RYS}"
result_dir="${RYS_RESULT_DIR:-/workspace/results}"
log_dir="${RYS_LOG_DIR:-/workspace/logs}"
mkdir -p "$result_dir" "$log_dir"
cd "$repo_dir"

log_file="$log_dir/qwen35_27b_large_comparison.log"
exit_file="$log_dir/qwen35_27b_large_comparison.exit"
rm -f "$exit_file"
exec >"$log_file" 2>&1
trap 'printf "%s\n" "$?" >"$exit_file"' EXIT

# Run the external comparison first, then our candidates in corrected small-set order.
configs=(
  --config 'baseline=baseline'
  --config 'author_33_34=blocks:33,34'
  --config 'author_31_34=blocks:31,34'
  --config 'author_30_35=blocks:30,35'
  --config 'author_26_34=blocks:26,34'
  --config 'beam_18=blocks:27,28;31,32;43,44;45,46'
  --config 'beam_4=blocks:21,22;27,28;31,32;41,42;43,44;45,46'
  --config 'beam_15=blocks:27,28;31,32;41,42;43,44;45,46'
  --config 'beam_17=blocks:27,28;43,44;45,46;47,48'
  --config 'beam_5=blocks:21,22;27,28;31,32;41,42;43,44;44,46;44,47'
  --config 'beam_22=blocks:27,28;43,44'
  --config 'beam_20=blocks:43,44;45,46;47,48'
  --config 'beam_7=blocks:21,22;27,28;31,32;41,42;43,44;45,46;42,47'
  --config 'beam_24=blocks:45,46'
  --config 'beam_12=blocks:27,28;31,32;41,42;43,44;45,46;44,47'
  --config 'beam_3=blocks:21,22;27,28;31,32;41,42;43,44;44,45;45,46;44,47'
  --config 'beam_11=blocks:21,22;27,28;31,32;41,42;43,44;45,46;41,47'
  --config 'beam_10=blocks:21,22;27,28;31,32;41,42;43,44;44,45;45,46;44,48'
  --config 'beam_21=blocks:43,44;45,46'
  --config 'beam_19=blocks:27,28;43,44;45,46'
  --config 'beam_1=blocks:21,22;27,28;31,32;41,42;43,44;44,46'
  --config 'beam_14=blocks:21,22;27,28;30,31;31,32;43,44;45,46'
  --config 'beam_8=blocks:21,22;27,28;31,32;41,42;43,44;44,45;45,46;42,47'
  --config 'beam_2=blocks:21,22;27,28;31,32;41,42;43,44;44,45;45,46'
  --config 'beam_16=blocks:27,28;31,32;43,44;45,46;47,48'
  --config 'beam_9=blocks:21,22;27,28;31,32;41,42;43,44;44,45;45,46;45,47'
  --config 'beam_23=blocks:43,44'
  --config 'beam_6=blocks:21,22;27,28;31,32;41,42;43,44;45,46;44,47'
  --config 'beam_13=blocks:21,22;27,28;31,32;31,33;41,42;43,44;45,46'
)

.venv/bin/python scripts/evaluate_exact_configs.py \
  --model-path Qwen/Qwen3.5-27B-FP8 \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --math-dataset-path datasets/math_120.json \
  --eq-dataset-path datasets/eq_140.json \
  --output "$result_dir/qwen35_27b_large_comparison.json" \
  "${configs[@]}"
