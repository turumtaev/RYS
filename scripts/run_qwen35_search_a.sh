#!/usr/bin/env bash
set -uo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"

repo_dir="${RYS_REPO_DIR:-/workspace/RYS}"
result_dir="${RYS_RESULT_DIR:-/workspace/results}"
log_dir="${RYS_LOG_DIR:-/workspace/logs}"
mkdir -p "$result_dir" "$log_dir"
cd "$repo_dir"

log_file="$log_dir/qwen35_27b_searchA_small16.log"
exit_file="$log_dir/qwen35_27b_searchA_small16.exit"
rm -f "$exit_file"
exec >"$log_file" 2>&1

.venv/bin/python scripts/beam_search.py \
  --model-path Qwen/Qwen3.5-27B-FP8 \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --dataset-limit 16 \
  --benchmark-batch-size 1 \
  --beam-width 2 \
  --replay-window 12 \
  --max-extra-layers 12 \
  --exact-top-k 24 \
  --exact-batch-size 1 \
  --math-max-new 64 \
  --eq-max-new 384 \
  --output "$result_dir/qwen35_27b_searchA_small16.json"
status=$?
printf '%s\n' "$status" > "$exit_file"
exit "$status"
