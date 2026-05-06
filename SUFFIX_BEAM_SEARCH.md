# Suffix Beam Search Plan

## Goal

Implement and validate the proxy-based search workflow for relayer configurations:

- keep the original benchmark/search problem from the blog
- replace full-answer generation during search with teacher-forced proxy scoring
- make local iteration possible on Mac
- reserve rented GPU time for larger runs after the local path is stable

This branch does **not** keep the old baseline worker path behind runtime flags.
Baseline behavior lives on `main`. Proxy-search work lives on `codex_proxy_search`.

## Current Stage

We are in:

- `Stage 2: local Mac validation`

Status:

- proxy scoring is implemented
- Mac environment is working
- focused tests are passing
- real worker smoke test is passing on CPU
- MPS is still unavailable on this machine

So the code is no longer blocked on basic implementation.
The next meaningful work is benchmark smoke coverage and small end-to-end search validation.

## What Is Already Done

### 1. Proxy scoring is implemented

- Math worker now supports teacher-forced proxy scoring.
- EQ worker now supports teacher-forced proxy scoring.
- Proxy score is based on target-token log-probability, not generated-answer grading.
- EQ proxy only scores the numeric answer tokens, not the emotion labels.
- Math proxy scores all answer tokens.

### 2. Mac local execution path is implemented

- workers accept `--device-map auto|cpu|mps|cuda:0`
- workers accept `--torch-dtype auto|float32|float16|bfloat16`
- workers accept `--dataset-limit` for fast local runs
- beam-search launcher forwards those options
- beam-search launcher runs Math and EQ sequentially when both target the same device

### 3. Repo environment works on Mac

- `uv` is installed locally
- local `.venv` was created with Python 3.11
- `pyproject.toml` now uses CUDA torch wheels only on Linux
- `uv.lock` was refreshed
- `pytest` is available through the `dev` dependency group

### 4. Validation already performed

- focused tests passed:
  - `tests/test_surrogate_utils.py`
  - `tests/test_hf_export.py`
- real Math smoke run passed with:
  - model: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - dataset limit: `1`
  - config `(0,0)` baseline
  - config `(0,1)` duplicated path

Smoke result summary:

- baseline proxy score: `-2.6175`
- duplicated proxy score: `-2.8945`

Important:

- these are log-probability proxy scores, so they are expected to be negative
- closer to `0` is better

## Known Issues / Open Questions

### 1. MPS is not available

Observed:

- `torch.backends.mps.is_built() == True`
- `torch.backends.mps.is_available() == False`

Impact:

- local smoke runs work on CPU
- larger local runs will be slow

### 2. Search quality is not validated yet

We only proved:

- environment works
- worker path works
- proxy path runs end to end

We have **not** yet shown:

- proxy score correlates well enough with real benchmark score
- beam-search using proxy score is behaving sensibly

## Planned Stages

### Stage 1. Proxy implementation

Status: done

- prompt/target construction
- target masking for EQ
- teacher-forced log-prob scoring
- worker integration

### Stage 2. Mac local validation

Status: mostly done

- environment setup
- tests
- real worker smoke run

Remaining:

- investigate MPS
- run EQ smoke test too

### Stage 3. Small end-to-end search smoke

Status: next

Goal:

- run a tiny beam-search smoke test with:
  - tiny HF model
  - tiny dataset limits
  - very small search width/depth

Purpose:

- verify orchestration, result files, and resume behavior
- verify that proxy-scored search produces sane candidate ordering

### Stage 4. Compare proxy search vs real benchmark

Status: pending

Goal:

- take a small candidate set
- score by proxy
- then rescore top candidates with fuller benchmark settings

Purpose:

- measure whether proxy ranking is useful
- decide whether the proxy is good enough for larger experiments

### Stage 5. GPU experiments

Status: pending

Only start after:

- Mac smoke path is stable
- small search smoke works
- proxy ranking looks directionally useful

## Immediate Next Tasks

In order:

1. Run EQ worker smoke test on Mac with the tiny model.
2. Build a minimal beam-search smoke command that uses:
   - tiny model
   - `--dataset-limit`
   - narrow beam width
   - shallow depth
3. Verify the beam-search output files are correct and resumable.
4. Investigate why MPS is unavailable.
5. Decide whether local CPU search is acceptable for tiny validation runs or whether all real search should move to rented GPU.

## Commands We Can Reuse

### Sync environment

```bash
UV_CACHE_DIR=.uv-cache ~/Library/Python/3.9/bin/uv sync --python 3.11 --group dev
```

### Run focused tests

```bash
.venv/bin/python -m pytest tests/test_surrogate_utils.py tests/test_hf_export.py
```

### Math smoke run

```bash
HF_HOME=.hf-cache .venv/bin/python -m src.workers.math_worker \
  --model-path HuggingFaceTB/SmolLM2-135M-Instruct \
  --dataset-path datasets/math_16.json \
  --results-file results/mac_smoke/math_results.pkl \
  --blocks 0,0 \
  --dataset-limit 1 \
  --batch-size 1 \
  --max-new 16 \
  --device-map cpu \
  --torch-dtype float32 \
  --attention-impl eager \
  --no-trust-remote-code
```

## Decision Rule For GPU Rental

Do **not** rent GPU yet just for implementation debugging.

Rent GPU when all three are true:

1. local worker smoke tests pass for Math and EQ
2. small beam-search smoke test works end to end
3. proxy ranking looks at least plausible on a small candidate set

## Branches

- `main`: baseline public repo behavior
- `codex_proxy_search`: proxy-search work
