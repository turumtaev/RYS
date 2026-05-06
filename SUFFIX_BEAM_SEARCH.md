# Benchmark-Conditioned Suffix Beam Search

This file is the canonical plan for this branch.

It has two jobs:

1. preserve the **original target algorithm**
2. record the **current implementation stage**

The most important correction is:

> the current branch does **not** yet implement the full suffix beam search algorithm.
> It implements the teacher-forced proxy scoring infrastructure that we plan to use inside that algorithm.

## 1. Original Target

The target is a new search method for finding the best **static relayer configuration** on the benchmark.

It is:

- not a token beam search
- not a runtime inference policy
- not a dynamic model that changes per user request

The output is still a fixed final relayer config, such as:

- a list of repeated blocks
- or an equivalent expanded layer path

## 2. Problem Setting

The blog's workflow is:

1. brute-force all single repeated blocks `(i, j)`
2. rank them on the benchmark
3. search for stronger multi-block configs because exhaustive search is too expensive

Your proposal targets the **same optimization problem**:

- objective: find the best fixed relayer config for the benchmark
- benchmark: the same Math/EQ probes used by the current repo
- novelty: evaluate search expansions more efficiently by reusing benchmark-side cached work

## 3. Core Idea

### Short version

At boundary `k`, treat layers `k+1 .. N-1` as a fixed suffix.

For each current beam candidate, compare:

1. plain continuation through layer `k`
2. replay continuation that repeats a recent block ending at `k`

Then score all children on the benchmark using the same suffix and keep only the top beam candidates.

So this is a beam search over **partial architectures**, not over tokens.

### More precise formulation

Let the base transformer layers be:

`f_0, f_1, ..., f_{N-1}`

At stage `k`, a beam element is a **partial config** that defines execution up to `k-1`.

The remaining suffix:

`S_{k+1} = f_{N-1} o ... o f_{k+1}`

is treated as fixed while we compare local choices ending at `k`.

For each beam candidate, generate children:

1. Plain child

   run layer `k` normally

2. Replay child

   choose `m` from a local window, for example:

   `m in [max(0, k - W + 1), ..., k - 1]`

   and append replay block `(m, k + 1)`

Operationally this means:

- run layer `k`
- feed its output back to layer `m`
- rerun layers `m .. k`
- then continue through suffix `k+1 .. N-1`

Each beam candidate therefore produces:

- one plain child
- up to `W` replay children

All of them are scored on the benchmark, and we keep the top `C`.

## 4. Why the Suffix View Helps

The useful observation is:

> if the suffix is fixed, benchmark quality depends on the representation handed to that suffix.

That makes an incremental search possible.

Instead of scoring every multi-block config from scratch, we can:

- grow the architecture boundary by boundary
- compare only local replay choices ending at the current boundary
- reuse benchmark-side work for shared prefixes and shared suffix structure
- prune aggressively with a beam

## 5. Where the Reuse Comes From

The reuse is for **benchmark evaluation during search**.

At a given step:

- benchmark inputs are the same for all children
- much of the prefix is shared
- the suffix boundary is shared

So reusable state may include:

- tokenized benchmark prompts
- prompt-side KV or equivalent cached activations
- hidden states at layer boundaries
- suffix-side computation that only needs to be recomputed after the modified replay window

The engineering goal is:

> do not reevaluate every child from scratch; rerun only the changed replay window and the dependent suffix work.

## 6. Search Objective

The ideal search uses benchmark quality, but exact generation-based scoring is too expensive for the inner loop.

So the practical plan is:

- use a fast deterministic proxy during search
- validate the shortlist with the current exact workers

Current proxy choice:

- Math: teacher-forced log-probability of the canonical correct integer answer
- EQ: teacher-forced log-probability of the canonical first-pass score block
- EQ masking: score only the numeric tokens, not the emotion labels

This proxy is already implemented in this branch.

## 7. What The Current Branch Actually Implements

This branch currently implements only the **supporting infrastructure** for the future suffix search:

### Implemented

- teacher-forced proxy scoring in `math_worker`
- teacher-forced proxy scoring in `eq_worker`
- EQ numeric-token masking
- Mac-compatible local environment
- small local smoke validation

### Not implemented yet

- boundary-by-boundary suffix beam search
- replay-window candidate expansion over partial architectures
- benchmark-side KV / activation reuse across children
- a new suffix-search driver script
- exact shortlist validation flow wired to the new search

So we are **not** at "suffix search works".
We are at "proxy scoring infra works and can be used to build suffix search".

## 8. Current Stage

We are in:

- `Stage B: local validation of the proxy/search infrastructure`

Status:

- proxy scoring is implemented
- Mac environment is working
- focused tests are passing
- real Math smoke run is passing on CPU
- MPS is still unavailable on this machine

## 9. Completed Work

### A. Proxy infrastructure

- Math worker supports teacher-forced proxy scoring
- EQ worker supports teacher-forced proxy scoring
- proxy score is target-token log-probability
- Math scores all answer tokens
- EQ scores only numeric answer tokens

### B. Mac local execution path

- workers accept `--device-map auto|cpu|mps|cuda:0`
- workers accept `--torch-dtype auto|float32|float16|bfloat16`
- workers accept `--dataset-limit`
- beam launcher forwards those options
- beam launcher serializes Math/EQ workers when they target the same device

### C. Environment and validation

- local `.venv` created with Python 3.11
- `uv` setup works on Mac
- `pyproject.toml` uses CUDA torch wheels only on Linux
- `uv.lock` refreshed
- `pytest` available via `dev` group

Validation already done:

- `tests/test_surrogate_utils.py`
- `tests/test_hf_export.py`
- real Math smoke run with tiny model

Smoke result summary:

- baseline proxy score `(0,0)`: `-2.6175`
- duplicated proxy score `(0,1)`: `-2.8945`

These are log-probability proxy scores, so negative values are expected.
Closer to `0` is better.

## 10. Known Issues

### MPS unavailable

Observed:

- `torch.backends.mps.is_built() == True`
- `torch.backends.mps.is_available() == False`

Impact:

- CPU smoke runs work
- larger local runs are slower than they should be

### Search algorithm still missing

We have proved:

- environment works
- proxy worker path works

We have not yet proved:

- suffix beam search orchestration works
- proxy ranking is useful enough inside that search
- cache reuse gives real speedup

## 11. Planned Stages

### Stage A. Proxy implementation

Status: done

- prompt/target construction
- target masking for EQ
- teacher-forced log-prob scoring
- worker integration

### Stage B. Local Mac validation

Status: mostly done

Remaining:

- run EQ smoke test
- understand MPS availability

### Stage C. Add suffix search driver

Status: next

Build the real algorithm as a separate path, not by replacing the current `scripts/beam_search.py` immediately.

Suggested script:

- `scripts/benchmark_suffix_beam_search.py`

Suggested responsibilities:

1. initialize beam from baseline prefix
2. expand children at boundary `k`
3. score children with the proxy
4. keep top `C`
5. continue through boundaries
6. save shortlist
7. optionally validate shortlist with exact workers

### Stage D. Add low-level suffix scorer

Status: pending

Suggested core module:

- `src/core/benchmark_suffix_scorer.py`

Needed capabilities:

- access decoder layers / final norm / LM head
- run model from an arbitrary boundary
- materialize replay-window children
- batch child scoring on benchmark data
- manage reusable benchmark-side caches or activations

### Stage E. Small end-to-end search smoke

Status: pending

Goal:

- tiny model
- tiny dataset limits
- small beam width
- small replay window

Purpose:

- validate orchestration
- validate output files
- validate resume behavior

### Stage F. Correlation / shortlist validation

Status: pending

Goal:

- rank a small candidate set by proxy
- then rescore the top candidates with the exact generation-based workers

Purpose:

- test whether proxy ranking is useful enough

### Stage G. GPU experiments

Status: pending

Only start after:

1. Math and EQ local smoke tests pass
2. suffix-search smoke test runs end to end
3. proxy ranking looks directionally useful

## 12. Immediate Next Tasks

In order:

1. Run EQ worker smoke test on Mac with the tiny model.
2. Restore the original target at the top of this plan file.
   Status: done.
3. Build a minimal `benchmark_suffix_beam_search.py` skeleton.
4. Implement boundary expansion logic for:
   - plain child
   - replay child `(m, k+1)` in a limited window
5. Hook the skeleton to the current proxy scorer first, even without cache reuse.
6. After that works, add benchmark-side reuse.

## 13. Commands We Can Reuse

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

## 14. Branches

- `main`: baseline public repo behavior
- `codex_proxy_search`: proxy infra and future suffix-search work
