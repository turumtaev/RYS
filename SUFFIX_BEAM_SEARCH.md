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

   `m in [max(0, k - W + 1), ..., k]`

   and append replay block `(m, k + 1)`

Operationally this means:

- run layer `k`
- feed its output back to layer `m`
- rerun layers `m .. k`
- then continue through suffix `k+1 .. N-1`

Here `W` is the maximum replay-block length. The case `m = k` repeats only
layer `k`; it is distinct from the plain child, which runs layer `k` once.

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

This branch implements the first correctness-first version of the suffix search.

### Implemented

- teacher-forced proxy scoring in `math_worker`
- teacher-forced proxy scoring in `eq_worker`
- EQ numeric-token masking
- uncached partial-layer execution for Llama/Qwen-style decoder stacks
- arbitrary and repeated layer paths without rebuilding the model stack
- boundary-by-boundary local replay expansion
- chunked benchmark proxy ranking and beam pruning
- candidate-owned CPU hidden states at every layer boundary
- prefix reuse across child expansions
- exact generation-based shortlist validation
- held-out validation dataset selection
- Mac-compatible local environment
- small local smoke validation

### Not implemented yet

- replay-budget-diverse beam retention
- intended large-model GPU validation

The cached search now runs end to end and matches full-prefix recomputation.
The next search-quality issue is preventing every beam candidate from spending
a small replay budget only in early layers.

## 8. Current Stage

We are in:

- `Search-quality refinement: replay-budget diversity`

Status:

- proxy scoring is implemented
- Mac environment is working
- focused tests are passing
- real Math and EQ smoke runs are passing
- MPS is available in normal, unsandboxed execution
- partial and repeated layer paths match Hugging Face full-model execution
- the cached suffix-search driver is implemented
- exact shortlist and held-out validation are implemented
- a full 30-boundary tiny-model MPS smoke search completed successfully
- a full 24-boundary Qwen search completed on both 16-example smoke subsets

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
- real EQ smoke run with tiny model
- synthetic Llama full/split/repeated-path equivalence
- synthetic Qwen2 full/sliding/repeated-path equivalence
- real 30-layer SmolLM integration on MPS with exact logits for normal and repeated paths

Smoke result summary:

- baseline proxy score `(0,0)`: `-2.6175`
- duplicated proxy score `(0,1)`: `-2.8945`
- EQ baseline proxy score `(0,0)`: `-2.9236`
- EQ duplicated proxy score `(0,1)`: `-3.9309`

These are log-probability proxy scores, so negative values are expected.
Closer to `0` is better.

## 10. Known Issues

### MPS visibility in sandboxed runs

Observed:

- `torch.backends.mps.is_built() == True`
- inside the restricted Codex sandbox: `is_available() == False`
- outside the sandbox: `is_available() == True`, one MPS device is visible, and tensor allocation succeeds

Impact:

- normal terminal runs can use `--device-map mps`
- Codex-run MPS experiments require elevated execution so Metal is visible

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

Status: done

- Math smoke test passed
- EQ smoke test passed
- MPS availability was verified outside the sandbox

### Stage C. Add low-level partial-layer / suffix scorer

Status: done

Suggested core module:

- `src/core/benchmark_suffix_scorer.py`

First supported architecture family:

- Llama/Qwen-style decoder models exposing `model.layers`

Needed capabilities:

- prepare embeddings, masks, positions, and RoPE inputs
- run an arbitrary contiguous layer range
- run a replay window
- apply final norm and LM head
- initially operate with `use_cache=False`

Required correctness tests:

1. partial runner over all layers matches normal `model.forward()` logits
2. repeated layer path matches the existing `LayerDuplicatedModel` proxy score

Both tests pass for synthetic Llama and Qwen2 models. The cached SmolLM model
also matches exactly on MPS for both normal and repeated paths.

### Stage D. Add suffix search driver

Status: done

The branch version of `scripts/beam_search.py` now implements the new algorithm.
The baseline branch retains the author's original search.

Responsibilities:

1. initialize beam from baseline prefix
2. expand children at boundary `k`
3. score children with the proxy
4. keep top `C`
5. continue through boundaries
6. save shortlist
7. optionally validate shortlist with exact workers

### Stage E. Small end-to-end search smoke

Status: done

Goal:

- tiny model
- tiny dataset limits
- small beam width
- small replay window

Purpose:

- validate orchestration
- validate output files

Observed result with one Math and one EQ example:

- model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- beam width: `2`
- replay window: `2`
- baseline proxy: `-2.770594`
- best final proxy: `-2.260116`
- search time on MPS: `23.4s`

This result proves the mechanics, not benchmark quality. Two examples are far
too few to judge whether the selected replay configuration generalizes.

### Stage F. Correlation / shortlist validation

Status: local held-out validation passed with a strict replay budget

Goal:

- rank a small candidate set by proxy
- then rescore the top candidates with the exact generation-based workers

Purpose:

- test whether proxy ranking is useful enough

Initial four-example result with the tiny model:

- baseline proxy: `-2.846952`
- best beam proxy: `-2.340805`
- baseline exact combined score: `0.435214`
- best-proxy beam exact combined score: `0.335484`
- baseline exact Math / EQ: `0.312095 / 0.558333`
- best-proxy beam exact Math / EQ: `0.049093 / 0.621875`
- proxy order: `beam_1, beam_2, baseline`
- exact order: `baseline, beam_2, beam_1`

The proxy-selected architecture improved exact EQ but substantially harmed
exact Math generation. Its Math outputs collapsed to a repeated large integer.
The mismatch remains with the workers' full generation limits, so it is not a
truncation artifact. This means the current teacher-forced combined objective
is not yet reliable enough to justify cache optimization or GPU-scale search.

However, this is still a tiny-model result. The 135M baseline is itself poor at
the benchmark, so the next correlation test should use a locally runnable model
that can produce valid Math and EQ answers before changing the proxy design.

Follow-up with `Qwen/Qwen2.5-0.5B-Instruct`:

- without a useful extra-layer cap, replay accumulation again caused repetitive
  Math generation and exact regression
- with `max_extra_layers=2`, the search retained two distinct candidates
- search set: first four Math and EQ examples
- held-out exact set: next four Math and EQ examples
- baseline exact combined: `0.414625`
- beam 1 exact combined: `0.509368`
- beam 2 exact combined: `0.466968`
- proxy order: `beam_1, beam_2, baseline`
- held-out exact order: `beam_1, beam_2, baseline`

Conclusion: the teacher-forced objective is useful for shortlist construction
when architecture growth is constrained, but large replay budgets cause the
greedy beam to over-relayer and collapse generation. Exact shortlist validation
remains required. The safe local default is therefore two extra layers; larger
budgets should be tested as an explicit sweep rather than inherited from the
author's different search algorithm.

### Stage G. GPU experiments

Status: pending

Only start after:

1. Math and EQ local smoke tests pass
2. suffix-search smoke test runs end to end
3. proxy ranking looks directionally useful

### Benchmark chunking

Status: done

- teacher-forced benchmark tensors remain in CPU memory
- only one configurable chunk is moved to the accelerator
- all child architectures are scored before that chunk is released
- `--benchmark-chunk-size` defaults to `8`

Qwen MPS equivalence probe with four Math and four EQ examples:

- chunk sizes `1` and `4` produced bit-identical scores and beam paths
- chunk size `1`: `30.3s` through boundary 2
- chunk size `4`: `14.2s` through boundary 2

Small chunks reduce peak residency but add substantial transfer and dispatch
overhead. Chunk size should therefore be the largest value that fits alongside
the model and any future boundary-state cache.

### CPU activation-cache profile

Status: implemented and validated

Qwen2.5-0.5B, float16, boundary 12, four Math plus four EQ examples:

- median prefix recomputation: `47.5ms` per example
- median CPU-to-MPS hidden-state reload: `0.168ms`
- median MPS-to-CPU offload: `0.480ms`
- recomputation / reload ratio: about `283x`
- eight-example cache size: `6.19 MiB` per candidate boundary

Combined 16+16 smoke-subset footprint:

- total teacher-forced tokens: `14,598`
- cache per candidate boundary: `24.95 MiB`
- beam 8: about `200 MiB`
- beam 12: about `299 MiB`
- beam 24: about `599 MiB`

Larger `math_120.json` plus `eq_140.json` footprint (`120 + 139` examples):

- total teacher-forced tokens: `65,905`
- cache per candidate boundary: `112.63 MiB`
- beam 2: about `225 MiB`
- beam 8: about `901 MiB`
- beam 12: about `1.35 GiB`
- beam 24: about `2.70 GiB`

CPU RAM and transfer speed are sufficient on the Mac. The implementation uses
two passes at each boundary:

1. score all children from cached parent states without retaining every child
2. prune globally, then recompute only the top children's short local expansion
   and offload those boundary states for the next step

This keeps resident CPU cache proportional to the beam width instead of the
full branching factor `C * (1 + W)`.

Implemented candidate state:

- partial layer path and replay blocks
- proxy scores
- one CPU hidden-state tensor per Math/EQ benchmark example at the current boundary

At boundary `k`, scoring now:

1. restores each parent's hidden state after `k-1`
2. runs layer `k` once per parent
3. creates plain and replay children from that state
4. runs only suffix `k+1 .. N-1` for proxy scoring
5. prunes globally
6. reruns only the winners' short local expansion and offloads their new states

Validation:

- synthetic cached search matches full-prefix recomputation across boundaries
- Qwen MPS scores and beam paths are bit-identical to uncached runs
- capped two-example search: `30.5s` uncached, `21.9s` cached (`1.39x`)
- continuously branching search: `87.3s` uncached, `52.7s` cached (`1.66x`)
- both 16-example smoke subsets completed all 24 boundaries with a stable `49.9 MiB` cache
- all 120 Math plus 139 EQ examples completed a two-boundary cache restore smoke
- large-dataset peak cache matched the estimate at `225.3 MiB` for beam width 2

## 12. Immediate Next Tasks

In order:

1. Preserve lower-budget candidates so a small replay budget does not get spent only in early layers.
2. Sweep larger replay budgets with exact held-out validation.
3. Profile the cached implementation on the intended GPU/model combination.

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
  --device-map mps \
  --torch-dtype float32 \
  --attention-impl eager \
  --no-trust-remote-code
```

### Full tiny suffix-search smoke

```bash
HF_HOME=.hf-cache .venv/bin/python scripts/beam_search.py \
  --model-path HuggingFaceTB/SmolLM2-135M-Instruct \
  --device-map mps \
  --torch-dtype float32 \
  --dataset-limit 1 \
  --beam-width 2 \
  --replay-window 2 \
  --output results/mac_smoke/suffix_beam_search_full.json
```

### Proxy search with exact shortlist validation

```bash
HF_HOME=.hf-cache .venv/bin/python scripts/beam_search.py \
  --model-path HuggingFaceTB/SmolLM2-135M-Instruct \
  --device-map mps \
  --torch-dtype float32 \
  --dataset-limit 4 \
  --beam-width 2 \
  --replay-window 2 \
  --exact-top-k 2 \
  --math-max-new 64 \
  --eq-max-new 384 \
  --output results/mac_smoke/suffix_beam_search_validation_4_full_generation.json
```

## 14. Branches

- `main`: baseline public repo behavior
- `codex_proxy_search`: proxy scoring and suffix-search implementation
