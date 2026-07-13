# Benchmark-Conditioned Suffix Beam Search

This file is the internal engineering log used while developing the feature
with Codex. It preserves design decisions, intermediate measurements, failed
experiments, and commands. It is intentionally chronological and repetitive;
[`README.md`](README.md) is the concise human-facing description.

It has three jobs:

1. preserve the **original target algorithm**
2. record the implementation history and experiment evidence
3. keep the current limitations and next engineering steps explicit

Current status, updated 2026-07-14:

- the boundary-cached, budget-indexed suffix beam search is implemented
- an end-to-end `Qwen/Qwen3.5-27B-FP8` search completed on an 80 GB H100
- the selected candidates and four RYS II Pareto configurations were evaluated
  on all 120 Math and 139 EQ examples with corrected full-scale EQ references
- the final results are summarized in `README.md`
- remaining work is optimization and search-space expansion, not initial
  implementation

> **Historical EQ metric warning:** local exact EQ and combined-score results
> recorded below before the Qwen3.5 GPU comparison used the upstream
> `reference_answer` field. That field normalizes the four EQ labels together,
> while the prompt and generated answers use independent 0-10 scores. Those
> historical exact EQ and combined numbers are preserved as debugging evidence
> and were not rerun. Proxy scores and Math scores are unaffected. The final
> 120+139 comparison uses `reference_answer_fullscale`.

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

> if the remaining model layers are fixed, benchmark quality depends on the
> hidden-state representation handed to those layers.

Here, prefix and suffix mean sections of the model's layer execution path, not
token prefixes or suffixes.

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

- Math: teacher-forced log-probability of the canonical correct integer answer and EOS
- EQ: teacher-forced log-probability of the canonical first-pass score block
- EQ masking: score each numeric token island plus its following delimiter or EOS

The terminator tokens are important: they reward assigning probability to a
complete answer rather than to the correct numeric prefix followed by more digits.

Qwen2.5-0.5B validation after adding terminators:

> The exact combined values in this subsection are covered by the historical
> EQ metric warning at the top of this file.

- Math target scores every answer token plus `<|im_end|>`
- EQ scores each numeric token plus newline, with `<|im_end|>` after the final score
- uncapped two-example baseline exact combined score: `0.462018`
- top termination-aware proxy candidate exact combined score: `0.485587`
- proxy and exact top rank agree
- the previous repeated giant-integer generation collapse was not observed

Before termination scoring, the comparable uncapped proxy winner scored about
`0.364` exactly and repeatedly generated extremely long integers. This supports
including answer termination in the search objective.

This proxy is already implemented in this branch.

## 7. What The Current Branch Actually Implements

This branch implements the first correctness-first end-to-end version of the
suffix search.

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
- budget-indexed beams with independent pruning per replay-layer budget
- Qwen3.5 partial-layer execution, hybrid attention, and cache-position support
- corrected `reference_answer_fullscale` EQ evaluation in Hugging Face and
  ExLlama workers
- explicit exact evaluation of arbitrary baseline, author, and search configs
- reproducible Qwen3.5-27B GPU launchers
- Mac-compatible local environment
- small local smoke validation
- complete 64-boundary Qwen3.5-27B GPU search
- full 120 Math + 139 EQ corrected comparison

### Not implemented yet

- reliable variable-length cross-example batching for Qwen3.5
- production-safe sibling-candidate batching
- nested replays that can modify the pending suffix of an earlier replay
- a corrected rerun of the author's full search candidate set
- broad Qwen3.5-27B hyperparameter tuning

The cached budget-indexed search now runs end to end and preserves candidates
that spend their replay budget at later boundaries.

## 8. Current Stage

We are in:

- `Post-experiment analysis and next search-space design`

Status:

- all planned correctness-first search components are implemented
- Mac CPU/MPS smoke and equivalence tests pass
- the Qwen3.5-27B H100 search and exact large-set comparison are complete
- `beam_23`, replay `(43,44)`, is the selected one-extra-layer result
- `beam_20`, replays `(43,44);(45,46);(47,48)`, is the selected
  three-extra-layer result and has the highest corrected average among the 29
  configurations evaluated in the comparison
- cross-example proxy batching remains disabled for Qwen3.5
- sibling batching is preserved only on `codex/sibling-batching-wip`
- nested replay paths remain unsupported

## 9. Completed Work

### A. Proxy infrastructure

- Math worker supports teacher-forced proxy scoring
- EQ worker supports teacher-forced proxy scoring
- proxy score is target-token log-probability
- Math scores all answer tokens
- Math also scores the final EOS token
- EQ scores numeric answer tokens plus the following newline or EOS token

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

### Qwen3.5 cross-example batching

The final GPU search used benchmark batch size 1. Batch size 2 initially
produced `NaN` EQ proxy scores. Fixing a cached-padding materialization bug
removed that NaN source, but variable-length batch scores still differed from
batch-1 scores at boundary 0. Almost every EQ example has a different length,
so ordinary length sorting does not produce useful equal-length groups.

Until packed-sequence or another padding-independent path is validated,
Qwen3.5 defaults to batch size 1 for correctness.

### Upstream EQ reference bug

This fork's Hugging Face and ExLlama exact evaluators now use
`reference_answer_fullscale`. The original upstream repository still needs the
isolated fix and regression test. Historical search rankings cannot be repaired
from the published Pareto table alone because the incorrect EQ score was part
of candidate selection.

### Sibling-candidate batching

Children of one parent can be batched after their local replays because they
have equal sequence length and execute the same remaining layer suffix. That
implementation is preserved at commit `92e3567` on branch
`codex/sibling-batching-wip`.

It was not used for the final search. With replay window 12, sibling batch size
grows to 13, and the longest EQ sequence made the 80 GB peak-memory risk too
high for the rented run. Its small BF16/FP8 score differences also need ranking
stability validation.

### Nested replay paths

The current candidate stores a flattened completed prefix. It can represent
`1,2,3,1,2,3`, but cannot insert another replay inside that replay to construct
`1,2,3,1,2,1,2,3`.

The proposed generalization stores a completed prefix and pending suffix for
every candidate. If, at boundary `k`, replaying `k-2,k-1,k` before the old
suffix is selected, retain:

```text
prefix = old_prefix + [k]
suffix = [k-2, k-1, k] + old_suffix
```

Candidates should then be scheduled by the smallest unfinished layer boundary.
This requires new deduplication, replay-budget, cache-lifetime, and pruning
rules.

## 11. Implementation and Experiment Stages

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

Actual implementation:

- `LlamaLikePartialRunner` and model-stack adapters in
  `src/workers/model_utils.py`

Supported architecture family:

- Llama/Qwen-style decoder models with discoverable stacks such as
  `model.layers` and `model.language_model.layers`

Implemented capabilities:

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
The original search remains available in the upstream
[`dnhkng/RYS`](https://github.com/dnhkng/RYS) repository.

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

Status: historical local experiment complete; exact EQ/combined values were
later found to use the wrong reference scale

Goal:

- rank a small candidate set by proxy
- then rescore the top candidates with the exact generation-based workers

Purpose:

- test whether proxy ranking is useful enough

Initial four-example result with the tiny model:

> Historical metric note: the proxy and Math values below remain useful, but
> the exact EQ and combined values predate the full-scale EQ reference fix.
> They must not be compared with the corrected Qwen3.5 results.

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
truncation artifact. At the time, this established that proxy ranking alone was
not sufficient and motivated answer-termination scoring, replay-budget caps,
and exact shortlist validation.

This was a tiny-model debugging result. The 135M baseline was itself poor at the
benchmark, so the next historical step used a stronger locally runnable model.

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

The conclusion at the time was that the teacher-forced objective was useful for
shortlist construction when architecture growth was constrained, while large
replay budgets could collapse generation. Exact shortlist validation is still
required, but the exact combined rankings in this local experiment are not
current benchmark evidence because of the EQ reference bug.

### Stage G. GPU experiments

Status: complete for the first Qwen3.5-27B experiment

Search setup:

- model: `Qwen/Qwen3.5-27B-FP8`
- hardware: one 80 GB H100
- proxy set: 16 Math + 16 EQ examples
- beam width: 2 per exact replay budget
- replay window: 12
- maximum extra layers: 12
- benchmark batch size: 1
- completed boundaries: all 64 layers
- search time: `36,958s` (`10.27h`)
- peak CUDA allocated/reserved: `29.9 / 42.5 GiB`
- peak CPU boundary cache: `3.54 GiB`

Exact comparison setup:

- 120 Math + 139 EQ examples
- corrected EQ field: `reference_answer_fullscale`
- baseline + four RYS II Pareto configs + 24 search candidates
- common Hugging Face/Transformers evaluator, batch size 1

Selected corrected results:

| Label | Replays | Extra | Math | EQ | Average |
|---|---|---:|---:|---:|---:|
| baseline | none | 0 | `0.969651` | `0.647302` | `0.808476` |
| `beam_23` | `(43,44)` | 1 | `0.998712` | `0.669784` | `0.834248` |
| `beam_20` | `(43,44);(45,46);(47,48)` | 3 | `0.994119` | `0.676349` | `0.835234` |

The full interpretation and comparison caveats are in `README.md`. In
particular, the author's reported configs were selected with the old EQ metric,
so this corrected rerun is not a symmetric comparison of search methods.

### Benchmark chunking

Status: done

- teacher-forced benchmark tensors remain in CPU memory
- only one configurable example group is moved to the accelerator
- that group is now evaluated as a real padded batch
- `--benchmark-chunk-size` remains as an alias for `--benchmark-batch-size`

### Padding-aware proxy batching

Status: infrastructure implemented; disabled for Qwen3.5 after CUDA profiling

- examples are right-padded within each batch
- examples are sorted by tokenized length within each benchmark before batching
- attention masks exclude padding from decoder attention
- full-sequence score masks support different prompt and target lengths
- EOS and EQ delimiter tokens remain scored
- proxy reductions happen per example before benchmark averaging
- masked reductions are vectorized and accumulated in float32
- cached hidden states remain unpadded in CPU RAM and are padded only on reload

Qwen MPS probe with four Math and four EQ examples:

- batch 1: `14.6s`
- batch 4 before length sorting/vectorized reduction: `16.4s`
- optimized batch 4: `15.6s`
- both runs retained identical beam paths
- maximum score difference was `1.22e-4` in float16

Batching did not help this MPS workload because EQ padding and large vocabulary
logits outweighed launch savings. CUDA profiling on Qwen3.5-27B later found
`NaN` EQ scores at batch size 2. A padding-cache fix removed the NaNs but did
not restore score equivalence with batch size 1. The automatic default is now
batch 1 for CPU/MPS and Qwen3.5; other CUDA architectures may use larger
batches after validation.

### Global-beam hyperparameter baseline

Status: historical local tuning pass complete; combined scores use the old EQ
reference scale

Setup:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- proxy search: first four Math and EQ examples
- exact validation: next four Math and EQ examples
- termination-aware objective

Results:

| Beam | Window | Budget | Search | Best exact |
|---:|---:|---:|---:|---:|
| 2 | 1 | 2 | `45.2s` | `0.509368` |
| 2 | 2 | 2 | `47.4s` | `0.509368` |
| 4 | 2 | 2 | `86.6s` | `0.509368` |
| 2 | 2 | 4 | `74.6s` | `0.490254` |

This pass selected beam width `2`, replay window `1`, and replay budget `2` for
the next local engineering comparison. It was a deliberately small, noisy pass
and is preserved as implementation history, not as current production tuning.

### Budget-indexed beam comparison

Status: implementation complete; quality comparison is historical because the
combined score uses the old EQ reference scale

The search now keeps an independent beam for every exact replay-layer budget.
At boundary `k`, budget `i` receives:

- a plain child from budget `i`
- a replay of length `L` from budget `i-L`

Using window `1` and maximum budget `2`:

| Method | Width | Search | Best exact |
|---|---:|---:|---:|
| Global | 2 total | `45.2s` | `0.509368` |
| Global | 4 total | `86.6s` | `0.509368` |
| Budget-indexed | 1 per budget | `89.3s` | `0.461896` |
| Budget-indexed | 2 per budget | `135.9s` | **`0.547777`** |

Budget-indexed width 2 found the later-layer configuration `(4,5);(11,12)`.
Its best exact candidate was second by proxy within budget 2, which motivated
retaining more than the single proxy winner for exact validation.

At approximately matched local runtime, global width 4 scored above
budget-indexed width 1 under the historical metric. The main durable result is
that budget-indexed pruning worked and preserved later-budget candidates; the
quality comparison itself should not be reused.

### Budget-indexed hyperparameter pass

Status: historical local tuning pass complete; combined scores use the old EQ
reference scale

The pass kept width `2` per budget and varied one parameter at a time from the
initial budget-indexed result:

| Width | Window | Budget | Search | Peak cache | Best exact |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 2 | `135.9s` | `31.0 MiB` | `0.547777` |
| 2 | 2 | 2 | `153.5s` | `31.0 MiB` | `0.547777` |
| 2 | 1 | 4 | `243.2s` | `55.8 MiB` | **`0.605051`** |

Window `2` retained the same best configurations as window `1` and only added
search cost. Budget `4` found two useful three-layer configurations. The best
held-out exact candidate replayed `(7,8);(11,12);(15,16)` and scored
`0.605051`; it ranked fourth by proxy, which reinforces the need to validate a
shortlist rather than only the proxy winner.

This pass selected width `2` per budget, replay window `1`, and maximum replay
budget `4` for subsequent engineering work. It used only four proxy examples
and four validation examples per benchmark and the pre-fix EQ evaluator, so it
is not a current hyperparameter recommendation.

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
2. prune independently within each budget, then recompute only the surviving
   children's short local expansion
   and offload those boundary states for the next step

This keeps resident CPU cache proportional to the retained candidates instead
of the full branching factor. Budget-indexed search retains up to
`1 + max_budget * beam_width` candidates because budget 0 has only the baseline.

Implemented candidate state:

- partial layer path and replay blocks
- proxy scores
- one CPU hidden-state tensor per Math/EQ benchmark example at the current boundary

At boundary `k`, scoring now:

1. restores each parent's hidden state after `k-1`
2. runs layer `k` once per parent
3. creates plain and replay children from that state
4. runs only suffix `k+1 .. N-1` for proxy scoring
5. prunes independently inside each replay-budget cell
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

1. Submit the isolated upstream fix that changes exact EQ scoring from
   `reference_answer` to `reference_answer_fullscale`, with a regression test.
2. Specify and test the generalized candidate state needed for nested replays:
   completed layer prefix, pending layer suffix, replay budget, and boundary
   cache ownership.
3. Only revisit Qwen3.5 batching if more GPU search is planned: packed-sequence
   cross-example batching first, then sibling batching with explicit OOM and
   ranking-stability tests.
4. For a symmetric comparison with RYS II, rerun the author's complete search
   candidate set or rerun the search itself under the corrected EQ objective
   and, ideally, the original ExLlamaV3 backend.

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

### Recorded Qwen3.5-27B GPU experiment

```bash
./scripts/run_qwen35_search_a.sh
./scripts/run_qwen35_large_comparison.sh
```

## 14. Branches

- `main`: published suffix-search implementation and corrected evaluator
- `suffix-beam-search`: same published implementation history as `main`
- `codex/sibling-batching-wip`: experimental sibling batching at commit
  `92e3567`; not used for the final GPU search
- [`dnhkng/RYS`](https://github.com/dnhkng/RYS): original upstream behavior
