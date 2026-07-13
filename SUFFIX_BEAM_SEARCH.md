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

- Math: teacher-forced log-probability of the canonical correct integer answer and EOS
- EQ: teacher-forced log-probability of the canonical first-pass score block
- EQ masking: score each numeric token island plus its following delimiter or EOS

The terminator tokens are important: they reward assigning probability to a
complete answer rather than to the correct numeric prefix followed by more digits.

Qwen2.5-0.5B validation after adding terminators:

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
- budget-indexed beams with independent pruning per replay-layer budget
- Mac-compatible local environment
- small local smoke validation

### Not implemented yet

- budget-indexed hyperparameter tuning
- intended large-model GPU validation

The cached budget-indexed search now runs end to end and preserves candidates
that spend their replay budget at later boundaries.

## 8. Current Stage

We are in:

- `Search-quality refinement: budget-indexed tuning`

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
- only one configurable example group is moved to the accelerator
- that group is now evaluated as a real padded batch
- `--benchmark-chunk-size` remains as an alias for `--benchmark-batch-size`

### Padding-aware proxy batching

Status: implemented; CUDA profiling pending

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
logits outweighed launch savings. The automatic default is therefore batch 1
for CPU/MPS and batch 8 for CUDA. CUDA batch size still needs profiling.

### Global-beam hyperparameter baseline

Status: small local tuning pass complete

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

The current global-beam baseline is therefore beam width `2`, replay window
`1`, and replay budget `2`. Wider search improved proxy scores but not held-out
exact quality. This is a deliberately small, noisy tuning pass; it selects a
reasonable baseline for testing budget-indexed beams, not final production
hyperparameters.

### Budget-indexed beam comparison

Status: implementation and first comparison complete

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

Budget-indexed width 2 found the later-layer configuration `(4,5);(11,12)`,
which improved held-out Math enough to beat the global baseline. Its best exact
candidate was second by proxy within budget 2, so exact shortlist validation
remains necessary.

At approximately matched local runtime, global width 4 beat budget-indexed
width 1. Budget indexing therefore improved maximum observed quality, not
compute efficiency. Because it showed a quality gain, a small budget-indexed
hyperparameter pass is justified next.

### Budget-indexed hyperparameter pass

Status: small local tuning pass complete

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

The selected local configuration is therefore width `2` per budget, replay
window `1`, and maximum replay budget `4`. This selection used the first four
examples for proxy search and the next four for exact validation, so it must be
confirmed on a fresh test split before treating the improvement as evidence of
generalization.

### Untouched Mac test

Status: complete

The fixed proxy shortlist was evaluated on examples `8-11`, which were not used
for proxy search or hyperparameter selection:

| Candidate | Replays | Exact score |
|---|---|---:|
| Baseline | none | `0.519584` |
| Selected | `(7,8);(11,12);(15,16)` | **`0.554819`** |

The absolute improvement was `+0.035235`. This is positive but substantially
smaller than the `+0.190426` improvement on the tuning validation split. The
0.5B experiment therefore validates the implementation and search signal, not
a general claim about relayering quality.

## 12. GPU Experiment Design

### Author hardware and workflow

The original Qwen2-72B discovery in RYS Part I was explicitly performed on two
RTX 4090s. RYS Part II describes the author's newer machine as two GH200
superchips with H100 GPUs and `192 GB` total HBM3, and says that most of the
newer scanning used FP8 models on that Hopper system. It does not establish
that every Qwen3.5 experiment required or occupied both GH200 GPUs. The Part II
search used the 16-question Math and EQ probes, then re-measured shortlisted
candidates on Math120 and EQ140 (actually 139 EQ examples).

Qwen3.5-27B has 64 layers, hidden size 5120, and a hybrid stack containing
Gated DeltaNet and attention layers. Qwen3.5-0.8B has the same hybrid pattern at
24 layers and hidden size 1024, making it the appropriate local architecture
gate before renting a GPU.

Local Qwen3.5-0.8B MPS validation is complete:

- native and partial-runner logits were bit-identical across the ordinary
  24-layer path (`max_abs = 0`)
- logits from both paths were finite
- a cached budget-indexed replay smoke completed through boundary 4
- replay paths crossed both DeltaNet and attention layers successfully

The local fallback uses the slower PyTorch DeltaNet implementation because the
optional Flash Linear Attention and causal-conv1d packages are not installed.
That affects speed, not the equivalence result.

A complete small search also passed end to end using four proxy examples and
the next four examples for exact validation:

| Candidate | Replays | Math | EQ | Combined |
|---|---|---:|---:|---:|
| Baseline | none | `0.220730` | `0.570476` | `0.395603` |
| Proxy winner | `(13,14);(19,20);(20,21)` | `0.220730` | `0.625476` | **`0.423103`** |

Search settings were width `2` per budget, window `1`, and budget `4`. The
search took `509.1s` and peaked at `64.8 MiB` of activation cache. The exact
improvement was `+0.0275`, entirely from EQ; the other three proxy-shortlisted
candidates regressed. This is enough to validate the complete Qwen3.5 path, but
not enough data to infer optimal 27B search parameters.

### Why the author's beam parameters do not map directly

The author's beam width `24` is a single global beam. Our current width is per
exact replay-budget cell, so the retained count is approximately:

```text
1 + beam_width_per_budget * max_extra_layers
```

Using width `24` and budget `56` would retain up to `1,345` candidates, not 24.
It would also require roughly `845 GiB` of CPU activation cache if search were
run on all Math120/EQ140 tokens. We should reuse the scale of the author's
search, not copy numerically incompatible flags.

For Qwen3.5-27B, the initial analogous configuration is:

- width `2` per budget
- replay window `12`, covering the author's useful 11-layer block
- maximum extra layers `12`, approximately the author's practical 20% overhead
- up to 25 retained candidates across budget cells, close to global width 24

The author's cap of 56 was a safety ceiling, not evidence that 56 extra layers
is a good initial budget. Test larger budgets only after the first search.

### Recommended Vast.ai machine

Use one on-demand H100 SXM 80 GB for the first campaign:

- GPU: `1x H100 SXM 80GB` (H100 PCIe 80 GB is an acceptable cheaper fallback)
- system RAM: at least `128 GB`; prefer `192-256 GB` if the premium is small
- CPU: at least 16 effective cores
- disk: `200 GB` local SSD, preferably at least `1 GB/s`
- CUDA: `12.8` or newer
- download: at least `500 Mbps`
- reliability: at least `98%`, verified host, direct SSH port
- rental type: on-demand for compatibility and the first full run

Avoid 24 GB consumer GPUs: the FP8 checkpoint plus activations and logits does
not leave safe headroom. Avoid multi-GPU offers initially because partial-layer
execution with a sharded model has not been validated. H200 is suitable but
unnecessary unless its price is close to H100.

Use this Docker image:

```text
pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
```

It matches the repository's locked PyTorch/CUDA versions. In the Vast.ai GUI,
select SSH launch mode, enable direct SSH, allocate 200 GB disk, and use the
image above. Do not select a vLLM or SGLang template: the search needs direct
access to intermediate layer states through Hugging Face Transformers.

Actual rented instance:

- one H100 SXM 80 GB at `$2.208/hour`
- CUDA driver capability 13.2; PyTorch `2.11.0+cu128`
- 2 TiB system RAM and 200 GB container disk
- `flash-linear-attention==0.5.1` / `fla-core==0.5.1`
- all 26 current tests pass locally; all 24 tests present at initial remote
  setup passed on the instance

### Staged paid experiment

#### Gate 1: environment and architecture compatibility

Status: passed

1. Run the complete unit test suite.
2. Load `Qwen/Qwen3.5-27B-FP8` on one GPU with BF16 activations.
3. Repeat the native-versus-partial comparison over `range(64)` to verify that
   FP8 loading and the CUDA fast path preserve the local 0.8B result.
4. Require finite logits and close proxy scores before any replay search.
5. Run a four-boundary search with two examples, width 2, window 2, budget 2.

Stop immediately if native/partial equivalence fails. Qwen3.5's DeltaNet layer
calling convention must then be implemented and tested before renting again.

Observed result: Qwen3.5-27B-FP8 loaded in about five seconds, exposed 64 text
layers, produced finite logits, and matched native execution exactly
(`max_abs = 0`) over the full ordinary layer path.

#### Gate 2: CUDA profiling

Status: passed; benchmark examples remain batch size 1

Run the same capped search with benchmark batch sizes `1`, `2`, `4`, and `8`.
Record wall time, peak VRAM, peak CPU cache, and CPU-to-GPU transfer time. Select
the fastest batch size that has at least 10 GB of VRAM headroom. This replaces
the current unverified CUDA default of 8.

Observed results:

- FLA's first Triton invocation compiled kernels in about 37 seconds; warm
  five-boundary Qwen3.5-0.8B search then took `4.2s`, versus `47.3s` before FLA
- Qwen3.5-27B batch 1 used `29.9 GiB` allocated / `33.0 GiB` reserved VRAM
- batch 2 produced `NaN` EQ proxy scores at boundary 1 despite ample VRAM
- batches 4 and 8 were not run after the batch-2 correctness failure
- Qwen3.5 now defaults to batch 1 on CUDA; other architectures retain CUDA
  default batch 8

The boundary-1 NaNs exposed a cache-padding bug: materialization cropped a short
row to its real sequence length, then restoration recreated its padded tail as
zero hidden vectors. Preserving the full padded hidden row removed the NaNs and
reduced the profile from `48.8s` to `32.1s`. However, batch-2 scores already
differed from batch 1 at boundary 0, proving that variable-length padding itself
changes fused DeltaNet results. All 16 EQ examples have different token lengths,
so equal-length batching would not accelerate the dominant workload. Qwen3.5
therefore remains restricted to batch 1 until packed-sequence support exists.

Batching across sibling architecture candidates is safe because every sibling
for one benchmark example has the same sequence length, attention mask, and
suffix boundary. The search now builds each sibling state independently, stacks
the states, and runs their common suffix once. Parent groups are filled with
discarded duplicate rows so every candidate at a boundary uses the same CUDA
batch shape; this avoids comparing scores produced by different kernel shapes.

On the four-example, four-boundary Qwen3.5-27B profile:

- separate suffixes: `48.8s`, `29.9 GiB` peak allocated
- sibling-batched suffixes: `39.5s`, `32.9 GiB` peak allocated
- the best replay remained `(2,3)` and the top three candidates kept the same
  order
- proxy scores changed slightly because BF16/FP8 batched kernels use different
  floating-point reduction paths; exact shortlist validation remains required
- a 13-sibling suffix over the longest small-probe EQ sequence (`904` tokens)
  produced finite logits at `34.1 GiB` peak allocated / `37.8 GiB` reserved,
  leaving safe headroom on the 80 GB H100
- benchmark examples are processed longest-first; shortest-first processing
  accumulated incompatible CUDA workspace blocks and reached `73.4 GiB`, while
  longest-first completed the same 16+16 boundary at `31.7 GiB` allocated /
  `32.5 GiB` reserved
- CUDA's allocator cache is cleared once between boundaries because sibling
  batches grow from 2 to 13 rows; otherwise workspaces for every earlier batch
  shape accumulate even though no GPU candidate state remains live

#### Search A: author-scale retained beam

Status: running on the rented H100 with sibling-candidate batching

Use only `math_16.json` and `eq_16.json` for search:

- width: `2` per budget
- replay window: `12`
- replay budget: `12`
- retained candidates: at most `25`
- exact validation: baseline plus the top 24 proxy candidates on the small probes

This is the primary experiment. It preserves about the same number of live
candidates as the author's width-24 beam while exploring local blocks up to the
length of his strongest single block.

A two-Math/two-EQ pilot through boundary 12 reached 25 retained candidates and
169 children per mature boundary. It completed in `1125.4s`, peaked at about
`436 MiB` of CPU activation cache, and projected the full proxy search at
roughly 9-11 hours (`$20-25`) before sibling batching. The first full 16+16 run
was stopped while cross-example batching was debugged. Its replacement keeps
benchmark batch size 1 and batches sibling candidates. Output and logs are
written incrementally to:

```text
/workspace/results/qwen35_27b_searchA_small16.json
/workspace/logs/qwen35_27b_searchA_small16.log
```

#### Search B: optional capacity check

Run only if Search A completes comfortably and proxy/exact shortlist quality is
not saturated:

- first option: width `4`, window `12`, budget `12` (up to 49 retained)
- second option: width `2`, window `12`, budget `24` (up to 49 retained)

Do not run both automatically. Choose width 4 if useful candidates are being
pruned within budget cells; choose budget 24 if the best candidates are hitting
the 12-layer cap.

#### Large-probe validation

Freeze the architecture shortlist before looking at large-probe scores. Fully
generate and score the baseline and shortlisted candidates on Math120 and all
139 EQ examples. Report absolute scores, deltas, extra-layer count, and a Pareto
frontier over quality versus added compute. The large probes are validation,
not inputs to the activation-cached search.

### Cost envelope

Vast.ai currently advertises H100 SXM around `$2/hour`, but marketplace prices,
storage, and bandwidth vary by host. Reserve:

- `$5-10` for compatibility and profiling
- `$20-40` for Search A
- `$20-50` for large-probe exact validation
- another `$20-40` only if Search B is justified

A practical initial balance is `$75`; do not fund a multi-day run until Gate 1
and Gate 2 provide measured throughput.

## 13. CPU Activation-Cache Profile

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

## 14. Immediate Next Tasks

In order:

1. Check that a 13-sibling suffix batch fits safely on the 80 GB H100.
2. Restart Search A and let its small-probe exact validation complete.
3. Inspect proxy/exact ranking and replay-budget saturation.
4. Add validation-only execution so the selected shortlist can be scored on
   Math120/EQ140 without rerunning proxy search.
5. Run staged large-probe validation, starting with baseline plus the best few
   small-probe candidates before expanding the shortlist.
6. Run Search B only if Search A shows a clear width or budget bottleneck.

## 15. Commands We Can Reuse

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

### Provision Vast.ai from the Mac

Install the CLI and authenticate it locally. Keep the API key in the CLI config;
never paste it into the repository or chat.

```bash
python3 -m pip install --user vastai
vastai set api-key YOUR_API_KEY
vastai show user
```

Create a dedicated SSH key and register only its public half:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vast_rys -C vast-rys
vastai create ssh-key ~/.ssh/vast_rys.pub
```

Search for the recommended machine. Prices are live marketplace values, so
inspect `dph`, storage, and bandwidth columns before choosing an offer ID.

```bash
vastai search offers \
  'gpu_name=H100_SXM num_gpus=1 gpu_ram>=80 cpu_ram>=128 cpu_cores_effective>=16 disk_space>=200 inet_down>=500 reliability>=0.98 verified=true direct_port_count>=1 rentable=true cuda_vers>=12.8' \
  -o 'dph'
```

Create the instance:

```bash
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel \
  --disk 200 \
  --ssh \
  --direct \
  --label rys-qwen35
vastai show instances
```

The Vast instance page supplies a command shaped like:

```bash
ssh -i ~/.ssh/vast_rys -p PORT root@PUBLIC_IP
```

Optionally add it to `~/.ssh/config` as `Host vast-rys`. Set
`ServerAliveInterval 30` and `ServerAliveCountMax 6` so idle sessions survive
short network interruptions.

Transfer the current working tree without local virtual environments, model
caches, or old results:

```bash
rsync -az --delete \
  --exclude .venv \
  --exclude .hf-cache \
  --exclude .uv-cache \
  --exclude results \
  -e 'ssh -i ~/.ssh/vast_rys -p PORT' \
  ./ root@PUBLIC_IP:/workspace/RYS/
```

The `--delete` flag is safe only for the dedicated `/workspace/RYS/` directory.
Omit it if that directory contains remote-only work.

### Initialize the remote environment

Run on the rented instance:

```bash
apt-get update
apt-get install -y curl git rsync tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

mkdir -p /workspace/.cache/huggingface /workspace/.cache/uv /workspace/results
export HF_HOME=/workspace/.cache/huggingface
export UV_CACHE_DIR=/workspace/.cache/uv

cd /workspace/RYS
uv sync --python 3.11 --group dev
uv run hf download Qwen/Qwen3.5-27B-FP8
uv run python -m pytest -q
nvidia-smi
free -h
```

Use `tmux new -s rys` for paid runs. Open a second SSH session for
`watch -n 1 nvidia-smi`; also monitor `free -h` because candidate activations
are deliberately stored in system RAM.

To let Codex operate the machine directly, add the local public key to Vast.ai
and provide only the generated SSH command (`host` and `port`). The private key
stays on this Mac. Codex can then invoke `ssh`, run experiments, poll logs, and
copy results back through the terminal. Do not provide the Vast API key unless
you explicitly want automated instance creation/destruction.

## 16. Branches

- `main`: baseline public repo behavior
- `codex_proxy_search`: proxy scoring and suffix-search implementation
