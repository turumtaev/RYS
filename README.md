# Benchmark-Conditioned Suffix Beam Search for RYS

This fork explores benchmark-conditioned suffix beam search for finding useful
layer repetitions in decoder language models. It builds on Daniel Han-Chen's
[RYS repository](https://github.com/dnhkng/RYS) and the experiments described in
[RYS Part II](https://dnhkng.github.io/posts/rys-ii/).

The upstream project searches for a fixed relayered architecture by evaluating
complete repeated blocks and composing the strongest blocks with beam search.
This fork searches for the same kind of static architecture, but grows it one
model boundary at a time and reuses benchmark hidden states across expansions.

The implementation, tests, GPU launchers, and corrected evaluation code are on
this branch. The longer chronological design log is in
[`SUFFIX_BEAM_SEARCH.md`](SUFFIX_BEAM_SEARCH.md).

## Method

The method starts from one observation:

> If the remaining model layers are fixed, benchmark quality depends on the
> hidden-state representation handed to those layers.

Here, **prefix** and **suffix** refer to parts of the model's layer execution
path, not to token prefixes or suffixes.

At boundary `k`, all candidate expansions are evaluated through the same model
suffix, layers `k+1 .. N-1`. The search can therefore compare the hidden states
produced by different local layer paths, retain the representations that score
best on the benchmark, and reuse the cached prefix states that produced them.

Let the original decoder contain layers `0 .. N-1`. At boundary `k`, every beam
candidate represents an execution path through layer `k-1` and owns the hidden
states produced by that prefix for every benchmark example. The search expands
the candidate with:

1. a plain continuation through layer `k`; or
2. layer `k` followed by a replay of a recent contiguous block `m .. k`.

The replay start `m` is restricted by `--replay-window`. For example, at layer
`k=3`, replaying `(1,4)` produces the partial path
`0,1,2,3,1,2,3`. Block ends are exclusive throughout the codebase.

Each child is completed with the unchanged suffix `k+1 .. N-1` and scored on
the benchmark. The implementation keeps a separate beam for every exact
extra-layer budget, which prevents all candidates from spending their replay
budget in early layers.

### Reused computation

The search keeps candidate boundary hidden states in CPU RAM. At every boundary
it:

1. restores each parent's hidden states after layer `k-1`;
2. runs layer `k` and each permitted local replay;
3. runs the common remaining suffix to calculate proxy scores;
4. prunes independently inside each replay-budget beam; and
5. offloads only the surviving child boundary states for the next step.

This avoids recomputing every retained prefix from token IDs. A two-pass
implementation also avoids storing the full branching factor: score all
children first, then recompute and retain only the winners' short expansions.

### Search objective

Running full autoregressive Math and EQ evaluation for every child is too
expensive. Search therefore uses a deterministic teacher-forced proxy:

- Math scores the log-probability of every token in the canonical numeric answer
  and the answer terminator.
- EQ scores only the four numeric answer spans and each following delimiter or
  terminator. Emotion names and formatting are present as context but do not
  contribute to the score.
- Math and EQ are averaged with equal weight.

Scoring the terminator matters because otherwise a configuration can maximize
the probability of the correct numeric prefix while continuing to generate
more digits. The final shortlist is always rerun with the original
generation-based Math and EQ metrics.

## What Is Implemented

- Teacher-forced Math and masked EQ objectives, including answer termination.
- Partial execution of Llama/Qwen-style decoder stacks exposing a layer list.
- Qwen3.5 support, including its hybrid attention and required cache positions.
- Arbitrary repeated layer paths without rebuilding model weights for proxy
  scoring.
- Boundary hidden-state caching in CPU RAM and suffix-only child evaluation.
- Budget-indexed beam search with local replay windows and replay-layer caps.
- Length-sorted, padding-aware benchmark batching infrastructure.
- Exact generation-based evaluation of search candidates and explicit configs.
- Corrected full-scale EQ references in Hugging Face and ExLlama workers.
- Mac CPU/MPS smoke tests and reproducible Qwen3.5-27B H100 launch scripts.

## Qwen3.5-27B Experiment

The main run used `Qwen/Qwen3.5-27B-FP8` on one 80 GB H100.

| Setting | Value |
|---|---:|
| Proxy examples | 16 Math + 16 EQ |
| Model layers | 64 |
| Beam width | 2 per exact replay budget |
| Replay window | 12 layers |
| Maximum extra layers | 12 |
| Proxy batch size | 1 |
| Proxy search time | 36,958 s (10.27 h) |
| Peak CUDA allocated | 29.9 GiB |
| Peak CUDA reserved | 42.5 GiB |
| Peak CPU boundary cache | 3.54 GiB |
| Exact comparison | 120 Math + 139 EQ |

The file remains named `eq_140.json` for compatibility, but contains 139
examples. The large benchmark extends the small probes; it is a broader
validation set, not a strictly disjoint holdout.

Run the same search with:

```bash
./scripts/run_qwen35_search_a.sh
```

Rerun the baseline, the author's four published large-set Pareto configs, and
the 24 proxy candidates with:

```bash
./scripts/run_qwen35_large_comparison.sh
```

Both launchers write resumable JSON results and logs outside the repository by
default (`/workspace/results` and `/workspace/logs`). Local `results/` and log
files are ignored by Git and are not part of the published source history.

## Results

### Published RYS II table

RYS Part II reports this Pareto frontier for Qwen3.5-27B:

| Config | Extra layers | Math delta | EQ delta | Delta sum |
|---|---:|---:|---:|---:|
| `(33,34)` | 1 (1.56%) | +0.0179 | +0.0945 | +0.1124 |
| `(31,34)` | 3 (4.69%) | +0.0207 | +0.0972 | +0.1179 |
| `(30,35)` | 5 (7.81%) | +0.0279 | +0.0979 | +0.1257 |
| `(26,34)` | 8 (12.50%) | +0.0279 | +0.1009 | +0.1288 |

During reproduction, we found a bug in the upstream EQ evaluator. Generated
scores are on the 0-10 scale requested by the prompt, but the evaluator compared
them with `reference_answer`, where the four reference scores are normalized
together. The datasets contain the intended integer labels under
`reference_answer_fullscale`. This fork changes exact EQ evaluation to use that
field.

The bug affects more than the reported EQ deltas. Because EQ score was part of
the search objective, it may also have affected which candidates were selected
for the reported Pareto frontier. Correcting the reference field does not
retroactively correct the original search; that would require reevaluating its
full candidate set and rerunning selection.

The table below is therefore a new common-evaluator rerun, not a corrected
version of the published table. It evaluates the baseline, the four
configurations reported in RYS II, and our candidates with the same corrected
EQ code. The RYS II configurations were selected under the original metric, so
the comparison should not be treated as a symmetric test of the two search
methods.

### Corrected common-evaluator comparison

These are Hugging Face/Transformers results on all 120 Math and 139 EQ examples.
`Average` is `(Math + EQ) / 2`; deltas are relative to the baseline from this
same run.

| Source | Config | Extra | Math | EQ | Average | Math delta | EQ delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | none | 0 | 0.969651 | 0.647302 | 0.808476 | 0 | 0 |
| RYS II config, rerun | `(33,34)` | 1 | 0.989684 | 0.668975 | 0.829329 | +0.020033 | +0.021673 |
| RYS II config, rerun | `(31,34)` | 3 | 0.998575 | 0.670234 | 0.834405 | +0.028925 | +0.022932 |
| RYS II config, rerun | `(30,35)` | 5 | 0.995503 | 0.672482 | 0.833992 | +0.025852 | +0.025180 |
| RYS II config, rerun | `(26,34)` | 8 | 0.997037 | 0.672302 | 0.834669 | +0.027386 | +0.025000 |
| This search, `beam_23` | `(43,44)` | 1 | **0.998712** | **0.669784** | **0.834248** | **+0.029061** | **+0.022482** |
| This search, `beam_20` | `(43,44);(45,46);(47,48)` | 3 | 0.994119 | **0.676349** | **0.835234** | +0.024468 | **+0.029047** |

The bold values compare configurations with the same number of extra layers.
Within this corrected rerun:

- `beam_23` scores `0.004919` higher on average than the one-layer `(33,34)`
  config at the same 1.56% overhead.
- `beam_20` scores `0.000829` higher on average than the three-layer `(31,34)`
  config at the same 4.69% overhead.
- `beam_20` has the highest average among the 29 configurations evaluated in
  this run. Its average is `0.000564` above the eight-layer `(26,34)` config
  while using three extra layers.
- For this limited candidate set, the Pareto frontier over extra layers and
  average score consists of the baseline, `beam_23`, and `beam_20`.

All configurations in this table were evaluated consistently, but this is not
an exact reproduction of the blog's absolute numbers. The blog used ExLlamaV3;
this experiment used Hugging Face Transformers with BF16 computation over the
FP8 checkpoint. Backend-level numerical differences can alter greedy outputs.
The Math metric also awards partial credit, so a near-1 Math score is not the
same as near-100% exact-match accuracy.

## Setup and Validation

Python dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
uv run pytest
```

The focused test suite covers target masks, teacher-forced scoring, partial and
repeated paths, cached versus uncached search, budget-indexed pruning, batching,
and export utilities. A real model diagnostic is also available:

```bash
uv run python scripts/check_partial_runner.py \
  --model-path Qwen/Qwen3.5-0.8B \
  --device-map mps \
  --torch-dtype float32 \
  --no-local-files-only
```

For a small end-to-end search:

```bash
HF_HOME=.hf-cache uv run python scripts/beam_search.py \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --device-map mps \
  --torch-dtype float16 \
  --dataset-limit 4 \
  --beam-width 2 \
  --replay-window 1 \
  --max-extra-layers 2 \
  --exact-top-k 2 \
  --output results/mac_smoke/suffix_search.json
```

Use `--device-map cpu` on a machine without MPS or CUDA. See
`scripts/beam_search.py --help` for separate search and exact-validation
datasets, offsets, generation limits, and cache controls.

## Known Limitations and Unfinished Work

### Cross-example batching on Qwen3.5

The production search used `--benchmark-batch-size 1`. Batch size 2 initially
produced `NaN` EQ proxy scores because cached short sequences were restored into
padded rows incorrectly. Fixing that bug removed the NaNs, but variable-length
batch scores still differed from batch-1 scores at the first boundary. Sorting
by length did not help enough because almost every EQ prompt has a distinct
length. Qwen3.5 therefore defaults to batch 1 until packed-sequence or another
padding-independent implementation is validated.

Exact generation also uses batch size 1 for the recorded comparison. Correctness
was preferred over throughput.

### Sibling-candidate batching

An alternative batches the children of one parent after their local replays,
because all siblings have equal sequence length and run the same suffix. The
implementation is preserved on branch `codex/sibling-batching-wip` at commit
`92e3567`. It was not used for the final search: sibling count grows to 13 with
a replay window of 12, and the longest EQ sequences made peak-memory behavior
too risky for the single rented 80 GB GPU. The branch is experimental and its
small numerical differences also need ranking-stability tests.

### Nested replay paths

The current search appends a replay to the path already fixed in the prefix. It
can construct:

```text
1,2,3,1,2,3
```

but it cannot discover a replay inside an earlier replay, such as:

```text
1,2,3,1,2,1,2,3
```

A generalized search should store both a prefix and a pending suffix for every
candidate. At boundary `k`, if executing `k`, then replaying `k-2,k-1,k`, then
the old suffix gives a better score, retain it as:

```text
prefix = old_prefix + [k]
suffix = [k-2, k-1, k] + old_suffix
```

The scheduler would process candidates with the smallest unfinished prefix
boundary first. This turns a replay into editable future structure instead of
permanently flattening it into the completed prefix. It needs a new candidate
state, deduplication rule, budget accounting, cache policy, and tests before it
can replace the current boundary-by-boundary algorithm.

### Evaluation and search quality

- The proxy is useful for pruning but did not rank the exact winner first on the
  16+16 search set; exact shortlist validation remains mandatory.
- The 16+16 proxy set was selected for speed, not statistical power.
- Hyperparameters were chosen pragmatically rather than with a full sweep on
  Qwen3.5-27B.
- The corrected author comparison should also be reproduced with ExLlamaV3 if
  exact agreement with the original execution backend is required.
- Raw GPU logs and result JSON are intentionally not committed. The launchers,
  exact configs, reported metrics, and source commit are preserved for reruns.

## Repository Layout

- `scripts/beam_search.py`: benchmark-conditioned suffix beam search.
- `scripts/evaluate_exact_configs.py`: generation-score explicit layer paths.
- `scripts/run_qwen35_*.sh`: recorded H100 experiment launchers.
- `scripts/check_partial_runner.py`: full-model versus partial-runner diagnostic.
- `src/workers/model_utils.py`: model loading, partial execution, and proxy input
  construction.
- `src/workers/math_worker.py`: Math prompts, proxy targets, generation, scoring.
- `src/workers/eq_worker.py`: EQ prompts, numeric masks, generation, scoring.
- `tests/test_surrogate_utils.py`: suffix-search and scoring tests.
- `SUFFIX_BEAM_SEARCH.md`: historical design decisions and experiment log.

For the original single-block scanner, multi-block beam search, surrogate
pipeline, exporter, and upstream usage instructions, see
[dnhkng/RYS](https://github.com/dnhkng/RYS).

## Citing This Work

If you use this implementation or its experimental results, please cite both
the original RYS II work and this repository:

```bibtex
@article{ng2026rysii,
  title  = {LLM Neuroanatomy II: Modern LLM Hacking and hints of a Universal Language?},
  author = {Ng, David Noel},
  year   = {2026},
  month  = {March},
  url    = {https://dnhkng.github.io/posts/rys-ii/}
}

@software{turumtaev2026suffixbeam,
  title  = {Benchmark-Conditioned Suffix Beam Search for RYS},
  author = {Turumtaev, Galim},
  year   = {2026},
  month  = {July},
  url    = {https://github.com/turumtaev/RYS}
}
```
