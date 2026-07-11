#!/usr/bin/env python3
"""Boundary-by-boundary beam search for benchmark-optimal layer replays.

This correctness-first implementation recomputes each complete candidate path.
Activation reuse can be added after the search behavior is validated end to end.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from src.core.layer_duplicator import build_model_with_layers
from src.workers.eq_worker import (
    pretokenize_eq_dataset,
    pretokenize_eq_teacher_forced_dataset,
    run_eq_test,
)
from src.workers.math_worker import (
    pretokenize_dataset,
    pretokenize_teacher_forced_dataset,
    run_math_test_batched_moe,
)
from src.workers.model_utils import (
    LlamaLikePartialRunner,
    get_text_num_layers,
    load_model_and_tokenizer,
    maybe_empty_cache,
    model_input_device,
    parse_device_map_arg,
    parse_torch_dtype_arg,
    score_teacher_forced_logits,
)


@dataclass(frozen=True)
class BeamCandidate:
    """A partial architecture through one original-layer boundary."""

    layer_path: tuple[int, ...]
    replays: tuple[tuple[int, int], ...]
    score: float | None = None
    math_score: float | None = None
    eq_score: float | None = None


def expand_candidate(
    candidate: BeamCandidate,
    *,
    layer_idx: int,
    replay_window: int,
    num_layers: int,
    max_extra_layers: int | None = None,
) -> list[BeamCandidate]:
    """Append plain layer ``k`` and local replay alternatives ending at ``k``."""
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"layer_idx must be in [0, {num_layers}).")
    if replay_window < 0:
        raise ValueError("replay_window must be >= 0.")

    plain_path = candidate.layer_path + (layer_idx,)
    children = [BeamCandidate(layer_path=plain_path, replays=candidate.replays)]
    first_replay_layer = max(0, layer_idx - replay_window + 1)
    for replay_start in range(first_replay_layer, layer_idx + 1):
        replay_path = tuple(range(replay_start, layer_idx + 1))
        child_path = plain_path + replay_path
        extra_layers = len(child_path) - (layer_idx + 1)
        if max_extra_layers is not None and extra_layers > max_extra_layers:
            continue
        children.append(
            BeamCandidate(
                layer_path=child_path,
                replays=candidate.replays + ((replay_start, layer_idx + 1),),
            )
        )
    return children


def expand_beam(
    beam: Iterable[BeamCandidate],
    *,
    layer_idx: int,
    replay_window: int,
    num_layers: int,
    max_extra_layers: int | None = None,
) -> list[BeamCandidate]:
    """Expand and deduplicate all candidates at one boundary."""
    unique: dict[tuple[int, ...], BeamCandidate] = {}
    for candidate in beam:
        for child in expand_candidate(
            candidate,
            layer_idx=layer_idx,
            replay_window=replay_window,
            num_layers=num_layers,
            max_extra_layers=max_extra_layers,
        ):
            unique.setdefault(child.layer_path, child)
    return list(unique.values())


def complete_layer_path(candidate: BeamCandidate, *, layer_idx: int, num_layers: int) -> tuple[int, ...]:
    """Append the unchanged suffix after the current boundary."""
    return candidate.layer_path + tuple(range(layer_idx + 1, num_layers))


def _load_dataset(path: str, limit: int | None, *, offset: int = 0) -> dict[str, Any]:
    with Path(path).open() as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, dict):
        raise ValueError(f"Expected an object dataset in {path}.")
    items = list(dataset.items())[offset:]
    if limit is not None:
        items = items[:limit]
    dataset = dict(items)
    if not dataset:
        raise ValueError(f"Dataset selection for {path} is empty (offset={offset}, limit={limit}).")
    return dataset


def _dataset_chunks(
    dataset: dict[str, dict[str, Any]],
    chunk_size: int,
) -> Iterable[list[dict[str, Any]]]:
    values = list(dataset.values())
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _move_cached_example(cached: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in cached.items()
    }


def score_candidates(
    runner: LlamaLikePartialRunner,
    candidates: list[BeamCandidate],
    *,
    layer_idx: int,
    math_dataset: dict[str, dict[str, Any]],
    eq_dataset: dict[str, dict[str, Any]],
    reduction: str,
    device: torch.device,
    chunk_size: int,
) -> list[BeamCandidate]:
    """Score complete suffixes while keeping only one benchmark chunk on device."""
    if not candidates:
        return []

    layer_paths = [
        complete_layer_path(candidate, layer_idx=layer_idx, num_layers=runner.num_layers)
        for candidate in candidates
    ]
    metric_scores: dict[str, list[list[float]]] = {
        "math": [[] for _ in candidates],
        "eq": [[] for _ in candidates],
    }
    with torch.inference_mode():
        for metric, dataset in (("math", math_dataset), ("eq", eq_dataset)):
            for cpu_chunk in _dataset_chunks(dataset, chunk_size):
                device_chunk = [_move_cached_example(cached, device) for cached in cpu_chunk]
                prepared_chunk = [
                    runner.prepare(
                        input_ids=cached["input_ids"],
                        attention_mask=cached["attention_mask"],
                    )
                    for cached in device_chunk
                ]
                for candidate_idx, layer_path in enumerate(layer_paths):
                    for cached, initial_state in zip(device_chunk, prepared_chunk):
                        final_state = runner.run_layer_indices(initial_state, layer_path)
                        logits = runner.logits(final_state)
                        result = score_teacher_forced_logits(
                            logits=logits,
                            input_ids=cached["input_ids"],
                            prompt_length=int(cached["prompt_length"]),
                            target_mask=cached.get("target_mask"),
                            reduction=reduction,
                        )
                        metric_scores[metric][candidate_idx].append(float(result["score"]))
                del prepared_chunk
                del device_chunk

    scored: list[BeamCandidate] = []
    for idx, candidate in enumerate(candidates):
        math_values = metric_scores["math"][idx]
        eq_values = metric_scores["eq"][idx]
        if not math_values or not eq_values:
            raise ValueError("Cannot score an empty proxy dataset.")
        math_score = sum(math_values) / len(math_values)
        eq_score = sum(eq_values) / len(eq_values)
        scored.append(
            BeamCandidate(
                layer_path=candidate.layer_path,
                replays=candidate.replays,
                score=(math_score + eq_score) / 2.0,
                math_score=math_score,
                eq_score=eq_score,
            )
        )
    return scored


def _candidate_json(candidate: BeamCandidate, *, layer_idx: int, num_layers: int) -> dict[str, Any]:
    data = asdict(candidate)
    data["layer_path"] = list(candidate.layer_path)
    data["replays"] = [list(block) for block in candidate.replays]
    data["complete_layer_path"] = list(
        complete_layer_path(candidate, layer_idx=layer_idx, num_layers=num_layers)
    )
    return data


def _write_progress(
    output_path: Path,
    *,
    args: argparse.Namespace,
    model_metadata: dict[str, Any],
    layer_idx: int,
    final_layer: int,
    beam: list[BeamCandidate],
    elapsed_seconds: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finished_proxy_search = layer_idx == final_layer
    status = "running"
    if finished_proxy_search:
        status = "proxy_complete" if args.exact_top_k > 0 else "complete"
    payload = {
        "algorithm": "benchmark_conditioned_suffix_beam_search",
        "status": status,
        "full_model_search": final_layer == model_metadata["num_layers"] - 1,
        "model": model_metadata,
        "search": {
            "beam_width": args.beam_width,
            "replay_window": args.replay_window,
            "max_extra_layers": args.max_extra_layers,
            "dataset_limit_per_benchmark": args.dataset_limit,
            "benchmark_chunk_size": args.benchmark_chunk_size,
            "example_reduction": args.reduction,
            "combined_score": "(math_score + eq_score) / 2",
            "requested_final_layer": final_layer,
            "last_completed_layer": layer_idx,
            "elapsed_seconds": elapsed_seconds,
        },
        "beam": [
            _candidate_json(candidate, layer_idx=layer_idx, num_layers=model_metadata["num_layers"])
            for candidate in beam
        ],
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output_path)


def _write_exact_validation(
    output_path: Path,
    results: list[dict[str, Any]],
    *,
    math_max_new: int,
    eq_max_new: int,
    batch_size: int,
    dataset_offset: int,
) -> None:
    payload = json.loads(output_path.read_text())
    proxy_order = [
        item["label"]
        for item in sorted(results, key=lambda item: item["proxy_score"], reverse=True)
    ]
    exact_order = [
        item["label"]
        for item in sorted(results, key=lambda item: item["score"], reverse=True)
    ]
    payload["status"] = "complete"
    payload["exact_validation"] = {
        "combined_score": "(math_score + eq_score) / 2",
        "math_max_new": math_max_new,
        "eq_max_new": eq_max_new,
        "batch_size": batch_size,
        "dataset_offset": dataset_offset,
        "examples_per_benchmark": len(results[0]["math_responses"]),
        "proxy_order": proxy_order,
        "exact_order": exact_order,
        "top_rank_agrees": proxy_order[0] == exact_order[0],
        "candidates": results,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output_path)


def run_exact_validation(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    math_dataset: dict[str, Any],
    eq_dataset: dict[str, Any],
    baseline_proxy_score: float,
    beam: list[BeamCandidate],
    last_layer: int,
    top_k: int,
    math_max_new: int,
    eq_max_new: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Generation-score the baseline and top proxy candidates."""
    exact_math = pretokenize_dataset(math_dataset, tokenizer, device)
    exact_eq = pretokenize_eq_dataset(eq_dataset, tokenizer, device)
    num_layers = get_text_num_layers(model)
    requested = [
        ("baseline", tuple(range(num_layers)), baseline_proxy_score, ()),
        *[
            (
                f"beam_{rank}",
                complete_layer_path(candidate, layer_idx=last_layer, num_layers=num_layers),
                candidate.score,
                candidate.replays,
            )
            for rank, candidate in enumerate(beam[:top_k], start=1)
        ],
    ]

    results: list[dict[str, Any]] = []
    seen_paths: set[tuple[int, ...]] = set()
    for label, layer_path, proxy_score, replays in requested:
        if layer_path in seen_paths:
            continue
        seen_paths.add(layer_path)
        run_model = (
            model
            if layer_path == tuple(range(num_layers))
            else build_model_with_layers(model, list(layer_path))
        )
        started = time.monotonic()
        math_result = run_math_test_batched_moe(
            run_model,
            exact_math,
            tokenizer,
            batch_size=batch_size,
            max_new_tokens=math_max_new,
            save_responses=True,
        )
        eq_result = run_eq_test(
            run_model,
            exact_eq,
            tokenizer,
            batch_size=batch_size,
            max_new_tokens=eq_max_new,
            save_responses=True,
        )
        math_score = float(math_result["score"])
        eq_score = float(eq_result["score"])
        result = {
            "label": label,
            "layer_path": list(layer_path),
            "replays": [list(block) for block in replays],
            "proxy_score": float(proxy_score),
            "math_score": math_score,
            "eq_score": eq_score,
            "score": (math_score + eq_score) / 2.0,
            "elapsed_seconds": time.monotonic() - started,
            "math_responses": math_result.get("responses", []),
            "eq_responses": eq_result.get("responses", []),
        }
        results.append(result)
        print(
            f"exact label={label} score={result['score']:.6f} "
            f"math={math_score:.6f} eq={eq_score:.6f} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if run_model is not model:
            del run_model
            maybe_empty_cache(device)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--math-dataset-path", default="datasets/math_16.json")
    parser.add_argument("--eq-dataset-path", default="datasets/eq_16.json")
    parser.add_argument("--output", default="results/suffix-beam-search.json")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--replay-window", type=int, default=4)
    parser.add_argument(
        "--max-extra-layers",
        type=int,
        default=2,
        help="Replay-layer budget. Increase only after proxy/exact correlation validation.",
    )
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument(
        "--benchmark-chunk-size",
        type=int,
        default=8,
        help="Tokenized examples moved from CPU to the accelerator at once.",
    )
    parser.add_argument(
        "--exact-dataset-offset",
        type=int,
        default=0,
        help="Skip this many examples before selecting the exact-validation set.",
    )
    parser.add_argument(
        "--exact-dataset-limit",
        type=int,
        default=None,
        help="Exact-validation examples per benchmark; defaults to --dataset-limit.",
    )
    parser.add_argument(
        "--exact-top-k",
        type=int,
        default=0,
        help="After proxy search, generation-score the baseline and top N beam candidates.",
    )
    parser.add_argument("--exact-batch-size", type=int, default=1)
    parser.add_argument("--math-max-new", type=int, default=64)
    parser.add_argument("--eq-max-new", type=int, default=384)
    parser.add_argument(
        "--stop-after-layer",
        type=int,
        default=None,
        help="Inclusive boundary used for small smoke runs.",
    )
    parser.add_argument("--reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--attention-impl", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.beam_width < 1:
        raise ValueError("--beam-width must be >= 1.")
    if args.replay_window < 0:
        raise ValueError("--replay-window must be >= 0.")
    if args.max_extra_layers < 0:
        raise ValueError("--max-extra-layers must be >= 0.")
    if args.dataset_limit is not None and args.dataset_limit < 1:
        raise ValueError("--dataset-limit must be >= 1.")
    if args.benchmark_chunk_size < 1:
        raise ValueError("--benchmark-chunk-size must be >= 1.")
    if args.exact_dataset_offset < 0:
        raise ValueError("--exact-dataset-offset must be >= 0.")
    if args.exact_dataset_limit is not None and args.exact_dataset_limit < 1:
        raise ValueError("--exact-dataset-limit must be >= 1.")
    if args.exact_top_k < 0:
        raise ValueError("--exact-top-k must be >= 0.")
    if args.exact_batch_size < 1:
        raise ValueError("--exact-batch-size must be >= 1.")
    if args.math_max_new < 1 or args.eq_max_new < 1:
        raise ValueError("--math-max-new and --eq-max-new must be >= 1.")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    device_map = parse_device_map_arg(args.device_map)
    torch_dtype = parse_torch_dtype_arg(args.torch_dtype, device_map=device_map)
    tokenizer, model, metadata = load_model_and_tokenizer(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=args.attention_impl,
    )
    runner = LlamaLikePartialRunner(model)
    device = model_input_device(model)

    math_raw = _load_dataset(args.math_dataset_path, args.dataset_limit)
    eq_raw = _load_dataset(args.eq_dataset_path, args.dataset_limit)
    math_dataset = pretokenize_teacher_forced_dataset(math_raw, tokenizer, torch.device("cpu"))
    eq_dataset = pretokenize_eq_teacher_forced_dataset(eq_raw, tokenizer, torch.device("cpu"))

    final_layer = runner.num_layers - 1
    if args.stop_after_layer is not None:
        if args.stop_after_layer < 0:
            raise ValueError("--stop-after-layer must be >= 0.")
        final_layer = min(final_layer, args.stop_after_layer)

    beam = [BeamCandidate(layer_path=(), replays=())]
    baseline_proxy_score: float | None = None
    started = time.monotonic()
    output_path = Path(args.output)
    for layer_idx in range(final_layer + 1):
        children = expand_beam(
            beam,
            layer_idx=layer_idx,
            replay_window=args.replay_window,
            num_layers=runner.num_layers,
            max_extra_layers=args.max_extra_layers,
        )
        scored = score_candidates(
            runner,
            children,
            layer_idx=layer_idx,
            math_dataset=math_dataset,
            eq_dataset=eq_dataset,
            reduction=args.reduction,
            device=device,
            chunk_size=args.benchmark_chunk_size,
        )
        scored.sort(key=lambda candidate: float(candidate.score), reverse=True)
        if layer_idx == 0:
            baseline = next(candidate for candidate in scored if not candidate.replays)
            baseline_proxy_score = float(baseline.score)
        beam = scored[: args.beam_width]
        elapsed = time.monotonic() - started
        best = beam[0]
        print(
            f"layer={layer_idx} children={len(children)} "
            f"best={best.score:.6f} math={best.math_score:.6f} "
            f"eq={best.eq_score:.6f} replays={best.replays} elapsed={elapsed:.1f}s",
            flush=True,
        )
        _write_progress(
            output_path,
            args=args,
            model_metadata=metadata,
            layer_idx=layer_idx,
            final_layer=final_layer,
            beam=beam,
            elapsed_seconds=elapsed,
        )

    if args.exact_top_k > 0:
        if baseline_proxy_score is None:
            raise RuntimeError("Baseline proxy score was not captured.")
        exact_limit = args.exact_dataset_limit
        if exact_limit is None:
            exact_limit = args.dataset_limit
        exact_math_raw = _load_dataset(
            args.math_dataset_path,
            exact_limit,
            offset=args.exact_dataset_offset,
        )
        exact_eq_raw = _load_dataset(
            args.eq_dataset_path,
            exact_limit,
            offset=args.exact_dataset_offset,
        )
        exact_results = run_exact_validation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            math_dataset=exact_math_raw,
            eq_dataset=exact_eq_raw,
            baseline_proxy_score=baseline_proxy_score,
            beam=beam,
            last_layer=final_layer,
            top_k=min(args.exact_top_k, len(beam)),
            math_max_new=args.math_max_new,
            eq_max_new=args.eq_max_new,
            batch_size=args.exact_batch_size,
        )
        _write_exact_validation(
            output_path,
            exact_results,
            math_max_new=args.math_max_new,
            eq_max_new=args.eq_max_new,
            batch_size=args.exact_batch_size,
            dataset_offset=args.exact_dataset_offset,
        )

    print(f"Saved {len(beam)} candidates to {output_path}")


if __name__ == "__main__":
    main()
