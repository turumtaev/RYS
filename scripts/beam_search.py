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

from src.workers.eq_worker import pretokenize_eq_teacher_forced_dataset
from src.workers.math_worker import pretokenize_teacher_forced_dataset
from src.workers.model_utils import (
    LlamaLikePartialRunner,
    load_model_and_tokenizer,
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


def _load_dataset(path: str, limit: int | None) -> dict[str, Any]:
    with Path(path).open() as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, dict):
        raise ValueError(f"Expected an object dataset in {path}.")
    if limit is not None:
        dataset = dict(list(dataset.items())[:limit])
    return dataset


def _score_proxy_dataset(
    runner: LlamaLikePartialRunner,
    dataset: dict[str, dict[str, Any]],
    layer_path: tuple[int, ...],
    *,
    reduction: str,
) -> float:
    scores: list[float] = []
    with torch.inference_mode():
        for cached in dataset.values():
            logits = runner.forward_path(
                input_ids=cached["input_ids"],
                attention_mask=cached["attention_mask"],
                layer_indices=layer_path,
            )
            result = score_teacher_forced_logits(
                logits=logits,
                input_ids=cached["input_ids"],
                prompt_length=int(cached["prompt_length"]),
                target_mask=cached.get("target_mask"),
                reduction=reduction,
            )
            scores.append(float(result["score"]))
    if not scores:
        raise ValueError("Cannot score an empty proxy dataset.")
    return sum(scores) / len(scores)


def score_candidate(
    runner: LlamaLikePartialRunner,
    candidate: BeamCandidate,
    *,
    layer_idx: int,
    math_dataset: dict[str, dict[str, Any]],
    eq_dataset: dict[str, dict[str, Any]],
    reduction: str,
) -> BeamCandidate:
    """Score a partial config after completing it with the normal suffix."""
    full_path = complete_layer_path(candidate, layer_idx=layer_idx, num_layers=runner.num_layers)
    math_score = _score_proxy_dataset(runner, math_dataset, full_path, reduction=reduction)
    eq_score = _score_proxy_dataset(runner, eq_dataset, full_path, reduction=reduction)
    return BeamCandidate(
        layer_path=candidate.layer_path,
        replays=candidate.replays,
        score=(math_score + eq_score) / 2.0,
        math_score=math_score,
        eq_score=eq_score,
    )


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
    payload = {
        "algorithm": "benchmark_conditioned_suffix_beam_search",
        "status": "complete" if layer_idx == final_layer else "running",
        "full_model_search": final_layer == model_metadata["num_layers"] - 1,
        "model": model_metadata,
        "search": {
            "beam_width": args.beam_width,
            "replay_window": args.replay_window,
            "max_extra_layers": args.max_extra_layers,
            "dataset_limit_per_benchmark": args.dataset_limit,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--math-dataset-path", default="datasets/math_16.json")
    parser.add_argument("--eq-dataset-path", default="datasets/eq_16.json")
    parser.add_argument("--output", default="results/suffix-beam-search.json")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--replay-window", type=int, default=4)
    parser.add_argument("--max-extra-layers", type=int, default=56)
    parser.add_argument("--dataset-limit", type=int, default=None)
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
    math_dataset = pretokenize_teacher_forced_dataset(math_raw, tokenizer, device)
    eq_dataset = pretokenize_eq_teacher_forced_dataset(eq_raw, tokenizer, device)

    final_layer = runner.num_layers - 1
    if args.stop_after_layer is not None:
        if args.stop_after_layer < 0:
            raise ValueError("--stop-after-layer must be >= 0.")
        final_layer = min(final_layer, args.stop_after_layer)

    beam = [BeamCandidate(layer_path=(), replays=())]
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
        scored = [
            score_candidate(
                runner,
                child,
                layer_idx=layer_idx,
                math_dataset=math_dataset,
                eq_dataset=eq_dataset,
                reduction=args.reduction,
            )
            for child in children
        ]
        scored.sort(key=lambda candidate: float(candidate.score), reverse=True)
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

    print(f"Saved {len(beam)} candidates to {output_path}")


if __name__ == "__main__":
    main()
