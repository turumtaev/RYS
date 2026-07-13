#!/usr/bin/env python3
"""Boundary-by-boundary beam search for benchmark-optimal layer replays.

Surviving candidates keep CPU hidden states at the current layer boundary.
Children restore those states, run only their local expansion and fixed suffix,
then materialize new boundary states only after global beam pruning.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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


@dataclass(frozen=True)
class CandidateExpansion:
    """A child architecture and the local operation that produced it."""

    child: BeamCandidate
    parent_path: tuple[int, ...]
    replay_start: int | None


ExampleKey = tuple[str, str]
BoundaryCache = dict[tuple[int, ...], dict[ExampleKey, torch.Tensor]]


@dataclass(frozen=True)
class ProxyBatch:
    qids: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    score_mask: torch.Tensor
    sequence_lengths: tuple[int, ...]


def expand_candidate(
    candidate: BeamCandidate,
    *,
    layer_idx: int,
    replay_window: int,
    num_layers: int,
    max_extra_layers: int | None = None,
) -> list[BeamCandidate]:
    """Append plain layer ``k`` and local replay alternatives ending at ``k``."""
    return [
        expansion.child
        for expansion in expand_candidate_with_parents(
            candidate,
            layer_idx=layer_idx,
            replay_window=replay_window,
            num_layers=num_layers,
            max_extra_layers=max_extra_layers,
        )
    ]


def expand_candidate_with_parents(
    candidate: BeamCandidate,
    *,
    layer_idx: int,
    replay_window: int,
    num_layers: int,
    max_extra_layers: int | None = None,
) -> list[CandidateExpansion]:
    """Expand one candidate while retaining the parent and replay operation."""
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"layer_idx must be in [0, {num_layers}).")
    if replay_window < 0:
        raise ValueError("replay_window must be >= 0.")

    plain_path = candidate.layer_path + (layer_idx,)
    expansions = [
        CandidateExpansion(
            child=BeamCandidate(layer_path=plain_path, replays=candidate.replays),
            parent_path=candidate.layer_path,
            replay_start=None,
        )
    ]
    first_replay_layer = max(0, layer_idx - replay_window + 1)
    for replay_start in range(first_replay_layer, layer_idx + 1):
        replay_path = tuple(range(replay_start, layer_idx + 1))
        child_path = plain_path + replay_path
        extra_layers = len(child_path) - (layer_idx + 1)
        if max_extra_layers is not None and extra_layers > max_extra_layers:
            continue
        expansions.append(
            CandidateExpansion(
                child=BeamCandidate(
                    layer_path=child_path,
                    replays=candidate.replays + ((replay_start, layer_idx + 1),),
                ),
                parent_path=candidate.layer_path,
                replay_start=replay_start,
            )
        )
    return expansions


def expand_beam(
    beam: Iterable[BeamCandidate],
    *,
    layer_idx: int,
    replay_window: int,
    num_layers: int,
    max_extra_layers: int | None = None,
) -> list[BeamCandidate]:
    """Expand and deduplicate all candidates at one boundary."""
    return [
        expansion.child
        for expansion in expand_beam_with_parents(
            beam,
            layer_idx=layer_idx,
            replay_window=replay_window,
            num_layers=num_layers,
            max_extra_layers=max_extra_layers,
        )
    ]


def expand_beam_with_parents(
    beam: Iterable[BeamCandidate],
    *,
    layer_idx: int,
    replay_window: int,
    num_layers: int,
    max_extra_layers: int | None = None,
) -> list[CandidateExpansion]:
    """Expand a beam and retain one parent operation per unique child path."""
    unique: dict[tuple[int, ...], CandidateExpansion] = {}
    for candidate in beam:
        for expansion in expand_candidate_with_parents(
            candidate,
            layer_idx=layer_idx,
            replay_window=replay_window,
            num_layers=num_layers,
            max_extra_layers=max_extra_layers,
        ):
            unique.setdefault(expansion.child.layer_path, expansion)
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


def _dataset_item_batches(
    dataset: dict[str, dict[str, Any]],
    batch_size: int,
) -> Iterable[list[tuple[str, dict[str, Any]]]]:
    items = sorted(
        dataset.items(),
        key=lambda item: int(item[1]["input_ids"].shape[1]),
    )
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _move_cached_example(cached: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in cached.items()
    }


def _collate_proxy_batch(
    items: list[tuple[str, dict[str, Any]]],
    *,
    device: torch.device,
    pad_token_id: int,
    padding_side: str = "right",
) -> ProxyBatch:
    """Pad pretokenized examples and build position and full-sequence score masks."""
    if not items:
        raise ValueError("Cannot collate an empty proxy batch.")
    if padding_side not in {"left", "right"}:
        raise ValueError(f"Unsupported padding side: {padding_side!r}.")
    sequence_lengths = tuple(int(cached["input_ids"].shape[1]) for _, cached in items)
    max_length = max(sequence_lengths)
    input_dtype = items[0][1]["input_ids"].dtype
    attention_dtype = items[0][1]["attention_mask"].dtype
    input_ids = torch.full(
        (len(items), max_length),
        pad_token_id,
        dtype=input_dtype,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(items), max_length),
        dtype=attention_dtype,
        device=device,
    )
    score_mask = torch.zeros(
        (len(items), max_length),
        dtype=torch.bool,
        device=device,
    )

    for row, (_, cached) in enumerate(items):
        length = sequence_lengths[row]
        offset = max_length - length if padding_side == "left" else 0
        ids = cached["input_ids"][0].to(device)
        attention = cached["attention_mask"][0].to(device)
        input_ids[row, offset : offset + length] = ids
        attention_mask[row, offset : offset + length] = attention
        prompt_length = int(cached["prompt_length"])
        target_mask = cached.get("target_mask")
        if target_mask is None:
            score_mask[row, offset + prompt_length : offset + length] = True
        else:
            target_mask = target_mask[0].to(device)
            if prompt_length + target_mask.numel() != length:
                raise ValueError("prompt_length + target_mask length must equal sequence length.")
            score_mask[row, offset + prompt_length : offset + length] = target_mask

    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    return ProxyBatch(
        qids=tuple(qid for qid, _ in items),
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        score_mask=score_mask,
        sequence_lengths=sequence_lengths,
    )


def _score_proxy_batch_logits(
    *,
    logits: torch.Tensor,
    batch: ProxyBatch,
    reduction: str,
) -> list[float]:
    """Reduce shifted token log-probabilities independently per example."""
    labels = batch.input_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1).gather(
        -1,
        labels.unsqueeze(-1),
    ).squeeze(-1)
    shifted_mask = batch.score_mask[:, 1:]
    counts = shifted_mask.sum(dim=1)
    if bool((counts == 0).any().item()):
        raise ValueError("Each proxy example must score at least one token.")
    sums = token_logprobs.float().masked_fill(~shifted_mask, 0).sum(dim=1)
    if reduction == "mean":
        scores = sums / counts
    elif reduction == "sum":
        scores = sums
    else:
        raise ValueError(f"Unsupported reduction: {reduction!r}")
    return [float(score) for score in scores.detach().cpu().tolist()]


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


def _group_expansions_by_parent(
    expansions: Iterable[CandidateExpansion],
) -> dict[tuple[int, ...], list[CandidateExpansion]]:
    grouped: dict[tuple[int, ...], list[CandidateExpansion]] = {}
    for expansion in expansions:
        grouped.setdefault(expansion.parent_path, []).append(expansion)
    return grouped


def _restore_parent_batch(
    initial_state: Any,
    *,
    parent_path: tuple[int, ...],
    metric: str,
    batch: ProxyBatch,
    parent_cache: BoundaryCache,
    layer_idx: int,
    device: torch.device,
) -> Any:
    if layer_idx == 0:
        if parent_path:
            raise ValueError("Boundary 0 must expand the empty baseline prefix.")
        return initial_state

    hidden_states = torch.empty_like(initial_state.hidden_states)
    for row, qid in enumerate(batch.qids):
        example_key = (metric, qid)
        try:
            cpu_hidden = parent_cache[parent_path][example_key]
        except KeyError as exc:
            raise KeyError(
                f"Missing parent boundary state for path={parent_path}, example={example_key}."
            ) from exc
        if cpu_hidden.shape[1] != hidden_states.shape[1]:
            raise ValueError(
                "Cached padded length changed between boundaries. "
                "Keep batch size and length-sorted batching fixed for a search run."
            )
        hidden_states[row : row + 1] = cpu_hidden.to(device)
    return replace(initial_state, hidden_states=hidden_states)


def _run_local_expansion(
    runner: LlamaLikePartialRunner,
    plain_state: Any,
    expansion: CandidateExpansion,
    *,
    layer_idx: int,
) -> Any:
    if expansion.replay_start is None:
        return plain_state
    return runner.run_range(plain_state, expansion.replay_start, layer_idx + 1)


def score_cached_expansions(
    runner: LlamaLikePartialRunner,
    expansions: list[CandidateExpansion],
    *,
    parent_cache: BoundaryCache,
    layer_idx: int,
    math_dataset: dict[str, dict[str, Any]],
    eq_dataset: dict[str, dict[str, Any]],
    reduction: str,
    device: torch.device,
    batch_size: int,
    pad_token_id: int,
    padding_side: str = "right",
) -> list[BeamCandidate]:
    """Score children from cached parent boundaries without rerunning prefixes."""
    grouped = _group_expansions_by_parent(expansions)
    metric_scores: dict[str, dict[tuple[int, ...], list[float]]] = {
        "math": {expansion.child.layer_path: [] for expansion in expansions},
        "eq": {expansion.child.layer_path: [] for expansion in expansions},
    }
    with torch.inference_mode():
        for metric, dataset in (("math", math_dataset), ("eq", eq_dataset)):
            for items in _dataset_item_batches(dataset, batch_size):
                batch = _collate_proxy_batch(
                    items,
                    device=device,
                    pad_token_id=pad_token_id,
                    padding_side=padding_side,
                )
                initial_state = runner.prepare(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids,
                )
                for parent_path, parent_expansions in grouped.items():
                    parent_state = _restore_parent_batch(
                        initial_state,
                        parent_path=parent_path,
                        metric=metric,
                        batch=batch,
                        parent_cache=parent_cache,
                        layer_idx=layer_idx,
                        device=device,
                    )
                    plain_state = runner.run_layer_indices(parent_state, (layer_idx,))
                    for expansion in parent_expansions:
                        child_state = _run_local_expansion(
                            runner,
                            plain_state,
                            expansion,
                            layer_idx=layer_idx,
                        )
                        final_state = runner.run_range(
                            child_state,
                            layer_idx + 1,
                            runner.num_layers,
                        )
                        logits = runner.logits(final_state)
                        metric_scores[metric][expansion.child.layer_path].extend(
                            _score_proxy_batch_logits(
                                logits=logits,
                                batch=batch,
                                reduction=reduction,
                            )
                        )

    scored: list[BeamCandidate] = []
    for expansion in expansions:
        path = expansion.child.layer_path
        math_values = metric_scores["math"][path]
        eq_values = metric_scores["eq"][path]
        if not math_values or not eq_values:
            raise ValueError("Cannot score an empty proxy dataset.")
        math_score = sum(math_values) / len(math_values)
        eq_score = sum(eq_values) / len(eq_values)
        scored.append(
            BeamCandidate(
                layer_path=path,
                replays=expansion.child.replays,
                score=(math_score + eq_score) / 2.0,
                math_score=math_score,
                eq_score=eq_score,
            )
        )
    return scored


def materialize_boundary_cache(
    runner: LlamaLikePartialRunner,
    expansions: list[CandidateExpansion],
    *,
    parent_cache: BoundaryCache,
    layer_idx: int,
    math_dataset: dict[str, dict[str, Any]],
    eq_dataset: dict[str, dict[str, Any]],
    device: torch.device,
    batch_size: int,
    pad_token_id: int,
    padding_side: str = "right",
) -> BoundaryCache:
    """Recreate only winning local expansions and offload their states to CPU."""
    grouped = _group_expansions_by_parent(expansions)
    next_cache: BoundaryCache = {
        expansion.child.layer_path: {} for expansion in expansions
    }
    with torch.inference_mode():
        for metric, dataset in (("math", math_dataset), ("eq", eq_dataset)):
            for items in _dataset_item_batches(dataset, batch_size):
                batch = _collate_proxy_batch(
                    items,
                    device=device,
                    pad_token_id=pad_token_id,
                    padding_side=padding_side,
                )
                initial_state = runner.prepare(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids,
                )
                for parent_path, parent_expansions in grouped.items():
                    parent_state = _restore_parent_batch(
                        initial_state,
                        parent_path=parent_path,
                        metric=metric,
                        batch=batch,
                        parent_cache=parent_cache,
                        layer_idx=layer_idx,
                        device=device,
                    )
                    plain_state = runner.run_layer_indices(parent_state, (layer_idx,))
                    for expansion in parent_expansions:
                        child_state = _run_local_expansion(
                            runner,
                            plain_state,
                            expansion,
                            layer_idx=layer_idx,
                        )
                        for row, qid in enumerate(batch.qids):
                            next_cache[expansion.child.layer_path][(metric, qid)] = (
                                child_state.hidden_states[row : row + 1]
                                .detach()
                                .to("cpu")
                                .contiguous()
                            )
    return next_cache


def boundary_cache_bytes(cache: BoundaryCache) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for candidate_cache in cache.values()
        for tensor in candidate_cache.values()
    )


def candidate_replay_budget(candidate: BeamCandidate, *, layer_idx: int) -> int:
    """Return extra executed layers in a partial path through ``layer_idx``."""
    budget = len(candidate.layer_path) - (layer_idx + 1)
    if budget < 0:
        raise ValueError("Candidate path does not reach the requested boundary.")
    return budget


def select_budget_indexed_beams(
    candidates: Iterable[BeamCandidate],
    *,
    layer_idx: int,
    beam_width: int,
    max_extra_layers: int,
) -> dict[int, list[BeamCandidate]]:
    """Keep an independent top beam for every exact replay-layer budget."""
    grouped: dict[int, list[BeamCandidate]] = {}
    for candidate in candidates:
        budget = candidate_replay_budget(candidate, layer_idx=layer_idx)
        if budget <= max_extra_layers:
            grouped.setdefault(budget, []).append(candidate)
    for budget_candidates in grouped.values():
        budget_candidates.sort(key=lambda candidate: float(candidate.score), reverse=True)
        del budget_candidates[beam_width:]
    return dict(sorted(grouped.items()))


def flatten_budget_beams(
    beam_by_budget: dict[int, list[BeamCandidate]],
) -> list[BeamCandidate]:
    candidates = [candidate for beam in beam_by_budget.values() for candidate in beam]
    candidates.sort(key=lambda candidate: float(candidate.score), reverse=True)
    return candidates


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
    beam_by_budget: dict[int, list[BeamCandidate]],
    elapsed_seconds: float,
    boundary_cache_size: int,
    peak_boundary_cache_size: int,
    peak_cuda_allocated: int | None,
    peak_cuda_reserved: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finished_proxy_search = layer_idx == final_layer
    status = "running"
    if finished_proxy_search:
        status = "proxy_complete" if args.exact_top_k > 0 else "complete"
    beam = flatten_budget_beams(beam_by_budget)
    payload = {
        "algorithm": "budget_indexed_benchmark_suffix_beam_search",
        "status": status,
        "full_model_search": final_layer == model_metadata["num_layers"] - 1,
        "model": model_metadata,
        "search": {
            "beam_width_per_budget": args.beam_width,
            "replay_window": args.replay_window,
            "max_extra_layers": args.max_extra_layers,
            "dataset_limit_per_benchmark": args.dataset_limit,
            "benchmark_batch_size": args.benchmark_batch_size,
            "padding_side": args.proxy_padding_side,
            "example_reduction": args.reduction,
            "combined_score": "(math_score + eq_score) / 2",
            "requested_final_layer": final_layer,
            "last_completed_layer": layer_idx,
            "elapsed_seconds": elapsed_seconds,
            "boundary_cache_mib": boundary_cache_size / 2**20,
            "peak_boundary_cache_mib": peak_boundary_cache_size / 2**20,
            "peak_cuda_allocated_mib": (
                peak_cuda_allocated / 2**20 if peak_cuda_allocated is not None else None
            ),
            "peak_cuda_reserved_mib": (
                peak_cuda_reserved / 2**20 if peak_cuda_reserved is not None else None
            ),
        },
        "beam": [
            _candidate_json(candidate, layer_idx=layer_idx, num_layers=model_metadata["num_layers"])
            for candidate in beam
        ],
        "beam_by_budget": {
            str(budget): [
                _candidate_json(
                    candidate,
                    layer_idx=layer_idx,
                    num_layers=model_metadata["num_layers"],
                )
                for candidate in budget_beam
            ]
            for budget, budget_beam in beam_by_budget.items()
        },
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
    math_dataset_path: str,
    eq_dataset_path: str,
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
        "math_dataset_path": math_dataset_path,
        "eq_dataset_path": eq_dataset_path,
        "examples_per_benchmark": len(results[0]["math_responses"]),
        "proxy_order": proxy_order,
        "exact_order": exact_order,
        "top_rank_agrees": proxy_order[0] == exact_order[0],
        "candidates": results,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output_path)


def default_benchmark_batch_size(device: torch.device, model_type: str | None) -> int:
    """Choose a correctness-first proxy batch size for the loaded architecture."""
    if device.type != "cuda" or model_type == "qwen3_5":
        return 1
    return 8


def resolve_exact_dataset_selection(
    args: argparse.Namespace,
) -> tuple[str, str, int | None]:
    """Resolve exact datasets without truncating explicitly separate validation files."""
    math_path = args.exact_math_dataset_path or args.math_dataset_path
    eq_path = args.exact_eq_dataset_path or args.eq_dataset_path
    limit = args.exact_dataset_limit
    uses_separate_dataset = bool(
        args.exact_math_dataset_path or args.exact_eq_dataset_path
    )
    if limit is None and not uses_separate_dataset:
        limit = args.dataset_limit
    return math_path, eq_path, limit


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
            "replay_budget": len(layer_path) - num_layers,
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
    parser.add_argument(
        "--exact-math-dataset-path",
        default=None,
        help="Math dataset used only for exact validation (default: search Math dataset).",
    )
    parser.add_argument(
        "--exact-eq-dataset-path",
        default=None,
        help="EQ dataset used only for exact validation (default: search EQ dataset).",
    )
    parser.add_argument("--output", default="results/suffix-beam-search.json")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=2,
        help="Candidates retained independently in each replay-budget cell.",
    )
    parser.add_argument("--replay-window", type=int, default=4)
    parser.add_argument(
        "--max-extra-layers",
        type=int,
        default=2,
        help="Replay-layer budget. Increase only after proxy/exact correlation validation.",
    )
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument(
        "--benchmark-batch-size",
        "--benchmark-chunk-size",
        dest="benchmark_batch_size",
        type=int,
        default=None,
        help="Right-padded proxy examples per step (default: 1 on CPU/MPS, 8 on CUDA).",
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
    if args.benchmark_batch_size is not None and args.benchmark_batch_size < 1:
        raise ValueError("--benchmark-batch-size must be >= 1.")
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if args.benchmark_batch_size is None:
        args.benchmark_batch_size = default_benchmark_batch_size(
            device,
            metadata.get("model_type"),
        )
    args.proxy_padding_side = (
        "left" if metadata.get("model_type") == "qwen3_5" else "right"
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("Batched proxy scoring requires tokenizer.pad_token_id.")
    pad_token_id = int(tokenizer.pad_token_id)

    math_raw = _load_dataset(args.math_dataset_path, args.dataset_limit)
    eq_raw = _load_dataset(args.eq_dataset_path, args.dataset_limit)
    math_dataset = pretokenize_teacher_forced_dataset(math_raw, tokenizer, torch.device("cpu"))
    eq_dataset = pretokenize_eq_teacher_forced_dataset(eq_raw, tokenizer, torch.device("cpu"))

    final_layer = runner.num_layers - 1
    if args.stop_after_layer is not None:
        if args.stop_after_layer < 0:
            raise ValueError("--stop-after-layer must be >= 0.")
        final_layer = min(final_layer, args.stop_after_layer)

    beam_by_budget = {0: [BeamCandidate(layer_path=(), replays=())]}
    parent_cache: BoundaryCache = {}
    baseline_proxy_score: float | None = None
    peak_boundary_cache_size = 0
    started = time.monotonic()
    output_path = Path(args.output)
    for layer_idx in range(final_layer + 1):
        parents = [
            candidate
            for budget_beam in beam_by_budget.values()
            for candidate in budget_beam
        ]
        expansions = expand_beam_with_parents(
            parents,
            layer_idx=layer_idx,
            replay_window=args.replay_window,
            num_layers=runner.num_layers,
            max_extra_layers=args.max_extra_layers,
        )
        scored = score_cached_expansions(
            runner,
            expansions,
            parent_cache=parent_cache,
            layer_idx=layer_idx,
            math_dataset=math_dataset,
            eq_dataset=eq_dataset,
            reduction=args.reduction,
            device=device,
            batch_size=args.benchmark_batch_size,
            pad_token_id=pad_token_id,
            padding_side=args.proxy_padding_side,
        )
        if layer_idx == 0:
            baseline = next(candidate for candidate in scored if not candidate.replays)
            baseline_proxy_score = float(baseline.score)
        beam_by_budget = select_budget_indexed_beams(
            scored,
            layer_idx=layer_idx,
            beam_width=args.beam_width,
            max_extra_layers=args.max_extra_layers,
        )
        beam = flatten_budget_beams(beam_by_budget)
        boundary_cache_size = 0
        if layer_idx < final_layer:
            expansion_by_path = {
                expansion.child.layer_path: expansion for expansion in expansions
            }
            winning_expansions = [
                expansion_by_path[candidate.layer_path] for candidate in beam
            ]
            parent_cache = materialize_boundary_cache(
                runner,
                winning_expansions,
                parent_cache=parent_cache,
                layer_idx=layer_idx,
                math_dataset=math_dataset,
                eq_dataset=eq_dataset,
                device=device,
                batch_size=args.benchmark_batch_size,
                pad_token_id=pad_token_id,
                padding_side=args.proxy_padding_side,
            )
            boundary_cache_size = boundary_cache_bytes(parent_cache)
            peak_boundary_cache_size = max(
                peak_boundary_cache_size,
                boundary_cache_size,
            )
        elapsed = time.monotonic() - started
        best = beam[0]
        print(
            f"layer={layer_idx} children={len(expansions)} "
            f"budgets={{{', '.join(f'{budget}:{len(items)}' for budget, items in beam_by_budget.items())}}} "
            f"best={best.score:.6f} math={best.math_score:.6f} "
            f"eq={best.eq_score:.6f} replays={best.replays} "
            f"cache={boundary_cache_size / 2**20:.1f}MiB elapsed={elapsed:.1f}s",
            flush=True,
        )
        _write_progress(
            output_path,
            args=args,
            model_metadata=metadata,
            layer_idx=layer_idx,
            final_layer=final_layer,
            beam_by_budget=beam_by_budget,
            elapsed_seconds=elapsed,
            boundary_cache_size=boundary_cache_size,
            peak_boundary_cache_size=peak_boundary_cache_size,
            peak_cuda_allocated=(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            peak_cuda_reserved=(
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            ),
        )

    beam = flatten_budget_beams(beam_by_budget)
    if args.exact_top_k > 0:
        if baseline_proxy_score is None:
            raise RuntimeError("Baseline proxy score was not captured.")
        exact_math_path, exact_eq_path, exact_limit = resolve_exact_dataset_selection(args)
        exact_math_raw = _load_dataset(
            exact_math_path,
            exact_limit,
            offset=args.exact_dataset_offset,
        )
        exact_eq_raw = _load_dataset(
            exact_eq_path,
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
            math_dataset_path=exact_math_path,
            eq_dataset_path=exact_eq_path,
        )

    print(f"Saved {len(beam)} candidates to {output_path}")


if __name__ == "__main__":
    main()
