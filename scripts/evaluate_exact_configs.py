#!/usr/bin/env python3
"""Generation-score explicit relayer configurations without running a search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from src.core.layer_config import expand_multi_block_config, parse_blocks_string
from src.core.layer_duplicator import build_model_with_layers
from src.workers.eq_worker import pretokenize_eq_dataset, run_eq_test
from src.workers.math_worker import pretokenize_dataset, run_math_test_batched_moe
from src.workers.model_utils import (
    get_text_num_layers,
    load_model_and_tokenizer,
    maybe_empty_cache,
    model_input_device,
    parse_device_map_arg,
    parse_torch_dtype_arg,
)


def parse_config(raw: str, num_layers: int) -> tuple[str, list[int], dict[str, Any]]:
    """Parse LABEL=baseline, LABEL=blocks:A,B;C,D, or LABEL=repeat:L,EXTRA."""
    if "=" not in raw:
        raise ValueError(f"Config must have LABEL=SPEC form: {raw!r}")
    label, spec = (part.strip() for part in raw.split("=", 1))
    if not label:
        raise ValueError(f"Config label is empty: {raw!r}")

    if spec == "baseline":
        return label, list(range(num_layers)), {"type": "baseline"}
    if spec.startswith("blocks:"):
        blocks = parse_blocks_string(spec)
        return (
            label,
            expand_multi_block_config(num_layers, blocks),
            {"type": "blocks", "blocks": [list(block) for block in blocks]},
        )
    if spec.startswith("repeat:"):
        parts = [part.strip() for part in spec.split(":", 1)[1].split(",")]
        if len(parts) != 2:
            raise ValueError(f"Repeat config must be repeat:LAYER,EXTRA: {raw!r}")
        layer, extra = map(int, parts)
        if layer < 0 or layer >= num_layers or extra < 1:
            raise ValueError(f"Invalid repeat config: {raw!r}")
        path = list(range(num_layers))
        path[layer + 1 : layer + 1] = [layer] * extra
        return label, path, {"type": "repeat", "layer": layer, "extra": extra}
    raise ValueError(f"Unsupported config spec: {spec!r}")


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--math-dataset-path", required=True)
    parser.add_argument("--eq-dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="LABEL=baseline, LABEL=blocks:A,B;C,D, or LABEL=repeat:LAYER,EXTRA",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--math-max-new", type=int, default=64)
    parser.add_argument("--eq-max-new", type=int, default=384)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    device_map = parse_device_map_arg(args.device_map)
    torch_dtype = parse_torch_dtype_arg(args.torch_dtype, device_map=device_map)
    tokenizer, model, metadata = load_model_and_tokenizer(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    device = model_input_device(model)
    num_layers = get_text_num_layers(model)
    configs = [parse_config(raw, num_layers) for raw in args.config]
    if len({label for label, _, _ in configs}) != len(configs):
        raise ValueError("Config labels must be unique.")

    math_dataset = json.loads(Path(args.math_dataset_path).read_text())
    eq_dataset = json.loads(Path(args.eq_dataset_path).read_text())
    exact_math = pretokenize_dataset(math_dataset, tokenizer, device)
    exact_eq = pretokenize_eq_dataset(eq_dataset, tokenizer, device)

    output_path = Path(args.output)
    if output_path.exists():
        payload = json.loads(output_path.read_text())
        results = payload.get("results", [])
    else:
        results = []
        payload = {
            "status": "running",
            "model": metadata,
            "math_dataset_path": args.math_dataset_path,
            "eq_dataset_path": args.eq_dataset_path,
            "math_examples": len(math_dataset),
            "eq_examples": len(eq_dataset),
            "eq_reference": "reference_answer_fullscale",
            "combined_score": "(math_score + eq_score) / 2",
            "results": results,
        }
    completed = {result["label"] for result in results}

    for label, layer_path, config_data in configs:
        if label in completed:
            print(f"exact label={label} skipped=already_complete", flush=True)
            continue
        run_model = (
            model
            if layer_path == list(range(num_layers))
            else build_model_with_layers(model, layer_path)
        )
        started = time.monotonic()
        math_result = run_math_test_batched_moe(
            run_model,
            exact_math,
            tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.math_max_new,
            save_responses=True,
        )
        eq_result = run_eq_test(
            run_model,
            exact_eq,
            tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.eq_max_new,
            save_responses=True,
        )
        math_score = float(math_result["score"])
        eq_score = float(eq_result["score"])
        result = {
            "label": label,
            "config": config_data,
            "layer_path": layer_path,
            "extra_layers": len(layer_path) - num_layers,
            "overhead": (len(layer_path) - num_layers) / num_layers,
            "math_score": math_score,
            "eq_score": eq_score,
            "score": (math_score + eq_score) / 2.0,
            "elapsed_seconds": time.monotonic() - started,
            "math_responses": math_result.get("responses", []),
            "eq_responses": eq_result.get("responses", []),
        }
        results.append(result)
        payload["results"] = results
        write_result(output_path, payload)
        print(
            f"exact label={label} score={result['score']:.6f} "
            f"math={math_score:.6f} eq={eq_score:.6f} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if run_model is not model:
            del run_model
            maybe_empty_cache(device)

    payload["status"] = "complete"
    baseline = next((result for result in results if result["label"] == "baseline"), None)
    if baseline is not None:
        for result in results:
            result["math_delta"] = result["math_score"] - baseline["math_score"]
            result["eq_delta"] = result["eq_score"] - baseline["eq_score"]
            result["delta_sum"] = result["math_delta"] + result["eq_delta"]
    payload["exact_order"] = [
        result["label"] for result in sorted(results, key=lambda item: item["score"], reverse=True)
    ]
    write_result(output_path, payload)


if __name__ == "__main__":
    main()
