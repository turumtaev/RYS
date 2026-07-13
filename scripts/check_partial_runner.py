#!/usr/bin/env python3
"""Verify that partial-layer execution reproduces a native full forward pass."""

from __future__ import annotations

import argparse
import json

import torch

from src.workers.model_utils import (
    LlamaLikePartialRunner,
    load_model_and_tokenizer,
    model_input_device,
    parse_device_map_arg,
    parse_torch_dtype_arg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="Calculate 17 multiplied by 23.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--attention-impl", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    device = model_input_device(model)
    batch = tokenizer(args.prompt, return_tensors="pt")
    inputs = {
        name: tensor.to(device)
        for name, tensor in batch.items()
        if name in {"input_ids", "attention_mask"}
    }
    runner = LlamaLikePartialRunner(model)

    with torch.no_grad():
        native = model(**inputs, use_cache=False).logits
        partial = runner.forward_path(**inputs)

    native_float = native.float()
    partial_float = partial.float()
    delta = (native_float - partial_float).abs()
    native_finite = bool(torch.isfinite(native).all().item())
    partial_finite = bool(torch.isfinite(partial).all().item())
    close = bool(
        torch.allclose(
            native_float,
            partial_float,
            rtol=args.rtol,
            atol=args.atol,
        )
    )
    result = {
        "model": metadata,
        "shape": list(native.shape),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "native_finite": native_finite,
        "partial_finite": partial_finite,
        "allclose": close,
        "rtol": args.rtol,
        "atol": args.atol,
    }
    print(json.dumps(result, indent=2))
    if not (native_finite and partial_finite and close):
        raise SystemExit("Partial runner does not match native forward pass.")


if __name__ == "__main__":
    main()
