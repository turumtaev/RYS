#!/usr/bin/env python
"""Shared model helpers for worker entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - unavailable in older transformers versions
    AutoModelForImageTextToText = None  # type: ignore[assignment]


def get_text_layer_owner(model: Any) -> tuple[Any, str, str]:
    """
    Locate the module that owns the text decoder layer ModuleList.

    Returns:
        (owner_object, attribute_name, dotted_path)
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "layers", "model.layers"

    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model, "layers", "model.language_model.layers"

    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model, "layers", "language_model.layers"

    raise AttributeError(
        "Could not locate text decoder layers on model. "
        "Expected one of: model.layers, model.language_model.layers, language_model.layers."
    )


def get_text_layers(model: Any):
    owner, attr, _ = get_text_layer_owner(model)
    return getattr(owner, attr)


def get_text_num_layers(model: Any) -> int:
    return len(get_text_layers(model))


def parse_device_map_arg(raw: str) -> str | dict[str, Any]:
    """
    Parse a device-map CLI argument.

    Accepts simple strings (e.g. "cuda:0", "auto") or JSON objects.
    """
    text = str(raw).strip()
    if not text:
        return "cuda:0"
    if text.startswith("{") or text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, (dict, list)):
            raise ValueError("device map JSON must be an object or list")
        return parsed
    return text


def parse_max_memory_json(raw: str | None) -> dict[str, Any] | None:
    """
    Parse optional max-memory JSON from CLI.

    Expected format:
      '{"cuda:0":"80GiB","cuda:1":"80GiB","cpu":"120GiB"}'
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("max-memory JSON must be an object")

    # Accelerate accepts integer GPU ids (0, 1, ...) rather than "cuda:0" keys.
    normalized: dict[Any, Any] = {}
    cuda_key = re.compile(r"^cuda:(\d+)$")
    for key, value in parsed.items():
        if isinstance(key, str):
            stripped = key.strip()
            match = cuda_key.match(stripped)
            if match:
                normalized[int(match.group(1))] = value
                continue
            if stripped.isdigit():
                normalized[int(stripped)] = value
                continue
            if stripped in {"cpu", "disk", "mps"}:
                normalized[stripped] = value
                continue
        normalized[key] = value
    return normalized


def select_generation_loader(config: Any) -> tuple[Any, str]:
    """Choose the appropriate AutoModel class for generation."""
    loader_pref = os.getenv("LEVELGEN_TEXT_LOADER", "auto").strip().lower()
    if loader_pref in {"causal", "causallm", "auto_causal"}:
        return AutoModelForCausalLM, "AutoModelForCausalLM"
    if loader_pref in {"itt", "image", "image_text"}:
        if AutoModelForImageTextToText is None:
            raise RuntimeError(
                "LEVELGEN_TEXT_LOADER requested ImageTextToText, but "
                "AutoModelForImageTextToText is unavailable in this transformers build."
            )
        return AutoModelForImageTextToText, "AutoModelForImageTextToText"

    architectures = list(getattr(config, "architectures", []) or [])
    is_conditional = any("ConditionalGeneration" in arch for arch in architectures)

    if is_conditional:
        if AutoModelForImageTextToText is None:
            raise RuntimeError(
                "Checkpoint advertises a ConditionalGeneration architecture, but "
                "AutoModelForImageTextToText is unavailable in this transformers build."
            )
        return AutoModelForImageTextToText, "AutoModelForImageTextToText"

    return AutoModelForCausalLM, "AutoModelForCausalLM"


def normalize_moe_fp8_config(config: Any) -> Any:
    """
    Patch known MoE config field placement issues for FP8 integrations.

    Some Qwen3.5 MoE checkpoints keep expert-related attributes under
    ``text_config`` while the FP8 integration expects them at the top level.
    """
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        return config

    fields = [
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "hidden_act",
        "intermediate_size",
        "rms_norm_eps",
    ]

    for field in fields:
        top_val = getattr(config, field, None)
        text_val = getattr(text_cfg, field, None)
        if top_val is None and text_val is not None:
            setattr(config, field, text_val)

    # Some FP8 integrations read many generic config fields from the top-level
    # config object. Mirror any missing text_config keys to avoid loader crashes.
    if hasattr(text_cfg, "to_dict"):
        for key, value in text_cfg.to_dict().items():
            if key in {"model_type", "architectures"}:
                continue
            if value is None:
                continue
            if getattr(config, key, None) is None:
                setattr(config, key, value)

    return config


def load_model_and_tokenizer(
    *,
    model_path: str,
    trust_remote_code: bool,
    local_files_only: bool,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict[str, Any] = "cuda:0",
    attn_implementation: str | None = None,
    max_memory: dict[str, Any] | None = None,
    cpu_offload: bool = False,
    offload_folder: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """
    Load tokenizer + model using architecture-aware loader selection.

    Returns:
        (tokenizer, model, metadata)
    """
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    config = normalize_moe_fp8_config(config)
    model_cls, loader_name = select_generation_loader(config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    if cpu_offload:
        load_kwargs["offload_state_dict"] = True
        if offload_folder:
            load_kwargs["offload_folder"] = offload_folder

    model = model_cls.from_pretrained(model_path, **load_kwargs)
    model.eval()

    owner, _, stack_path = get_text_layer_owner(model)
    num_layers = len(getattr(owner, "layers"))

    metadata = {
        "architectures": list(getattr(config, "architectures", []) or []),
        "model_type": getattr(config, "model_type", None),
        "loader": loader_name,
        "text_stack": stack_path,
        "num_layers": num_layers,
        "device_map_arg": device_map,
        "hf_device_map": getattr(model, "hf_device_map", None),
    }
    return tokenizer, model, metadata


def is_moe_model(model) -> bool:
    """
    Detect if a model is a Mixture of Experts (MoE) model.

    Checks for common MoE indicators in the config and model structure.
    """
    config = model.config

    # Check config attributes that indicate MoE
    if hasattr(config, "num_experts") and config.num_experts > 1:
        return True
    if hasattr(config, "num_local_experts") and config.num_local_experts > 1:
        return True
    if hasattr(config, "n_routed_experts") and config.n_routed_experts > 1:
        return True

    # Check for MoE-specific layer structure
    try:
        layers = get_text_layers(model)
    except AttributeError:
        layers = None

    if layers is not None and len(layers) > 0:
        first_layer = layers[0]
        if hasattr(first_layer, "mlp"):
            mlp = first_layer.mlp
            if hasattr(mlp, "experts") and hasattr(mlp.experts, "__len__"):
                return True
            if hasattr(mlp, "gate") or hasattr(mlp, "router"):
                return True
            if hasattr(mlp, "shared_expert") or hasattr(mlp, "shared_experts"):
                return True

    return False


def strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks from model output.

    Handles both complete and incomplete thinking blocks.
    Works with Qwen3, GPT-OSS, and other thinking models.
    """
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)
    return result.strip()


@dataclass(frozen=True)
class TeacherForcedInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_length: int
    target_length: int
    target_mask: torch.Tensor | None = None


def _move_tensor_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device | str | None,
) -> dict[str, torch.Tensor]:
    if device is None:
        return batch
    return {k: v.to(device) for k, v in batch.items()}


def build_teacher_forced_inputs(
    tokenizer: Any,
    *,
    prompt_text: str,
    target_text: str,
    target_char_spans: list[tuple[int, int]] | None = None,
    device: torch.device | str | None = None,
    add_special_tokens: bool = True,
) -> TeacherForcedInputs:
    """Tokenize a prompt/target pair for teacher-forced scoring."""
    prompt_batch = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=add_special_tokens)
    full_tokenizer_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "add_special_tokens": add_special_tokens,
    }
    need_offsets = bool(target_char_spans)
    if need_offsets:
        full_tokenizer_kwargs["return_offsets_mapping"] = True
    full_batch = tokenizer(prompt_text + target_text, **full_tokenizer_kwargs)

    prompt_ids = prompt_batch["input_ids"]
    full_ids = full_batch["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if full_len <= prompt_len:
        raise ValueError("Target text produced no additional tokens.")
    if not torch.equal(full_ids[:, :prompt_len], prompt_ids):
        raise ValueError(
            "Prompt tokenization is not a prefix of prompt+target tokenization. "
            "Use a canonical target serialization that preserves the tokenizer boundary."
        )

    target_mask = None
    if target_char_spans:
        if "offset_mapping" not in full_batch:
            raise ValueError(
                "Masked teacher-forced scoring requires tokenizer offset mappings. "
                "Use a fast tokenizer or disable target masking."
            )

        prompt_char_len = len(prompt_text)
        target_offsets = full_batch["offset_mapping"][:, prompt_len:, :]
        mask_rows: list[list[bool]] = []
        for row in target_offsets.tolist():
            row_mask: list[bool] = []
            for start, end in row:
                rel_start = max(0, int(start) - prompt_char_len)
                rel_end = max(0, int(end) - prompt_char_len)
                is_scored = False
                if rel_end > rel_start:
                    for span_start, span_end in target_char_spans:
                        if rel_start < span_end and rel_end > span_start:
                            is_scored = True
                            break
                row_mask.append(is_scored)
            mask_rows.append(row_mask)
        target_mask = torch.tensor(mask_rows, dtype=torch.bool)

    full_batch = _move_tensor_batch(
        {k: v for k, v in full_batch.items() if k != "offset_mapping"},
        device,
    )
    if target_mask is not None and device is not None:
        target_mask = target_mask.to(device)
    return TeacherForcedInputs(
        input_ids=full_batch["input_ids"],
        attention_mask=full_batch["attention_mask"],
        prompt_length=prompt_len,
        target_length=full_len - prompt_len,
        target_mask=target_mask,
    )


def gather_target_token_logprobs(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """Extract token logprobs for the target suffix of a prompt+target sequence."""
    if input_ids.ndim != 2 or logits.ndim != 3:
        raise ValueError("Expected input_ids shape [batch, seq] and logits shape [batch, seq, vocab].")
    if input_ids.shape[0] != logits.shape[0] or input_ids.shape[1] != logits.shape[1]:
        raise ValueError("Input/logit shape mismatch.")
    if prompt_length <= 0 or prompt_length >= input_ids.shape[1]:
        raise ValueError("prompt_length must be in [1, seq_len - 1].")

    target_positions = torch.arange(prompt_length, input_ids.shape[1], device=input_ids.device)
    next_token_logits = logits[:, target_positions - 1, :]
    target_token_ids = input_ids[:, target_positions]
    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    return log_probs.gather(-1, target_token_ids.unsqueeze(-1)).squeeze(-1)


def reduce_logprobs(
    token_logprobs: torch.Tensor,
    reduction: str = "mean",
    mask: torch.Tensor | None = None,
) -> float:
    """Reduce token-level logprobs to a scalar score."""
    if mask is not None:
        if mask.shape != token_logprobs.shape:
            raise ValueError("Mask shape must match token_logprobs shape.")
        token_logprobs = token_logprobs.masked_select(mask)
    if token_logprobs.numel() == 0:
        raise ValueError("Cannot reduce an empty token logprob tensor.")
    if reduction == "mean":
        return float(token_logprobs.mean().item())
    if reduction == "sum":
        return float(token_logprobs.sum().item())
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def score_teacher_forced_inputs(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
    target_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> dict[str, Any]:
    """Teacher-force a pretokenized prompt/target pair."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    token_logprobs = gather_target_token_logprobs(
        logits=outputs.logits,
        input_ids=input_ids,
        prompt_length=prompt_length,
    )

    return {
        "score": reduce_logprobs(token_logprobs, reduction=reduction, mask=target_mask),
        "sum_logprob": float(token_logprobs.sum().item()),
        "mean_logprob": float(token_logprobs.mean().item()),
        "target_token_count": int(token_logprobs.numel()),
        "scored_token_count": int(target_mask.sum().item()) if target_mask is not None else int(token_logprobs.numel()),
    }


def score_prompt_target(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    target_char_spans: list[tuple[int, int]] | None = None,
    device: torch.device | str | None = None,
    add_special_tokens: bool = True,
    reduction: str = "mean",
) -> dict[str, Any]:
    """Teacher-force a canonical target and return token-level logprob metrics."""
    tf_inputs = build_teacher_forced_inputs(
        tokenizer,
        prompt_text=prompt_text,
        target_text=target_text,
        target_char_spans=target_char_spans,
        device=device,
        add_special_tokens=add_special_tokens,
    )
    result = score_teacher_forced_inputs(
        model=model,
        input_ids=tf_inputs.input_ids,
        attention_mask=tf_inputs.attention_mask,
        prompt_length=tf_inputs.prompt_length,
        target_mask=tf_inputs.target_mask,
        reduction=reduction,
    )
    result["prompt_token_count"] = int(tf_inputs.prompt_length)
    result["target_length"] = int(tf_inputs.target_length)
    return result
