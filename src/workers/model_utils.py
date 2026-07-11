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
        text = "auto"
    if text.startswith("{") or text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, (dict, list)):
            raise ValueError("device map JSON must be an object or list")
        return parsed
    if text == "auto":
        if torch.cuda.is_available():
            return "auto"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return text


def parse_torch_dtype_arg(
    raw: str | None,
    *,
    device_map: str | dict[str, Any] | None = None,
) -> torch.dtype:
    """
    Parse a torch dtype CLI argument with a practical auto mode.

    Auto defaults:
    - CPU/MPS: float32 for maximum compatibility on local dev machines
    - CUDA/auto sharding: bfloat16 as the existing repo default
    """
    text = "auto" if raw is None else str(raw).strip().lower()
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"fp32", "float32", "float"}:
        return torch.float32
    if text != "auto":
        raise ValueError(f"Unsupported torch dtype: {raw!r}")

    if isinstance(device_map, str) and device_map in {"cpu", "mps"}:
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16


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

    explicit_device: str | None = None
    if isinstance(device_map, str) and device_map in {"cpu", "mps"}:
        explicit_device = device_map

    load_kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if explicit_device is None:
        load_kwargs["device_map"] = device_map
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    if cpu_offload and explicit_device is None:
        load_kwargs["offload_state_dict"] = True
        if offload_folder:
            load_kwargs["offload_folder"] = offload_folder

    model = model_cls.from_pretrained(model_path, **load_kwargs)
    if explicit_device is not None:
        model.to(explicit_device)
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
        "execution_device": explicit_device,
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
    }
    return tokenizer, model, metadata


def model_input_device(model: Any) -> torch.device:
    """Return the device that should be used for model inputs."""
    return next(model.parameters()).device


def maybe_empty_cache(device: torch.device | str | None = None) -> None:
    """Best-effort cache cleanup across CUDA and MPS backends."""
    dev_type = None
    if isinstance(device, torch.device):
        dev_type = device.type
    elif isinstance(device, str):
        dev_type = device.split(":", 1)[0]

    if dev_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    if dev_type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


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


@dataclass(frozen=True)
class PartialForwardState:
    """Hidden state plus the immutable inputs required by decoder layers."""

    hidden_states: torch.Tensor
    attention_masks: Any
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]


class LlamaLikePartialRunner:
    """Run arbitrary layer paths for current Llama/Qwen-style HF models.

    This first implementation intentionally disables KV caching. That makes
    replaying an original layer safe because attention ``layer_idx`` values are
    not used to address cache slots.
    """

    def __init__(self, model: Any):
        decoder, layers_attr, stack_path = get_text_layer_owner(model)
        missing = [
            name
            for name in ("embed_tokens", "norm", "rotary_emb")
            if not hasattr(decoder, name)
        ]
        if missing:
            raise TypeError(
                f"Unsupported decoder at {stack_path}: missing {', '.join(missing)}. "
                "Expected a Llama/Qwen-style decoder stack."
            )

        output_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_head is None:
            raise TypeError("Model does not expose an output embedding / LM head.")

        self.model = model
        self.decoder = decoder
        self.layers = getattr(decoder, layers_attr)
        self.output_head = output_head
        self.stack_path = stack_path

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def prepare(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> PartialForwardState:
        """Embed a full sequence and prepare mask/position inputs once."""
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")

        hidden_states = self.decoder.embed_tokens(input_ids)
        cache_position = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
        )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_kwargs = {
            "config": self.decoder.config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        full_mask = create_causal_mask(**mask_kwargs)
        if getattr(self.decoder, "has_sliding_layers", False):
            attention_masks: Any = {
                "full_attention": full_mask,
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        else:
            attention_masks = full_mask

        position_embeddings = self.decoder.rotary_emb(
            hidden_states,
            position_ids=position_ids,
        )
        return PartialForwardState(
            hidden_states=hidden_states,
            attention_masks=attention_masks,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    def run_layer_indices(
        self,
        state: PartialForwardState,
        layer_indices: list[int] | tuple[int, ...] | range,
    ) -> PartialForwardState:
        """Run an arbitrary layer path, including repeated indices."""
        hidden_states = state.hidden_states
        for raw_idx in layer_indices:
            idx = int(raw_idx)
            if idx < 0 or idx >= self.num_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {self.num_layers}).")

            layer = self.layers[idx]
            attention_mask = state.attention_masks
            if isinstance(attention_mask, dict):
                attention_type = getattr(layer, "attention_type", "full_attention")
                if attention_type not in attention_mask:
                    raise ValueError(f"No prepared attention mask for type {attention_type!r}.")
                attention_mask = attention_mask[attention_type]

            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=state.position_embeddings,
                position_ids=state.position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=state.cache_position,
            )
            hidden_states = layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output

        return PartialForwardState(
            hidden_states=hidden_states,
            attention_masks=state.attention_masks,
            position_ids=state.position_ids,
            cache_position=state.cache_position,
            position_embeddings=state.position_embeddings,
        )

    def run_range(
        self,
        state: PartialForwardState,
        start: int,
        end: int,
    ) -> PartialForwardState:
        """Run the half-open original-layer range ``[start, end)``."""
        if start < 0 or end < start or end > self.num_layers:
            raise ValueError(f"Invalid layer range [{start}, {end}) for {self.num_layers} layers.")
        return self.run_layer_indices(state, range(start, end))

    def logits(self, state: PartialForwardState) -> torch.Tensor:
        """Apply final normalization and the model's output head."""
        return self.output_head(self.decoder.norm(state.hidden_states))

    def forward_path(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_indices: list[int] | tuple[int, ...] | range | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one complete custom layer path and return token logits."""
        state = self.prepare(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        path = range(self.num_layers) if layer_indices is None else layer_indices
        state = self.run_layer_indices(state, path)
        return self.logits(state)


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

    return score_teacher_forced_logits(
        logits=outputs.logits,
        input_ids=input_ids,
        prompt_length=prompt_length,
        target_mask=target_mask,
        reduction=reduction,
    )


def score_teacher_forced_logits(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_length: int,
    target_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> dict[str, Any]:
    """Score precomputed logits with the standard teacher-forced objective."""
    token_logprobs = gather_target_token_logprobs(
        logits=logits,
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
