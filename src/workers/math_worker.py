#!/usr/bin/env python
"""
Math benchmark worker for relayer experiments.

Worker that pulls configs from a shared work queue and processes them.
Supports both dense and MoE decoder stacks.

Multiple workers can run in parallel on different GPUs, dynamically
claiming work from the shared queue.

Usage:
    # Launch on GPU 0
    CUDA_VISIBLE_DEVICES=0 python -m src.workers.math_worker --model-path /path/to/model &

    # Launch on GPU 1
    CUDA_VISIBLE_DEVICES=1 python -m src.workers.math_worker --model-path /path/to/model &

    # With custom queue/results files
    CUDA_VISIBLE_DEVICES=0 python -m src.workers.math_worker \
        --queue-file work_queue.json \
        --results-file results/shared_results.pkl
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.layer_duplicator_moe import (
    build_model_with_layers_moe,
    expand_multi_block_config,
)
from src.core.layer_duplicator import build_model_with_layers
from src.core.layer_config import (
    is_baseline_layers,
    layer_spec_string,
    parse_blocks_string,
    parse_layer_list_string,
    parse_queue_entry_layers,
)
from src.workers.model_utils import (
    build_teacher_forced_inputs,
    is_moe_model,
    load_model_and_tokenizer,
    parse_device_map_arg,
    parse_max_memory_json,
    score_teacher_forced_inputs,
    strip_thinking,
)
from src.workers.shared_queue import SharedWorkQueue, format_eta

PADDING_MODE_MASKED = "masked"
PADDING_MODE_INPROMPT_SPACE = "inprompt_space"
MATH_SYSTEM_PROMPT = (
    "You are a highly intelligent AI. You have extraordinary intuition and can "
    "easily make accurate estimations. For the following questions, you will "
    "always provide an answer, even if you are not certain."
)


def calculate_score(actual, estimate):
    """Calculate score comparing actual vs estimated answer."""
    try:
        actual_str = str(int(actual))
        estimate_str = str(int(estimate))
    except (ValueError, OverflowError):
        return 0

    max_length = max(len(actual_str), len(estimate_str))
    actual_padded = actual_str.ljust(max_length, "0")
    estimate_padded = estimate_str.ljust(max_length, "0")
    padding_size = max_length - min(len(actual_str), len(estimate_str))

    actual_int = int(actual_padded)
    estimate_int = int(estimate_padded)

    if max(actual_int, estimate_int) == 0:
        return 0
    relative_diff = abs(actual_int - estimate_int) / max(actual_int, estimate_int)
    correction_factor = 1 - (padding_size / max_length)
    score = (1 - relative_diff) * correction_factor

    return max(0, min(score, 1))


def generate_messages(question, *, use_no_think_prefix: bool = True):
    """Generate chat messages for the math question."""
    user_text = f"/no_think {question}" if use_no_think_prefix else question
    return [
        {
            "role": "system",
            "content": MATH_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]


def serialize_math_target(answer) -> str:
    """Convert a math answer to the canonical proxy target string."""
    try:
        return str(int(answer))
    except (TypeError, ValueError, OverflowError):
        return str(answer).strip()


def build_math_target(answer) -> tuple[str, list[tuple[int, int]]]:
    """Build the canonical math target and the spans to score."""
    target_text = serialize_math_target(answer)
    return target_text, [(0, len(target_text))]


def build_math_prompt_target(tokenizer, question: str, answer, *, use_no_think_prefix: bool = True) -> tuple[str, str, list[tuple[int, int]]]:
    """Render the canonical math prompt and target for teacher-forced proxy scoring."""
    messages = generate_messages(question, use_no_think_prefix=use_no_think_prefix)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    target_text, target_char_spans = build_math_target(answer)
    return prompt, target_text, target_char_spans

def extract_integers(text):
    """Extract all integers from generated text."""
    split_parts = re.split(r'\D+', text)
    return [int(part) for part in split_parts if part.isdigit()]


def pretokenize_dataset(dataset, tokenizer, device, *, use_no_think_prefix: bool = True):
    """Pre-tokenize all questions once."""
    print("Pre-tokenizing dataset...")
    tokenized = {}

    for qid, sample in tqdm(dataset.items(), desc="Tokenizing"):
        messages = generate_messages(sample["question"], use_no_think_prefix=use_no_think_prefix)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode for faster inference
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        tokenized[qid] = {
            'input_ids': inputs['input_ids'].to(device),
            'attention_mask': inputs['attention_mask'].to(device),
            'answer': sample['answer']
        }

    print(f"Pre-tokenized {len(tokenized)} questions")
    return tokenized


def pretokenize_teacher_forced_dataset(dataset, tokenizer, device, *, use_no_think_prefix: bool = True) -> dict:
    """Pre-tokenize prompt+target pairs once for teacher-forced proxy scoring."""
    print("Pre-tokenizing teacher-forced math dataset...")
    tokenized = {}

    for qid, sample in tqdm(dataset.items(), desc="Tokenizing teacher-forced"):
        prompt_text, target_text, target_char_spans = build_math_prompt_target(
            tokenizer,
            sample["question"],
            sample["answer"],
            use_no_think_prefix=use_no_think_prefix,
        )
        inputs = build_teacher_forced_inputs(
            tokenizer,
            prompt_text=prompt_text,
            target_text=target_text,
            target_char_spans=target_char_spans,
            device=device,
        )
        tokenized[qid] = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "prompt_length": inputs.prompt_length,
            "target_length": inputs.target_length,
            "target_mask": inputs.target_mask,
            "target_text": target_text,
        }

    print(f"Pre-tokenized {len(tokenized)} teacher-forced math questions")
    return tokenized


def run_math_teacher_forced_proxy(
    model,
    teacher_forced_dataset: dict,
    *,
    reduction: str = "mean",
    save_responses: bool = True,
):
    """Score math benchmark examples with teacher-forced target logprobs."""
    responses = [] if save_responses else None
    scores = []
    mean_logprobs = []
    sum_logprobs = []
    total_target_tokens = 0

    for qid, cached in teacher_forced_dataset.items():
        result = score_teacher_forced_inputs(
            model=model,
            input_ids=cached["input_ids"],
            attention_mask=cached["attention_mask"],
            prompt_length=int(cached["prompt_length"]),
            target_mask=cached.get("target_mask"),
            reduction=reduction,
        )
        scores.append(float(result["score"]))
        mean_logprobs.append(float(result["mean_logprob"]))
        sum_logprobs.append(float(result["sum_logprob"]))
        total_target_tokens += int(result["target_token_count"])

        if save_responses:
            responses.append(
                {
                    "qid": qid,
                    "score": float(result["score"]),
                    "mean_logprob": float(result["mean_logprob"]),
                    "sum_logprob": float(result["sum_logprob"]),
                    "target_token_count": int(result["target_token_count"]),
                    "scored_token_count": int(result["scored_token_count"]),
                    "target_text": cached["target_text"],
                }
            )

    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_mean_logprob = sum(mean_logprobs) / len(mean_logprobs) if mean_logprobs else 0.0
    total_sum_logprob = sum(sum_logprobs)

    result = {
        "score": float(avg_score),
        "mean_logprob": float(avg_mean_logprob),
        "sum_logprob": float(total_sum_logprob),
        "target_token_count": int(total_target_tokens),
        "scored_token_count": int(sum(int(r["scored_token_count"]) for r in responses)) if save_responses else int(total_target_tokens),
        "scoring_mode": "teacher_forced_proxy",
        "reduction": reduction,
    }
    if save_responses:
        result["responses"] = responses
    return result


def run_math_test_batched_moe(
    model,
    tokenized_dataset,
    tokenizer,
    batch_size=1,
    max_new_tokens=256,
    save_responses=True,
    padding_mode: str = PADDING_MODE_MASKED,
    prompt_pad_id: int | None = None,
):
    """
    Run math test with configurable batch size.

    Returns:
        If save_responses=False: float (average score)
        If save_responses=True: dict with 'score' and 'responses'
    """
    scores = []
    responses = [] if save_responses else None
    questions = list(tokenized_dataset.items())
    default_pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for i in range(0, len(questions), batch_size):
        batch_items = questions[i:i+batch_size]

        if batch_size == 1:
            qid, cached = batch_items[0]
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=cached['input_ids'],
                    attention_mask=cached['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=default_pad_id,
                )

            input_length = cached['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Strip thinking blocks, fallback to raw if empty
            stripped_text = strip_thinking(raw_text)
            integers = extract_integers(stripped_text)
            if len(integers) == 0:
                integers = extract_integers(raw_text)

            answer = cached['answer']

            if len(integers) == 0:
                question_score = 0
            else:
                integer_scores = [calculate_score(answer, i) for i in integers]
                question_score = max(integer_scores)

            scores.append(question_score)

            if save_responses:
                responses.append({
                    'qid': qid,
                    'raw_output': raw_text,
                    'extracted': integers,
                    'reference': answer,
                    'score': question_score
                })

        else:
            # Batched processing with padding
            batch_input_ids = []
            batch_attention_masks = []
            batch_answers = []
            batch_input_lengths = []
            batch_qids = []

            max_len = max(item[1]['input_ids'].shape[1] for item in batch_items)
            pad_id = int(
                prompt_pad_id
                if (padding_mode == PADDING_MODE_INPROMPT_SPACE and prompt_pad_id is not None)
                else default_pad_id
            )

            for qid, cached in batch_items:
                input_ids = cached['input_ids']
                attention_mask = cached['attention_mask']

                pad_length = max_len - input_ids.shape[1]
                if pad_length > 0:
                    input_ids = torch.cat([
                        torch.full(
                            (1, pad_length),
                            pad_id,
                            device=input_ids.device,
                            dtype=input_ids.dtype,
                        ),
                        input_ids
                    ], dim=1)
                    if padding_mode == PADDING_MODE_INPROMPT_SPACE:
                        left_mask = torch.ones(
                            (1, pad_length),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        )
                    else:
                        left_mask = torch.zeros(
                            (1, pad_length),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        )
                    attention_mask = torch.cat([
                        left_mask,
                        attention_mask
                    ], dim=1)

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)
                batch_answers.append(cached['answer'])
                batch_input_lengths.append(max_len)
                batch_qids.append(qid)

            batch_input_ids = torch.cat(batch_input_ids, dim=0)
            batch_attention_masks = torch.cat(batch_attention_masks, dim=0)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_masks,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )

            for j, (output, input_length, answer, qid) in enumerate(zip(outputs, batch_input_lengths, batch_answers, batch_qids)):
                generated_ids = output[input_length:]
                raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Strip thinking blocks, fallback to raw if empty
                stripped_text = strip_thinking(raw_text)
                integers = extract_integers(stripped_text)
                if len(integers) == 0:
                    integers = extract_integers(raw_text)

                if len(integers) == 0:
                    question_score = 0
                else:
                    integer_scores = [calculate_score(answer, i) for i in integers]
                    question_score = max(integer_scores)

                scores.append(question_score)

                if save_responses:
                    responses.append({
                        'qid': qid,
                        'raw_output': raw_text,
                        'extracted': integers,
                        'reference': answer,
                        'score': question_score
                    })

    avg_score = sum(scores) / len(scores) if scores else 0

    if save_responses:
        return {'score': avg_score, 'responses': responses}
    return avg_score


def run_math_preflight(
    model,
    tokenized_dataset: dict,
    tokenizer,
    *,
    samples: int,
    batch_size: int,
    max_new_tokens: int,
    padding_mode: str,
    prompt_pad_id: int,
    min_extract_rate: float,
) -> dict[str, float]:
    """Quick baseline probe to catch totally broken generations before long sweeps."""
    probe_items = list(tokenized_dataset.items())[:samples]
    if not probe_items:
        raise RuntimeError("Preflight requested but tokenized dataset is empty.")

    probe_dataset = dict(probe_items)
    probe_batch = max(1, min(batch_size, len(probe_dataset)))
    probe_result = run_math_test_batched_moe(
        model,
        probe_dataset,
        tokenizer,
        batch_size=probe_batch,
        max_new_tokens=max_new_tokens,
        save_responses=True,
        padding_mode=padding_mode,
        prompt_pad_id=prompt_pad_id,
    )

    responses = probe_result.get("responses", [])
    extracted_count = sum(1 for row in responses if row.get("extracted"))
    extract_rate = extracted_count / len(responses) if responses else 0.0

    if extracted_count == 0:
        raise RuntimeError(
            "Math preflight failed: zero parseable integers extracted across probe samples. "
            "This usually indicates an incompatible checkpoint loader/runtime."
        )
    if extract_rate < min_extract_rate:
        raise RuntimeError(
            "Math preflight failed: parseable integer extraction rate "
            f"{extract_rate:.2%} is below required threshold {min_extract_rate:.2%}."
        )

    return {
        "samples": float(len(responses)),
        "score": float(probe_result.get("score", 0.0)),
        "extract_rate": float(extract_rate),
    }


def main():
    parser = argparse.ArgumentParser(description="Math benchmark worker for shared relayer queues")
    parser.add_argument("--queue-file", type=str, default="work_queue.json",
                        help="Path to shared queue file (canonical layers entries or legacy key entries)")
    parser.add_argument("--results-file", type=str, default="results/math_results.pkl",
                        help="Path to shared results file")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to Hugging Face model directory")
    parser.add_argument("--dataset-path", type=str, default="datasets/math_16.json",
                        help="Path to math dataset")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--max-new", type=int, default=256,
                        help="max_new_tokens for math generation")
    parser.add_argument("--padding-mode", type=str, choices=[PADDING_MODE_MASKED, PADDING_MODE_INPROMPT_SPACE],
                        default=PADDING_MODE_MASKED,
                        help="Padding behavior for batched generation")
    parser.add_argument("--prompt-pad-id", type=int, default=None,
                        help="Token id used for inprompt-space left padding")
    parser.add_argument("--use-no-think-prefix", action=argparse.BooleanOptionalAction, default=True,
                        help="Prefix math prompts with /no_think")
    parser.add_argument("--adaptive-batch-retry", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-retry with smaller batch size on OOM/context failures")
    parser.add_argument("--min-batch-size", type=int, default=1,
                        help="Smallest batch size allowed during adaptive retries")
    parser.add_argument("--max-retries-per-phase", type=int, default=8,
                        help="Maximum adaptive retries for math inference")
    parser.add_argument("--strategic", action="store_true",
                        help="Use strategic layer config subset")
    parser.add_argument("--attention-impl", type=str, default="eager",
                        choices=["eager", "flash_attention_2", "sdpa"],
                        help="Attention implementation")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to enable HuggingFace trust_remote_code when loading model/tokenizer")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False,
                        help="Load model/tokenizer from local files only (no Hub fetch)")
    parser.add_argument("--device-map", type=str, default="cuda:0",
                        help="HF device_map value: e.g. 'cuda:0', 'auto', or JSON object string")
    parser.add_argument("--max-memory-json", type=str, default=None,
                        help="Optional max_memory JSON, e.g. '{\"cuda:0\":\"80GiB\",\"cuda:1\":\"80GiB\"}'")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable HF state-dict offload during loading (useful with device_map auto)")
    parser.add_argument("--offload-folder", type=str, default=None,
                        help="Optional folder for HF offload files when --cpu-offload is enabled")
    parser.add_argument("--worker-id", type=str, default=None,
                        help="Worker ID for logging (auto-detect from CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip baseline generation sanity check before queue/custom run")
    parser.add_argument("--preflight-samples", type=int, default=8,
                        help="Number of questions to sample for math preflight")
    parser.add_argument("--preflight-max-new", type=int, default=128,
                        help="max_new_tokens for math preflight probe")
    parser.add_argument("--preflight-min-extract-rate", type=float, default=0.5,
                        help="Minimum parseable-integer extraction rate in preflight (0..1)")
    parser.add_argument("--proxy-reduction", type=str, choices=["mean", "sum"], default="mean",
                        help="Reduction used for teacher-forced proxy scoring")

    # Direct layer specification (single-config mode, bypasses queue)
    parser.add_argument("--layer-list", type=str, default=None,
                        help="Direct layer list, e.g., '0,1,2,3,2,3,4,5'. Bypasses queue.")
    parser.add_argument("--blocks", type=str, default=None,
                        help="Multi-block spec, e.g., '3,6;4,6'. Bypasses queue.")
    parser.add_argument("--config-file", type=str, default=None,
                        help="File with configs (one per line). Lines starting with 'layers:' are layer lists, "
                             "otherwise treated as block specs. Bypasses queue.")

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_new < 1:
        raise ValueError("--max-new must be >= 1")
    if args.min_batch_size < 1:
        raise ValueError("--min-batch-size must be >= 1")
    if args.max_retries_per_phase < 0:
        raise ValueError("--max-retries-per-phase must be >= 0")
    if args.preflight_samples < 1:
        raise ValueError("--preflight-samples must be >= 1")
    if args.preflight_max_new < 1:
        raise ValueError("--preflight-max-new must be >= 1")
    if not (0.0 <= args.preflight_min_extract_rate <= 1.0):
        raise ValueError("--preflight-min-extract-rate must be in [0, 1]")

    try:
        resolved_device_map = parse_device_map_arg(args.device_map)
    except Exception as exc:
        raise ValueError(f"Invalid --device-map value: {exc}") from exc
    try:
        resolved_max_memory = parse_max_memory_json(args.max_memory_json)
    except Exception as exc:
        raise ValueError(f"Invalid --max-memory-json value: {exc}") from exc

    # Auto-detect worker ID from GPU
    if args.worker_id is None:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        args.worker_id = f"GPU{cuda_visible}"

    print("=" * 80)
    print(f"MoE Worker [{args.worker_id}]")
    print("=" * 80)
    print(f"Queue file: {args.queue_file}")
    print(f"Results file: {args.results_file}")
    print(f"Model: {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Math max_new: {args.max_new}")
    print(f"Teacher-forced proxy: enabled (reduction={args.proxy_reduction})")
    print(f"Padding mode: {args.padding_mode}")
    print(f"Adaptive retry: {args.adaptive_batch_retry} (min={args.min_batch_size}, max_retries={args.max_retries_per_phase})")
    print(
        "Preflight: "
        + (
            "disabled"
            if args.skip_preflight
            else (
                "enabled "
                f"(samples={args.preflight_samples}, max_new={args.preflight_max_new}, "
                f"min_extract_rate={args.preflight_min_extract_rate:.2f})"
            )
        )
    )
    print(f"Trust remote: {args.trust_remote_code}")
    print(f"Local files only: {args.local_files_only}")
    print(f"Device map: {resolved_device_map}")
    if resolved_max_memory is not None:
        print(f"Max memory: {resolved_max_memory}")
    print(f"CPU offload: {args.cpu_offload} (folder={args.offload_folder})")

    # Initialize work queue
    queue = SharedWorkQueue(args.queue_file, args.results_file)

    # Skip queue check if in custom config mode
    if not (args.layer_list or args.blocks or args.config_file):
        remaining, completed = queue.get_queue_status()
        print(f"\nQueue status: {remaining} remaining, {completed} completed")

        if remaining == 0:
            print("Queue is empty, nothing to do!")
            return

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} math questions")

    # Load model
    print(f"\nLoading model from {args.model_path}")
    attn_impl = args.attention_impl if args.attention_impl != "eager" else None
    tokenizer, model, load_meta = load_model_and_tokenizer(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        device_map=resolved_device_map,
        attn_implementation=attn_impl,
        max_memory=resolved_max_memory,
        cpu_offload=args.cpu_offload,
        offload_folder=args.offload_folder,
    )
    num_layers = int(load_meta["num_layers"])
    print(f"Loader: {load_meta['loader']} | architectures={load_meta['architectures']} | text_stack={load_meta['text_stack']}")
    print(f"Resolved HF device map: {load_meta.get('hf_device_map')}")
    print(f"Model has {num_layers} text layers")

    # Detect model type (MoE vs dense)
    model_is_moe = is_moe_model(model)
    model_type = "MoE" if model_is_moe else "Dense"
    print(f"Model type: {model_type}")

    # Select appropriate layer duplication function
    if model_is_moe:
        build_duplicated_model = build_model_with_layers_moe
    else:
        build_duplicated_model = build_model_with_layers

    teacher_forced_dataset = pretokenize_teacher_forced_dataset(
        dataset,
        tokenizer,
        model.device,
        use_no_think_prefix=args.use_no_think_prefix,
    )

    if not args.skip_preflight:
        print("Preflight skipped in proxy branch.")

    def run_math_proxy(run_model):
        """Run teacher-forced proxy scoring for math."""
        result = run_math_teacher_forced_proxy(
            run_model,
            teacher_forced_dataset,
            reduction=args.proxy_reduction,
            save_responses=True,
        )
        return result, 1, 0

    # Single-config mode: direct layer list or blocks specification
    if args.layer_list or args.blocks or args.config_file:
        print(f"\n{'='*80}")
        print("Custom config mode (bypassing queue)")
        print(f"{'='*80}")

        # Build list of configs to process
        configs_to_run = []

        if args.config_file:
            # Read configs from file
            print(f"Loading configs from {args.config_file}")
            with open(args.config_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue  # Skip empty lines and comments

                    if line.startswith('layers:'):
                        # Explicit layer list: "layers:0,1,2,3,2,3,4,5"
                        layer_str = line[7:].strip()
                        layer_indices = parse_layer_list_string(layer_str)
                        config_name = f"layers:{layer_str}"
                    else:
                        # Block spec: "3,6" or "3,6;4,6"
                        blocks = parse_blocks_string(line)
                        layer_indices = expand_multi_block_config(num_layers, blocks)
                        config_name = f"blocks:{line}"

                    configs_to_run.append((config_name, layer_indices))
            print(f"Loaded {len(configs_to_run)} configs from file")

        elif args.layer_list:
            layer_indices = parse_layer_list_string(args.layer_list)
            configs_to_run.append((f"layer_list:{args.layer_list}", layer_indices))

        else:  # args.blocks
            blocks = parse_blocks_string(args.blocks)
            layer_indices = expand_multi_block_config(num_layers, blocks)
            configs_to_run.append((f"blocks:{args.blocks}", layer_indices))

        # Validate all configs
        for config_name, layer_indices in configs_to_run:
            for idx in layer_indices:
                if idx < 0 or idx >= num_layers:
                    print(f"ERROR: Config '{config_name}' has layer index {idx} out of range [0, {num_layers})")
                    return

        # Prepare results storage
        results_path = None
        all_results = {}
        if args.results_file:
            results_path = Path(args.results_file)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            if results_path.exists():
                with open(results_path, 'rb') as f:
                    all_results = pickle.load(f)

        # Process each config
        total_start = time.time()
        for i, (config_name, layer_indices) in enumerate(configs_to_run):
            print(f"\n[{i+1}/{len(configs_to_run)}] Config: {config_name}")
            print(f"Layer indices ({len(layer_indices)} layers): {layer_indices[:20]}{'...' if len(layer_indices) > 20 else ''}")
            key = tuple(layer_indices)
            if key in all_results:
                existing = all_results[key]
                existing_score = existing.get("score") if isinstance(existing, dict) else None
                if isinstance(existing_score, (int, float)):
                    print(f"Skipping (already in results): score={existing_score:.4f}")
                else:
                    print("Skipping (already in results).")
                continue

            config_start = time.time()

            if layer_indices == list(range(num_layers)):
                print("Running baseline (original model)...")
                result, effective_batch, retries = run_math_proxy(model)
            else:
                print("Building duplicated model...")
                dup_model = build_duplicated_model(model, layer_indices)
                result, effective_batch, retries = run_math_proxy(dup_model)
                del dup_model
                torch.cuda.empty_cache()

            elapsed = time.time() - config_start
            score = result['score']
            print(
                f"Result: score={score:.4f} ({elapsed:.1f}s, "
                f"mean_logprob={result['mean_logprob']:.4f}, target_tokens={result['target_token_count']})"
            )

            # Save result incrementally
            all_results[key] = result
            if results_path:
                with open(results_path, 'wb') as f:
                    pickle.dump(all_results, f)

        total_elapsed = time.time() - total_start
        print(f"\n{'='*80}")
        print(f"Completed {len(configs_to_run)} configs in {total_elapsed:.1f}s")
        if results_path:
            print(f"Results saved to {results_path}")
        print(f"{'='*80}")

        return

    print("Queue mode: canonical layer-list configs (legacy (i,j) keys auto-converted).")

    # Worker loop
    print(f"\n{'='*80}")
    print("Starting worker loop...")
    print(f"{'='*80}\n")

    configs_processed = 0
    start_time = time.time()

    while True:
        # Get next config from queue
        entry = queue.get_next_config()

        if entry is None:
            print(f"\n[{args.worker_id}] Queue empty, worker finished!")
            break

        try:
            parsed_entry = parse_queue_entry_layers(num_layers, entry)
        except Exception as exc:
            print(f"[{args.worker_id}] WARNING: Invalid queue entry {entry!r}: {exc}")
            continue
        config_idx = parsed_entry["idx"]
        layer_indices = parsed_entry["layers"]
        key = parsed_entry["layer_key"]
        config_spec = parsed_entry["spec"]

        # Run experiment
        config_start_time = time.time()

        if is_baseline_layers(layer_indices, num_layers):
            # Baseline - use original model
            result, effective_batch, retries = run_math_proxy(model)
        else:
            # Build duplicated model (auto-selects MoE or dense)
            dup_model = build_duplicated_model(model, layer_indices)
            result, effective_batch, retries = run_math_proxy(dup_model)

            # Explicit memory cleanup
            del dup_model
            torch.cuda.empty_cache()

        config_time = time.time() - config_start_time
        configs_processed += 1

        # Extract score for logging
        score = result['score']

        # Save result (with responses)
        queue.save_result(key, result)

        # Progress report
        remaining, completed = queue.get_queue_status()
        elapsed = time.time() - start_time
        rate = configs_processed / elapsed if elapsed > 0 else 0

        if remaining > 0 and rate > 0:
            eta = remaining / rate
            eta_str = format_eta(eta)
        else:
            eta_str = "N/A"

        print(f"[{args.worker_id}] Config {config_idx} {config_spec} ({layer_spec_string(layer_indices)}): score={score:.4f} "
              f"({config_time:.1f}s, mean_logprob={result['mean_logprob']:.4f}, "
              f"target_tokens={result['target_token_count']}) | "
              f"Queue: {remaining} left | Rate: {rate:.2f}/s | ETA: {eta_str}")

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Worker [{args.worker_id}] Summary")
    print(f"{'='*80}")
    print(f"Configs processed: {configs_processed}")
    print(f"Total time: {format_eta(total_time)}")
    print(f"Average rate: {configs_processed/total_time:.2f} configs/s" if total_time > 0 else "N/A")


if __name__ == "__main__":
    main()
