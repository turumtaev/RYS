from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from scripts.beam_search import (
    BeamCandidate,
    _dataset_chunks,
    _load_dataset,
    _write_exact_validation,
    complete_layer_path,
    expand_beam,
    expand_candidate,
    score_candidates,
)
from src.core.layer_duplicator import build_model_with_layers
from src.workers.eq_worker import build_eq_first_pass_target, serialize_eq_first_pass_target
from src.workers.math_worker import build_math_target, serialize_math_target
from src.workers.model_utils import (
    LlamaLikePartialRunner,
    gather_target_token_logprobs,
    reduce_logprobs,
    score_teacher_forced_inputs,
    score_teacher_forced_logits,
)
from src.utils.surrogate_utils import (
    count_vector_to_layers,
    counts_from_csv,
    counts_to_csv,
    key_to_count_vector,
    relative_overhead_from_counts,
    stable_quantile_bins,
)


class SurrogateUtilsTests(unittest.TestCase):
    @staticmethod
    def _tiny_llama() -> LlamaForCausalLM:
        torch.manual_seed(7)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
        )
        return LlamaForCausalLM(config).eval()

    @staticmethod
    def _tiny_qwen2() -> Qwen2ForCausalLM:
        torch.manual_seed(11)
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            use_sliding_window=True,
            sliding_window=2,
            layer_types=["full_attention", "sliding_attention", "full_attention"],
            pad_token_id=0,
        )
        return Qwen2ForCausalLM(config).eval()

    def test_key_to_count_vector_roundtrip(self):
        key = (0, 1, 2, 1, 2, 3)
        counts = key_to_count_vector(key, num_layers=4)
        self.assertEqual(counts, [1, 2, 2, 1])
        decoded = count_vector_to_layers(counts, num_layers=4)
        self.assertEqual(decoded, [0, 1, 1, 2, 2, 3])

    def test_counts_csv_parse(self):
        counts = [1, 2, 3, 1]
        raw = counts_to_csv(counts)
        self.assertEqual(raw, "1,2,3,1")
        self.assertEqual(counts_from_csv(raw, expected_len=4), counts)

    def test_relative_overhead(self):
        counts = [1, 2, 2, 1]
        overhead = relative_overhead_from_counts(counts, num_layers=4)
        self.assertAlmostEqual(overhead, 0.5)

    def test_stable_quantile_bins(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        bins = stable_quantile_bins(values, bins=3)
        self.assertEqual(len(bins), 5)
        self.assertTrue(all(0 <= b < 3 for b in bins))

    def test_serialize_math_target_uses_integer_string(self):
        self.assertEqual(serialize_math_target(25774), "25774")
        self.assertEqual(serialize_math_target("00042"), "42")
        self.assertEqual(build_math_target("00042"), ("42", [(0, 2)]))

    def test_serialize_eq_first_pass_target_uses_fullscale_integers(self):
        reference = {
            "emotion1": "Intimidated",
            "emotion2": "Hopeful",
            "emotion3": "Disheartened",
            "emotion4": "Smug",
            "emotion1_score": 3,
            "emotion2_score": 4,
            "emotion3_score": 6,
            "emotion4_score": 0,
        }
        expected = "\n".join(
            [
                "First pass scores:",
                "Intimidated: 3",
                "Hopeful: 4",
                "Disheartened: 6",
                "Smug: 0",
            ]
        )
        self.assertEqual(serialize_eq_first_pass_target(reference), expected)
        target_text, spans = build_eq_first_pass_target(reference)
        self.assertEqual(target_text, expected)
        self.assertEqual(
            [target_text[start:end] for start, end in spans],
            ["3", "4", "6", "0"],
        )

    def test_gather_target_token_logprobs_uses_shifted_logits(self):
        input_ids = torch.tensor([[10, 11, 2, 3]])
        logits = torch.full((1, 4, 5), -10.0)
        logits[0, 1, 2] = 4.0
        logits[0, 2, 3] = 5.0

        token_logprobs = gather_target_token_logprobs(
            logits=logits,
            input_ids=input_ids,
            prompt_length=2,
        )

        expected_first = torch.log_softmax(logits[0, 1], dim=-1)[2].item()
        expected_second = torch.log_softmax(logits[0, 2], dim=-1)[3].item()
        self.assertEqual(token_logprobs.shape, (1, 2))
        self.assertTrue(math.isclose(token_logprobs[0, 0].item(), expected_first, rel_tol=1e-6))
        self.assertTrue(math.isclose(token_logprobs[0, 1].item(), expected_second, rel_tol=1e-6))
        self.assertTrue(
            math.isclose(
                reduce_logprobs(token_logprobs, reduction="sum"),
                expected_first + expected_second,
                rel_tol=1e-6,
            )
        )
        mask = torch.tensor([[False, True]])
        self.assertTrue(
            math.isclose(
                reduce_logprobs(token_logprobs, reduction="sum", mask=mask),
                expected_second,
                rel_tol=1e-6,
            )
        )

    def test_teacher_forced_precomputed_logits_match_model_scoring(self):
        model = self._tiny_llama()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones_like(input_ids)
        target_mask = torch.tensor([[True, False]])

        expected = score_teacher_forced_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_length=2,
            target_mask=target_mask,
        )
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        actual = score_teacher_forced_logits(
            logits=logits,
            input_ids=input_ids,
            prompt_length=2,
            target_mask=target_mask,
        )

        self.assertEqual(actual, expected)

    def test_suffix_beam_expands_plain_and_local_replay_paths(self):
        candidate = BeamCandidate(layer_path=(0, 1), replays=())
        children = expand_candidate(
            candidate,
            layer_idx=2,
            replay_window=2,
            num_layers=5,
        )

        self.assertEqual(
            [child.layer_path for child in children],
            [
                (0, 1, 2),
                (0, 1, 2, 1, 2),
                (0, 1, 2, 2),
            ],
        )
        self.assertEqual(
            [child.replays for child in children],
            [(), ((1, 3),), ((2, 3),)],
        )
        self.assertEqual(
            complete_layer_path(children[-1], layer_idx=2, num_layers=5),
            (0, 1, 2, 2, 3, 4),
        )

    def test_suffix_beam_respects_extra_layer_cap(self):
        candidate = BeamCandidate(layer_path=(0, 1, 0, 1), replays=((0, 2),))
        children = expand_beam(
            [candidate],
            layer_idx=2,
            replay_window=2,
            num_layers=4,
            max_extra_layers=4,
        )

        self.assertEqual(
            [child.layer_path for child in children],
            [
                (0, 1, 0, 1, 2),
                (0, 1, 0, 1, 2, 1, 2),
                (0, 1, 0, 1, 2, 2),
            ],
        )

    def test_exact_validation_records_proxy_and_exact_rank_orders(self):
        results = [
            {
                "label": "baseline",
                "proxy_score": -3.0,
                "score": 0.6,
                "math_responses": [{}, {}],
            },
            {
                "label": "beam_1",
                "proxy_score": -2.0,
                "score": 0.4,
                "math_responses": [{}, {}],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.json"
            output_path.write_text('{"status": "proxy_complete"}\n')
            _write_exact_validation(
                output_path,
                results,
                math_max_new=64,
                eq_max_new=384,
                batch_size=1,
                dataset_offset=2,
            )
            payload = json.loads(output_path.read_text())

        validation = payload["exact_validation"]
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(validation["proxy_order"], ["beam_1", "baseline"])
        self.assertEqual(validation["exact_order"], ["baseline", "beam_1"])
        self.assertFalse(validation["top_rank_agrees"])
        self.assertEqual(validation["math_max_new"], 64)
        self.assertEqual(validation["eq_max_new"], 384)
        self.assertEqual(validation["dataset_offset"], 2)
        self.assertEqual(validation["examples_per_benchmark"], 2)

    def test_load_dataset_applies_offset_before_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.json"
            dataset_path.write_text(json.dumps({"a": 1, "b": 2, "c": 3}) + "\n")
            selected = _load_dataset(str(dataset_path), 1, offset=1)

        self.assertEqual(selected, {"b": 2})

    def test_chunked_candidate_scoring_is_chunk_size_invariant(self):
        model = self._tiny_llama()
        runner = LlamaLikePartialRunner(model)
        dataset = {
            "a": {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "prompt_length": 2,
                "target_mask": torch.tensor([[True, True]]),
            },
            "b": {
                "input_ids": torch.tensor([[5, 6, 7, 8]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "prompt_length": 2,
                "target_mask": torch.tensor([[True, False]]),
            },
        }
        candidates = [
            BeamCandidate(layer_path=(0,), replays=()),
            BeamCandidate(layer_path=(0, 0), replays=((0, 1),)),
        ]

        chunked = score_candidates(
            runner,
            candidates,
            layer_idx=0,
            math_dataset=dataset,
            eq_dataset=dataset,
            reduction="mean",
            device=torch.device("cpu"),
            chunk_size=1,
        )
        unchunked = score_candidates(
            runner,
            candidates,
            layer_idx=0,
            math_dataset=dataset,
            eq_dataset=dataset,
            reduction="mean",
            device=torch.device("cpu"),
            chunk_size=2,
        )

        self.assertEqual(len(list(_dataset_chunks(dataset, 1))), 2)
        for chunked_candidate, unchunked_candidate in zip(chunked, unchunked):
            self.assertAlmostEqual(chunked_candidate.score, unchunked_candidate.score)
            self.assertAlmostEqual(chunked_candidate.math_score, unchunked_candidate.math_score)
            self.assertAlmostEqual(chunked_candidate.eq_score, unchunked_candidate.eq_score)

    def test_partial_runner_matches_full_forward_and_split_execution(self):
        model = self._tiny_llama()
        runner = LlamaLikePartialRunner(model)
        input_ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])

        with torch.no_grad():
            expected = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            actual = runner.forward_path(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            state = runner.prepare(input_ids=input_ids, attention_mask=attention_mask)
            state = runner.run_range(state, 0, 1)
            state = runner.run_range(state, 1, runner.num_layers)
            split_actual = runner.logits(state)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(split_actual, expected, rtol=1e-5, atol=1e-6)

    def test_partial_runner_repeated_path_matches_duplicated_model(self):
        model = self._tiny_llama()
        runner = LlamaLikePartialRunner(model)
        layer_path = [0, 0, 1, 2]
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones_like(input_ids)
        duplicated_model = build_model_with_layers(model, layer_path)

        with torch.no_grad():
            expected = duplicated_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            actual = runner.forward_path(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_indices=layer_path,
            )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_partial_runner_matches_qwen2_with_sliding_attention(self):
        model = self._tiny_qwen2()
        runner = LlamaLikePartialRunner(model)
        layer_path = [0, 1, 1, 2]
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)
        duplicated_model = build_model_with_layers(model, layer_path)

        with torch.no_grad():
            expected = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            actual = runner.forward_path(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            expected_repeat = duplicated_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            actual_repeat = runner.forward_path(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_indices=layer_path,
            )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_repeat, expected_repeat, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
