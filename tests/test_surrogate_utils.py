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
    _dataset_item_batches,
    _load_dataset,
    _write_exact_validation,
    boundary_cache_bytes,
    candidate_replay_budget,
    complete_layer_path,
    expand_beam,
    expand_beam_with_parents,
    expand_candidate,
    materialize_boundary_cache,
    score_cached_expansions,
    score_candidates,
    select_budget_indexed_beams,
)
from src.core.layer_duplicator import build_model_with_layers
from src.workers.eq_worker import build_eq_first_pass_target, serialize_eq_first_pass_target
from src.workers.math_worker import build_math_target, serialize_math_target
from src.workers.model_utils import (
    LlamaLikePartialRunner,
    build_teacher_forced_inputs,
    expand_mask_islands_right,
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

    def test_expand_mask_islands_scores_each_following_token(self):
        mask = torch.tensor(
            [[False, True, True, True, False, False, True, True, False]]
        )

        expanded = expand_mask_islands_right(mask)

        self.assertEqual(
            expanded.tolist(),
            [[False, True, True, True, True, False, True, True, True]],
        )
        self.assertEqual(mask.tolist(), [[False, True, True, True, False, False, True, True, False]])

    def test_teacher_forced_inputs_append_and_score_answer_terminators(self):
        class CharacterTokenizer:
            eos_token_id = 99

            def __call__(
                self,
                text,
                *,
                return_tensors,
                add_special_tokens,
                return_offsets_mapping=False,
            ):
                del return_tensors, add_special_tokens
                result = {
                    "input_ids": torch.tensor([[ord(char) % 90 for char in text]]),
                    "attention_mask": torch.ones((1, len(text)), dtype=torch.long),
                }
                if return_offsets_mapping:
                    result["offset_mapping"] = torch.tensor(
                        [[[idx, idx + 1] for idx in range(len(text))]]
                    )
                return result

        inputs = build_teacher_forced_inputs(
            CharacterTokenizer(),
            prompt_text="ab",
            target_text="12\nx3",
            target_char_spans=[(0, 2), (4, 5)],
        )

        self.assertEqual(inputs.input_ids[0, -1].item(), 99)
        self.assertEqual(inputs.target_length, 6)
        self.assertEqual(
            inputs.target_mask.tolist(),
            [[True, True, True, False, True, True]],
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

    def test_budget_indexed_beam_retains_each_exact_budget(self):
        candidates = [
            BeamCandidate((0, 1, 2), (), score=-3.0),
            BeamCandidate((0, 1, 2, 2), ((2, 3),), score=-2.0),
            BeamCandidate((0, 1, 1, 2), ((1, 2),), score=-1.5),
            BeamCandidate((0, 1, 2, 1, 2), ((1, 3),), score=-1.0),
        ]

        beams = select_budget_indexed_beams(
            candidates,
            layer_idx=2,
            beam_width=1,
            max_extra_layers=2,
        )

        self.assertEqual(list(beams), [0, 1, 2])
        self.assertEqual(beams[0][0].layer_path, (0, 1, 2))
        self.assertEqual(beams[1][0].layer_path, (0, 1, 1, 2))
        self.assertEqual(beams[2][0].layer_path, (0, 1, 2, 1, 2))

    def test_budgeted_expansion_transitions_from_previous_boundary(self):
        parents = [
            BeamCandidate((0, 1), (), score=-2.0),
            BeamCandidate((0, 1, 1), ((1, 2),), score=-1.0),
        ]
        expansions = expand_beam_with_parents(
            parents,
            layer_idx=2,
            replay_window=2,
            num_layers=4,
            max_extra_layers=2,
        )

        budgets = {
            candidate_replay_budget(expansion.child, layer_idx=2)
            for expansion in expansions
        }
        self.assertEqual(budgets, {0, 1, 2})
        self.assertTrue(
            any(
                expansion.parent_path == (0, 1)
                and expansion.replay_start == 1
                and candidate_replay_budget(expansion.child, layer_idx=2) == 2
                for expansion in expansions
            )
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
                "input_ids": torch.tensor([[5, 6, 7, 8, 9]]),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
                "prompt_length": 2,
                "target_mask": torch.tensor([[True, False, True]]),
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
        batches = list(_dataset_item_batches(dataset, 2))
        self.assertEqual(
            [cached["input_ids"].shape[1] for _, cached in batches[0]],
            [4, 5],
        )
        for chunked_candidate, unchunked_candidate in zip(chunked, unchunked):
            self.assertAlmostEqual(chunked_candidate.score, unchunked_candidate.score)
            self.assertAlmostEqual(chunked_candidate.math_score, unchunked_candidate.math_score)
            self.assertAlmostEqual(chunked_candidate.eq_score, unchunked_candidate.eq_score)

    def test_cached_boundary_search_matches_full_prefix_recomputation(self):
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
                "input_ids": torch.tensor([[5, 6, 7, 8, 9]]),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
                "prompt_length": 2,
                "target_mask": torch.tensor([[True, False, True]]),
            },
        }
        beam = [BeamCandidate(layer_path=(), replays=())]
        parent_cache = {}

        for layer_idx in range(2):
            expansions = expand_beam_with_parents(
                beam,
                layer_idx=layer_idx,
                replay_window=2,
                num_layers=runner.num_layers,
                max_extra_layers=2,
            )
            cached = score_cached_expansions(
                runner,
                expansions,
                parent_cache=parent_cache,
                layer_idx=layer_idx,
                math_dataset=dataset,
                eq_dataset=dataset,
                reduction="mean",
                device=torch.device("cpu"),
                batch_size=2,
                pad_token_id=0,
            )
            recomputed = score_candidates(
                runner,
                [expansion.child for expansion in expansions],
                layer_idx=layer_idx,
                math_dataset=dataset,
                eq_dataset=dataset,
                reduction="mean",
                device=torch.device("cpu"),
                chunk_size=1,
            )

            cached_by_path = {candidate.layer_path: candidate for candidate in cached}
            for candidate in recomputed:
                actual = cached_by_path[candidate.layer_path]
                self.assertTrue(math.isclose(actual.score, candidate.score, abs_tol=1e-6))
                self.assertTrue(math.isclose(actual.math_score, candidate.math_score, abs_tol=1e-6))
                self.assertTrue(math.isclose(actual.eq_score, candidate.eq_score, abs_tol=1e-6))

            cached.sort(key=lambda candidate: float(candidate.score), reverse=True)
            beam = cached[:2]
            if layer_idx == 0:
                expansion_by_path = {
                    expansion.child.layer_path: expansion for expansion in expansions
                }
                parent_cache = materialize_boundary_cache(
                    runner,
                    [expansion_by_path[candidate.layer_path] for candidate in beam],
                    parent_cache=parent_cache,
                    layer_idx=layer_idx,
                    math_dataset=dataset,
                    eq_dataset=dataset,
                    device=torch.device("cpu"),
                    batch_size=2,
                    pad_token_id=0,
                )
                self.assertGreater(boundary_cache_bytes(parent_cache), 0)
                self.assertTrue(all(len(states) == 4 for states in parent_cache.values()))

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
