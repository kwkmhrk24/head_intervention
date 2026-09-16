import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.evaluate_gcg_rank_distribution_shift_cancellation import (
    DATASET_GROUPS,
    INTERVENTION_TIMING,
    load_holdout_development_records,
    parse_args,
    run_gpu,
)
from scripts.evaluate_gcg_rank_distribution_shift_cancellation_independent100 import (
    EXPECTED_INDICES,
    parse_args as parse_independent_args,
    validate_independent_identity,
    verify_baseline_reproduction,
)
from scripts.prepare_gcg_rank_distribution_shift_cancellation import (
    parse_args as parse_prepare_args,
)
from utils.gcg_rank_distribution_shift_cancellation import (
    ALL_RANK_COUNT,
    ALPHA_DEFAULT,
    METHOD,
    RankDistributionShiftCancellationController,
    compute_mean_shift,
    offline_shift_crossing_diagnostics,
    validate_shift_payload,
)
from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    SELECTION_COUNT,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    strict_lower_indices,
    write_json,
    write_jsonl,
)
from utils.gcg_first_token_holdout import (
    EVALUATION_ROLE,
    EXCLUDED_AMBIGUOUS_INDICES,
    HOLDOUT_INDICES,
    _records_sha256,
)


SCORE_SHAPE = (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
SHIFT_SHAPE = (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)


def make_scores(values):
    """Place one unsorted head's strict-lower values in raw S with S_sym=0.25."""

    rows, cols = strict_lower_indices(16)
    scores = torch.zeros((1, NUM_HEADS, 20, 20), dtype=torch.float32)
    scores[0, 0, -16 + rows, -16 + cols] = values + 0.25
    scores[0, 0, -16 + cols, -16 + rows] = -values + 0.25
    return scores


class ShiftConstructionTests(unittest.TestCase):
    def test_independent_signed_sort_and_paired_mean_delta(self):
        original = torch.zeros(SCORE_SHAPE, dtype=torch.float32)
        gcg = torch.zeros_like(original)
        original[0, 0, 0, :3] = torch.tensor([3.0, -2.0, 1.0])
        gcg[0, 0, 0, :3] = torch.tensor([2.0, 4.0, -3.0])
        original[1, 0, 0, :3] = torch.tensor([5.0, -1.0, 2.0])
        gcg[1, 0, 0, :3] = torch.tensor([3.0, 6.0, -4.0])
        # The source positions differ, and GCG must be sorted separately.
        expected = (
            torch.sort(gcg, dim=-1, stable=True).values
            - torch.sort(original, dim=-1, stable=True).values
        ).mean(0)
        shift = compute_mean_shift(original, gcg)
        torch.testing.assert_close(shift, expected, rtol=0, atol=0)
        self.assertEqual(tuple(shift.shape), SHIFT_SHAPE)
        self.assertAlmostEqual(shift[0, 0, 0].item(), -0.2, places=6)
        self.assertAlmostEqual(shift[0, 0, -1].item(), 0.1, places=6)

    def test_artifact_requires_method_hashes_and_conditions(self):
        metadata = {
            "source_selection_manifest_sha256": "a" * 64,
            "source_score_artifact_sha256": "b" * 64,
            "original_score_sha256": "c" * 64,
            "jailbreak_score_sha256": "d" * 64,
            "selection_prompt_count": SELECTION_COUNT,
            "rank_sort": "signed_value_ascending",
            "rank_tie_break": "stable strict-lower row-major position order",
            "alpha_default": ALPHA_DEFAULT,
            "scientific_conditions": {"selection_data": "Selection20"},
        }
        payload = {"schema_version": 1, "method": METHOD,
                   "mean_shift": torch.zeros(SHIFT_SHAPE), "metadata": metadata}
        validate_shift_payload(payload)
        invalid = dict(payload)
        invalid["metadata"] = dict(metadata, selection_prompt_count=48)
        with self.assertRaisesRegex(ValueError, "Selection20"):
            validate_shift_payload(invalid)

    def test_development_entrypoints_reject_independent100(self):
        args = parse_args(["--shift-artifact-dir", "/tmp/shift"])
        self.assertEqual(args.stage, "cpu-validate")
        self.assertEqual(args.alpha, 1.0)
        self.assertEqual(tuple(args.dataset_groups), DATASET_GROUPS)
        self.assertEqual(DATASET_GROUPS, ("selection20", "holdout48"))
        self.assertEqual(
            tuple(parse_args(["--shift-artifact-dir", "/tmp/shift",
                              "--dataset-groups", "holdout48"]).dataset_groups),
            ("holdout48",),
        )
        self.assertEqual(INTERVENTION_TIMING, "first_token_only")
        self.assertEqual(ALL_RANK_COUNT, 122880)
        with self.assertRaises(SystemExit):
            parse_args(["--shift-artifact-dir", "/tmp/shift",
                        "--dataset-groups", "independent100"])
        with self.assertRaises(SystemExit):
            parse_prepare_args(["--output-dir", "/tmp/out",
                                "--dataset-groups", "independent100"])


class DevelopmentEvaluatorTests(unittest.TestCase):
    def test_holdout_uses_fixed_audited_manifest_without_advbench(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            (run_dir / "baseline").mkdir()
            records = []
            baseline = []
            raw_hash = "a" * 64
            for index in range(50):
                included = index not in EXCLUDED_AMBIGUOUS_INDICES
                for duplicate in range(2 if not included else 1):
                    original = f"Original {index}"
                    gcg = f"{original} GCG {duplicate}"
                    records.append({
                        "dataset_index": index,
                        "raw_file_position": len(records),
                        "raw_archive_index": index,
                        "original_prompt": original,
                        "GCG_prompt": gcg,
                        "original_prompt_sha256": sha256_text(original),
                        "GCG_prompt_sha256": sha256_text(gcg),
                        "raw_artifact_sha256": raw_hash,
                        "included": included,
                        "exclusion_reason": None if included else "ambiguous_duplicate",
                    })
                    if included:
                        baseline.append({
                            "prompt_id": f"llama:gcg-holdout:{index}",
                            "row_position": len(baseline),
                            "dataset_index": index,
                            "prompt": gcg,
                            "response": "response",
                            "response_token_ids": [1],
                            "response_token_count": 1,
                            "success": index % 2 == 0,
                            "generation_failed": False,
                        })
            manifest = {
                "evaluation_role": EVALUATION_ROLE,
                "fully_unseen_across_all_prior_research": False,
                "unused_for_current_gcg_head_selection": True,
                "unused_for_current_reference_construction": True,
                "unused_for_current_gcg_evaluation": True,
                "original_behaviors_previously_used_in_autodan_research": True,
                "raw_artifact_sha256": raw_hash,
                "advbench_sha256": "b" * 64,
                "records": records,
                "records_sha256": _records_sha256(records),
            }
            manifest_path = run_dir / "holdout_manifest.json"
            write_json(manifest_path, manifest)
            write_json(run_dir / "run_manifest.json", {
                "status": "complete",
                "holdout_manifest_sha256_fixed_before_baseline": sha256_file(manifest_path),
                "holdout_manifest_records_sha256": manifest["records_sha256"],
                "included_dataset_indices": list(HOLDOUT_INDICES),
                "excluded_dataset_indices": list(EXCLUDED_AMBIGUOUS_INDICES),
                "included_count": 48,
                "raw_artifact_sha256": raw_hash,
                "advbench_sha256": manifest["advbench_sha256"],
            })
            write_jsonl(run_dir / "baseline/prompt_results.jsonl", baseline)
            loaded, provenance = load_holdout_development_records(run_dir)
            self.assertEqual(len(loaded), 48)
            self.assertEqual(tuple(row["dataset_index"] for row in loaded), HOLDOUT_INDICES)
            self.assertFalse(provenance["advbench_or_raw_archive_opened"])
            self.assertEqual(loaded[0]["baseline_response"], "response")

    def test_same_frozen_shift_runs_both_groups_and_saves_separate_results(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "evaluation"
            args = parse_args([
                "--shift-artifact-dir", str(Path(temp) / "frozen-shift"),
                "--stage", "run", "--output-dir", str(output_dir),
            ])
            mean_shift = torch.zeros(SHIFT_SHAPE, dtype=torch.float32)
            groups = {
                "selection20": [{"dataset_index": 68, "input_ids": torch.tensor([[1, 2]])}],
                "holdout48": [{"dataset_index": 0, "input_ids": torch.tensor([[3, 4]])}],
            }
            seen = []

            def fake_baseline(records, **kwargs):
                return [{**row, "response": "response", "response_token_ids": [1],
                         "success": True, "first_logits": torch.zeros(2),
                         "generation_failed": False} for row in records]

            def fake_condition(records, baseline_by_index, *, selection, label, **kwargs):
                self.assertIs(selection, mean_shift)
                self.assertTrue(torch.equal(
                    records[0]["input_ids"], baseline_by_index[records[0]["dataset_index"]]["input_ids"]
                ))
                seen.append(label)
                return ([{
                    "dataset_group": label, "input_ids_match_baseline": True,
                    "first_token_logit_metrics": {"argmax_equal": True},
                    "diagnostics": {"selected_rank_intervention_count": ALL_RANK_COUNT},
                    "transition": "still_attack_success", "success": True,
                }], {
                    "dataset_group": label, "prompt_count": 1, "input_ids_match_count": 1,
                    "selected_rank_intervention_count": ALL_RANK_COUNT, "alpha": 1.0,
                    "attack_success_count": 1, "asr": 1.0,
                    "newly_defended_count": 0, "still_attack_success_count": 1,
                    "newly_worsened_count": 0, "still_defended_count": 0,
                    "response_changed_count": 0, "first_token_argmax_changed_count": 0,
                    "selected_value_crossing_count": 0,
                    "later_forward_intervention_count": 0,
                })

            with (
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.torch.cuda.is_available", return_value=True),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.torch.cuda.get_device_name", return_value="mock GPU"),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.load_validated_shift",
                      return_value=(mean_shift, {"source_selection_manifest_sha256": "s" * 64},
                                    {"mean_shift_sha256": "d" * 64}, {"verified": True}, None, None)),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.load_development_groups",
                      return_value=(groups, {"selection20": {}, "holdout48": {}})),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.seed_everything"),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.load_model_and_tokenizer",
                      return_value=(object(), object())),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.validate_loaded_model",
                      return_value={"validated": True}),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.format_records",
                      side_effect=lambda rows, tokenizer: rows),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.start_cuda_measurement",
                      return_value=0.0),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.finish_cuda_measurement",
                      return_value={}),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.run_baseline",
                      side_effect=fake_baseline),
                patch("scripts.evaluate_gcg_rank_distribution_shift_cancellation.run_condition",
                      side_effect=fake_condition),
            ):
                result = run_gpu(args)
            self.assertEqual(seen, list(DATASET_GROUPS))
            self.assertEqual(result["dataset_groups"], list(DATASET_GROUPS))
            manifest = read_json(output_dir / "run_manifest.json")
            self.assertTrue(manifest["holdout48_used"])
            self.assertFalse(manifest["holdout48_used_for_shift"])
            self.assertFalse(manifest["independent100_used"])
            self.assertEqual(manifest["shift_artifact_verification"]["mean_shift_sha256"], "d" * 64)
            for group_name in DATASET_GROUPS:
                baseline_summary = read_json(output_dir / f"{group_name}_baseline_summary.json")
                condition_summary = read_json(output_dir / f"{group_name}_summary.json")
                rows = read_jsonl(output_dir / f"{group_name}_prompt_results.jsonl")
                self.assertEqual(baseline_summary["attack_success_count"], 1)
                self.assertEqual(baseline_summary["asr"], 1.0)
                self.assertEqual(condition_summary["asr"], 1.0)
                self.assertTrue(condition_summary["runtime_invariants_passed"])
                self.assertEqual(rows[0]["dataset_group"], group_name)


class Independent100EvaluatorTests(unittest.TestCase):
    @staticmethod
    def frozen_rows():
        records = []
        prior = []
        for index in EXPECTED_INDICES:
            prompt = f"behavior {index} suffix"
            formatted = f"<s>[INST] {prompt} [/INST]"
            ids = [1, index, 2]
            records.append({
                "dataset_index": index,
                "prompt": prompt,
                "formatted_prompt": formatted,
                "saved_formatted_prompt": formatted,
                "saved_non_target_input_ids": ids,
                "input_ids": torch.tensor([ids]),
            })
            prior.append({
                "dataset_index": index,
                "prompt": prompt,
                "formatted_prompt": formatted,
                "input_ids": ids,
                "response": "baseline",
                "response_token_ids": [42],
                "success": index < 273,
            })
        return records, prior

    def test_dedicated_entrypoint_is_fixed_and_development_cli_still_rejects(self):
        args = parse_independent_args(["--shift-artifact-dir", "/tmp/shift"])
        self.assertEqual(args.stage, "cpu-validate")
        self.assertFalse(hasattr(args, "alpha"))
        self.assertFalse(hasattr(args, "dataset_groups"))
        with self.assertRaises(SystemExit):
            parse_independent_args([
                "--shift-artifact-dir", "/tmp/shift", "--alpha", "0.5"
            ])
        with self.assertRaises(SystemExit):
            parse_args([
                "--shift-artifact-dir", "/tmp/shift",
                "--dataset-groups", "independent100",
            ])

    def test_fixed_position_inputs_match_and_overlap_is_zero(self):
        records, prior = self.frozen_rows()
        audit = validate_independent_identity(records, prior)
        self.assertEqual(audit["record_count"], 100)
        self.assertEqual(audit["position_input_ids_match_count"], 100)
        self.assertEqual(audit["selection20_overlap_count"], 0)
        self.assertEqual(audit["holdout48_overlap_count"], 0)
        records[5]["input_ids"][0, -1] = 99
        with self.assertRaisesRegex(ValueError, "input identity mismatch"):
            validate_independent_identity(records, prior)

    def test_baseline_53_gate_precedes_intervention(self):
        records, prior = self.frozen_rows()
        baseline = []
        for record, saved in zip(records, prior):
            baseline.append({
                **record,
                "response": saved["response"],
                "response_token_ids": saved["response_token_ids"],
                "success": saved["success"],
                "generation_failed": False,
            })
        verify_baseline_reproduction(baseline, prior)
        baseline[0]["response_token_ids"] = [999]
        with self.assertRaisesRegex(RuntimeError, "refusing final intervention"):
            verify_baseline_reproduction(baseline, prior)


class ShiftRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.cols = strict_lower_indices(16)
        self.physical_values = torch.arange(120, dtype=torch.float32).roll(17)
        self.scores = make_scores(self.physical_values)
        self.shift = torch.zeros(SHIFT_SHAPE, dtype=torch.float32)
        self.shift[0, 0, 1] = 2.0  # Q'(1)=-1, crossing Q'(0)=0.

    def controller(self, *, alpha=1.0):
        controller = RankDistributionShiftCancellationController(
            mean_shift=self.shift, alpha=alpha
        )
        controller.set_prompt_context(prompt_length=20)
        return controller

    def test_runtime_formula_scatter_and_no_resort_on_crossing(self):
        controller = self.controller()
        before = self.scores.clone()
        lower_before = ((before[0, 0, -16:, -16:]
                         - before[0, 0, -16:, -16:].T) / 2)[self.rows, self.cols]
        sorted_before, positions = torch.sort(lower_before, stable=True)
        result = controller.apply(0, before)
        lower_after = ((result[0, 0, -16:, -16:]
                        - result[0, 0, -16:, -16:].T) / 2)[self.rows, self.cols]
        expected = sorted_before - self.shift[0, 0]
        torch.testing.assert_close(lower_after[positions], expected, rtol=0, atol=0)
        self.assertEqual(lower_after[positions[0]].item(), 0.0)
        self.assertEqual(lower_after[positions[1]].item(), -1.0)
        self.assertGreater(controller.summary()["selected_value_crossing_count"], 0)
        self.assertGreater(controller.summary()["adjacent_rank_pair_crossing_count"], 0)
        self.assertEqual(controller.summary()["selected_rank_intervention_count"],
                         NUM_HEADS * ELEMENTS_PER_HEAD)

    def test_antisymmetric_raw_delta_symmetric_diagonal_and_outer_region(self):
        controller = self.controller(alpha=0.5)
        before = self.scores.clone()
        result = controller.apply(0, before)
        delta = result - before
        torch.testing.assert_close(delta + delta.transpose(-1, -2),
                                   torch.zeros_like(delta), rtol=0, atol=0)
        block_before = before[0, 0, -16:, -16:]
        block_after = result[0, 0, -16:, -16:]
        torch.testing.assert_close((block_after + block_after.T) / 2,
                                   (block_before + block_before.T) / 2,
                                   rtol=0, atol=0)
        torch.testing.assert_close(torch.diagonal(result, dim1=-2, dim2=-1),
                                   torch.diagonal(before, dim1=-2, dim2=-1),
                                   rtol=0, atol=0)
        self.assertTrue(torch.equal(result[:, :, :4, :], before[:, :, :4, :]))
        self.assertTrue(torch.equal(result[:, :, -16:, :4], before[:, :, -16:, :4]))
        self.assertEqual(controller.summary()["max_constructed_delta_antisymmetry_error_float32"], 0.0)
        self.assertEqual(controller.summary()["max_symmetric_component_error_float32"], 0.0)

    def test_first_token_only_and_alpha_zero_complete_noop(self):
        active = self.controller()
        active.apply(0, self.scores)
        later = torch.randn((1, NUM_HEADS, 21, 21), dtype=torch.float32)
        self.assertIs(active.apply(0, later), later)
        self.assertEqual(active.summary()["later_forward_intervention_count"], 0)
        self.assertEqual(active.summary()["later_forward_skip_count"], 1)
        zero = self.controller(alpha=0.0)
        before = self.scores.clone()
        self.assertIs(zero.apply(0, self.scores), self.scores)
        self.assertTrue(torch.equal(self.scores, before))
        self.assertEqual(zero.summary()["selected_rank_intervention_count"], 0)
        self.assertEqual(zero.summary()["alpha_zero_bypass_count"], 1)

    def test_ties_use_stable_strict_lower_position_order(self):
        tied_scores = make_scores(torch.zeros(120, dtype=torch.float32))
        tied_shift = torch.zeros(SHIFT_SHAPE, dtype=torch.float32)
        tied_shift[0, 0, 0] = 1.0
        controller = RankDistributionShiftCancellationController(
            mean_shift=tied_shift, alpha=1.0
        )
        controller.set_prompt_context(prompt_length=20)
        result = controller.apply(0, tied_scores)
        asym = (result[0, 0, -16:, -16:]
                - result[0, 0, -16:, -16:].T) / 2
        lower = asym[self.rows, self.cols]
        self.assertEqual(lower[0].item(), -1.0)
        self.assertTrue(torch.equal(lower[1:], torch.zeros(119)))

    def test_common_shift_preserves_prompt_difference(self):
        a = torch.arange(120, dtype=torch.float32)
        b = torch.arange(120, dtype=torch.float32) + 7
        shift = torch.arange(120, dtype=torch.float32) * 0.5
        torch.testing.assert_close((a - shift) - (b - shift), a - b,
                                   rtol=0, atol=0)

    def test_offline_crossing_is_detected_without_correction(self):
        values = torch.zeros(SCORE_SHAPE, dtype=torch.float32)
        values[:, 0, 0] = torch.arange(120, dtype=torch.float32)
        result = offline_shift_crossing_diagnostics(values, self.shift)
        self.assertEqual(result["rank_element_count"], SELECTION_COUNT * ALL_RANK_COUNT)
        self.assertEqual(result["adjacent_rank_pair_crossing_count"], SELECTION_COUNT)
        self.assertEqual(result["rank_value_crossing_count"], 2 * SELECTION_COUNT)
        self.assertEqual(result["prompt_with_crossing_count"], SELECTION_COUNT)
        self.assertEqual(result["definition"],
                         "strict adjacent inversion in Q-alpha*D; no correction or re-sort")


if __name__ == "__main__":
    unittest.main()
