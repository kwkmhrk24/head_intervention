import math
import unittest

import torch

from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    NUM_HEADS,
    NUM_LAYERS,
    REFERENCE_METADATA_FIELDS,
    TAIL_TOKENS,
    strict_lower_indices,
)
from utils.reference_asym_score_intervention import (
    EXPECTED_SCORE_CAPTURE_POSITION,
    ReferenceAsymmetryController,
    build_antisymmetric_reference_matrix,
    validate_alpha,
    validate_runtime_artifact_payloads,
)


EXPECTED_HEADS = ((31, 22), (13, 4), (5, 30), (9, 18), (27, 24))
SPLIT_HASH = "a" * 64
SCORE_HASH = "b" * 64
MODEL_PATH = "/models/llama"


def make_reference_payload():
    lower = strict_lower_indices(TAIL_TOKENS)
    metadata = {field: None for field in REFERENCE_METADATA_FIELDS}
    metadata.update(
        {
            "reference_name": "original reference",
            "alternative_name": "non-jailbreak reference",
            "tail_tokens": TAIL_TOKENS,
            "selection_manifest_sha256": SPLIT_HASH,
            "score_artifact_sha256": SCORE_HASH,
            "model_path": MODEL_PATH,
            "model_config": {},
            "torch_dtype": "torch.float16",
            "attention_implementation": "eager",
            "score_capture_position": EXPECTED_SCORE_CAPTURE_POSITION,
            "score_calculation_dtype": "torch.float16",
            "stored_value_dtype": "torch.float32",
            "selection_row_positions": list(range(20)),
            "selection_dataset_indices": list(range(20)),
        }
    )
    shape = (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
    return {
        "mean": torch.zeros(shape, dtype=torch.float32),
        "std": torch.zeros(shape, dtype=torch.float32),
        "sem": torch.zeros(shape, dtype=torch.float32),
        "tril_rows": lower[0].clone(),
        "tril_cols": lower[1].clone(),
        "metadata": metadata,
    }


def make_top5_payload():
    return {
        "selection_manifest_sha256": SPLIT_HASH,
        "score_artifact_sha256": SCORE_HASH,
        "heads": [
            {"rank": rank, "layer": layer, "head": head}
            for rank, (layer, head) in enumerate(EXPECTED_HEADS, 1)
        ],
    }


def make_verification():
    return {
        "verified": True,
        "selection_manifest_sha256": SPLIT_HASH,
        "score_artifact_sha256": SCORE_HASH,
    }


def make_controller(alpha, lower_values=None):
    reference = torch.zeros((2, 4, ELEMENTS_PER_HEAD), dtype=torch.float32)
    if lower_values is None:
        lower_values = torch.linspace(-1.0, 1.0, ELEMENTS_PER_HEAD)
    reference[0, 1] = lower_values
    return ReferenceAsymmetryController(
        reference_mean=reference,
        layer_head_pairs=[(0, 1)],
        alpha=alpha,
    )


class ReferenceMatrixTests(unittest.TestCase):
    def test_strict_lower_expands_to_antisymmetric_matrix(self):
        values = torch.arange(ELEMENTS_PER_HEAD, dtype=torch.float32)
        matrix = build_antisymmetric_reference_matrix(values)
        indices = torch.tril_indices(16, 16, offset=-1)
        self.assertEqual(tuple(matrix.shape), (16, 16))
        self.assertTrue(torch.equal(matrix[indices[0], indices[1]], values))
        self.assertTrue(
            torch.equal(matrix[indices[1], indices[0]], -values)
        )
        self.assertTrue(torch.equal(torch.diagonal(matrix), torch.zeros(16)))
        self.assertTrue(torch.equal(matrix + matrix.T, torch.zeros_like(matrix)))

    def test_wrong_tril_order_is_rejected(self):
        values = torch.zeros(ELEMENTS_PER_HEAD)
        indices = torch.tril_indices(16, 16, offset=-1).flip(1)
        with self.assertRaisesRegex(ValueError, "tril_indices order"):
            build_antisymmetric_reference_matrix(values, tril_indices=indices)


class InterpolationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.scores = torch.randn(1, 4, 20, 20, dtype=torch.float32)

    def test_alpha_zero_is_exact_object_identity(self):
        controller = make_controller(0.0)
        before = self.scores.clone()
        result = controller.apply(0, self.scores)
        self.assertIs(result, self.scores)
        self.assertTrue(torch.equal(result, before))
        self.assertEqual(controller.summary()["alpha_zero_bypass_count"], 1)

    def test_alpha_one_asymmetry_equals_reference(self):
        controller = make_controller(1.0)
        result = controller.apply(0, self.scores)
        block = result[0, 1, -16:, -16:]
        observed = (block - block.T) / 2
        expected = controller._reference_cpu[(0, 1)]
        torch.testing.assert_close(observed, expected, atol=2e-7, rtol=0)

    def test_alpha_half_matches_manual_interpolation(self):
        controller = make_controller(0.5)
        original = self.scores[0, 1, -16:, -16:]
        current = (original - original.T) / 2
        reference = controller._reference_cpu[(0, 1)]
        expected = original + 0.5 * (reference - current)
        result = controller.apply(0, self.scores)
        torch.testing.assert_close(
            result[0, 1, -16:, -16:],
            expected,
            atol=0,
            rtol=0,
        )

    def test_only_selected_tail_block_changes(self):
        controller = make_controller(0.5)
        result = controller.apply(0, self.scores)
        self.assertIsNot(result, self.scores)
        self.assertFalse(
            torch.equal(
                result[0, 1, -16:, -16:],
                self.scores[0, 1, -16:, -16:],
            )
        )
        self.assertTrue(torch.equal(result[:, [0, 2, 3]], self.scores[:, [0, 2, 3]]))
        self.assertTrue(torch.equal(result[:, 1, :4, :], self.scores[:, 1, :4, :]))
        self.assertTrue(torch.equal(result[:, 1, -16:, :4], self.scores[:, 1, -16:, :4]))
        summary = controller.summary()
        self.assertEqual(summary["max_unselected_head_change"], 0.0)
        self.assertEqual(summary["max_outside_tail_change"], 0.0)
        selected = summary["per_head"][0]
        self.assertIn("max_delta_antisymmetry_error", selected)
        self.assertIn("max_symmetric_component_error", selected)
        self.assertEqual(selected["max_unselected_head_change"], 0.0)
        self.assertEqual(selected["max_outside_tail_change"], 0.0)

    def test_unselected_layer_returns_same_tensor_object(self):
        controller = make_controller(0.5)
        result = controller.apply(1, self.scores)
        self.assertIs(result, self.scores)
        self.assertEqual(controller.summary()["attention_forward_call_count"], 0)

    def test_float32_symmetric_component_and_diagonal_are_preserved(self):
        controller = make_controller(0.5)
        before = self.scores[0, 1, -16:, -16:]
        result = controller.apply(0, self.scores)
        after = result[0, 1, -16:, -16:]
        before_sym = (before + before.T) / 2
        after_sym = (after + after.T) / 2
        torch.testing.assert_close(after_sym, before_sym, atol=2e-7, rtol=0)
        delta = after - before
        torch.testing.assert_close(delta + delta.T, torch.zeros_like(delta), atol=2e-7, rtol=0)
        torch.testing.assert_close(
            torch.diagonal(after),
            torch.diagonal(before),
            atol=0,
            rtol=0,
        )

    def test_short_sequence_skips_and_counts(self):
        controller = make_controller(0.5)
        scores = torch.randn(1, 4, 15, 15)
        before = scores.clone()
        result = controller.apply(0, scores)
        self.assertIs(result, scores)
        self.assertTrue(torch.equal(result, before))
        summary = controller.summary()
        self.assertEqual(summary["short_sequence_skip_count"], 1)
        self.assertEqual(summary["intervention_call_count"], 0)
        self.assertEqual(summary["per_head"][0]["short_sequence_skip_count"], 1)
        self.assertEqual(controller.head_metric_records(), [])


class AlphaValidationTests(unittest.TestCase):
    def test_valid_boundaries(self):
        self.assertEqual(validate_alpha(0), 0.0)
        self.assertEqual(validate_alpha(1), 1.0)

    def test_values_outside_unit_interval_are_rejected(self):
        for value in (-0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_alpha(value)

    def test_nan_and_infinity_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_alpha(value)


class RuntimeArtifactValidationTests(unittest.TestCase):
    def validate(self, verification=None, top5=None, reference=None):
        return validate_runtime_artifact_payloads(
            verification=verification or make_verification(),
            top5_payload=top5 or make_top5_payload(),
            reference_payload=reference or make_reference_payload(),
            expected_heads=EXPECTED_HEADS,
            expected_selection_manifest_sha256=SPLIT_HASH,
            expected_score_artifact_sha256=SCORE_HASH,
            expected_model_path=MODEL_PATH,
        )

    def test_valid_payloads_return_top5(self):
        self.assertEqual(self.validate(), EXPECTED_HEADS)

    def test_top5_mismatch_is_rejected(self):
        payload = make_top5_payload()
        payload["heads"][4]["head"] = 25
        with self.assertRaisesRegex(ValueError, "top5 heads mismatch"):
            self.validate(top5=payload)

    def test_reference_shape_mismatch_is_rejected(self):
        payload = make_reference_payload()
        payload["mean"] = torch.zeros(31, 32, 120)
        with self.assertRaisesRegex(ValueError, "shape/dtype mismatch"):
            self.validate(reference=payload)

    def test_manifest_hash_mismatch_is_rejected(self):
        verification = make_verification()
        verification["selection_manifest_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "selection_manifest_sha256 mismatch"):
            self.validate(verification=verification)

    def test_tail_tokens_mismatch_is_rejected(self):
        payload = make_reference_payload()
        payload["metadata"]["tail_tokens"] = 15
        with self.assertRaisesRegex(ValueError, "tail_tokens mismatch"):
            self.validate(reference=payload)


if __name__ == "__main__":
    unittest.main()
