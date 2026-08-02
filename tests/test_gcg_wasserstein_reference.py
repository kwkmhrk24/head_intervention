import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.prepare_gcg_wasserstein_reference import main as prepare_main
from utils.gcg_wasserstein_reference import (
    ELEMENTS_PER_HEAD,
    EXPECTED_SELECTION_DATASET_INDICES,
    EXPECTED_SELECTION_ROW_POSITIONS,
    NUM_HEADS,
    NUM_LAYERS,
    REFERENCE_METADATA_FIELDS,
    build_split_manifest,
    compute_head_wasserstein_statistics,
    compute_original_reference,
    ensure_output_directory_absent,
    equal_weight_1d_wasserstein,
    strict_lower_indices,
    validate_reference_payload,
    verify_declared_file_hashes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = REPO_ROOT / "data/derived/llama2/gcg/original_prompts.jsonl"
JAILBREAK = REPO_ROOT / "data/derived/llama2/gcg/jailbreak_prompts.jsonl"
BASELINE = REPO_ROOT / (
    "data/outputs/multidataset_asym_score_intervention/"
    "beta05_full_20260722_041303/llama_gcg/no_intervention/prompt_results.jsonl"
)
BASELINE_MANIFEST = REPO_ROOT / (
    "data/outputs/multidataset_asym_score_intervention/"
    "beta05_full_20260722_041303/llama_gcg/run_manifest.json"
)


def real_split_manifest():
    return build_split_manifest(
        original_input=ORIGINAL,
        jailbreak_input=JAILBREAK,
        baseline_results=BASELINE,
        baseline_run_manifest=BASELINE_MANIFEST,
        repo_root=REPO_ROOT,
        created_at="2026-01-01T00:00:00+00:00",
    )


class SplitTests(unittest.TestCase):
    def test_fixed_seed42_dual_index_split(self):
        manifest = real_split_manifest()
        self.assertEqual(
            manifest["selection_row_positions"],
            EXPECTED_SELECTION_ROW_POSITIONS,
        )
        self.assertEqual(
            manifest["selection_dataset_indices"],
            EXPECTED_SELECTION_DATASET_INDICES,
        )
        selection = set(manifest["selection_row_positions"])
        evaluation = set(manifest["evaluation_row_positions"])
        self.assertFalse(selection & evaluation)
        self.assertEqual(selection | evaluation, set(range(160)))
        self.assertEqual(manifest["selection_count"], 20)
        self.assertEqual(manifest["evaluation_count"], 140)
        self.assertEqual(manifest["baseline_success_count_selection"], 20)
        self.assertEqual(manifest["baseline_success_count_evaluation"], 54)
        self.assertAlmostEqual(manifest["baseline_asr_evaluation"], 54 / 140)
        records = manifest["records"]
        self.assertEqual(
            [records[index]["dataset_index"] for index in sorted(selection)],
            EXPECTED_SELECTION_DATASET_INDICES,
        )

    def test_prepare_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dry-run-output"
            prepare_main(
                [
                    "--original-input",
                    str(ORIGINAL),
                    "--jailbreak-input",
                    str(JAILBREAK),
                    "--baseline-results",
                    str(BASELINE),
                    "--baseline-run-manifest",
                    str(BASELINE_MANIFEST),
                    "--output-dir",
                    str(output),
                    "--dry-run",
                ]
            )
            self.assertFalse(output.exists())


class WassersteinTests(unittest.TestCase):
    def test_identical_and_permuted_vectors_are_zero(self):
        x = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float32)
        self.assertEqual(equal_weight_1d_wasserstein(x, x).item(), 0.0)
        permutation = x[torch.tensor([2, 0, 1])]
        self.assertEqual(
            equal_weight_1d_wasserstein(x, permutation).item(),
            0.0,
        )

    def test_hand_computed_example(self):
        x = torch.tensor([0.0, 1.0], dtype=torch.float32)
        y = torch.tensor([0.0, 3.0], dtype=torch.float32)
        self.assertEqual(equal_weight_1d_wasserstein(x, y).item(), 1.0)

    def test_batch_layer_head_shape(self):
        x = torch.zeros(2, 3, 4, 5, dtype=torch.float32)
        y = torch.ones_like(x)
        result = equal_weight_1d_wasserstein(x, y)
        self.assertEqual(result.shape, (2, 3, 4))
        self.assertTrue(torch.equal(result, torch.ones(2, 3, 4)))

    def test_nonfinite_and_shape_mismatch_are_rejected(self):
        finite = torch.zeros(2, 3, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            equal_weight_1d_wasserstein(finite, torch.zeros(2, 4))
        for bad_value in (float("nan"), float("inf")):
            bad = finite.clone()
            bad[0, 0] = bad_value
            with self.assertRaisesRegex(ValueError, "finite"):
                equal_weight_1d_wasserstein(bad, finite)

    def test_full_head_ranking_ties_use_layer_then_head(self):
        original = torch.zeros(
            20,
            NUM_LAYERS,
            NUM_HEADS,
            ELEMENTS_PER_HEAD,
            dtype=torch.float32,
        )
        jailbreak = torch.ones_like(original)
        distances, ranking = compute_head_wasserstein_statistics(
            original,
            jailbreak,
        )
        self.assertEqual(distances.shape, (20, 32, 32))
        self.assertEqual(
            [(row["layer"], row["head"]) for row in ranking[:4]],
            [(0, 0), (0, 1), (0, 2), (0, 3)],
        )
        self.assertTrue(all(row["mean"] == 1.0 for row in ranking))
        self.assertEqual([row["rank"] for row in ranking], list(range(1, 1025)))


class TrilIndexTests(unittest.TestCase):
    def test_tail16_strict_lower_indices(self):
        indices = strict_lower_indices(16)
        expected = torch.tril_indices(16, 16, offset=-1)
        self.assertEqual(indices.shape, (2, 120))
        self.assertTrue(torch.equal(indices, expected))
        self.assertTrue(torch.all(indices[0] > indices[1]))


class ReferenceTests(unittest.TestCase):
    def test_reference_shapes_values_and_dtype(self):
        base = torch.arange(20, dtype=torch.float32).view(20, 1, 1, 1)
        original = base.expand(20, 32, 32, 120).contiguous()
        reference = compute_original_reference(original)
        for name in ("mean", "std", "sem"):
            self.assertEqual(reference[name].shape, (32, 32, 120))
            self.assertEqual(reference[name].dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                reference["mean"],
                torch.full((32, 32, 120), 9.5, dtype=torch.float32),
            )
        )
        expected_std = torch.arange(20, dtype=torch.float32).std(correction=1)
        self.assertTrue(
            torch.allclose(
                reference["std"],
                torch.full((32, 32, 120), expected_std),
            )
        )

    def test_reference_required_metadata(self):
        tensors = {
            "mean": torch.zeros(32, 32, 120, dtype=torch.float32),
            "std": torch.zeros(32, 32, 120, dtype=torch.float32),
            "sem": torch.zeros(32, 32, 120, dtype=torch.float32),
            "tril_rows": torch.tril_indices(16, 16, offset=-1)[0],
            "tril_cols": torch.tril_indices(16, 16, offset=-1)[1],
        }
        metadata = {key: "value" for key in REFERENCE_METADATA_FIELDS}
        metadata["reference_name"] = "original reference"
        metadata["alternative_name"] = "non-jailbreak reference"
        validate_reference_payload({**tensors, "metadata": metadata})
        metadata.pop("score_artifact_sha256")
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_reference_payload({**tensors, "metadata": metadata})


class ArtifactTests(unittest.TestCase):
    def test_existing_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                ensure_output_directory_absent(directory)

    def test_declared_hash_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"original")
            expected_hash = hashlib.sha256(b"original").hexdigest()
            manifest = {
                "required_files": ["artifact.bin"],
                "files": {
                    "artifact.bin": {
                        "sha256": expected_hash,
                        "size_bytes": len(b"original"),
                    }
                },
            }
            verify_declared_file_hashes(root, manifest)
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_declared_file_hashes(root, manifest)


if __name__ == "__main__":
    unittest.main()
