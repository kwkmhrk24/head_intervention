"""Offline artifacts for the Llama-2 GCG Wasserstein reference experiment.

This module deliberately does not modify the existing beta score-intervention
path.  GPU-only score extraction is isolated here, while split construction,
Wasserstein statistics, reference statistics, and artifact validation remain
CPU-testable.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from utils.artifact_io import (
    ensure_output_directory_absent,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
    write_jsonl,
)
from utils.experiment_runtime import git_state


TOTAL_COUNT = 160
BASELINE_SUCCESS_COUNT = 74
SELECTION_COUNT = 20
EVALUATION_COUNT = 140
SEED = 42
TAIL_TOKENS = 16
ELEMENTS_PER_HEAD = TAIL_TOKENS * (TAIL_TOKENS - 1) // 2
NUM_LAYERS = 32
NUM_HEADS = 32
SAMPLING_ALGORITHM = (
    "sorted(random.Random(42).sample(baseline_success_row_positions, 20))"
)
EXPECTED_SELECTION_ROW_POSITIONS = [
    8,
    10,
    12,
    13,
    24,
    27,
    28,
    31,
    36,
    57,
    63,
    67,
    74,
    83,
    103,
    121,
    148,
    155,
    157,
    158,
]
EXPECTED_SELECTION_DATASET_INDICES = [
    68,
    70,
    72,
    73,
    84,
    87,
    88,
    91,
    96,
    117,
    123,
    127,
    134,
    143,
    163,
    181,
    208,
    215,
    217,
    218,
]

CORE_ARTIFACT_FILES = [
    "split_manifest.json",
    "split_manifest.csv",
    "selection_original_scores.pt",
    "selection_jailbreak_scores.pt",
    "score_extraction_manifest.json",
    "per_prompt_wasserstein.pt",
    "per_prompt_wasserstein.csv",
    "head_wasserstein_ranking.csv",
    "top5_heads.json",
    "original_reference.pt",
    "original_reference_summary.json",
]

SPLIT_RECORD_FIELDS = [
    "row_position",
    "dataset_index",
    "prompt_id",
    "split",
    "baseline_success",
    "original_prompt_sha256",
    "jailbreak_prompt_sha256",
]

REFERENCE_METADATA_FIELDS = {
    "reference_name",
    "alternative_name",
    "tail_tokens",
    "selection_row_positions",
    "selection_dataset_indices",
    "selection_manifest_sha256",
    "score_artifact_sha256",
    "model_path",
    "model_config",
    "torch_dtype",
    "attention_implementation",
    "score_capture_position",
    "score_calculation_dtype",
    "stored_value_dtype",
}


def composite_score_sha256(original_sha256: str, jailbreak_sha256: str) -> str:
    return sha256_text(f"{original_sha256}:{jailbreak_sha256}")


def _require_prompt_row(row: Mapping[str, object], *, source: str, row_position: int):
    dataset_index = row.get("index")
    prompt = row.get("prompt")
    if type(dataset_index) is not int:
        raise ValueError(
            f"{source}: row_position={row_position} has a non-integer index"
        )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            f"{source}: row_position={row_position} has no non-empty prompt"
        )
    return dataset_index, prompt


def build_split_manifest(
    *,
    original_input: str | Path,
    jailbreak_input: str | Path,
    baseline_results: str | Path,
    baseline_run_manifest: str | Path,
    repo_root: str | Path,
    created_at: str | None = None,
) -> dict:
    """Build and fully validate the fixed dual-index GCG split."""

    original_input = Path(original_input)
    jailbreak_input = Path(jailbreak_input)
    baseline_results = Path(baseline_results)
    baseline_run_manifest = Path(baseline_run_manifest)
    original = read_jsonl(original_input)
    jailbreak = read_jsonl(jailbreak_input)
    baseline = read_jsonl(baseline_results)
    run_manifest = read_json(baseline_run_manifest)

    counts = {
        "original": len(original),
        "jailbreak": len(jailbreak),
        "baseline": len(baseline),
    }
    if set(counts.values()) != {TOTAL_COUNT}:
        raise ValueError(f"expected exactly {TOTAL_COUNT} aligned rows: {counts}")

    jailbreak_sha256 = sha256_file(jailbreak_input)
    if run_manifest.get("input_sha256") != jailbreak_sha256:
        raise ValueError("baseline run manifest input_sha256 does not match jailbreak input")
    if run_manifest.get("normalized_prompt_count") != TOTAL_COUNT:
        raise ValueError("baseline run manifest normalized_prompt_count mismatch")
    if run_manifest.get("attack") != "gcg":
        raise ValueError("baseline run manifest attack is not gcg")
    generation = run_manifest.get("generation_settings")
    if not isinstance(generation, dict):
        raise ValueError("baseline run manifest lacks generation_settings")
    expected_generation = {
        "do_sample": False,
        "max_new_tokens": 32,
        "use_cache": False,
        "seed": 42,
        "system_prompt": None,
        "batch_size": 1,
    }
    for key, expected in expected_generation.items():
        if generation.get(key) != expected:
            raise ValueError(
                f"baseline run manifest generation setting mismatch: "
                f"{key}={generation.get(key)!r}, expected={expected!r}"
            )

    records = []
    seen_dataset_indices = set()
    baseline_success_row_positions = []
    for row_position, (original_row, jailbreak_row, baseline_row) in enumerate(
        zip(original, jailbreak, baseline)
    ):
        original_index, original_prompt = _require_prompt_row(
            original_row, source=str(original_input), row_position=row_position
        )
        jailbreak_index, jailbreak_prompt = _require_prompt_row(
            jailbreak_row, source=str(jailbreak_input), row_position=row_position
        )
        if original_index != jailbreak_index:
            raise ValueError(
                f"original/jailbreak dataset_index mismatch at row_position={row_position}"
            )
        dataset_index = original_index
        if dataset_index in seen_dataset_indices:
            raise ValueError(f"duplicate dataset_index: {dataset_index}")
        seen_dataset_indices.add(dataset_index)
        if dataset_index != row_position + 60:
            raise ValueError(
                f"unexpected GCG dataset mapping at row_position={row_position}: "
                f"dataset_index={dataset_index}"
            )
        if baseline_row.get("source_index") != dataset_index:
            raise ValueError(
                f"baseline source_index mismatch at row_position={row_position}"
            )
        if baseline_row.get("raw_index") != dataset_index:
            raise ValueError(
                f"baseline raw_index mismatch at row_position={row_position}"
            )
        if baseline_row.get("prompt") != jailbreak_prompt:
            raise ValueError(
                f"baseline/jailbreak prompt mismatch at dataset_index={dataset_index}"
            )
        prompt_id = baseline_row.get("prompt_id")
        if not isinstance(prompt_id, (str, int)) or isinstance(prompt_id, bool):
            raise ValueError(f"invalid baseline prompt_id at dataset_index={dataset_index}")
        baseline_success = baseline_row.get("success")
        if type(baseline_success) is not bool:
            raise ValueError(
                f"invalid baseline success at dataset_index={dataset_index}"
            )
        if baseline_success:
            baseline_success_row_positions.append(row_position)
        records.append(
            {
                "row_position": row_position,
                "dataset_index": dataset_index,
                "prompt_id": prompt_id,
                "split": None,
                "baseline_success": baseline_success,
                "original_prompt_sha256": sha256_text(original_prompt),
                "jailbreak_prompt_sha256": sha256_text(jailbreak_prompt),
            }
        )

    if len(baseline_success_row_positions) != BASELINE_SUCCESS_COUNT:
        raise ValueError(
            f"baseline success count mismatch: "
            f"{len(baseline_success_row_positions)} != {BASELINE_SUCCESS_COUNT}"
        )
    selection_row_positions = sorted(
        random.Random(SEED).sample(
            baseline_success_row_positions,
            SELECTION_COUNT,
        )
    )
    if selection_row_positions != EXPECTED_SELECTION_ROW_POSITIONS:
        raise ValueError(
            "fixed seed-42 selection differs from the expected row positions"
        )
    selection_set = set(selection_row_positions)
    evaluation_row_positions = [
        row_position
        for row_position in range(TOTAL_COUNT)
        if row_position not in selection_set
    ]
    for record in records:
        record["split"] = (
            "selection"
            if record["row_position"] in selection_set
            else "evaluation"
        )

    selection_dataset_indices = [
        records[row_position]["dataset_index"]
        for row_position in selection_row_positions
    ]
    if selection_dataset_indices != EXPECTED_SELECTION_DATASET_INDICES:
        raise ValueError(
            "fixed seed-42 selection differs from the expected dataset indices"
        )
    evaluation_dataset_indices = [
        records[row_position]["dataset_index"]
        for row_position in evaluation_row_positions
    ]
    selection_success = sum(
        records[row_position]["baseline_success"]
        for row_position in selection_row_positions
    )
    evaluation_success = sum(
        records[row_position]["baseline_success"]
        for row_position in evaluation_row_positions
    )
    if len(selection_row_positions) != SELECTION_COUNT:
        raise AssertionError("selection count mismatch")
    if len(evaluation_row_positions) != EVALUATION_COUNT:
        raise AssertionError("evaluation count mismatch")
    if selection_success != SELECTION_COUNT:
        raise AssertionError("selection must contain only baseline successes")
    if evaluation_success != BASELINE_SUCCESS_COUNT - SELECTION_COUNT:
        raise AssertionError("evaluation baseline success count mismatch")
    if selection_set & set(evaluation_row_positions):
        raise AssertionError("selection/evaluation overlap")
    if selection_set | set(evaluation_row_positions) != set(range(TOTAL_COUNT)):
        raise AssertionError("selection/evaluation do not partition all rows")

    git_head, git_branch = git_state(repo_root)
    manifest = {
        "schema_version": 1,
        "dataset": "Llama-2 GCG 160 paired prompts",
        "dataset_index_definition": "AdvBench dataset index; primary prompt key",
        "row_position_definition": "zero-based row in the paired 160-row JSONL inputs",
        "seed": SEED,
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "selection_count": SELECTION_COUNT,
        "evaluation_count": EVALUATION_COUNT,
        "baseline_success_count_all": BASELINE_SUCCESS_COUNT,
        "baseline_success_count_selection": selection_success,
        "baseline_success_count_evaluation": evaluation_success,
        "baseline_asr_all": BASELINE_SUCCESS_COUNT / TOTAL_COUNT,
        "baseline_asr_evaluation": evaluation_success / EVALUATION_COUNT,
        "original_input": str(original_input.resolve()),
        "jailbreak_input": str(jailbreak_input.resolve()),
        "baseline_results": str(baseline_results.resolve()),
        "baseline_run_manifest": str(baseline_run_manifest.resolve()),
        "original_input_sha256": sha256_file(original_input),
        "jailbreak_input_sha256": jailbreak_sha256,
        "baseline_results_sha256": sha256_file(baseline_results),
        "baseline_run_manifest_sha256": sha256_file(baseline_run_manifest),
        "created_at": created_at or utc_now(),
        "git_head": git_head,
        "git_branch": git_branch,
        "selection_row_positions": selection_row_positions,
        "selection_dataset_indices": selection_dataset_indices,
        "evaluation_row_positions": evaluation_row_positions,
        "evaluation_dataset_indices": evaluation_dataset_indices,
        "records": records,
    }
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, object]) -> None:
    required = {
        "seed",
        "sampling_algorithm",
        "selection_count",
        "evaluation_count",
        "baseline_success_count_all",
        "baseline_success_count_selection",
        "baseline_success_count_evaluation",
        "baseline_asr_all",
        "baseline_asr_evaluation",
        "original_input_sha256",
        "jailbreak_input_sha256",
        "baseline_results_sha256",
        "baseline_run_manifest_sha256",
        "created_at",
        "git_head",
        "git_branch",
        "selection_row_positions",
        "selection_dataset_indices",
        "evaluation_row_positions",
        "evaluation_dataset_indices",
        "records",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"split manifest missing fields: {sorted(missing)}")
    if manifest["seed"] != SEED or manifest["sampling_algorithm"] != SAMPLING_ALGORITHM:
        raise ValueError("split manifest sampling metadata mismatch")
    records = manifest["records"]
    if not isinstance(records, list) or len(records) != TOTAL_COUNT:
        raise ValueError("split manifest must contain 160 records")
    if any(set(record) != set(SPLIT_RECORD_FIELDS) for record in records):
        raise ValueError("split manifest record schema mismatch")
    row_positions = [record["row_position"] for record in records]
    if row_positions != list(range(TOTAL_COUNT)):
        raise ValueError("split manifest records are not in row_position order")
    dataset_indices = [record["dataset_index"] for record in records]
    if len(dataset_indices) != len(set(dataset_indices)):
        raise ValueError("split manifest contains duplicate dataset_index")
    selection = list(manifest["selection_row_positions"])
    evaluation = list(manifest["evaluation_row_positions"])
    if selection != EXPECTED_SELECTION_ROW_POSITIONS:
        raise ValueError("split manifest selection row positions mismatch")
    if list(manifest["selection_dataset_indices"]) != EXPECTED_SELECTION_DATASET_INDICES:
        raise ValueError("split manifest selection dataset indices mismatch")
    if set(selection) & set(evaluation):
        raise ValueError("split manifest selection/evaluation overlap")
    if set(selection) | set(evaluation) != set(range(TOTAL_COUNT)):
        raise ValueError("split manifest selection/evaluation union mismatch")
    if sum(bool(records[index]["baseline_success"]) for index in selection) != 20:
        raise ValueError("split manifest selection baseline success mismatch")
    if sum(bool(records[index]["baseline_success"]) for index in evaluation) != 54:
        raise ValueError("split manifest evaluation baseline success mismatch")


def selection_records(manifest: Mapping[str, object]) -> list[dict]:
    validate_split_manifest(manifest)
    records = manifest["records"]
    return [dict(records[row]) for row in manifest["selection_row_positions"]]


def write_split_artifacts(manifest: Mapping[str, object], output_dir: str | Path):
    validate_split_manifest(manifest)
    output_dir = Path(output_dir)
    write_json(output_dir / "split_manifest.json", dict(manifest))
    csv_path = output_dir / "split_manifest.csv"
    if csv_path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(manifest["records"])


def strict_lower_indices(
    tail_tokens: int = TAIL_TOKENS,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if type(tail_tokens) is not int or tail_tokens < 2:
        raise ValueError("tail_tokens must be an integer >= 2")
    indices = torch.tril_indices(
        tail_tokens,
        tail_tokens,
        offset=-1,
        device=device,
    )
    expected = tail_tokens * (tail_tokens - 1) // 2
    if tuple(indices.shape) != (2, expected):
        raise AssertionError("unexpected strict-lower index shape")
    if not bool((indices[0] > indices[1]).all().item()):
        raise AssertionError("strict-lower indices contain a non-lower position")
    return indices


def equal_weight_1d_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x and y must be torch tensors")
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {tuple(x.shape)} != {tuple(y.shape)}")
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("x and y must have a non-empty sample dimension")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("x and y must use floating-point dtypes")
    if not bool(torch.isfinite(x).all().item()) or not bool(
        torch.isfinite(y).all().item()
    ):
        raise ValueError("x and y must contain only finite values")
    x_sorted = torch.sort(x, dim=-1).values
    y_sorted = torch.sort(y, dim=-1).values
    return torch.mean(torch.abs(x_sorted - y_sorted), dim=-1)


def validate_score_tensor(values: torch.Tensor, *, prompt_count: int = SELECTION_COUNT):
    expected = (prompt_count, NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
    if not isinstance(values, torch.Tensor):
        raise TypeError("score values must be a torch tensor")
    if tuple(values.shape) != expected:
        raise ValueError(
            f"score tensor shape mismatch: {tuple(values.shape)} != {expected}"
        )
    if values.dtype != torch.float32:
        raise ValueError(f"stored score tensor must be float32, found {values.dtype}")
    if values.device.type != "cpu":
        raise ValueError("stored score tensor must be on CPU")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("score tensor contains NaN or infinity")


def compute_head_wasserstein_statistics(
    original: torch.Tensor,
    jailbreak: torch.Tensor,
) -> tuple[torch.Tensor, list[dict]]:
    validate_score_tensor(original)
    validate_score_tensor(jailbreak)
    distances = equal_weight_1d_wasserstein(original, jailbreak)
    expected = (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS)
    if tuple(distances.shape) != expected:
        raise AssertionError("unexpected per-prompt Wasserstein shape")
    mean = distances.mean(dim=0)
    median = torch.quantile(distances, 0.5, dim=0)
    std = distances.std(dim=0, correction=1)
    sem = std / math.sqrt(SELECTION_COUNT)
    minimum = distances.amin(dim=0)
    maximum = distances.amax(dim=0)
    rows = []
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "mean": float(mean[layer, head].item()),
                    "median": float(median[layer, head].item()),
                    "std": float(std[layer, head].item()),
                    "sem": float(sem[layer, head].item()),
                    "min": float(minimum[layer, head].item()),
                    "max": float(maximum[layer, head].item()),
                }
            )
    rows.sort(key=lambda row: (-row["mean"], row["layer"], row["head"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["is_top5"] = rank <= 5
    return distances.contiguous(), rows


def compute_original_reference(original: torch.Tensor) -> dict:
    validate_score_tensor(original)
    mean = original.mean(dim=0).to(torch.float32).contiguous()
    std = original.std(dim=0, correction=1).to(torch.float32).contiguous()
    sem = (std / math.sqrt(original.shape[0])).to(torch.float32).contiguous()
    expected = (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
    if tuple(mean.shape) != expected or tuple(std.shape) != expected or tuple(
        sem.shape
    ) != expected:
        raise AssertionError("unexpected original reference shape")
    if not all(bool(torch.isfinite(value).all().item()) for value in (mean, std, sem)):
        raise ValueError("original reference contains NaN or infinity")
    return {"mean": mean, "std": std, "sem": sem}


def validate_reference_payload(payload: Mapping[str, object]) -> None:
    required = {"mean", "std", "sem", "tril_rows", "tril_cols", "metadata"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"reference artifact missing fields: {sorted(missing)}")
    expected = (NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD)
    for name in ("mean", "std", "sem"):
        tensor = payload[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"reference {name} is not a tensor")
        if tuple(tensor.shape) != expected or tensor.dtype != torch.float32:
            raise ValueError(f"reference {name} shape/dtype mismatch")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"reference {name} contains NaN or infinity")
    for name in ("tril_rows", "tril_cols"):
        tensor = payload[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (
            ELEMENTS_PER_HEAD,
        ):
            raise ValueError(f"reference {name} shape mismatch")
        if tensor.dtype != torch.int64:
            raise ValueError(f"reference {name} must be int64")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("reference metadata is not an object")
    missing_metadata = REFERENCE_METADATA_FIELDS - set(metadata)
    if missing_metadata:
        raise ValueError(
            f"reference metadata missing fields: {sorted(missing_metadata)}"
        )
    if metadata["reference_name"] != "original reference":
        raise ValueError("reference_name must be 'original reference'")
    if metadata["alternative_name"] != "non-jailbreak reference":
        raise ValueError("alternative_name must be 'non-jailbreak reference'")


def save_torch_artifact(path: str | Path, payload) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    torch.save(payload, path)


def write_per_prompt_wasserstein_csv(
    path: str | Path,
    distances: torch.Tensor,
    selected_records: Sequence[Mapping[str, object]],
) -> None:
    expected = (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS)
    if tuple(distances.shape) != expected:
        raise ValueError("per-prompt Wasserstein shape mismatch")
    if len(selected_records) != SELECTION_COUNT:
        raise ValueError("selected record count mismatch")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    fields = [
        "selection_position",
        "row_position",
        "dataset_index",
        "prompt_id",
        "layer",
        "head",
        "wasserstein",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prompt_position, record in enumerate(selected_records):
            for layer in range(NUM_LAYERS):
                for head in range(NUM_HEADS):
                    writer.writerow(
                        {
                            "selection_position": prompt_position,
                            "row_position": record["row_position"],
                            "dataset_index": record["dataset_index"],
                            "prompt_id": record["prompt_id"],
                            "layer": layer,
                            "head": head,
                            "wasserstein": float(
                                distances[prompt_position, layer, head].item()
                            ),
                        }
                    )


def write_head_ranking_csv(path: str | Path, rows: Sequence[Mapping[str, object]]):
    if len(rows) != NUM_LAYERS * NUM_HEADS:
        raise ValueError("head ranking must contain 1024 rows")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    fields = [
        "rank",
        "layer",
        "head",
        "mean",
        "median",
        "std",
        "sem",
        "min",
        "max",
        "is_top5",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_declared_artifact_manifest(
    *,
    output_dir: str | Path,
    file_names: Sequence[str],
    metadata: Mapping[str, object],
) -> dict:
    output_dir = Path(output_dir)
    missing = [name for name in file_names if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot declare missing artifacts: {missing}")
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        **dict(metadata),
        "required_files": list(file_names),
        "files": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "size_bytes": (output_dir / name).stat().st_size,
            }
            for name in file_names
        },
    }
    write_json(output_dir / "artifact_manifest.json", payload)
    return payload


def verify_declared_file_hashes(
    output_dir: str | Path,
    artifact_manifest: Mapping[str, object],
) -> None:
    output_dir = Path(output_dir)
    required_files = artifact_manifest.get("required_files")
    files = artifact_manifest.get("files")
    if not isinstance(required_files, list) or not isinstance(files, dict):
        raise ValueError("artifact manifest lacks required_files/files")
    if set(required_files) != set(files):
        raise ValueError("artifact manifest required_files/files mismatch")
    for name in required_files:
        entry = files.get(name)
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("invalid artifact manifest file entry")
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing artifact: {path}")
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            raise ValueError(
                f"artifact hash mismatch for {name}: "
                f"actual={actual}, expected={entry.get('sha256')}"
            )


def verify_artifact_directory(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    manifest = read_json(output_dir / "artifact_manifest.json")
    required = manifest.get("required_files")
    if required != CORE_ARTIFACT_FILES:
        raise ValueError("artifact manifest does not declare the expected core files")
    verify_declared_file_hashes(output_dir, manifest)

    split_sha = sha256_file(output_dir / "split_manifest.json")
    score_manifest = read_json(output_dir / "score_extraction_manifest.json")
    reference_summary = read_json(output_dir / "original_reference_summary.json")
    if score_manifest.get("selection_manifest_sha256") != split_sha:
        raise ValueError("score extraction manifest split hash mismatch")
    if reference_summary.get("selection_manifest_sha256") != split_sha:
        raise ValueError("reference summary split hash mismatch")

    original_sha = sha256_file(output_dir / "selection_original_scores.pt")
    jailbreak_sha = sha256_file(output_dir / "selection_jailbreak_scores.pt")
    score_sha = composite_score_sha256(original_sha, jailbreak_sha)
    if score_manifest.get("score_artifact_sha256") != score_sha:
        raise ValueError("score extraction manifest composite score hash mismatch")
    if reference_summary.get("score_artifact_sha256") != score_sha:
        raise ValueError("reference summary score hash mismatch")

    original_payload = torch.load(
        output_dir / "selection_original_scores.pt",
        map_location="cpu",
        weights_only=True,
    )
    jailbreak_payload = torch.load(
        output_dir / "selection_jailbreak_scores.pt",
        map_location="cpu",
        weights_only=True,
    )
    validate_score_tensor(original_payload["values"])
    validate_score_tensor(jailbreak_payload["values"])
    reference_payload = torch.load(
        output_dir / "original_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    validate_reference_payload(reference_payload)
    if reference_payload["metadata"]["selection_manifest_sha256"] != split_sha:
        raise ValueError("reference artifact split hash mismatch")
    if reference_payload["metadata"]["score_artifact_sha256"] != score_sha:
        raise ValueError("reference artifact score hash mismatch")
    top5 = read_json(output_dir / "top5_heads.json")
    if not isinstance(top5.get("heads"), list) or len(top5["heads"]) != 5:
        raise ValueError("top5_heads.json does not contain five heads")
    return {
        "verified": True,
        "artifact_count": len(required),
        "selection_manifest_sha256": split_sha,
        "score_artifact_sha256": score_sha,
        "top5_heads": top5["heads"],
    }


def _capture_layer_inputs(model, input_ids: torch.Tensor) -> dict[int, dict]:
    captured = {}
    handles = []
    layers = model.model.layers
    for layer_index, layer in enumerate(layers):
        attention = layer.self_attn

        def hook(module, module_inputs, module_kwargs, layer_index=layer_index):
            if module_inputs:
                hidden_states = module_inputs[0]
            else:
                hidden_states = module_kwargs.get("hidden_states")
            position_ids = module_kwargs.get("position_ids")
            if hidden_states is None or position_ids is None:
                raise RuntimeError(
                    f"unable to capture hidden_states/position_ids at layer {layer_index}"
                )
            captured[layer_index] = {
                "hidden_states": hidden_states.detach(),
                "position_ids": position_ids.detach(),
            }

        handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        with torch.inference_mode():
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(range(len(layers))):
        raise RuntimeError("failed to capture all Llama attention layer inputs")
    return captured


def extract_prompt_scores(
    *,
    model,
    tokenizer,
    prompt: str,
    row_position: int,
    dataset_index: int,
    tail_tokens: int = TAIL_TOKENS,
    component: str = "antisymmetric",
) -> tuple[torch.Tensor, int]:
    """Extract eager pre-mask component values for one prompt's strict lower tail."""

    from utils.prompt_utils import apply_template

    formatted_prompt = apply_template(
        [prompt],
        system_prompt=None,
        tokenizer=tokenizer,
        model_config={"model_name": "Llama"},
    )[0]
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return extract_prompt_scores_from_input_ids(
        model=model,
        input_ids=inputs.input_ids,
        row_position=row_position,
        dataset_index=dataset_index,
        tail_tokens=tail_tokens,
        component=component,
    )


def extract_prompt_scores_from_input_ids(
    *,
    model,
    input_ids: torch.Tensor,
    row_position: int,
    dataset_index: int,
    tail_tokens: int = TAIL_TOKENS,
    component: str = "antisymmetric",
) -> tuple[torch.Tensor, int]:
    """Run the established score extraction on preformatted exact input IDs."""

    if component not in ("antisymmetric", "symmetric"):
        raise ValueError(
            "component must be 'antisymmetric' or 'symmetric', "
            f"found {component!r}"
        )

    from transformers.models.llama import modeling_llama

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise TypeError("input_ids must be a rank-2 torch tensor")
    if input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
        raise ValueError("score extraction requires one unpadded torch.long input")
    input_ids = input_ids.to(model.device)
    sequence_length = int(input_ids.shape[-1])
    if sequence_length < tail_tokens:
        raise ValueError(
            "sequence shorter than tail window: "
            f"row_position={row_position}, dataset_index={dataset_index}, "
            f"sequence_length={sequence_length}, tail_tokens={tail_tokens}"
        )
    captured = _capture_layer_inputs(model, input_ids)
    lower = strict_lower_indices(tail_tokens, device=model.device)
    destination = torch.empty(
        (NUM_LAYERS, NUM_HEADS, lower.shape[1]),
        dtype=torch.float32,
        device="cpu",
    )

    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        if type(attention) is not modeling_llama.LlamaAttention:
            raise TypeError(
                f"expected eager LlamaAttention at layer={layer_index}, "
                f"found={type(attention).__name__}"
            )
        if attention.config.pretraining_tp != 1:
            raise ValueError("score extraction currently requires pretraining_tp == 1")
        hidden_states = captured[layer_index]["hidden_states"]
        position_ids = captured[layer_index]["position_ids"]
        batch_size, query_length, _ = hidden_states.shape
        if batch_size != 1 or query_length != sequence_length:
            raise ValueError("score extraction requires unpadded batch size 1")

        with torch.inference_mode():
            query_states = attention.q_proj(hidden_states)
            key_states = attention.k_proj(hidden_states)
            value_states = attention.v_proj(hidden_states)
            query_states = query_states.view(
                batch_size, query_length, attention.num_heads, attention.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                batch_size,
                query_length,
                attention.num_key_value_heads,
                attention.head_dim,
            ).transpose(1, 2)
            value_states = value_states.view(
                batch_size,
                query_length,
                attention.num_key_value_heads,
                attention.head_dim,
            ).transpose(1, 2)
            cos, sin = attention.rotary_emb(value_states, position_ids)
            query_states, key_states = modeling_llama.apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )
            key_states = modeling_llama.repeat_kv(
                key_states,
                attention.num_key_value_groups,
            )
            attention_scores = torch.matmul(
                query_states,
                key_states.transpose(2, 3),
            ) / math.sqrt(attention.head_dim)
            expected_shape = (1, NUM_HEADS, sequence_length, sequence_length)
            if tuple(attention_scores.shape) != expected_shape:
                raise ValueError(
                    f"attention score shape mismatch at layer={layer_index}: "
                    f"{tuple(attention_scores.shape)} != {expected_shape}"
                )
            if attention_scores.dtype != torch.float16:
                raise ValueError(
                    f"attention score dtype mismatch at layer={layer_index}: "
                    f"{attention_scores.dtype} != torch.float16"
                )
            score_tail = attention_scores[
                :, :, -tail_tokens:, -tail_tokens:
            ]
            transpose = score_tail.transpose(-1, -2)
            component_tail = (
                (score_tail - transpose) / 2
                if component == "antisymmetric"
                else (score_tail + transpose) / 2
            )
            lower_values = component_tail[:, :, lower[0], lower[1]]
            destination[layer_index].copy_(
                lower_values[0].to(device="cpu", dtype=torch.float32)
            )
        del captured[layer_index]
    validate_score_tensor(destination.unsqueeze(0), prompt_count=1)
    return destination, sequence_length


def extract_selection_scores(
    *,
    model,
    tokenizer,
    original_rows: Sequence[Mapping[str, object]],
    jailbreak_rows: Sequence[Mapping[str, object]],
    selected_records: Sequence[Mapping[str, object]],
    progress: Callable[[str], None] | None = None,
    component: str = "antisymmetric",
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    if len(selected_records) != SELECTION_COUNT:
        raise ValueError("selection score extraction requires exactly 20 records")
    original_values = torch.empty(
        (SELECTION_COUNT, NUM_LAYERS, NUM_HEADS, ELEMENTS_PER_HEAD),
        dtype=torch.float32,
    )
    jailbreak_values = torch.empty_like(original_values)
    lengths = []
    for selection_position, record in enumerate(selected_records):
        row_position = int(record["row_position"])
        dataset_index = int(record["dataset_index"])
        original_row = original_rows[row_position]
        jailbreak_row = jailbreak_rows[row_position]
        if (
            original_row.get("index") != dataset_index
            or jailbreak_row.get("index") != dataset_index
        ):
            raise ValueError(
                f"selection input alignment mismatch at dataset_index={dataset_index}"
            )
        if sha256_text(original_row["prompt"]) != record["original_prompt_sha256"]:
            raise ValueError(f"original prompt hash mismatch at dataset_index={dataset_index}")
        if sha256_text(jailbreak_row["prompt"]) != record["jailbreak_prompt_sha256"]:
            raise ValueError(
                f"jailbreak prompt hash mismatch at dataset_index={dataset_index}"
            )
        if progress is not None:
            progress(
                f"[{selection_position + 1}/{SELECTION_COUNT}] "
                f"row_position={row_position} dataset_index={dataset_index}"
            )
        original_score, original_length = extract_prompt_scores(
            model=model,
            tokenizer=tokenizer,
            prompt=original_row["prompt"],
            row_position=row_position,
            dataset_index=dataset_index,
            component=component,
        )
        jailbreak_score, jailbreak_length = extract_prompt_scores(
            model=model,
            tokenizer=tokenizer,
            prompt=jailbreak_row["prompt"],
            row_position=row_position,
            dataset_index=dataset_index,
            component=component,
        )
        original_values[selection_position].copy_(original_score)
        jailbreak_values[selection_position].copy_(jailbreak_score)
        lengths.append(
            {
                "selection_position": selection_position,
                "row_position": row_position,
                "dataset_index": dataset_index,
                "original_sequence_length": original_length,
                "jailbreak_sequence_length": jailbreak_length,
            }
        )
    validate_score_tensor(original_values)
    validate_score_tensor(jailbreak_values)
    return original_values, jailbreak_values, lengths


def llama_model_config_summary(model) -> dict:
    config = model.config
    return {
        "model_type": config.model_type,
        "architectures": list(config.architectures or []),
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.hidden_size // config.num_attention_heads),
        "pretraining_tp": int(config.pretraining_tp),
        "max_position_embeddings": int(config.max_position_embeddings),
        "rope_theta": float(config.rope_theta),
    }
