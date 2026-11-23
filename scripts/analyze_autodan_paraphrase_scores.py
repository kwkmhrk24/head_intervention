"""Capture and summarize Full Scores for the fixed 12 AutoDAN paraphrase pairs."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_autodan_id15_controls import edit_distance
from scripts.analyze_autodan_saved_pair_scores import metrics
from scripts.audit_autodan_original_extension import BASE, DERIVED, FAST, MODEL, locate
from scripts.build_autodan_fastchat_pairs import AUDIT, TOKENIZER_FILES
from scripts.evaluate_gcg_position_element_topk_intervention import format_autodan_records
from utils.artifact_io import read_json, read_jsonl, sha256_file, sha256_text, utc_now, write_json, write_jsonl
from utils.full_attention_scores import SCORE_METHOD, extract_to_npy
from utils.gcg_shuffled_common import SCORES, write_csv

IDS = (4, 20, 28, 30, 33, 36, 57, 82, 89, 106, 112, 114)
CONDITIONS = ("baseline", "paraphrase")
OUTPUT = BASE / "paraphrase_score_pairs12"
RAW = ROOT / "data/raw/llama2"
FORMAT = "official_fastchat_34062e9"
SHAPE = (32, 32)


def require(value, message):
    if not value:
        raise ValueError(message)


def git(path, *args):
    return subprocess.check_output(["git", "-c", f"safe.directory={path}", "-C",
                                    str(path), *args], text=True).strip()


def prepare(out):
    from transformers import AutoTokenizer, __version__

    raw_path = RAW / "llama2_AutoDAN.json"
    para_path = RAW / "llama2_AutoDAN_paraphrase_gpt4omini.jsonl"
    original_path = DERIVED / "original_prompts.jsonl"
    paired_path = FAST / "paired_results.jsonl"
    reference_path = FAST / "run_manifest.json"
    reference = read_json(reference_path)
    weights_path = SCORES / "model_weights.json"
    previous = read_json(SCORES / "plan.json")
    model_weights = read_json(weights_path)
    revision = git(MODEL, "rev-parse", "HEAD")
    require(revision == reference["model_repository_head"] == previous["model_revision"], "model revision mismatch")
    require(__version__ == reference["transformers_version"] == previous["software_versions"]["transformers"],
            "Transformers version mismatch")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    require(type(tok).__name__ == reference["tokenizer_class"] and tok.is_fast, "tokenizer mismatch")
    tokenizer_files = [MODEL / name for name in TOKENIZER_FILES]
    known_files = read_json(AUDIT / "manifest.json")["inputs"]
    require(all(sha256_file(p) == known_files[str(p.resolve())] for p in tokenizer_files),
            "tokenizer files differ from saved audit")
    raw = json.loads(raw_path.read_text())
    paras = read_jsonl(para_path)
    originals = read_jsonl(original_path)
    historical = read_jsonl(paired_path)
    require(len(raw) == len(paras) == len(originals) == len(historical) == 130, "unexpected source counts")
    para_by_text = {r["jailbreak_prompt"]: r["paraphrased_jailbreak_prompt"] for r in paras}
    require(len(para_by_text) == 130 and set(para_by_text) == {r["user prompt"] for r in raw},
            "paraphrase source text alignment failed")
    rows = []
    for i in IDS:
        r, h = raw[i], historical[i]
        old, orig = r["user prompt"], originals[i]["prompt"]
        require(r["index"] == h["dataset_index"] == i and old == h["prompt"], f"legacy key mismatch {i}")
        records = format_autodan_records(
            [{"dataset_index": i, "prompt": text} for text in (old, para_by_text[old])], tok, prompt_format=FORMAT)
        require(records[0]["formatted_prompt"] == h["official_formatted_prompt"] and
                records[0]["input_ids"][0].tolist() == h["official_input_ids"], f"historical FastChat mismatch {i}")
        for condition, record in zip(CONDITIONS, records, strict=True):
            ids, span, _, retention, boundary = locate(
                tok, record["formatted_prompt"], record["prompt"], orig, True)
            require(ids == record["input_ids"][0].tolist(), f"offset tokenization mismatch {i}/{condition}")
            rows.append({
                "original_id": i, "condition": condition, "user_label": "success" if condition == "baseline" else "failure",
                "label_source": "user manual confirmation (2026-11-23 request)",
                "label_evaluation_model_template_rule": "unknown",
                "prompt_sha256": sha256_text(record["prompt"]), "input_token_ids": ids,
                "input_token_ids_sha256": sha256_text(json.dumps(ids, separators=(",", ":"))),
                "input_length": len(ids), "original_retention": retention, "original_token_span": span,
                "original_token_ids": ids[slice(*span)] if span else None, "original_boundary_crossing": boundary,
                "historical_baseline_input_ids_exact": condition == "baseline" or None,
                "score_path": f"{condition}/id_{i:03d}.npy",
            })
        a, b = rows[-2:]
        diff = {"token_length_delta": b["input_length"] - a["input_length"],
                "token_edit_distance": edit_distance(a["input_token_ids"], b["input_token_ids"])}
        a.update(diff)
        b.update(diff)
    files = [raw_path, para_path, original_path, paired_path, reference_path, *tokenizer_files,
             Path(__file__).resolve(), ROOT/"utils/full_attention_scores.py",
             ROOT/"scripts/analyze_autodan_saved_pair_scores.py",
             ROOT/"scripts/evaluate_gcg_position_element_topk_intervention.py",
             ROOT/"scripts/analyze_attention_scores.py", weights_path, SCORES/"plan.json"]
    estimated = sum(4 * 32 * 32 * r["input_length"]**2 for r in rows)
    require(shutil.disk_usage(out.parent).free > 2 * estimated, "insufficient disk space")
    out.mkdir(exist_ok=False)
    for d in (*CONDITIONS, "metrics"):
        (out / d).mkdir()
    write_json(out/"plan.json", {
        "created_at": utc_now(), "target_ids": list(IDS), "git_head": git(ROOT, "rev-parse", "HEAD"),
        "branch": git(ROOT, "branch", "--show-current"), "model_revision": revision, "model": str(MODEL),
        "tokenizer": type(tok).__name__, "fastchat_format": FORMAT, "model_weights": model_weights,
        "fastchat_conditions": reference["condition_b_format"], "score": SCORE_METHOD,
        "expected_software_versions": previous["software_versions"], "estimated_payload_bytes": estimated,
        "input_hashes": {str(p): sha256_file(p) for p in files},
    })
    write_jsonl(out/"frozen_inputs.jsonl", rows)
    for a,b in zip(rows[::2], rows[1::2], strict=True):
        print(a["original_id"], a["input_length"], b["input_length"], a["token_edit_distance"],
              a["original_retention"], b["original_retention"], flush=True)


def measure_scores(path, length):
    scores = np.load(path, mmap_mode="r", allow_pickle=False)
    require(scores.dtype == np.float32 and scores.shape == (*SHAPE, length, length), "saved score shape or dtype mismatch")
    require(np.isfinite(scores).all(), "saved score is nonfinite")
    results = []
    for layer, block in enumerate(scores):
        for head, matrix in enumerate(block):
            v = metrics(matrix)  # Checks reconstruction and both energy decompositions in FP64.
            energy = v["fro_norm"]**2
            require(energy > 0, "zero score energy cannot be normalized")
            results.append({"layer": layer, "head": head, "N_full": v["rms"],
                            "positive_energy_fraction": v["energy_pos_sum"]/energy,
                            "negative_energy_fraction": v["energy_neg_sum"]/energy, **v})
    return results


def capture(out):
    import torch
    import transformers
    from scripts.evaluate_gcg_first_token_reference_ablation import load_model_and_tokenizer

    plan = read_json(out/"plan.json")
    inputs = read_jsonl(out/"frozen_inputs.jsonl")
    require(len(inputs) == 24 and [(r["original_id"], r["condition"]) for r in inputs] ==
            [(i,c) for i in IDS for c in CONDITIONS], "frozen input keys mismatch")
    require(all(sha256_file(p) == digest for p,digest in plan["input_hashes"].items()), "frozen source changed")
    require(all(sha256_text(json.dumps(r["input_token_ids"], separators=(",", ":"))) ==
                r["input_token_ids_sha256"] for r in inputs), "frozen Token ID hash mismatch")
    require(git(MODEL, "rev-parse", "HEAD") == plan["model_revision"], "local model revision changed")
    software = plan["expected_software_versions"]
    require(torch.__version__ == software["torch"] and transformers.__version__ == software["transformers"],
            "runtime versions differ from previous Full Score extraction")
    require(torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 20*1024**3, "CUDA or available VRAM")
    model_weights = plan["model_weights"]
    print("Verifying model weights", flush=True)
    for name, digest in {"model.safetensors.index.json": model_weights["index_sha256"],
                         **model_weights["shard_sha256"]}.items():
        require(sha256_file(MODEL/name) == digest, f"model weights differ: {name}")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    write_json(out/"runtime.json", {"torch":torch.__version__, "transformers":transformers.__version__,
               "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"memory_bytes":list(torch.cuda.mem_get_info()),
               "execution_context":"sandbox exempt after CUDA preflight in normal sandbox failed",
               "eval":True, "use_cache":False,"allow_tf32":False,"generation":False,"lm_head_called":False})
    model, _ = load_model_and_tokenizer(MODEL)
    manifest, per_head = [], []
    for row in inputs:
        i,c,L = row["original_id"],row["condition"],row["input_length"]
        dst = out/row["score_path"]
        require(not dst.exists(), f"existing score {dst}")
        with TemporaryDirectory(dir=out, prefix=".capture-") as temp:
            path = Path(temp)/"scores.npy"
            ids = torch.tensor([row["input_token_ids"]],dtype=torch.long,device=model.device)
            capture_info = extract_to_npy(model.model, ids, path, validate_native=True)
            require(capture_info["position_ids"] == [list(range(L))], "native position order mismatch")
            require(len(capture_info["native_qk_reference_checks"]) == 32, "missing native Q/K check")
            measures = measure_scores(path,L)
            path.rename(dst)
        write_json(dst.with_suffix(".json"), capture_info)
        manifest.append({k:v for k,v in row.items() if k != "input_token_ids"} | {
            "model_revision":plan["model_revision"],"score_sha256":sha256_file(dst),
            "shape":capture_info["shape"],"dtype":capture_info["dtype"],"score_definition":SCORE_METHOD,
            "mask_applied_to_saved_score":False,"softmax_applied_to_saved_score":False,
        })
        per_head.extend({"original_id":i,"condition":c,**v} for v in measures)
        print(f"Verified {i} {c}: {capture_info['shape']}",flush=True)
        if len(manifest) == 2:
            unchanged = all(sha256_file(p) == digest for p, digest in plan["input_hashes"].items())
            require(unchanged, "source changed during smoke test")
            write_json(out/"smoke_validation.json", {"original_id":4,"passed":True,
                "both_inputs_finite_fp32":True,"energy_and_reconstruction_checked":True,
                "source_hashes_unchanged":unchanged,
                "model_generation":False})
    write_jsonl(out/"manifest.jsonl",manifest)
    save_metrics(out, plan, manifest, per_head)


def save_metrics(out, plan, manifest, per_head):
    frame = pd.DataFrame(per_head)
    require(len(frame) == 24*1024 and not frame.duplicated(
        ["original_id","condition","layer","head"]).any(), "incomplete per-Head metrics")
    write_csv(out/"metrics/per_head.csv", frame)
    indexed = frame.set_index(["condition","original_id","layer","head"])
    delta = (indexed.loc["baseline"] - indexed.loc["paraphrase"]).add_prefix("delta_")
    delta = delta.reset_index()
    write_csv(out/"metrics/paired_per_head.csv", delta)
    pairs = delta.groupby("original_id",as_index=False).mean(numeric_only=True).drop(columns=["layer","head"])
    for key in ("input_length","token_length_delta","token_edit_distance"):
        for c in CONDITIONS if key == "input_length" else ("baseline",):
            values = {r["original_id"]:r[key] for r in manifest if r["condition"] == c}
            pairs[f"{c}_{key}" if key == "input_length" else key] = pairs["original_id"].map(values)
    write_csv(out/"metrics/paired_summary.csv", pairs)
    layers = delta.groupby(["original_id","layer"],as_index=False).mean(numeric_only=True).drop(columns="head")
    write_csv(out/"metrics/pair_layer_summary.csv", layers)
    layer_means = layers.groupby("layer",as_index=False).mean(numeric_only=True).drop(columns="original_id")
    write_csv(out/"metrics/layer_summary.csv", layer_means)
    source_unchanged = all(sha256_file(p)==d for p,d in plan["input_hashes"].items())
    require(source_unchanged, "source changed during run")
    write_json(out/"validation.json",{
        "inputs":len(manifest),"pairs":len(pairs),"score_files":len(manifest),"per_head_rows":len(frame),
        "paired_head_rows":len(delta),"baseline_historical_input_ids_exact":12,
        "all_scores_finite_fp32":True,"reconstruction_and_energy_verified_heads":len(frame),
        "maximum_ratio_identity_error":float((frame["R_sym"]**2+frame["R_asym"]**2-1).abs().max()),
        "maximum_sign_energy_identity_error":float((frame["positive_energy_fraction"]+frame["negative_energy_fraction"]-1).abs().max()),
        "source_hashes_unchanged":source_unchanged,"generate_called":False,"lm_head_called":False,
        "saved_score_bytes":sum((out/r["score_path"]).stat().st_size for r in manifest),
    })
    print("Metrics and validation saved.",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","capture"))
    args = parser.parse_args()
    {"prepare":prepare,"capture":capture}[args.stage](OUTPUT)


if __name__ == "__main__":
    main()
