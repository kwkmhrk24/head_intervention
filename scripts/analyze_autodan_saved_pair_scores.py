#!/usr/bin/env python3
"""Describe saved Strong pilot labels using forward-only Full Score capture."""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.artifact_io import read_json, read_jsonl, sha256_file, utc_now
from utils.gcg_shuffled_common import SCORES, write_csv, write_json
from utils.prompt_utils import apply_template

SOURCE = ROOT/"results/harmbench_autodan_old_strong_pilot_recheck/trajectory_run0_current_baseline.jsonl"
SOURCE_SUMMARY = SOURCE.with_name("trajectory_run0_current_baseline_summary.json")
KEYS = ("success", "failure")


def tokenizer_for(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, local_files_only=True)


def prepare(out):
    source = read_json(SOURCE_SUMMARY)
    prior = read_json(SCORES/"plan.json")
    model = Path(source["model_path"])
    if model != Path(prior["model"]) or source["transformers_version"] != prior["software_versions"]["transformers"]:
        raise ValueError("saved model path or Transformers version differs from Score reference")
    revision = subprocess.check_output(["git", "-c", f"safe.directory={model}", "-C", str(model),
                                        "rev-parse", "HEAD"], text=True).strip()
    if revision != prior["model_revision"]:
        raise ValueError("model revision differs from Score reference")
    tok = tokenizer_for(model)
    template_hash = hashlib.sha256(tok.chat_template.encode()).hexdigest()
    rows = [(line, r) for line, r in enumerate(read_jsonl(SOURCE), 1) if r["raw_index"] == 15]
    from utils.evaluate_utils import check_success
    candidates = []
    for line, r in rows:
        if (r["behavior_id"] != "advbench_16" or r["harmbench_run_index"] != 0
                or r["chat_template_sha256"] != template_hash or r["generation_failed"]
                or r["decoding_config"] != source["decoding_config"]
                or check_success(r["response"]) != r["keyword_success"]):
            raise ValueError("saved evaluation metadata mismatch")
        body, goal = r["attack_prompt"], r["goal"]
        if not body.endswith(goal) or body.count(goal) != 1:
            raise ValueError("Original must be the unique body suffix")
        text = apply_template([body], None, tok, {"model_name":"Llama"})[0]
        offset = text.index(body)
        start = offset + len(body) - len(goal)
        stop = start + len(goal)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc.offset_mapping
        # Only wholly contained attack-prefix tokens, excluding boundary-spanning tokens.
        attack = [i for i,(a,b) in enumerate(offsets) if offset <= a < b <= start]
        original = [i for i,(a,b) in enumerate(offsets) if a < stop and start < b and a < b]
        if original != list(range(original[0], original[-1]+1)):
            raise ValueError("noncontiguous Original Token range")
        if attack and attack != list(range(attack[0], attack[-1]+1)):
            raise ValueError("noncontiguous attack Token range")
        candidates.append(dict(
            source_line=line, candidate_id=r["attack_prompt_sha256"], step=r["harmbench_step_index"],
            run=0, original_id=15, success=r["keyword_success"], response_sha256=r["response_sha256"],
            response_token_count=r["response_length_tokens"], matched_refusals=r["matched_refusal_substrings"],
            input_ids=enc.input_ids, length=len(enc.input_ids),
            original_span=[original[0],original[-1]+1],
            original_ids=[enc.input_ids[i] for i in original],
            original_boundary_crossing=any(offsets[i][0]<start or offsets[i][1]>stop for i in original),
            attack_span=[attack[0],attack[-1]+1] if attack else None,
        ))
    if len({r["goal"] for _,r in rows}) != 1 or [sum(r["success"] == v for r in candidates) for v in (True,False)] != [7,11]:
        raise ValueError("unexpected Original or labels")
    ranked = sorted((abs(a["length"]-b["length"]), abs(a["step"]-b["step"]), a["step"], b["step"], a,b)
                    for a in candidates if a["success"] for b in candidates if not b["success"]
                    and a["original_ids"] == b["original_ids"])
    _, _, _, _, success, failure = ranked[0]
    selected = dict(zip(KEYS, (success, failure)))
    if any(not r["attack_span"] for r in selected.values()):
        raise ValueError("selected attack region is empty")
    required = sum(32*32*r["length"]**2*4 for r in selected.values())
    if shutil.disk_usage(out.parent).free < 2*required:
        raise ValueError("insufficient disk for captures plus margin")
    files = [SOURCE, SOURCE_SUMMARY, SCORES/"plan.json", SCORES/"model_weights.json",
             Path(__file__).resolve(), ROOT/"utils/full_attention_scores.py",
             model/"tokenizer_config.json", model/"tokenizer.json"]
    out.mkdir(exist_ok=False)
    write_json(out/"plan.json", dict(
        created_at=utc_now(), model=str(model), revision=revision, template_sha256=template_hash,
        source=str(SOURCE), source_summary=str(SOURCE_SUMMARY),
        input_hashes={str(p):sha256_file(p) for p in files},
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        selected=selected, ranked_match_count=len(ranked), candidate_count=len(candidates),
        selection_order="identical Original Token IDs, absolute input length difference, absolute step difference, step order",
        response_storage="exact response remains in the linked immutable source JSONL, keyed by source_line and hash",
        score_payload_bytes=required,
    ))
    print({k:(r["step"],r["length"],r["original_span"],r["attack_span"]) for k,r in selected.items()},flush=True)


def capture(out):
    import torch
    import transformers
    from scripts.evaluate_gcg_first_token_reference_ablation import load_model_and_tokenizer
    from utils.full_attention_scores import extract_to_npy
    plan = read_json(out/"plan.json")
    if any(sha256_file(p) != digest for p,digest in plan["input_hashes"].items()):
        raise ValueError("frozen inputs or code changed")
    prior = read_json(SCORES/"plan.json")
    if transformers.__version__ != prior["software_versions"]["transformers"] or torch.__version__ != prior["software_versions"]["torch"]:
        raise ValueError("Score extraction software differs from reference")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20*1024**3:
        raise ValueError("preflight CUDA / free VRAM failed")
    path = Path(plan["model"])
    provenance = read_json(SCORES/"model_weights.json")
    print("Verifying model bytes",flush=True)
    for filename, expected in {"model.safetensors.index.json":provenance["index_sha256"],
                               **provenance["shard_sha256"]}.items():
        if sha256_file(path/filename) != expected:
            raise ValueError("local weight bytes differ from previous extraction")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    write_json(out/"runtime.json",dict(gpu=torch.cuda.get_device_name(), gpu_memory_bytes=list(torch.cuda.mem_get_info()),
                     cuda=torch.version.cuda,torch=torch.__version__,transformers=transformers.__version__,
                     weight_hashes=provenance,cuda_access="sandbox exempt after normal CUDA unavailable",
                     allow_tf32=False, generate_responses=False))
    model, _ = load_model_and_tokenizer(path)
    for label, row in plan["selected"].items():
        print(f"Forward-only Score capture: {label}, L={row['length']}",flush=True)
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=model.device)
        target = out/f"{label}.npy"
        metadata = extract_to_npy(model.model,ids,target,validate_native=True)
        if metadata["position_ids"] != [list(range(row["length"]))]:
            raise ValueError("unexpected actual position IDs")
        write_json(out/f"{label}_score.json",dict(**metadata, sha256=sha256_file(target)))


def metrics(block):
    s = np.asarray(block, dtype=np.float64)
    length = s.shape[0]
    sym, asym = (s+s.T)/2, (s-s.T)/2
    energy = float(np.vdot(s,s))
    parts = [float(np.vdot(x,x)) for x in (sym,asym)]
    np.testing.assert_allclose(sum(parts), energy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sym+asym,s,rtol=0,atol=1e-12)
    negative = float(np.square(s[s<0]).sum())
    positive = float(np.square(s[s>0]).sum())
    np.testing.assert_allclose(positive+negative,energy,rtol=1e-12,atol=1e-12)
    norm = np.sqrt(energy)
    result = dict(signed_mean=s.mean(), rms=norm/length, fro_norm=norm,
                  E_pos=positive/s.size, E_neg=negative/s.size, energy_pos_sum=positive, energy_neg_sum=negative)
    for name, part in zip(("sym","asym"),parts):
        component = np.sqrt(part)
        result.update({f"norm_{name}":component, f"N_{name}":component/length,
                       f"R_{name}":component/norm if norm else np.nan})
    result["energy_ratio_asym"] = parts[1]/energy if energy else np.nan
    return result


def analyze(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan = read_json(out/"plan.json")
    if any(sha256_file(p) != digest for p,digest in plan["input_hashes"].items()):
        raise ValueError("frozen inputs changed")
    records = []
    for condition in KEYS:
        file = out/f"{condition}.npy"
        meta = read_json(out/f"{condition}_score.json")
        if sha256_file(file) != meta["sha256"]:
            raise ValueError("Score hash mismatch")
        score = np.load(file,mmap_mode="r",allow_pickle=False)
        l = plan["selected"][condition]["length"]
        if score.shape != (32,32,l,l) or score.dtype != np.float32:
            raise ValueError("unexpected saved Score")
        a,b = plan["selected"][condition]["attack_span"]
        for layer in range(32):
            for head in range(32):
                block = score[layer,head]
                for region,data in (("full",block),("attack",block[a:b,a:b])):
                    records.append(dict(condition=condition,region=region,layer=layer,head=head,
                                        length=len(data),**metrics(data)))
            if layer % 8 == 7:
                print(condition, "analyzed through layer", layer, flush=True)
    table = pd.DataFrame(records).set_index(["condition","region","layer","head"]).sort_index()
    s,f = (table.loc[c] for c in KEYS)
    if not s.index.equals(f.index):
        raise ValueError("Head correspondence mismatch")
    delta = s.drop(columns="length")-f.drop(columns="length")
    for part in ("sym","asym"):
        col = f"N_{part}"
        delta[f"P_{part}"] = delta[col].div(f[col].replace(0,np.nan))*100
    write_csv(out/"score_metrics.csv",table.reset_index())
    write_csv(out/"head_deltas.csv",delta.reset_index())
    summary = {}
    for region, d in delta.groupby(level="region"):
        counts = {}
        for name in ("R","N","P","norm"):
            a,b = d[f"{name}_sym"],d[f"{name}_asym"]
            counts[name] = dict(asym_dominant=int((b.abs()>a.abs()+1e-12).sum()),
                                sym_dominant=int((a.abs()>b.abs()+1e-12).sum()),
                                missing=int((a.isna()|b.isna()).sum()),
                                same_sign=int((a*b>0).sum()), opposite_sign=int((a*b<0).sum()))
        summary[region] = dict(mean_delta=d.mean().to_dict(), dominance=counts,
                               success_mean=s.loc[region].mean().to_dict(),
                               failure_mean=f.loc[region].mean().to_dict())
        cols = ["signed_mean","rms","E_pos","E_neg","R_sym","R_asym","N_sym","N_asym","P_sym","P_asym"]
        fig,axes = plt.subplots(2,5,figsize=(15,6),layout="constrained")
        for ax,col in zip(axes.flat,cols):
            values = d[col].to_numpy().reshape(32,32)
            limit = np.nanmax(abs(d[[col[:1]+"_sym",col[:1]+"_asym"]].to_numpy())) if col[:1] in ("R","N","P") else np.nanmax(abs(values))
            im = ax.imshow(values,cmap="RdBu_r",vmin=-limit,vmax=limit,interpolation="nearest")
            ax.set(title=f"Δ {col}",xlabel="Head",ylabel="Layer")
            fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(f"AutoDAN ID 15 | success − failure | {region}")
        fig.savefig(out/f"{region}_heatmaps.png",dpi=180)
        plt.close(fig)
    write_json(out/"summary.json",summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","capture","analyze"))
    parser.add_argument("--output-dir",required=True,type=Path)
    args = parser.parse_args()
    {"prepare":prepare,"capture":capture,"analyze":analyze}[args.stage](args.output_dir.resolve())
