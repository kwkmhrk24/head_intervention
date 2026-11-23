#!/usr/bin/env python3
"""Fixed-candidate descriptive SF/SS/FF score comparison for legacy ID 15."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations, permutations
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_autodan_saved_pair_scores import SOURCE, SOURCE_SUMMARY, tokenizer_for
from utils.artifact_io import read_json, read_jsonl, sha256_file, utc_now
from utils.gcg_shuffled_common import SCORES, write_csv, write_json
from utils.prompt_utils import apply_template
from utils.evaluate_utils import check_success

PRIOR=ROOT/"data/outputs/llama2/autodan/pair_score_id15_20261123"
REGIONS=("full","attack")
CATEGORIES=("SF","SS","FF")
KEYS=["region","layer","head"]


def edit_distance(a,b):
    """Unit-cost token Levenshtein distance using a bit-parallel DP column."""
    mask={}
    for i,t in enumerate(a):
        mask[t]=mask.get(t,0)|(1<<i)
    vp,vn,distance=-1,0,len(a)
    if not a:
        return len(b)
    high=1<<(len(a)-1)
    for token in b:
        eq=mask.get(token,0)
        x=eq|vn
        d0=(((x&vp)+vp)^vp)|x
        hp=vn|~(d0|vp)
        hn=vp&d0
        distance+=bool(hp&high)-bool(hn&high)
        hp=(hp<<1)|1
        hn<<=1
        vp=hn|~(d0|hp)
        vn=hp&d0
    return distance


def candidates():
    source=read_json(SOURCE_SUMMARY)
    t=tokenizer_for(source["model_path"])
    template_hash=hashlib.sha256(t.chat_template.encode()).hexdigest()
    goal_text=set()
    result={}
    for line,r in enumerate(read_jsonl(SOURCE),1):
        if r["raw_index"]!=15:
            continue
        if (r["behavior_id"]!="advbench_16" or r["harmbench_run_index"]!=0 or r["generation_failed"]
                or r["chat_template_sha256"]!=template_hash or r["decoding_config"]!=source["decoding_config"]
                or check_success(r["response"])!=r["keyword_success"]):
            raise ValueError("saved condition/labels mismatch")
        goal,body=r["goal"],r["attack_prompt"]
        goal_text.add(goal)
        if not body.endswith(goal) or body.count(goal)!=1:
            raise ValueError("Original is not unique suffix")
        text=apply_template([body],None,t,{"model_name":"Llama"})[0]
        offset=text.index(body)
        start=offset+len(body)-len(goal)
        end=start+len(goal)
        enc=t(text,add_special_tokens=False,return_offsets_mapping=True)
        original=[i for i,(a,b) in enumerate(enc.offset_mapping) if a<end and start<b and a<b]
        attack=[i for i,(a,b) in enumerate(enc.offset_mapping) if offset<=a<b<=start]
        for ix in (original,attack):
            if ix and ix!=list(range(ix[0],ix[-1]+1)):
                raise ValueError("noncontiguous Token range")
        step=r["harmbench_step_index"]
        if step in result:
            raise ValueError("duplicate step")
        result[step]=dict(step=step,success=r["keyword_success"],source_line=line,
                          candidate_id=r["attack_prompt_sha256"],response_sha256=r["response_sha256"],
                          input_ids=enc.input_ids,length=len(enc.input_ids),
                          original_span=[original[0],original[-1]+1],original_ids=[enc.input_ids[i] for i in original],
                          attack_span=[attack[0],attack[-1]+1] if attack else None)
    if (len(goal_text)!=1 or len({r["candidate_id"] for r in result.values()})!=18
            or Counter(r["success"] for r in result.values())!={True:7,False:11}):
        raise ValueError("unexpected corpus")
    return result, template_hash


def pair_rows(rows):
    anchor=rows[5]["original_ids"]
    pairs=[]
    for i,j in combinations(sorted(rows),2):
        a,b=rows[i],rows[j]
        group="SF" if a["success"]!=b["success"] else "SS" if a["success"] else "FF"
        lhs,rhs=(a,b) if group=="SF" and a["success"] else (b,a)
        length=abs(lhs["length"]-rhs["length"])
        edit=edit_distance(lhs["input_ids"],rhs["input_ids"])
        original_equal=lhs["original_ids"]==rhs["original_ids"]
        pairs.append(dict(group=group,pair=f"{lhs['step']}-{rhs['step']}",lhs=lhs["step"],rhs=rhs["step"],
                          length_abs=length,length_relative=length/max(lhs["length"],rhs["length"]),
                          step_abs=j-i,edit_distance=edit,edit_relative=edit/max(lhs["length"],rhs["length"]),
                          original_text_equal=True,original_ids_equal=original_equal,
                          original_start_delta=lhs["original_span"][0]-rhs["original_span"][0],
                          original_end_delta=lhs["original_span"][1]-rhs["original_span"][1],
                          anchor_tokens=original_equal and lhs["original_ids"]==anchor,
                          attack_spans_present=bool(lhs["attack_span"] and rhs["attack_span"])))
    return pairs


def select(pairs):
    # Select a balanced number of matched triples using metadata only.
    near={g:[p for p in pairs if p["group"]==g and p["anchor_tokens"] and p["attack_spans_present"]
             and p["length_abs"]<=25 and p["step_abs"]<=3] for g in CATEGORIES}
    anchor=next(p for p in near["SF"] if p["pair"]=="5-4")
    count=min(3,*(len(v) for v in near.values()))
    if count<3:
        raise ValueError("insufficient comparable pairs")
    def cost(triples):
        keys=("length_abs","step_abs","edit_relative")
        normalizers=(25.,3.,1.)
        return sum((max(t[k] for t in triple)-min(t[k] for t in triple))/s
                   for triple in triples for k,s in zip(keys,normalizers,strict=True)), tuple(
                       p["pair"] for triple in triples for p in triple)
    possible=(tuple(zip((anchor,*sf),ss,ff,strict=True))
              for sf in combinations([p for p in near["SF"] if p is not anchor],count-1)
              for ss in permutations(near["SS"],count) for ff in permutations(near["FF"],count))
    return [dict(p,triple=i) for i, t in enumerate(min(possible,key=cost),1) for p in t], {
        g:len(v) for g,v in near.items()}


def prepare(out):
    rows,template_hash=candidates()
    pairs=pair_rows(rows)
    selected, eligible=select(pairs)
    out.mkdir(exist_ok=False)
    deps=[SOURCE,SOURCE_SUMMARY,SCORES/"plan.json",SCORES/"model_weights.json",
          Path(__file__).resolve(),ROOT/"utils/full_attention_scores.py"]
    write_json(out/"plan.json",dict(
        created_at=utc_now(), original_id=15, run=0, template_sha256=template_hash,
        source_hashes={str(p):sha256_file(p) for p in deps},
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        rules="require anchor Original Token IDs; both attack regions present; length difference <=25; step difference <=3; force 5-4; three triples minimizing sum of within-triple ranges (length/25 + step/3 + normalized Levenshtein), lexicographic pair-ID tie break",
        candidate_count=len(rows), eligible_counts=eligible, selected=selected, candidates=rows))
    write_csv(out/"all_pair_metadata.csv",pd.DataFrame(pairs))
    write_csv(out/"selected_pairs.csv",pd.DataFrame(selected))
    print([(r["group"],r["pair"],r["length_abs"],r["step_abs"],r["edit_distance"]) for r in selected],flush=True)


def reduce_layer(scores, attack_span):
    """Same FP64 element metrics as the earlier pair, bounded by four Heads."""
    import torch
    output={}
    a,b=attack_span
    for region, block in (("full",scores),("attack",scores[:,a:b,a:b])):
        fields=[]
        for raw in block.split(4):
            x=raw.to(torch.float64)
            length=x.shape[-1]
            squared=x.square()
            energy=squared.sum(dim=(-2,-1))
            pos=torch.where(x>0,squared,0).sum(dim=(-2,-1))
            neg=torch.where(x<0,squared,0).sum(dim=(-2,-1))
            transposed=x.transpose(-2,-1)
            sy=((x+transposed)*.5).square().sum(dim=(-2,-1))
            ay=((x-transposed)*.5).square().sum(dim=(-2,-1))
            torch.testing.assert_close(sy+ay,energy,atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(pos+neg,energy,atol=1e-12,rtol=1e-12)
            norm=energy.sqrt()
            fields.append(torch.stack([x.mean(dim=(-2,-1)),norm/length,norm,
                                       pos/length**2,neg/length**2,sy.sqrt()/norm,ay.sqrt()/norm,
                                       sy.sqrt()/length,ay.sqrt()/length],dim=-1).cpu().numpy())
        output[region]=np.concatenate(fields)
    return output


METRICS=["signed_mean","rms","fro_norm","E_pos","E_neg","R_sym","R_asym","N_sym","N_asym"]


def capture(out):
    import torch
    import transformers
    from scripts.evaluate_gcg_first_token_reference_ablation import load_model_and_tokenizer
    from utils.full_attention_scores import extract_to_npy
    plan=read_json(out/"plan.json")
    if any(sha256_file(p)!=v for p,v in plan["source_hashes"].items()):
        raise ValueError("frozen input or code changed")
    prior=read_json(PRIOR/"plan.json")
    refs=read_json(SCORES/"plan.json")
    model_path=Path(prior["model"])
    revision=subprocess.check_output(["git","-c",f"safe.directory={model_path}","-C",str(model_path),
                                      "rev-parse","HEAD"],text=True).strip()
    sp=refs["software_versions"]
    if (revision!=refs["model_revision"] or torch.__version__!=sp["torch"]
            or transformers.__version__!=sp["transformers"]):
        raise ValueError("revision or software differs from the prior extraction")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<24*1024**3:
        raise ValueError("CUDA/free VRAM check failed")
    hashes=read_json(SCORES/"model_weights.json")
    print("Checking model weights",flush=True)
    for name,expected in {"model.safetensors.index.json":hashes["index_sha256"],**hashes["shard_sha256"]}.items():
        if sha256_file(model_path/name)!=expected:
            raise ValueError("model weights differ from the previous run")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    write_json(out/"runtime.json",dict(created_at=utc_now(),revision=revision,weights=hashes,
               gpu=torch.cuda.get_device_name(),memory=torch.cuda.mem_get_info(),torch=torch.__version__,
               transformers=transformers.__version__,allow_tf32=False,generated_responses=False,
               execution="CUDA outside normal sandbox after visibility check"))
    model,_=load_model_and_tokenizer(model_path)
    steps=sorted({pair[k] for pair in plan["selected"] for k in ("lhs","rhs")}-{4,5})
    for step in steps:
        row=plan["candidates"][str(step)]
        print(f"Forward-only step {step} (L={row['length']})",flush=True)
        ids=torch.tensor([row["input_ids"]],dtype=torch.long,device=model.device)
        layers={}
        def consume(index,scores):
            if index in layers:
                raise ValueError("duplicate Score layer")
            layers[index]=reduce_layer(scores,row["attack_span"])
        meta=extract_to_npy(model.model,ids,None,validate_native=True,layer_consumer=consume)
        if (meta["position_ids"]!=[list(range(row["length"]))] or sorted(layers)!=list(range(32))
                or len(meta["native_qk_reference_checks"])!=32):
            raise ValueError("captured positions or layers differ")
        frames=[]
        for region in REGIONS:
            values=np.concatenate([layers[i][region] for i in range(32)])
            if values.shape!=(1024,len(METRICS)) or not np.isfinite(values).all():
                raise ValueError("incomplete or nonfinite metrics")
            frame=pd.DataFrame(values,columns=METRICS)
            frame=frame.assign(region=region,layer=np.repeat(range(32),32),head=np.tile(range(32),32),step=step)
            frames.append(frame)
        path=out/f"step_{step}_metrics.csv"
        write_csv(path,pd.concat(frames,ignore_index=True))
        write_json(out/f"step_{step}_capture.json",dict(**meta,metrics_sha256=sha256_file(path)))
        print(f"Saved validated step {step}",flush=True)


def load_metrics(out, plan):
    prior=read_json(PRIOR/"plan.json")
    ledger=read_json(PRIOR/"output_hashes.json")
    for name in ("plan.json","score_metrics.csv","head_deltas.csv"):
        if sha256_file(PRIOR/name)!=ledger[name]:
            raise ValueError("prior artifact hash mismatch")
    base=pd.read_csv(PRIOR/"score_metrics.csv",float_precision="round_trip")
    frames=[]
    for c,step in (("success",5),("failure",4)):
        row=plan["candidates"][str(step)]
        old=prior["selected"][c]
        if any(row[k]!=old[k] for k in ("input_ids","candidate_id","attack_span","original_ids")):
            raise ValueError("baseline candidate differs")
        frames.append(base[base.condition==c][KEYS+METRICS].assign(step=step))
    steps=sorted({pair[k] for pair in plan["selected"] for k in ("lhs","rhs")}-{4,5})
    for step in steps:
        path=out/f"step_{step}_metrics.csv"
        if sha256_file(path)!=read_json(out/f"step_{step}_capture.json")["metrics_sha256"]:
            raise ValueError("streaming capture checksum mismatch")
        frames.append(pd.read_csv(path,float_precision="round_trip"))
    frame=pd.concat(frames,ignore_index=True).set_index(["step",*KEYS],verify_integrity=True).sort_index()
    expected=pd.MultiIndex.from_product([sorted({4,5,*steps}),sorted(REGIONS),range(32),range(32)],
                                       names=["step",*KEYS])
    if not frame.index.equals(expected):
        raise ValueError("incomplete candidate metric grid")
    np.testing.assert_allclose(frame.R_sym**2+frame.R_asym**2,1,atol=1e-12,rtol=0)
    np.testing.assert_allclose(frame.E_pos+frame.E_neg,frame.rms**2,atol=1e-12,rtol=1e-12)
    return frame


def analyze(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan=read_json(out/"plan.json")
    frame=load_metrics(out,plan)
    delta=[]
    for pair in plan["selected"]:
        a,b=(frame.loc[pair[k],METRICS] for k in ("lhs","rhs"))
        d=a-b
        for part in ("sym","asym"):
            n=f"N_{part}"
            d[f"P_{part}"]=100*d[n]/b[n].replace(0,np.nan)
        delta.append(d.reset_index().assign(group=pair["group"],pair=pair["pair"],triple=pair["triple"]))
    values=pd.concat(delta,ignore_index=True)
    numeric=METRICS+["P_sym","P_asym"]
    anchor=values[values.pair=="5-4"].set_index(KEYS).sort_index()
    saved=pd.read_csv(PRIOR/"head_deltas.csv",float_precision="round_trip").set_index(KEYS).sort_index()
    np.testing.assert_allclose(anchor[numeric],saved[numeric],atol=1e-12,rtol=1e-12)
    pair_summary=[]
    for (region,group,pair), d in values.groupby(["region","group","pair"]):
        row=dict(region=region,group=group,pair=pair)
        for col in numeric:
            a=d[col]
            row.update({f"{col}_mean":a.mean(), f"{col}_median":a.median(),
                        f"{col}_mean_abs":a.abs().mean(),f"{col}_up_fraction":(a>0).mean()})
        for m in ("R","N","P"):
            a,b=d[f"{m}_sym"],d[f"{m}_asym"]
            row[f"{m}_asym_dominant"]=int((b.abs()>a.abs()+1e-12).sum())
            row[f"{m}_sym_dominant"]=int((a.abs()>b.abs()+1e-12).sum())
        pair_summary.append(row)
    per_pair=pd.DataFrame(pair_summary)
    group_heads=values.groupby(["region","group","layer","head"])[numeric].mean()
    group_absolute=values.assign(**{col:values[col].abs() for col in numeric}).groupby(
        ["region","group","layer","head"])[numeric].mean()
    groups=per_pair.groupby(["region","group"])[[k for k in per_pair if k.endswith(("_mean","_median","_mean_abs","_up_fraction","_dominant"))]].median()
    for name,data in (("candidate_metrics",frame.reset_index()),("head_deltas",values),("pair_summary",per_pair),
                      ("group_summary",groups.reset_index()),("group_head_means",group_heads.reset_index()),
                      ("group_head_mean_absolute",group_absolute.reset_index())):
        write_csv(out/f"{name}.csv",data)
    cols=["signed_mean","rms","E_neg","N_sym","N_asym"]
    for region in REGIONS:
        fh,axh=plt.subplots(3,len(cols),figsize=(16,8),layout="constrained")
        fd,axd=plt.subplots(2,len(cols),figsize=(16,6),layout="constrained")
        for i,col in enumerate(cols):
            limit=group_heads.loc[region,col].abs().max()
            for j,g in enumerate(CATEGORIES):
                x=group_heads.loc[(region,g),col].to_numpy().reshape(32,32)
                img=axh[j,i].imshow(x,cmap="RdBu_r",vmin=-limit,vmax=limit,interpolation="nearest")
                axh[j,i].set(title=f"{g} · {col}",xlabel="Head",ylabel="Layer")
            fh.colorbar(img,ax=axh[:,i],shrink=.68)
            for axis, data, title in ((axd[0,i],group_heads,"Signed"),(axd[1,i],group_absolute,"Absolute")):
                axis.boxplot([data.loc[(region,g),col] for g in CATEGORIES],showfliers=False)
                axis.set(xticks=[1,2,3],xticklabels=CATEGORIES,title=f"{title} · {col}")
                axis.axhline(0,color=".6",lw=.6)
        fh.savefig(out/f"{region}_heatmaps.png",dpi=160)
        fd.savefig(out/f"{region}_head_distributions.png",dpi=160)
        plt.close(fh)
        plt.close(fd)
    checks=dict(status="passed",prior_anchor_max_difference=float(abs(anchor[numeric]-saved[numeric]).to_numpy().max()),
                candidate_count=len(frame)//2048,pair_count=len(per_pair)//2,
                ratio_identity_max=float(abs(frame.R_sym**2+frame.R_asym**2-1).max()),
                energy_identity_max=float(abs(frame.E_pos+frame.E_neg-frame.rms**2).max()),
                missing_percent=int(values[["P_sym","P_asym"]].isna().sum().sum()))
    write_json(out/"validation.json",checks)
    print(checks,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","capture","analyze"))
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    {"prepare":prepare,"capture":capture,"analyze":analyze}[a.stage](a.output_dir.resolve())
