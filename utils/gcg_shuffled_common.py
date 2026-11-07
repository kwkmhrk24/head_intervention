"""Shared CPU infrastructure scoped to the legacy GCG/Shuffled phase.

Legacy module names re-export these functions. Numerical operations and ordering
are preserved; CLI/provenance compatibility is audited over all twenty pairs.
"""
from pathlib import Path
from collections import Counter
import json
import warnings
import numpy as np
import pandas as pd
from utils.artifact_io import read_json,read_jsonl,sha256_file
from utils.gcg_shuffled_metrics import TIE_ATOL,TIE_RTOL
ROOT=Path(__file__).resolve().parents[1]
SCORES=ROOT/'data/outputs/full_attention_scores/legacy_selection20_20261107T053240Z'
DATASET=ROOT/'data/outputs/shuffled_gcg_comparison/legacy_selection20_20261106T171044Z/dataset.jsonl'

def write_json(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')


def write_csv(path,frame):
    frame.to_csv(path,index=False,mode='x',float_format='%.17g')


def inspect_inputs():
    extraction_plan=read_json(SCORES/'plan.json')
    validation=read_json(SCORES/'validation_final.json')
    if sha256_file(SCORES/'manifest.jsonl')!=validation['manifest_sha256']:
        raise ValueError('score manifest hash mismatch')
    if sha256_file(DATASET)!=extraction_plan['dataset_sha256']:
        raise ValueError('comparison dataset differs from score extraction source')
    dataset=read_jsonl(DATASET)
    comparison_plan=read_json(DATASET.parent/'plan.json')
    if [r['dataset_index'] for r in dataset]!=comparison_plan['selected_indices'] or len(dataset)!=20:
        raise ValueError('exact selection20 required')
    by_index={r['dataset_index']:r for r in dataset}
    if len(by_index)!=20 or any(i>=220 for i in by_index):
        raise ValueError('duplicate/nonlegacy index')
    primary=[]
    for row in dataset:
        labels={c:row['conditions'][c]['evaluation']['success'] for c in ('original','gcg','shuffled')}
        if any(type(v) is not bool for v in labels.values()):
            raise ValueError('historical automatic labels missing')
        if not labels['original'] and labels['gcg'] and not labels['shuffled']:
            primary.append(row['dataset_index'])
        gcg,shuffled=(row['conditions'][c] for c in ('gcg','shuffled'))
        u0,u1=gcg['token_ranges']['suffix']
        if u1-u0!=20 or shuffled['token_ranges']!=gcg['token_ranges']:
            raise ValueError('paired suffix ranges differ')
        g,s=gcg['input_ids'],shuffled['input_ids']
        if len(g)!=len(s) or g[:u0]!=s[:u0] or g[u1:]!=s[u1:] or Counter(g[u0:u1])!=Counter(s[u0:u1]):
            raise ValueError('paired input token multiset/flanks differ')
    if len(primary)!=18 or sorted(set(by_index)-set(primary))!=[91,123]:
        raise ValueError('predefined primary18/auxiliary20 membership mismatch')
    entries=read_jsonl(SCORES/'manifest.jsonl')
    expected={(i,c) for i in by_index for c in ('original','gcg','shuffled')}
    if len(entries)!=60 or {(e['dataset_index'],e['condition']) for e in entries}!=expected:
        raise ValueError('missing/duplicate input correspondence')
    references=[]
    for entry in entries:
        index,condition=entry['dataset_index'],entry['condition']
        pointer_path=SCORES/entry['completion_file'];pointer=read_json(pointer_path)
        metadata_path=SCORES/pointer['metadata_file']
        if sha256_file(metadata_path)!=pointer['metadata_sha256']:
            raise ValueError('score metadata hash mismatch')
        metadata=read_json(metadata_path);item=metadata['input'];source=by_index[index]['conditions'][condition]
        if item['dataset_index']!=index or item['condition']!=condition or metadata['score_file']!=entry['score_file']:
            raise ValueError('score Index/condition mismatch')
        for field in ('input_ids','token_ranges','final_input_token_position','final_input_token_id'):
            if item[field]!=source[field]:raise ValueError(f'input metadata mismatch: {field}')
        length=len(source['input_ids']);cursor=0
        for region in ('template_prefix','original','suffix','template_closing'):
            a,b=source['token_ranges'][region]
            if a!=cursor or not a<=b<=length:raise ValueError('invalid region partition')
            cursor=b
        if cursor!=length or item['sequence_length']!=length or source['final_input_token_position']!=length-1 or source['final_input_token_id']!=source['input_ids'][-1]:
            raise ValueError('final query or sequence metadata mismatch')
        extraction=metadata['extraction']
        if metadata['plan_sha256']!=sha256_file(SCORES/'plan.json') or extraction['causal_mask_applied_to_saved_scores'] or extraction['softmax_applied'] or not extraction['future_direction_entries_preserved']:
            raise ValueError('source scores not verified raw pre-mask/pre-softmax')
        path=SCORES/entry['score_file']
        if sha256_file(path)!=metadata['score_sha256']:raise ValueError('score file hash mismatch')
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        if values.shape!=(32,32,length,length) or values.dtype!=np.float32:
            raise ValueError('source score shape/dtype mismatch')
        references.append(dict(dataset_index=index,condition=condition,path=str(path),sha256=metadata['score_sha256'],
            metadata_path=str(metadata_path),metadata_sha256=pointer['metadata_sha256'],input=item))
    return dataset,primary,references


def summary_frame(values,indices,names,entities):
    stats=paired_stats(values,indices);count=len(names)
    rows=pd.DataFrame([dict(**entity,metric=name,sample_count=len(indices)) for entity in entities for name in names])
    for name,array in stats.items():rows[name]=array.reshape(-1)
    return rows


def paired_stats(values,indices):
    """values[sample, condition G/S, head-or-view, metric]."""
    if values.ndim!=4 or values.shape[1]!=2:raise ValueError('paired sample/condition tensor required')
    x=values[indices,0];y=values[indices,1];d=x-y
    if np.isinf(values).any():raise ValueError('infinite input metric')
    valid=np.isfinite(x)&np.isfinite(y);count=valid.sum(axis=0)

    def mean(z):
        n=np.isfinite(z).sum(axis=0);result=np.full(z.shape[1:],np.nan)
        np.divide(np.nansum(z,axis=0),n,out=result,where=n>0)
        return result

    def median(z):
        with warnings.catch_warnings():
            # An entirely undefined ratio stays undefined, with explicit count.
            warnings.simplefilter('ignore',RuntimeWarning)
            return np.nanmedian(z,axis=0)

    d=np.where(valid,d,np.nan);dm=mean(d);variance=np.full(dm.shape,np.nan)
    np.divide(np.nansum((d-dm)**2,axis=0),count-1,out=variance,where=count>1)
    tolerance=TIE_ATOL+TIE_RTOL*np.maximum(np.abs(x),np.abs(y))
    return dict(gcg_mean=mean(x),shuffled_mean=mean(y),gcg_median=median(x),shuffled_median=median(y),
        delta_mean=dm,delta_median=median(d),delta_std=np.sqrt(variance),delta_mean_absolute=mean(np.abs(d)),
        gcg_valid_count=np.isfinite(x).sum(axis=0),shuffled_valid_count=np.isfinite(y).sum(axis=0),
        pair_valid_count=count,gcg_larger_count=((d>tolerance)&valid).sum(axis=0),
        shuffled_larger_count=((d < -tolerance)&valid).sum(axis=0),
        equal_count=((np.abs(d)<=tolerance)&valid).sum(axis=0))


def reduce_views(values,views):
    """Features are computed per head before equal-weight aggregation."""
    return np.stack([values[v['members']].mean(axis=0) for v in views])


def region_layout(scores,ranges):
    """Validated ordered OO/UU/UO FP64 blocks and absolute-position masks."""
    if scores.ndim<2 or scores.shape[-1]!=scores.shape[-2]:raise ValueError('square scores required')
    a,b=ranges['original'];c,d=ranges['suffix']
    if not 0<=a<b<=c<d<=scores.shape[-1] or d-c!=20:raise ValueError('invalid Original/Suffix ranges')
    o=np.arange(a,b);u=np.arange(c,d)
    blocks={'OO':np.asarray(scores[...,a:b,a:b],dtype=np.float64),
            'UU':np.asarray(scores[...,c:d,c:d],dtype=np.float64),
            'UO':np.asarray(scores[...,c:d,a:b],dtype=np.float64)}
    masks={'OO':o[None,:]<=o[:,None],'UU':u[None,:]<=u[:,None],'UO':o[None,:]<=u[:,None]}
    if not masks['UO'].all() or not all(np.isfinite(b).all() for b in blocks.values()):
        raise ValueError('nonfinite score or future UO')
    return blocks,masks,{'original_length':b-a,'suffix_length':d-c}


def head_id(layer,head):
    if not isinstance(layer,(int,np.integer)) or not isinstance(head,(int,np.integer)) or not 0<=layer<32 or not 0<=head<32:
        raise ValueError('Layer/Head must be integer 0-31')
    return int(layer*32+head)


def head_entities():
    return [dict(layer=h//32,head=h%32) for h in range(1024)]
