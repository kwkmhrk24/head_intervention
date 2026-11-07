#!/usr/bin/env python3
"""Existing attention masses plus OO/UU/UO block norms, on CPU only."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd

from scripts.analyze_gcg_shuffled_four_metrics import DATASET,SCORES,inspect_inputs,write_csv,write_json
from scripts.analyze_attention_head_metric_relationships import pearson_correlation,spearman_correlation
from utils.artifact_io import read_json,sha256_file,utc_now
from utils.gcg_shuffled_region_metrics import METRICS,region_metrics,paired_stats
from utils.gcg_shuffled_token_metrics import reduce_views

TOKEN=ROOT/'data/outputs/gcg_shuffled_token_analysis/legacy_selection20_20261107T075659Z'
FOUR=ROOT/'data/outputs/gcg_shuffled_four_metrics/legacy_selection20_20261107T060324Z'
OO_RTOL=1e-5
OO_ATOL=1e-5


def summary_frame(values,indices,names,entities):
    stats=paired_stats(values,indices);count=len(names)
    rows=pd.DataFrame([dict(**entity,metric=name,sample_count=len(indices)) for entity in entities for name in names])
    for name,array in stats.items():rows[name]=array.reshape(-1)
    return rows


def check_original(gcg,shuffled,reference):
    if gcg.shape!=shuffled.shape or gcg.shape!=reference.shape:
        raise ValueError('Original block reference shape mismatch')
    np.testing.assert_allclose(gcg,shuffled,rtol=OO_RTOL,atol=OO_ATOL)
    np.testing.assert_allclose(gcg,reference,rtol=OO_RTOL,atol=OO_ATOL)
    return np.max(np.abs(gcg.astype(float)-shuffled),axis=(-2,-1)),np.max(np.abs(gcg.astype(float)-reference),axis=(-2,-1))


def head_maps(regions,attention):
    result=attention[['layer','head']].drop_duplicates().sort_values(['layer','head'])
    for frame,names in ((regions,METRICS),(attention,('M_U','M_O','D'))):
        for name in names:
            part=frame[frame.metric==name][['layer','head','delta_mean']].rename(columns={'delta_mean':'delta_'+name})
            result=result.merge(part,on=['layer','head'],validate='one_to_one')
    return result


def association_rows(maps,subset):
    pairs=[('delta_M_U','negative_delta_M_O'),('delta_M_U','delta_RMS_UU_full'),
           ('delta_M_U','delta_RMS_UO_full'),('delta_M_O','delta_RMS_UO_full'),
           ('delta_D','delta_R_UU_UO_full'),('delta_D','delta_R_UU_UO_causal'),
           ('delta_RMS_UU_full','delta_RMS_UU_causal'),('delta_R_UU_UO_full','delta_R_UU_UO_causal')]
    maps=maps.copy();maps['negative_delta_M_O']=-maps.delta_M_O
    result=[]
    for centered in (False,True):
        frame=maps.copy()
        if centered:
            numeric=[name for name in frame if name not in ('layer','head')]
            frame[numeric]-=frame.groupby('layer')[numeric].transform('mean')
        for xname,yname in pairs:
            x=frame[xname].to_numpy();y=frame[yname].to_numpy();valid=np.isfinite(x)&np.isfinite(y)
            try:r=pearson_correlation(x[valid],y[valid]);rho=spearman_correlation(x[valid],y[valid])
            except ValueError:r=rho=None
            result.append(dict(subset=subset,layer_centered=centered,x=xname,y=yname,head_count=int(valid.sum()),
                pearson=r,spearman=rho,positive_x_positive_y=int(((x>1e-12)&(y>1e-12)&valid).sum()),
                positive_x_negative_y=int(((x>1e-12)&(y < -1e-12)&valid).sum()),
                negative_x_positive_y=int(((x < -1e-12)&(y>1e-12)&valid).sum()),
                negative_x_negative_y=int(((x < -1e-12)&(y < -1e-12)&valid).sum()),
                interpretation='descriptive dependent Head summaries; no significance or causality claim'))
    return result


def arrange_existing_attention(directory,token_manifest):
    query=pd.read_csv(TOKEN/'four_query_summary.csv')
    write_csv(directory/'attention_existing_query_summary.csv',query)
    rows=[]
    for n in (18,20):
        with np.load(TOKEN/f'row_summary_{n}.npz',allow_pickle=False) as archive:
            suffix=archive['mean__D_suffix_logmass_delta'];original=archive['mean__D_original_logmass_delta_negated']
        for v,view in enumerate(token_manifest['views']):
            for slot in range(5):
                key=str(slot+1) if slot<4 else 'mean4'
                a=float(suffix[v,slot]) if slot<4 else float(suffix[v].mean())
                b=float(original[v,slot]) if slot<4 else float(original[v].mean())
                target=query[(query.subset==n)&(query.view_id==view['view_id'])&(query.query_slot==key)&(query.metric=='D')].iloc[0].delta_mean
                np.testing.assert_allclose(a+b,target,rtol=1e-11,atol=1e-11)
                rows.append(dict(subset=n,view_id=view['view_id'],query_slot=key,suffix_logmass_change=a,
                                 original_logmass_decrease=b,delta_D=target))
    write_csv(directory/'attention_existing_D_decomposition.csv',pd.DataFrame(rows))
    return query


def run(directory):
    if directory.exists():raise FileExistsError('new output directory required')
    dataset,primary,references=inspect_inputs()
    token_manifest=read_json(TOKEN/'manifest.json');old_hashes=read_json(TOKEN/'output_hashes.json')
    if token_manifest['all_indices']!=[r['dataset_index'] for r in dataset] or token_manifest['primary_indices']!=primary:
        raise ValueError('existing token analysis cohort mismatch')
    used=['manifest.json','selected_heads.csv','four_query_summary.csv','all_head_scalars.npz','row_summary_18.npz','row_summary_20.npz']
    for name in used:
        if sha256_file(TOKEN/name)!=old_hashes[name]:raise ValueError('previous result hash mismatch')
    views=token_manifest['views'];indices=token_manifest['all_indices'];positions18=[indices.index(i) for i in primary]
    with np.load(TOKEN/'all_head_scalars.npz',allow_pickle=False) as archive:
        # Reuse existing softmax/logsumexp results, never recompute them.
        attention=archive['query_metrics'][:,:,:,3,:]
        norm_reference=archive['norm_metrics'][:,:,:,0]
        existing_view_queries=archive['view_queries']
    if attention.shape!=(20,2,1024,3) or norm_reference.shape!=(20,2,1024):raise ValueError('existing scalar shape mismatch')
    sources=[Path(__file__),ROOT/'utils/gcg_shuffled_region_metrics.py',ROOT/'utils/gcg_shuffled_region_plots.py',
        ROOT/'tests/test_gcg_shuffled_regions.py',ROOT/'scripts/analyze_gcg_shuffled_four_metrics.py',
        ROOT/'utils/gcg_shuffled_token_metrics.py',ROOT/'utils/gcg_shuffled_metrics.py',
        DATASET,DATASET.parent/'plan.json',SCORES/'plan.json',SCORES/'manifest.jsonl',SCORES/'validation_final.json',
        FOUR/'summary_18.csv',FOUR/'summary_20.csv',FOUR/'output_hashes.json']+[TOKEN/n for n in used]
    four_hashes=read_json(FOUR/'output_hashes.json')
    for n in (18,20):
        if sha256_file(FOUR/f'summary_{n}.csv')!=four_hashes[f'summary_{n}.csv']:
            raise ValueError('existing per-Head attention summary hash mismatch')
    hashes={str(path):sha256_file(path) for path in sources}
    directory.mkdir(parents=True,exist_ok=False)
    score_plan=read_json(SCORES/'plan.json')
    write_json(directory/'manifest.json',dict(created_at=utc_now(),role='exploratory_primary18_region_comparison',
        primary_indices=primary,all_indices=indices,views=views,source_hashes=hashes,score_references=references,
        comparison_dataset=str(DATASET),score_root=str(SCORES),
        reused_attention=str(TOKEN),metrics=list(METRICS),
        aliases=dict(requested_R_UO='R_UU_OO_full',requested_R_internal_external='R_UU_UO_full'),
        normalization='F/sqrt(valid element count); ratios per sample/Head before averaging; no epsilon',
        causal_policy='absolute Key <= Query including diagonal; UO entirely visible; auxiliary OO causal is also computed for a consistently causal OO ratio',
        primary_policy='full unmasked scores; causal metrics separately labeled, never merged',
        average_order='per-sample per-Head norm/RMS/ratio; equal Head mean within existing fixed views; sample mean; never norm of mean matrix',
        undefined_policy='zero denominator => blank CSV ratio, explicit valid counts; no epsilon or inf',
        comparison_tolerance=dict(rtol=1e-12,atol=1e-12),oo_invariance_tolerance=dict(rtol=OO_RTOL,atol=OO_ATOL),
        head_selection='reuse previous fixed45 individuals /12 groups; no new ranking',
        historical_labels='43 refusal-substring automatic labels; human outcomes not confirmed',
        model_info={k:score_plan[k] for k in ('model_name','model_revision','model_config','computation','dataset_sha256')},
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=version('matplotlib'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
        model_loaded=False,forward=False,attention_recomputed=False,independent100_used=False))
    arrange_existing_attention(directory,token_manifest)
    write_csv(directory/'selected_heads_reference.csv',pd.read_csv(TOKEN/'selected_heads.csv'))
    lookup={(r['dataset_index'],r['condition']):r for r in references}
    tensors=[];counts=[];invariance=[];checks=[];zero_rows=[]
    flat=np.arange(1024)
    for i,index in enumerate(indices):
        original_ref=lookup[index,'original'];original=np.load(original_ref['path'],mmap_mode='r',allow_pickle=False)
        oa,ob=original_ref['input']['token_ranges']['original'];reference_block=original[:,:,oa:ob,oa:ob]
        pair=[];oo_blocks=[];count_reference=None;max_norm_error=0.0
        for c,condition in enumerate(('gcg','shuffled')):
            ref=lookup[index,condition];item=ref['input'];a,b=item['token_ranges']['original']
            original_ids=original_ref['input']['input_ids'][oa:ob]
            if item['input_ids'][a:b]!=original_ids:raise ValueError('actual Original token IDs differ')
            scores=np.load(ref['path'],mmap_mode='r',allow_pickle=False)
            values,count=region_metrics(scores,item['token_ranges']);oo_blocks.append(scores[:,:,a:b,a:b])
            np.testing.assert_allclose(values['F_UU_full'].reshape(1024),norm_reference[i,c],rtol=1e-11,atol=1e-11)
            max_norm_error=max(max_norm_error,float(np.max(np.abs(values['F_UU_full'].reshape(1024)-norm_reference[i,c]))))
            if count_reference is not None and count_reference!=count:raise ValueError('paired element counts differ')
            count_reference=count
            matrix=np.stack([values[name].reshape(1024) for name in METRICS],axis=-1);pair.append(matrix)
            for h,j in zip(*np.where(np.isnan(matrix))):
                zero_rows.append(dict(dataset_index=index,condition=condition,layer=int(h//32),head=int(h%32),metric=METRICS[j],reason='zero denominator'))
        diff,refdiff=check_original(oo_blocks[0],oo_blocks[1],reference_block)
        invariance.append(pd.DataFrame(dict(dataset_index=index,layer=flat//32,head=flat%32,
            max_abs_gcg_shuffled=diff.reshape(1024),max_abs_gcg_original_condition=refdiff.reshape(1024))))
        tensors.append(np.stack(pair));counts.append(dict(dataset_index=index,**count_reference))
        checks.append(dict(dataset_index=index,oo_gcg_shuffled_max=float(diff.max()),oo_vs_original_condition_max=float(refdiff.max()),existing_F_UU_max_error=max_norm_error))
    values=np.stack(tensors)
    view_values=np.stack([np.stack([reduce_views(values[i,c],views) for c in range(2)]) for i in range(20)])
    view_attention=existing_view_queries[:,:,:,3,:].transpose(0,2,1,3)
    np.savez_compressed(directory/'per_sample_values.npz',region_metrics=values,attention_q4=attention,view_region_metrics=view_values,view_attention_q4=view_attention)
    frames=[]
    for i,index in enumerate(indices):
        for c,condition in enumerate(('gcg','shuffled')):
            frame=pd.DataFrame(dict(dataset_index=index,condition=condition,layer=flat//32,head=flat%32,
                primary18=index in primary,**{name:values[i,c,:,j] for j,name in enumerate(METRICS)}))
            frames.append(frame)
    write_csv(directory/'metrics.csv',pd.concat(frames,ignore_index=True))
    pair_frames=[pd.DataFrame(dict(dataset_index=index,layer=flat//32,head=flat%32,
        **{'delta_'+name:values[i,0,:,j]-values[i,1,:,j] for j,name in enumerate(METRICS)})) for i,index in enumerate(indices)]
    write_csv(directory/'paired_differences.csv',pd.concat(pair_frames,ignore_index=True))
    write_csv(directory/'element_counts.csv',pd.DataFrame(counts));write_csv(directory/'oo_invariance.csv',pd.concat(invariance,ignore_index=True))
    write_csv(directory/'zero_denominators.csv',pd.DataFrame(zero_rows,columns=['dataset_index','condition','layer','head','metric','reason']))
    entities=[dict(layer=h//32,head=h%32) for h in flat]
    view_entities=[dict(view_id=v['view_id'],view_kind=v['kind'],head_count=len(v['members'])) for v in views]
    associations=[];group_details=[];overview=[]
    for n,positions in ((18,positions18),(20,list(range(20)))):
        summary=summary_frame(values,positions,METRICS,entities)
        # M_U/D summaries already exist; only the missing M_O Head summary needs
        # aggregation. No attention scores or probabilities are recalculated.
        old_attention=pd.read_csv(FOUR/f'summary_{n}.csv')
        old_attention=old_attention[old_attention.metric.isin(['M_U','D'])].rename(columns={'equal_within_tolerance_count':'equal_count'})
        mo=summary_frame(attention[:,:,:,1:2],positions,('M_O',),entities)
        common=[c for c in mo.columns if c!='delta_mean_absolute']
        attn=pd.concat([old_attention[common],mo[common]],ignore_index=True).sort_values(['layer','head','metric'])
        view_summary=summary_frame(view_values,positions,METRICS,view_entities)
        write_csv(directory/f'summary_{n}.csv',summary);write_csv(directory/f'attention_head_summary_{n}.csv',attn)
        write_csv(directory/f'view_summary_{n}.csv',view_summary)
        maps=head_maps(summary,attn);write_csv(directory/f'head_maps_{n}.csv',maps)
        associations+=association_rows(maps,n)
        for row in view_summary[view_summary.view_id=='all_heads'].to_dict('records'):overview.append(dict(subset=n,**row))
        fixed=pd.read_csv(TOKEN/'selected_heads.csv')
        for member in fixed.itertuples():
            part=summary[(summary.layer==member.layer)&(summary['head']==member.head)]
            for record in part.to_dict('records'):group_details.append(dict(subset=n,selected_group=member.group_id,old_metric=member.metric,old_rank=member.rank,**record))
    write_csv(directory/'overview.csv',pd.DataFrame(overview));write_csv(directory/'associations.csv',pd.DataFrame(associations))
    write_csv(directory/'fixed_group_member_effects.csv',pd.DataFrame(group_details))
    # When OO is invariant, normalization is algebraically rescaled UU change.
    errors=[]
    for policy in ('full','causal'):
        ratio=METRICS.index('R_UU_OO_'+policy);uu=METRICS.index('RMS_UU_'+policy);oo=METRICS.index('RMS_OO_'+policy)
        numerator=values[:,0,:,uu]-values[:,1,:,uu];denom=values[:,1,:,oo]
        valid=denom>0;target=np.full_like(denom,np.nan);np.divide(numerator,denom,out=target,where=valid)
        actual=values[:,0,:,ratio]-values[:,1,:,ratio]
        errors.append(float(np.max(np.abs(actual[valid]-target[valid]))) if valid.any() else None)
        np.testing.assert_allclose(actual[valid],target[valid],rtol=1e-9,atol=1e-9)
    validation=dict(status='passed',samples=20,primary18=primary,metric_rows=40960,paired_rows=20480,
        zero_denominator_rows=len(zero_rows),all_input_correspondence_and_hashes_verified=True,
        oo_max_absolute_error=max(c['oo_gcg_shuffled_max'] for c in checks),
        oo_vs_original_condition_max_absolute_error=max(c['oo_vs_original_condition_max'] for c in checks),
        existing_F_UU_max_absolute_error=max(c['existing_F_UU_max_error'] for c in checks),
        invariant_OO_normalized_delta_identity_max_errors=errors,
        finite_except_explicit_undefined_ratios=not np.isinf(values).any(),checks=checks)
    write_json(directory/'validation.json',validation)
    from utils.gcg_shuffled_region_plots import make_figures
    make_figures(directory)
    write_report(directory,validation)
    if any(sha256_file(p)!=h for p,h in hashes.items()) or any(sha256_file(r['path'])!=r['sha256'] for r in references):
        raise ValueError('source data or code changed during analysis')
    write_json(directory/'output_hashes.json',{str(p.relative_to(directory)):sha256_file(p) for p in directory.rglob('*') if p.is_file()})
    print(pd.DataFrame(overview)[['subset','metric','gcg_mean','shuffled_mean','delta_mean','delta_std','gcg_larger_count','shuffled_larger_count']].to_string(index=False))
    print(f'Completed {directory}')


def write_report(directory,validation):
    overview=pd.read_csv(directory/'overview.csv');query=pd.read_csv(directory/'attention_existing_query_summary.csv')
    a=query[(query.subset==18)&(query.view_id=='all_heads')&(query.query_slot=='4')].set_index('metric')
    lines=['# Original / Suffix領域Score分析','',
        '既存selection20のGCG/Shuffled対応ペア。主要primary18はGCG自動成功・Shuffled自動失敗で、人手確認済み成否ではない。補助は同じ20件。CPU上の既存データ分析のみ。','',
        '## Attention割合（既存値の再利用）','',
        '| 指標 | GCG | Shuffled | 差 |','|---|---:|---:|---:|']
    for metric in ('M_U','M_O','D'):
        row=a.loc[metric];lines.append(f'| {metric} | {row.gcg_mean:.6g} | {row.shuffled_mean:.6g} | {row.delta_mean:+.6g} |')
    lines+=['','Softmax/logsumexpは再実行せず、既存four_query_summary.csvとall_head_scalars.npzを再利用。Dの既存分解はattention_existing_D_decomposition.csv。','',
        '## 全Head平均の領域指標（主要18件）','',
        '各サンプル・HeadでNorm/RMS/比率を計算後、等重みHead平均、サンプル平均。平均行列のNormではない。','',
        '| 指標 | GCG | Shuffled | 平均差 | 差分中央値 | 差分SD |','|---|---:|---:|---:|---:|---:|']
    for row in overview[overview.subset==18].itertuples():
        lines.append(f'| {row.metric} | {row.gcg_mean:.6g} | {row.shuffled_mean:.6g} | {row.delta_mean:+.6g} | {row.delta_median:+.6g} | {row.delta_std:.6g} |')
    lines+=['','## 検証と意味','',
        f'Original内部ScoreのGCG/Shuffled最大絶対差={validation["oo_max_absolute_error"]}。Original単独条件との最大差={validation["oo_vs_original_condition_max_absolute_error"]}。許容値はrtol=atol=1e-5。',
        f'既存F_Uとの最大誤差={validation["existing_F_UU_max_absolute_error"]:.6g}。ゼロ分母による未定義行={validation["zero_denominator_rows"]}。',
        'OOが不変なら、UU/OO比の変化は固定OO基準でのUU変化の再表現であり、独立した機構的証拠ではない。',
        'M_O低下をOO生Score低下と同一視しない。M_Oは最終Queryの正規化Attention割合で、OOはOriginal Query×Original Keyの生Score。UOもSuffix Query×Original Keyの生Score規模であり、Attention割合ではない。',
        'fullは未来側を含むMask前全要素。causalは絶対Key<=Queryの有効要素だけで、RMS分母もその個数。UUは400/210要素、UOは20*m要素で全て可視。OOも補助的にm²/m(m+1)/2で分け、causal UU/OO比は両側をcausalにした別指標。',
        '主要UU/OO比はfullのまま維持。ゼロ分母にはepsilonを加えず比率を空欄にし、valid countとzero_denominators.csvに明示。','',
        '## 保存・再現','',
        'metrics.csv、paired_differences.csv、summary_18/20.csv、view_summary_18/20.csv、head_maps_18/20.csv、fixed_group_member_effects.csv、overview.csv、associations.csv、element_counts.csv、oo_invariance.csv。',
        '既存の45個別Head・12増減群・全Headを固定して使用し、新しい指標を見てHeadを選び直していない。Headは独立標本ではない。同一18件での選択バイアスがあり、有意性・因果・成功一般への普遍性は主張しない。',
        '結果の解釈と最初に見る図はinterpretation.md。','',
        '```bash',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_gcg_shuffled_regions -v',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_regions.py --output-dir data/outputs/gcg_shuffled_region_analysis/NEW_RUN',
        '```','',
        'NEW_RUNは未作成の新規パス。モデルロード、Forward、Suffix再生成、Score再抽出、応答生成、ASR再評価、介入、独立100件の利用は実施しない。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)


if __name__=='__main__':main()
