#!/usr/bin/env python3
"""CPU-only exploration of six scalar outcomes in four suffix metric families."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-gcg-shuffled-metrics')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.artifact_io import read_json,read_jsonl,sha256_file,utc_now
from utils.gcg_shuffled_metrics import METRICS,TIE_ATOL,TIE_RTOL,calculate_metrics,paired_differences,summarize
from scripts.analyze_attention_head_metric_relationships import pearson_correlation,spearman_correlation

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


def frames_from_inputs(primary,references):
    frames=[];validations=[]
    layers,heads=np.indices((32,32))
    for reference in references:
        if reference['condition']=='original':continue
        item=reference['input'];scores=np.load(reference['path'],mmap_mode='r',allow_pickle=False)
        values,validation=calculate_metrics(scores,item['token_ranges']['original'],item['token_ranges']['suffix'],item['final_input_token_position'])
        frame=pd.DataFrame(dict(dataset_index=reference['dataset_index'],condition=reference['condition'],
            layer=layers.ravel(),head=heads.ravel(),primary_18=reference['dataset_index'] in primary,
            **{k:v.ravel() for k,v in values.items()}))
        frame['R_asym_defined']=np.isfinite(frame.R_asym)
        frames.append(frame)
        validations.append(dict(dataset_index=reference['dataset_index'],condition=reference['condition'],**validation))
    metrics=pd.concat(frames,ignore_index=True).sort_values(['dataset_index','condition','layer','head']).reset_index(drop=True)
    if len(metrics)!=20*2*32*32 or metrics.duplicated(['dataset_index','condition','layer','head']).any():
        raise ValueError('missing or duplicate metric rows')
    paired=paired_differences(metrics)
    if len(paired)!=20*32*32:raise ValueError('missing paired differences')
    return metrics,paired,validations


def associations(summary,subset):
    joined=summary[summary.metric=='M_U'][['layer','head','delta_mean']].merge(
        summary[summary.metric=='R_asym'][['layer','head','delta_mean']],on=['layer','head'],suffixes=('_M_U','_R_asym'),validate='one_to_one')
    rows=[]
    for centered in (False,True):
        frame=joined.copy()
        if centered:
            for col in ('delta_mean_M_U','delta_mean_R_asym'):
                frame[col]-=frame.groupby('layer')[col].transform('mean')
        x=frame.delta_mean_M_U.to_numpy();y=frame.delta_mean_R_asym.to_numpy();valid=np.isfinite(x)&np.isfinite(y)
        try:r=pearson_correlation(x[valid],y[valid]);rho=spearman_correlation(x[valid],y[valid])
        except ValueError:r=rho=None
        sx=np.where(np.abs(x)<=TIE_ATOL,0,np.sign(x));sy=np.where(np.abs(y)<=TIE_ATOL,0,np.sign(y))
        nonzero=valid&(sx!=0)&(sy!=0)
        rows.append(dict(subset=subset,layer_centered=centered,head_count=int(valid.sum()),pearson=r,spearman=rho,
            both_nonzero_count=int(nonzero.sum()),same_sign_count=int((nonzero&(sx==sy)).sum()),
            opposite_sign_count=int((nonzero&(sx!=sy)).sum()),
            note='descriptive correlations across dependent Layer/Head summaries; not causal or independent tests'))
    return rows,joined


def outputs_and_figures(directory,metrics,paired,summaries):
    figures=directory/'figures';figures.mkdir()
    grids=directory/'heatmap_values';grids.mkdir()
    overviews=[];extremes=[];relations=[];color_limits={}
    for metric in METRICS:
        maxima=[float(s[s.metric==metric].delta_mean.abs().max()) for s in summaries.values()]
        maximum=max([v for v in maxima if np.isfinite(v)],default=0.0)
        color_limits[metric]=maximum if maximum>0 else 1.0
        for n,summary in summaries.items():
            part=summary[summary.metric==metric]
            grid=part.pivot(index='layer',columns='head',values='delta_mean').reindex(index=range(32),columns=range(32))
            if np.isinf(grid.to_numpy()).any():raise ValueError('infinite heatmap values')
            grid.to_csv(grids/f'delta_{metric}_{n}.csv',mode='x',float_format='%.17g')
            fig,ax=plt.subplots(figsize=(10,8.5))
            cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('lightgrey')
            image=ax.imshow(np.ma.masked_invalid(grid.to_numpy()),cmap=cmap,vmin=-color_limits[metric],vmax=color_limits[metric],origin='upper')
            ax.set(xticks=range(32),yticks=range(32),xlabel='Head',ylabel='Layer',title=f'Mean GCG - Shuffled: {metric} | n={n}')
            ax.tick_params(labelsize=8);fig.colorbar(image,ax=ax,label=f'Delta {metric}')
            fig.tight_layout();fig.savefig(figures/f'delta_{metric}_{n}.png',dpi=200);plt.close(fig)
    for n,summary in summaries.items():
        write_csv(directory/f'summary_{n}.csv',summary)
        sym_table=summary[summary.metric.isin(['F','F_sym','F_asym','R_asym'])].pivot(index=['layer','head'],columns='metric',
            values=['gcg_mean','shuffled_mean','delta_mean','delta_median','delta_std'])
        sym_table.columns=['_'.join(c) for c in sym_table.columns]
        write_csv(directory/f'sym_asym_changes_{n}.csv',sym_table.reset_index())
        indices=sorted(metrics[metrics.primary_18].dataset_index.unique()) if n==18 else sorted(metrics.dataset_index.unique())
        for metric in METRICS:
            part=summary[summary.metric==metric].copy();part['abs_delta']=part.delta_mean.abs()
            for rank,row in enumerate(part.sort_values(['abs_delta','layer','head'],ascending=[False,True,True]).head(5).to_dict('records'),1):
                extremes.append(dict(subset=n,abs_rank=rank,**row))
            sample_means=paired[paired.dataset_index.isin(indices)].groupby('dataset_index')['delta_'+metric].mean()
            tolerance=TIE_ATOL+TIE_RTOL*np.maximum(part.gcg_mean.abs(),part.shuffled_mean.abs())
            overviews.append(dict(subset=n,metric=metric,gcg_mean_across_heads=float(part.gcg_mean.mean()),
                shuffled_mean_across_heads=float(part.shuffled_mean.mean()),delta_mean_across_heads=float(part.delta_mean.mean()),
                delta_median_across_heads=float(part.delta_mean.median()),
                positive_head_count=int((part.delta_mean>tolerance).sum()),negative_head_count=int((part.delta_mean < -tolerance).sum()),
                numerical_zero_head_count=int((part.delta_mean.abs()<=tolerance).sum()),
                sample_global_delta_sd=float(sample_means.std(ddof=1)),sample_global_delta_median=float(sample_means.median()),
                positive_sample_global_delta_count=int((sample_means>TIE_ATOL).sum()),sample_count=n))
        relation,joined=associations(summary,n);relations+=relation
        write_csv(directory/f'mass_asym_head_maps_{n}.csv',joined)
        fig,ax=plt.subplots(figsize=(8,6))
        plot=ax.scatter(joined.delta_mean_M_U,joined.delta_mean_R_asym,c=joined.layer,s=15,cmap='viridis',alpha=.7)
        ax.axhline(0,color='grey',linewidth=.7);ax.axvline(0,color='grey',linewidth=.7)
        ax.set(xlabel='Mean Delta M_U',ylabel='Mean Delta R_asym',title=f'Layer/Head mean differences | n={n}')
        fig.colorbar(plot,ax=ax,label='Layer');fig.tight_layout();fig.savefig(figures/f'mass_vs_asym_{n}.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,12))
    for ax,metric in zip(axes.flat,('M_U','D','F','R_asym')):
        grid=summaries[18].query('metric == @metric').pivot(index='layer',columns='head',values='delta_mean')
        image=ax.imshow(grid,cmap='RdBu_r',vmin=-color_limits[metric],vmax=color_limits[metric])
        ax.set(title=f'Mean Delta {metric} | n=18',xlabel='Head',ylabel='Layer',xticks=range(0,32,4),yticks=range(0,32,4))
        fig.colorbar(image,ax=ax,shrink=.85)
    fig.tight_layout();fig.savefig(figures/'overview_primary18.png',dpi=200);plt.close(fig)
    overview=pd.DataFrame(overviews);top=pd.DataFrame(extremes);association=pd.DataFrame(relations)
    write_csv(directory/'overview.csv',overview);write_csv(directory/'descriptive_head_extremes.csv',top);write_csv(directory/'associations.csv',association)
    # Compare fixed head maps between subsets without treating heads as independent samples.
    comparison=[]
    for metric in METRICS:
        a=summaries[18].query('metric == @metric').set_index(['layer','head']).delta_mean
        b=summaries[20].query('metric == @metric').set_index(['layer','head']).delta_mean.reindex(a.index)
        comparison.append(dict(metric=metric,map_pearson=pearson_correlation(a,b),
            map_spearman=spearman_correlation(a,b),max_absolute_map_change=float(np.max(np.abs(a-b))),
            sign_changed_beyond_tolerance_count=int(((a>TIE_ATOL)&(b < -TIE_ATOL) | (a < -TIE_ATOL)&(b>TIE_ATOL)).sum())))
    write_csv(directory/'subset_comparison.csv',pd.DataFrame(comparison))
    return overview,top,association,color_limits,comparison


def write_report(directory,overview,top,relations,comparison,primary,validations):
    lines=['# GCGとShuffledの4指標分析','',
        '既存の成功例から選んだ20件の探索的分析。主要18件は既存43拒否表現ルールでOriginal失敗・GCG成功・Shuffled失敗。人手で確認された実成否ではない。',
        f'主要Index：{primary}。補助20件では91・123も保持。OriginalにはSuffix指標を適用していない。','',
        '## 計算・検証','',
        '全20件×2条件×32 Layer×32 Headの40,960行、ペア差分20,480行。float64で安定Softmax/logsumexp、未加工20×20生ScoreのFrobeniusノルムとSym/Asymを計算。',
        'R_asymは二乗ノルム比であり、ノルム比そのものではない。B=0のみ未定義として空欄。差分SDはddof=1。大小件数は絶対1e-12＋相対1e-12以内を数値的同等として数える。',
        f'ゼロB件数：{sum(v["zero_block_count"] for v in validations)}。Softmax領域underflow件数：{sum(v["softmax_region_underflow_count"] for v in validations)}。',
        '入力・範囲・最終Query・ハッシュ、M_U範囲、Dのlog(M_U/M_O)同値性、NumPy参照ノルム、再構成・二乗ノルム恒等式、全グリッドの欠損を検証。詳細はvalidation.json。','',
        '## 主要18件の全Layer/Head平均','',
        '各Headで18ペアを平均後、全1024 Headを等重み平均。Headは独立な標本とは扱わず、有意差検定はしていない。','',
        '| 指標 | GCG平均 | Shuffled平均 | 平均差 | 正方向Head | 負方向Head | 数値的同等 | 正方向のサンプル平均 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in overview[overview.subset==18].to_dict('records'):
        lines.append(f'| {r["metric"]} | {r["gcg_mean_across_heads"]:.6g} | {r["shuffled_mean_across_heads"]:.6g} | {r["delta_mean_across_heads"]:+.6g} | {r["positive_head_count"]} | {r["negative_head_count"]} | {r["numerical_zero_head_count"]} | {r["positive_sample_global_delta_count"]}/18 |')
    mu=overview[(overview.subset==18)&(overview.metric=='M_U')].iloc[0]
    direction='一致' if mu.delta_mean_across_heads>0 else '逆'
    lines+=['','## 仮説に対する観察','',
        f'仮説Aの予測（ΔM_U>0）と全体平均の符号は{direction}。正方向Headは{int(mu.positive_head_count)}/1024、負方向Headは{int(mu.negative_head_count)}/1024。局所的な増加と全体の増加を区別する。',
        '仮説BはF、F_sym、F_asym、R_asymそれぞれの変化方向とサンプル間の一貫性を分けて評価する。単に差が非ゼロであることから、攻撃成功に特有・因果的な構造と断定しない。']
    lines+=['','## 差の大きいHead（記述のみ）','','値による防御位置決定は実施していない。以下は平均差の絶対値上位3箇所。','']
    for metric in METRICS:
        parts=top[(top.subset==18)&(top.metric==metric)].head(3)
        lines.append('- '+metric+': '+', '.join(f'L{int(r.layer)}H{int(r.head)} ({r.delta_mean:+.6g}, GCG大 {int(r.gcg_larger_count)}/18, Shuffled大 {int(r.shuffled_larger_count)}/18)' for r in parts.itertuples()))
    lines+=['','## Attention集中度と反対称比率の同所性','','同じLayer/Headの平均ΔM_Uと平均ΔR_asymの記述的相関。Layer平均を引いた値も併記し、層による共通傾向を区別。相関は因果を示さない。','']
    for r in relations.to_dict('records'):
        lines.append(f'- n={r["subset"]}, layer_centered={r["layer_centered"]}: Pearson={r["pearson"]:.6g}, Spearman={r["spearman"]:.6g}, 同符号={r["same_sign_count"]}, 逆符号={r["opposite_sign_count"]}。')
    lines+=['','## 18件と20件の感度確認','','| 指標 | 18件平均差 | 20件平均差 | Head-map Pearson | 符号反転Head数 |','|---|---:|---:|---:|---:|']
    for metric,c in zip(METRICS,comparison):
        a=overview[(overview.subset==18)&(overview.metric==metric)].iloc[0]
        b=overview[(overview.subset==20)&(overview.metric==metric)].iloc[0]
        lines.append(f'| {metric} | {a.delta_mean_across_heads:+.6g} | {b.delta_mean_across_heads:+.6g} | {c["map_pearson"]:.6g} | {c["sign_changed_beyond_tolerance_count"]} |')
    lines+=['','## 解釈上の限界・次の仮説','','平均差が正でも全Headの集中を意味しない。Layer/Headとサンプルによって方向・大きさが異なる可能性を、保存した平均・中央値・SD・大小件数で確認する。',
        'Fは行列の同時行列並べ替えに不変であり、差が小さくても構造同一とは言えない。R_asymの変化はFの総量変化と分けて読む。',
        '生ScoreのSym/Asymは未来方向を含むため、反対称成分を実際の情報流の方向やAttention Weightと同一視しない。',
        'GCGとShuffledはToken集合・長さが同じでも順序・RoPE位置対応・各層の文脈表現が変わる。観察差だけからJailbreakの原因を確定できない。',
        '成功例の選択と18件への結果条件付けを伴う小標本の探索であり、独立データへの一般化を主張しない。',
        '次の検証候補：集中度の変化が特定層で再現するか、反対称比率と集中度がLayer要因を除いて対応するか、これらが実際の有害な応諾と関係するか。検証ルールを事前固定した別実験と人手確認が必要。今回は追加実験なし。','',
        '## 保存先と再現','','metrics.csv、paired_differences.csv、summary_18.csv、summary_20.csv、sym_asym_changes_18/20.csv、overview.csv、associations.csv、subset_comparison.csv、heatmap_values/、figures/。',
        '各指標の図は18件と20件で同じ対称色範囲を用い、0を中心にした。全指標の平均・中央値・SD・件数は集計CSVを参照。','',
        '![主要4指標](figures/overview_primary18.png)','',
        '```bash',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_gcg_shuffled_four_metrics -v',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_four_metrics.py --output-dir data/outputs/gcg_shuffled_four_metrics/NEW_RUN',
        '```','',
        'NEW_RUNは未作成の新規ディレクトリ。CPUのみ。モデルロード、Forward、ASR再判定、Suffix再生成、Score再抽出、介入、independent100参照は行わない。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();directory=args.output_dir
    if directory.exists():raise FileExistsError('new output directory required')
    dataset,primary,references=inspect_inputs()
    directory.mkdir(parents=True,exist_ok=False)
    initial_hashes={str(p):sha256_file(p) for p in [DATASET,DATASET.parent/'plan.json',SCORES/'manifest.jsonl',SCORES/'plan.json',
        Path(__file__),ROOT/'utils/gcg_shuffled_metrics.py',ROOT/'tests/test_gcg_shuffled_four_metrics.py',
        ROOT/'utils/artifact_io.py',ROOT/'scripts/analyze_attention_head_metric_relationships.py']}
    write_json(directory/'manifest.json',dict(created_at=utc_now(),role='exploratory_outcome_conditioned_selection20',
        dataset=str(DATASET),score_root=str(SCORES),primary_indices=primary,auxiliary_indices=[r['dataset_index'] for r in dataset],
        primary_rule='historical automatic Original=false, GCG=true, Shuffled=false',
        human_reviewed=False,metric_families=4,scalar_metrics=list(METRICS),auxiliary_mass='M_O',
        metric_definitions=dict(M_U='final-query softmax mass over Suffix; all input keys in denominator',
            D='logsumexp(S[q,U])-logsumexp(S[q,O])',F='Frobenius norm of unmasked B=S[U,U]',
            F_sym='norm((B+B.T)/2)',F_asym='norm((B-B.T)/2)',R_asym='squared F_asym / squared F; zero B undefined'),
        computation_dtype='float64 from readonly float32 scores',delta='GCG - Shuffled',
        std_ddof=1,tie_atol=TIE_ATOL,tie_rtol=TIE_RTOL,undefined_ratio_policy='empty CSV field; explicit valid counts; no epsilon substitution',
        hypothesis_inference='descriptive; no significance tests or causal inference; dependent heads',
        associations='Pearson and average-rank Spearman of head mean delta M_U vs delta R_asym; raw and layer-centered',
        source_hashes=initial_hashes,score_references=references,independent100_used=False,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip()))
    metrics,paired,validations=frames_from_inputs(primary,references)
    write_csv(directory/'metrics.csv',metrics);write_csv(directory/'paired_differences.csv',paired)
    summaries={n:summarize(metrics,paired,indices) for n,indices in [(18,primary),(20,[r['dataset_index'] for r in dataset])]}
    for n,summary in summaries.items():
        if len(summary)!=32*32*len(METRICS) or not summary.sample_count.eq(n).all():raise ValueError('incomplete summary grid')
    overview,top,relations,limits,comparison=outputs_and_figures(directory,metrics,paired,summaries)
    write_json(directory/'validation.json',dict(status='passed',metric_rows=len(metrics),paired_rows=len(paired),
        primary_n=18,auxiliary_n=20,all_input_hashes_and_ranges_checked=True,checks=validations,
        undefined_ratio_rows=int((~metrics.R_asym_defined).sum()),
        summaries_rows={str(k):len(v) for k,v in summaries.items()},heatmap_symmetric_color_limits=limits))
    write_report(directory,overview,top,relations,comparison,primary,validations)
    if any(sha256_file(p)!=digest for p,digest in initial_hashes.items()) or any(sha256_file(r['path'])!=r['sha256'] for r in references):
        raise ValueError('inputs or sources changed during analysis')
    write_json(directory/'output_hashes.json',{str(p.relative_to(directory)):sha256_file(p) for p in directory.rglob('*') if p.is_file()})
    print(overview.to_string(index=False));print(relations.to_string(index=False))
    print(f'Completed {directory}')


if __name__=='__main__':
    main()
