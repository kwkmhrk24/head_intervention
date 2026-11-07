#!/usr/bin/env python3
"""CPU-only signed region means, with immutable references to the prior RMS run."""
from pathlib import Path
import argparse
import json
import platform
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.analyze_gcg_shuffled_four_metrics import inspect_inputs, write_csv, write_json
from scripts.analyze_gcg_shuffled_regions import summary_frame
from utils.artifact_io import sha256_file, read_json, utc_now
from utils.gcg_shuffled_token_metrics import reduce_views
from utils.gcg_shuffled_signed_means import METRICS, signed_means
from utils.gcg_shuffled_region_plots import plt, finish
OLD = ROOT/'data/outputs/gcg_shuffled_region_analysis/legacy_selection20_20261107T094154Z'


def run(out):
    if out.exists(): raise FileExistsError('output directory must be new')
    old = read_json(OLD/'manifest.json'); old_hashes = read_json(OLD/'output_hashes.json')
    used = ['manifest.json', 'element_counts.csv', 'selected_heads_reference.csv', 'per_sample_values.npz', 'overview.csv']
    for name in used:
        if sha256_file(OLD/name) != old_hashes[name]: raise ValueError('old result hash mismatch: '+name)
    _, primary, refs = inspect_inputs()
    indices = old['all_indices']; views = old['views']
    if primary != old['primary_indices'] or set(indices) != {r['dataset_index'] for r in refs}:
        raise ValueError('cohort mismatch')
    lookup = {(r['dataset_index'], r['condition']): r for r in refs}
    old_counts = pd.read_csv(OLD/'element_counts.csv').set_index('dataset_index')
    fixed = pd.read_csv(OLD/'selected_heads_reference.csv')
    with np.load(OLD/'per_sample_values.npz', allow_pickle=False) as z:
        rms = z['region_metrics']
    if rms.shape != (20, 2, 1024, len(old['metrics'])): raise ValueError('RMS correspondence shape')
    sources = [Path(__file__), ROOT/'utils/gcg_shuffled_common.py', ROOT/'utils/gcg_shuffled_plot_common.py', ROOT/'utils/gcg_shuffled_signed_means.py', ROOT/'tests/test_gcg_shuffled_signed_means.py',
               ROOT/'docs/gcg_shuffled_signed_mean_analysis.md', ROOT/'scripts/analyze_gcg_shuffled_regions.py',
               ROOT/'utils/gcg_shuffled_region_metrics.py', ROOT/'utils/gcg_shuffled_token_metrics.py',
               ROOT/'scripts/analyze_gcg_shuffled_four_metrics.py'] + [OLD/n for n in used]
    hashes = {str(p): sha256_file(p) for p in sources}
    values = []; counts = []; max_oo = 0.
    for index in indices:
        pair = []; pair_counts = []
        for condition in ('gcg', 'shuffled'):
            ref = lookup[index, condition]; x = np.load(ref['path'], mmap_mode='r', allow_pickle=False)
            v, n = signed_means(x, ref['input']['token_ranges'])
            pair.append(np.stack([v[name].reshape(1024) for name in METRICS], axis=-1)); pair_counts.append(n)
        if pair_counts[0] != pair_counts[1] or pair_counts[0] != old_counts.loc[index].to_dict():
            raise ValueError('element counts differ from prior RMS run')
        for j in (0, 3):
            np.testing.assert_array_equal(pair[0][:, j], pair[1][:, j])
            max_oo = max(max_oo, float(np.abs(pair[0][:, j]-pair[1][:, j]).max()))
        counts.append(dict(dataset_index=index, **pair_counts[0])); values.append(pair)
    values = np.asarray(values)
    if values.shape != (20, 2, 1024, 6) or not np.isfinite(values).all(): raise ValueError('missing/nonfinite means')
    view_values = np.array([[reduce_views(values[i, c], views) for c in range(2)] for i in range(20)])
    out.mkdir(parents=True)
    manifest = dict(created_at=utc_now(), old_region_analysis=str(OLD), primary_indices=primary, all_indices=indices,
        metrics=list(METRICS), views=views, score_references=refs, source_hashes=hashes, model_info=old['model_info'],
        condition_order=['gcg', 'shuffled'], head_order='layer*32+head; zero-based', device='CPU',
        formula='sum(raw signed scores in valid region)/valid element count; no abs/square/clip',
        mask='full=all; causal=absolute Key<=Query, including diagonal; no -inf or zero padding',
        aggregation='per sample/condition/Head/region mean; equal Head/view mean; sample statistics, SD ddof=1',
        direction_tolerance='same prior paired_stats: atol=1e-12 + rtol=1e-12 * max(abs(condition values))',
        group_order='same views order as old fixed_head_groups.png: all_heads followed by fixed groups',
        labels='43-refusal-expression automatic labels, human outcomes unconfirmed',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        prohibited_operations_executed=False)
    write_json(out/'manifest.json', manifest)
    flat=np.arange(1024); entities=[dict(layer=int(h//32),head=int(h%32)) for h in flat]
    frames=[]; pairs=[]; samples=[]; overview=[]; members=[]; rms_comparison=[]
    for i,index in enumerate(indices):
        for c,condition in enumerate(('gcg','shuffled')):
            frames.append(pd.DataFrame(dict(dataset_index=index,condition=condition,primary18=index in primary,
                layer=flat//32,head=flat%32,**{name:values[i,c,:,j] for j,name in enumerate(METRICS)})))
        pairs.append(pd.DataFrame(dict(dataset_index=index,layer=flat//32,head=flat%32,
            **{'delta_'+name:values[i,0,:,j]-values[i,1,:,j] for j,name in enumerate(METRICS)})))
        for j,name in enumerate(METRICS):
            samples.append(dict(dataset_index=index,primary18=index in primary,metric=name,
                gcg_mean=values[i,0,:,j].mean(),shuffled_mean=values[i,1,:,j].mean(),
                delta_mean=(values[i,0,:,j]-values[i,1,:,j]).mean()))
    for n,pos in ((18,[indices.index(i) for i in primary]),(20,list(range(20)))):
        heads=summary_frame(values,pos,METRICS,entities)
        vs=summary_frame(view_values,pos,METRICS,[dict(view_id=v['view_id'],view_kind=v['kind'],head_count=len(v['members'])) for v in views])
        assert len(heads) == 1024*6 and not heads.duplicated(['layer', 'head', 'metric']).any()
        assert (heads.pair_valid_count == n).all()
        assert np.isfinite(heads.select_dtypes(include='number')).all().all()
        write_csv(out/f'head_summary_{n}.csv', heads); write_csv(out/f'view_summary_{n}.csv',vs)
        for row in vs[vs.view_id=='all_heads'].to_dict('records'):
            part=heads[heads.metric==row['metric']]
            overview.append(dict(subset=n,**row,positive_head_count=int((part.delta_mean>1e-12).sum()),
                negative_head_count=int((part.delta_mean < -1e-12).sum()),equal_head_count=int((part.delta_mean.abs()<=1e-12).sum())))
        for member in fixed.itertuples():
            part=heads[(heads.layer==member.layer)&(heads['head']==member.head)]
            for row in part.to_dict('records'):
                members.append(dict(subset=n,selected_group=member.group_id,old_metric=member.metric,old_rank=member.rank,**row))
        for j,name in enumerate(METRICS):
            base=name.removeprefix('mu_'); k=old['metrics'].index('RMS_'+base)
            sm=values[pos,0,:,j]-values[pos,1,:,j]; rd=rms[pos,0,:,k]-rms[pos,1,:,k]
            for h in flat:
                rms_comparison.append(dict(subset=n,layer=int(h//32),head=int(h%32),metric=name,
                    signed_mean_delta=float(sm[:,h].mean()),existing_rms_delta=float(rd[:,h].mean())))
    write_csv(out/'metrics.csv',pd.concat(frames)); write_csv(out/'paired_differences.csv',pd.concat(pairs))
    write_csv(out/'element_counts.csv',pd.DataFrame(counts)); write_csv(out/'sample_summary.csv',pd.DataFrame(samples))
    write_csv(out/'overview.csv',pd.DataFrame(overview)); write_csv(out/'fixed_group_member_effects.csv',pd.DataFrame(members))
    write_csv(out/'selected_heads_reference.csv',fixed); write_csv(out/'rms_comparison.csv',pd.DataFrame(rms_comparison))
    np.savez_compressed(out/'signed_means.npz',values=values,view_values=view_values)
    make_figures(out, manifest)
    write_report(out)
    for p,h in hashes.items():
        if sha256_file(p)!=h: raise ValueError('source changed during analysis')
    for ref in refs:
        if sha256_file(ref['path'])!=ref['sha256']:raise ValueError('score changed')
    validation=dict(status='passed',samples=20,primary_samples=18,metric_rows=40960,paired_rows=20480,
        finite=True,missing_pairs=0,oo_max_signed_mean_difference=max_oo,element_counts_match_prior=True,
        input_correspondence_and_hashes_verified=True,old_sources_unchanged=True,
        valid_pair_count_per_head_primary=18,valid_pair_count_per_head_all=20)
    write_json(out/'validation.json',validation)
    write_json(out/'output_hashes.json',{str(p.relative_to(out)):sha256_file(p) for p in out.rglob('*') if p.is_file()})
    print(pd.DataFrame(overview)[['subset','metric','gcg_mean','shuffled_mean','delta_mean','delta_std','gcg_larger_count','shuffled_larger_count','positive_head_count','negative_head_count']].to_string(index=False))
    print(out)


def make_figures(out, m):
    f=out/'figures';f.mkdir(); p=pd.read_csv(out/'overview.csv');p=p[p.subset==18].set_index('metric')
    title='n=18 | raw signed score; per Head/sample region mean, then equal Head/sample mean'
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for r,policy in enumerate(('full','causal')):
        names=[f'mu_{b}_{policy}' for b in ('OO','UU','UO')]; x=np.arange(3)
        for c,color,shift in [('gcg','tab:red',-.18),('shuffled','tab:blue',.18)]:
            axes[r,0].bar(x+shift,p.loc[names,c+'_mean'],.36,label=c,color=color)
        axes[r,0].legend();axes[r,0].set(xticks=x,xticklabels=['OO','UU','UO'],title=policy,ylabel='Signed mean score')
        axes[r,1].bar(x,p.loc[names].delta_mean,color=np.where(p.loc[names].delta_mean>=0,'tab:red','tab:blue'))
        axes[r,1].errorbar(x,p.loc[names].delta_mean,yerr=p.loc[names].delta_std,fmt='none',color='black',capsize=3)
        axes[r,1].set(xticks=x,xticklabels=['OO','UU','UO'],title=policy+' delta; bars=sample SD',ylabel='GCG-Shuffled (score)')
        for ax in axes[r]:ax.axhline(0,color='grey',lw=.7)
    finish(fig,f/'signed_mean_overview.png',title)
    h=pd.read_csv(out/'head_summary_18.csv');fig,axes=plt.subplots(2,3,figsize=(15,9))
    for c,region in enumerate(('OO','UU','UO')):
        limit=max(float(h[h.metric==f'mu_{region}_{policy}'].delta_mean.abs().max()) for policy in ('full','causal')) or 1
        for r,policy in enumerate(('full','causal')):
            ax=axes[r,c];grid=h[h.metric==f'mu_{region}_{policy}'].pivot(index='layer',columns='head',values='delta_mean')
            im=ax.imshow(grid,cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set(title=region+' '+policy,xlabel='Head (0-31)',ylabel='Layer (0-31)');fig.colorbar(im,ax=ax,label='GCG-Shuffled score')
    finish(fig,f/'signed_mean_head_heatmaps.png',title+' | shared Full/Causal scales per region')
    groups=pd.read_csv(out/'view_summary_18.csv');members=pd.read_csv(out/'fixed_group_member_effects.csv');members=members[members.subset==18]
    ordered=[v for v in m['views'] if v['kind']=='all_heads']+[v for v in m['views'] if v['kind'] not in ('individual','all_heads')]
    fig,axes=plt.subplots(2,2,figsize=(18,14))
    for ax,name in zip(axes.flat,['mu_UU_full','mu_UO_full','mu_UU_causal','mu_UO_causal']):
        for i,v in enumerate(ordered):
            row=groups[(groups.view_id==v['view_id'])&(groups.metric==name)].iloc[0]
            ax.errorbar(row.delta_mean,i,xerr=row.delta_std,fmt='D',color='black',capsize=2,markersize=4)
            part=members[(members.selected_group==v['view_id'])&(members.metric==name)]
            ax.scatter(part.delta_mean,i+np.linspace(-.15,.15,len(part)),s=20,color='tab:blue')
        ax.axvline(p.loc[name].delta_mean,color='orange',ls='--');ax.axvline(0,color='grey',lw=.7)
        ax.set_yticks(range(len(ordered)),[v['view_id'] for v in ordered],fontsize=8);ax.invert_yaxis()
        ax.set(title=name,xlabel='GCG-Shuffled score')
    finish(fig,f/'signed_mean_fixed_groups.png',title+' | fixed old groups: diamond=mean, bar=sample SD, dots=member means; orange=all Heads')
    sample=pd.read_csv(out/'sample_summary.csv');fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,name in zip(axes.flat,['mu_UU_full','mu_UO_full','mu_UU_causal','mu_UO_causal']):
        part=sample[(sample.primary18)&(sample.metric==name)]
        ax.bar(np.arange(18),part.delta_mean,color=np.where(part.delta_mean>=0,'tab:red','tab:blue'))
        ax.axhline(part.delta_mean.mean(),color='orange',ls='--');ax.axhline(0,color='grey',lw=.7)
        ax.set_xticks(range(18),part.dataset_index,rotation=60,fontsize=8);ax.set(title=name,xlabel='Dataset Index',ylabel='GCG-Shuffled score')
    finish(fig,f/'signed_mean_samples.png',title+' | orange=sample mean')
    comp=pd.read_csv(out/'rms_comparison.csv');comp=comp[comp.subset==18];fig,axes=plt.subplots(2,2,figsize=(12,9))
    for ax,name in zip(axes.flat,['mu_UU_full','mu_UO_full','mu_UU_causal','mu_UO_causal']):
        part=comp[comp.metric==name];im=ax.scatter(part.existing_rms_delta,part.signed_mean_delta,c=part.layer,cmap='viridis',s=9,alpha=.65)
        ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7)
        ax.set(title=name,xlabel='Existing RMS delta (score magnitude)',ylabel='Signed mean delta (score)');fig.colorbar(im,ax=ax,label='Layer')
    finish(fig,f/'rms_vs_signed_mean.png','n=18 paired mean differences | 1024 dependent Heads; different metrics, no causal inference')


def write_report(out):
    table=pd.read_csv(out/'overview.csv')
    lines=['# Signed Mean Score追加分析','',
        '主要18件は既存ルール上のGCG自動成功・Shuffled自動失敗条件。実際の有害応諾は人手確認していない。全20件は補助分析。',
        '各サンプル・各Head・各領域の有効Scoreを符号付きでFP64算術平均した後、Head等重み、サンプル等重みで集計。SDはサンプル間のペア差、ddof=1。領域間で要素を結合しない。', '',
        '| N | 指標 | GCG | Shuffled | 差 | 差のSD | サンプル増/減/同等 | Head増/減/同等 |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for r in table.itertuples():
        lines.append(f'| {r.subset} | {r.metric} | {r.gcg_mean:.6f} | {r.shuffled_mean:.6f} | {r.delta_mean:+.6f} | {r.delta_std:.6f} | {r.gcg_larger_count}/{r.shuffled_larger_count}/{r.equal_count} | {r.positive_head_count}/{r.negative_head_count}/{r.equal_head_count} |')
    lines += ['', '## 固定Head群（主要18件）', '', '| 固定群 | 指標 | 差 | サンプル差SD | 群内Head増/減/同等 |', '|---|---|---:|---:|---|']
    groups=pd.read_csv(out/'view_summary_18.csv');members=pd.read_csv(out/'fixed_group_member_effects.csv')
    for r in groups[groups.view_kind=='selected_group'].itertuples():
        if 'mu_OO' in r.metric: continue
        part=members[(members.subset==18)&(members.selected_group==r.view_id)&(members.metric==r.metric)]
        lines.append(f'| {r.view_id} | {r.metric} | {r.delta_mean:+.6f} | {r.delta_std:.6f} | {(part.delta_mean>1e-12).sum()}/{(part.delta_mean < -1e-12).sum()}/{(part.delta_mean.abs()<=1e-12).sum()} |')
    lines += ['', '## 解釈上の制約',
        'RMS増加は符号付き平均増加と同義ではない。RMSは規模、Signed Meanは符号付き水準を示す。両方増加しても正負の分布全体の変化は確定できない。',
        'Fullは未来方向を含む。CausalはKey<=Queryの実要素だけを平均し、対角を含む。UUは400/210要素、UOは20*mで両方同じ、OOはm²/m(m+1)/2。',
        'OO・UU・UOのQueryは最終Queryとは異なる。Region平均の増加をSoftmax Attention割合増加、Attention Hijackingや攻撃成功の因果機構と同一視しない。',
        'Headは独立標本ではない。固定群は同じ18件で既に選択されたため独立検証ではなく、有意性・一般化を主張しない。正負別エネルギー、追加Softmax、新規推論は実施していない。',
        '', '## 再現', '```bash',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_gcg_shuffled_signed_means tests.test_gcg_shuffled_regions tests.test_gcg_shuffled_tokens tests.test_gcg_shuffled_four_metrics',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_signed_means.py --output-dir data/outputs/gcg_shuffled_signed_means/NEW_RUN', '```',
        'NEW_RUNは未作成の新しいディレクトリ。入力・出力ハッシュ、計算定義・環境・Token範囲はmanifest.jsonとoutput_hashes.json。検証結果はvalidation.json。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
