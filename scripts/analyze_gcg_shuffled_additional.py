#!/usr/bin/env python3
"""CPU sign energy, same-query attention and row-centered diagnostics."""
from pathlib import Path
import argparse
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from utils.gcg_shuffled_common import inspect_inputs,write_json,write_csv,summary_frame
from utils.artifact_io import read_json,sha256_file,utc_now
from utils.gcg_shuffled_token_metrics import reduce_views
from utils.gcg_shuffled_additional_metrics import SIGN_METRICS,QUERY_FIELDS,QUERY_LABELS,sign_metrics,query_diagnostics
from utils.gcg_shuffled_statistics import pearson_correlation,spearman_correlation
REGION=ROOT/'data/outputs/gcg_shuffled_region_analysis/legacy_selection20_20261107T094154Z'
SIGNED=ROOT/'data/outputs/gcg_shuffled_signed_means/legacy_selection20_20261108T_signed_mean_v1'
TOKEN=ROOT/'data/outputs/gcg_shuffled_token_analysis/legacy_selection20_20261107T075659Z'
DEFAULT_ROOT=ROOT/'data/outputs/gcg_vs_shuffled/legacy_selection20'


def export_tables(out,values,names,entities,views,indices,primary,query=False):
    """No reweighting: compute individual features, equal Head view, sample stats."""
    np.savez_compressed(out/'values.npz',values=values)
    if query:
        vv=np.array([[reduce_views(v,views) for v in pair] for pair in values])
        flat=values.reshape(20,2,1024*25,len(names));vf=vv.reshape(20,2,len(views)*25,len(names))
        ventities=[dict(view_id=v['view_id'],view_kind=v['kind'],head_count=len(v['members']),query=q) for v in views for q in QUERY_LABELS]
    else:
        flat=values;vv=np.array([[reduce_views(v,views) for v in pair] for pair in values]);vf=vv
        ventities=[dict(view_id=v['view_id'],view_kind=v['kind'],head_count=len(v['members'])) for v in views]
    frames=[];paired=[];sample=[];overview=[];members=[]
    fixed=pd.read_csv(REGION/'selected_heads_reference.csv')
    for i,index in enumerate(indices):
        for c,condition in enumerate(('gcg','shuffled')):
            f=pd.DataFrame(entities);f.insert(0,'condition',condition);f.insert(0,'dataset_index',index)
            f=pd.concat([f,pd.DataFrame(flat[i,c],columns=names)],axis=1);frames.append(f)
        d=pd.DataFrame(entities);d.insert(0,'dataset_index',index)
        paired.append(pd.concat([d,pd.DataFrame(flat[i,0]-flat[i,1],columns=['delta_'+n for n in names])],axis=1))
    write_csv(out/'metrics.csv',pd.concat(frames,ignore_index=True));write_csv(out/'paired_differences.csv',pd.concat(paired,ignore_index=True))
    for n,pos in ((18,[indices.index(i) for i in primary]),(20,list(range(20)))):
        summary=summary_frame(flat,pos,names,entities);group=summary_frame(vf,pos,names,ventities)
        write_csv(out/f'head_summary_{n}.csv',summary);write_csv(out/f'view_summary_{n}.csv',group)
        overview.extend(dict(subset=n,**r) for r in group[group.view_id=='all_heads'].to_dict('records'))
        member_parts=[]
        for member in fixed.itertuples():
            part=summary[(summary.layer==member.layer)&(summary['head']==member.head)].copy()
            part.insert(0,'selected_group',member.group_id);part.insert(0,'old_rank',member.rank);part.insert(0,'subset',n)
            member_parts.append(part)
        members.append(pd.concat(member_parts,ignore_index=True))
    write_csv(out/'overview.csv',pd.DataFrame(overview));write_csv(out/'fixed_group_member_effects.csv',pd.concat(members,ignore_index=True))
    # Already-computed per Head features averaged within each view, per sample.
    for i,index in enumerate(indices):
        for vi,v in enumerate(views):
            if v['kind']=='individual':continue
            for qi,q in enumerate(QUERY_LABELS if query else [None]):
                for j,name in enumerate(names):
                    x=vv[i,0,vi,qi,j] if query else vv[i,0,vi,j];y=vv[i,1,vi,qi,j] if query else vv[i,1,vi,j]
                    sample.append(dict(dataset_index=index,primary18=index in primary,view_id=v['view_id'],query=q,
                                       metric=name,gcg=x,shuffled=y,delta=x-y))
    write_csv(out/'sample_summary.csv',pd.DataFrame(sample))
    return vv


def run(root,run_id):
    paths={key:root/folder/run_id for key,folder in [('sign','30_sign_decomposition'),('query','40_query_softmax'),('center','50_row_centered'),('integrated','60_integrated_analysis')]}
    if any(p.exists() for p in paths.values()):raise FileExistsError('all run paths must be new')
    dataset,primary,refs=inspect_inputs();old=read_json(REGION/'manifest.json');indices=[r['dataset_index'] for r in dataset]
    if indices!=old['all_indices'] or primary!=old['primary_indices']:raise ValueError('old cohort differs')
    views=old['views'];lookup={(r['dataset_index'],r['condition']):r for r in refs}
    old_files=[REGION/n for n in ['manifest.json','per_sample_values.npz','element_counts.csv','selected_heads_reference.csv']]+[SIGNED/'signed_means.npz',SIGNED/'manifest.json',TOKEN/'all_head_scalars.npz',TOKEN/'four_query_summary.csv']
    for source in (REGION,SIGNED,TOKEN):
        hashes=read_json(source/'output_hashes.json')
        for p in old_files:
            if p.parent==source and sha256_file(p)!=hashes[p.name]:raise ValueError('old artifact hash mismatch')
    code=[Path(__file__),ROOT/'utils/gcg_shuffled_statistics.py',ROOT/'utils/artifact_io.py',ROOT/'utils/gcg_shuffled_additional_metrics.py',ROOT/'utils/gcg_shuffled_additional_plots.py',ROOT/'tests/test_gcg_shuffled_additional.py']
    code += [ROOT/p for p in ['scripts/analyze_gcg_shuffled_four_metrics.py','scripts/analyze_gcg_shuffled_regions.py','utils/gcg_shuffled_region_metrics.py','utils/gcg_shuffled_token_metrics.py','utils/gcg_shuffled_metrics.py']]
    # Include shared implementation once it is extracted in the compatibility phase.
    shared=ROOT/'utils/gcg_shuffled_common.py'
    if shared.exists():code.extend([shared,ROOT/'utils/gcg_shuffled_plot_common.py'])
    source_hashes={str(p):sha256_file(p) for p in code+old_files}
    with np.load(REGION/'per_sample_values.npz') as z:rms=z['region_metrics']
    with np.load(SIGNED/'signed_means.npz') as z:mu=z['values']
    with np.load(TOKEN/'all_head_scalars.npz') as z:oldq=z['query_metrics']
    oldcounts=pd.read_csv(REGION/'element_counts.csv').set_index('dataset_index')
    for p in paths.values():p.mkdir(parents=True)
    (paths['query']/'row_profiles').mkdir()
    energies=[];queries=[];counts=[];qcounts=[];validations=[]
    for i,index in enumerate(indices):
        ep=[];qp=[];pair_profiles={};paired_counts=[]
        for c,condition in enumerate(('gcg','shuffled')):
            ref=lookup[index,condition];item=ref['input'];scores=np.load(ref['path'],mmap_mode='r',allow_pickle=False)
            e,n=sign_metrics(scores,item['token_ranges']);q,profile=query_diagnostics(scores,item)
            if n!=oldcounts.loc[index].to_dict():raise ValueError('old region element counts differ')
            paired_counts.append(n);ev=np.stack([e[name].reshape(1024) for name in SIGN_METRICS],axis=-1)
            energy_error=mean_error=0.
            for policy in ('full','causal'):
                for region in ('OO','UU','UO'):
                    total=e[f'E_pos_{region}_{policy}']+e[f'E_neg_{region}_{policy}']
                    reference=rms[i,c,:,old['metrics'].index(f'RMS_{region}_{policy}')].reshape(32,32)**2
                    np.testing.assert_allclose(total,reference,rtol=1e-12,atol=1e-12)
                    energy_error=max(energy_error,float(np.abs(total-reference).max()))
                    signed=e[f'mu_pos_{region}_{policy}']+e[f'mu_neg_{region}_{policy}']
                    oldmu=mu[i,c,:,['mu_OO_full','mu_UU_full','mu_UO_full','mu_OO_causal','mu_UU_causal','mu_UO_causal'].index(f'mu_{region}_{policy}')].reshape(32,32)
                    np.testing.assert_allclose(signed,oldmu,rtol=1e-12,atol=1e-12)
                    mean_error=max(mean_error,float(np.abs(signed-oldmu).max()))
            np.testing.assert_allclose(q[:,20:24,:3],oldq[i,c],rtol=1e-12,atol=1e-11)
            validations.append(dict(dataset_index=index,condition=condition,energy_identity_max_error=energy_error,
                signed_identity_max_error=mean_error,all_four_queries_old_max_error=float(np.max(np.abs(q[:,20:24,:3]-oldq[i,c]))),
                **{k:profile[k] for k in ('centered_softmax_max_error','centered_mean_max_error','centered_D_max_error')}))
            for name in ('raw','centered','weight'):pair_profiles[condition+'_'+name]=reduce_views(profile[name],views)
            pair_profiles[condition+'_input_ids']=np.array(item['input_ids'])
            for qi,pos in enumerate(profile['query_positions']):
                qcounts.append(dict(dataset_index=index,condition=condition,query=QUERY_LABELS[qi],absolute_query=int(pos),
                    visible_count=int(profile['visible'][qi].sum()),**{k:int(v[qi]) for k,v in profile['counts'].items()}))
            ep.append(ev);qp.append(q)
        if paired_counts[0]!=paired_counts[1]:raise ValueError('pair count mismatch')
        # Original has identical prefixes and must remain unchanged.
        for policy in ('full','causal'):
            for field in ('E_pos','E_neg','mu_pos','mu_neg'):
                j=SIGN_METRICS.index(f'{field}_OO_{policy}');np.testing.assert_array_equal(ep[0][:,j],ep[1][:,j])
        counts.append(dict(dataset_index=index,**paired_counts[0]))
        np.savez_compressed(paths['query']/'row_profiles'/f'index_{index:03d}.npz',**pair_profiles,query_positions=profile['query_positions'],visible=profile['visible'])
        energies.append(ep);queries.append(qp)
        print('Computed:',index,flush=True)
    energies=np.asarray(energies);queries=np.asarray(queries)
    heads=[dict(layer=h//32,head=h%32) for h in range(1024)]
    export_tables(paths['sign'],energies,SIGN_METRICS,heads,views,indices,primary)
    qentities=[dict(layer=h//32,head=h%32,query=q) for h in range(1024) for q in QUERY_LABELS]
    export_tables(paths['query'],queries,QUERY_FIELDS,qentities,views,indices,primary,query=True)
    write_csv(paths['sign']/'element_counts.csv',pd.DataFrame(counts));write_csv(paths['query']/'visible_counts.csv',pd.DataFrame(qcounts))
    # Shared query/centered values stored once, with relative links from the centered stage.
    import os
    for name in ('metrics.csv','paired_differences.csv','values.npz','head_summary_18.csv','head_summary_20.csv','view_summary_18.csv','view_summary_20.csv','sample_summary.csv','overview.csv','fixed_group_member_effects.csv'):
        (paths['center']/name).symlink_to(os.path.relpath(paths['query']/name,paths['center']))
    names=[];parts=[]
    for policy in ('full','causal'):
        for region in ('OO','UU','UO'):
            r=f'{region}_{policy}'
            for f in ('RMS','mu'):
                name=f'{f}_{r}';names.append(name)
                parts.append(rms[:,:,:,old['metrics'].index(name)] if f=='RMS' else mu[:,:,:,['mu_OO_full','mu_UU_full','mu_UO_full','mu_OO_causal','mu_UU_causal','mu_UO_causal'].index(name)])
            for f in ('E_pos','E_neg','mu_pos','mu_neg'):
                name=f'{f}_{r}';names.append(name);parts.append(energies[:,:,:,SIGN_METRICS.index(name)])
    for slot,label in ((24,'suffix_mean20'),(23,'q4')):
        for f in ('M_U','M_O','D','conditional_M_U','row_mean','raw_O','raw_U','centered_O','centered_U','gap_U_minus_O'):
            names.append(f'{label}_{f}');parts.append(queries[:,:,:,slot,QUERY_FIELDS.index(f)])
    integrated=np.stack(parts,axis=-1);export_tables(paths['integrated'],integrated,names,heads,views,indices,primary)
    correlations=[]
    pairs=[('RMS_UU_full','E_neg_UU_full'),('RMS_UU_full','E_pos_UU_full'),('mu_UU_full','mu_neg_UU_full'),
        ('suffix_mean20_gap_U_minus_O','suffix_mean20_D'),('suffix_mean20_centered_U','suffix_mean20_M_U'),
        ('q4_gap_U_minus_O','q4_D'),('RMS_UU_full','suffix_mean20_M_U'),('mu_UU_full','q4_M_U')]
    for n,pos in ((18,[indices.index(i) for i in primary]),(20,list(range(20)))):
        d=integrated[pos,0]-integrated[pos,1]
        for unit,x in [('head_mean',d.mean(axis=0)),('sample_head_mean',d.mean(axis=1))]:
            for a,b in pairs:
                xx=x[:,names.index(a)];yy=x[:,names.index(b)]
                correlations.append(dict(subset=n,unit=unit,x=a,y=b,count=len(xx),pearson=pearson_correlation(xx,yy),spearman=spearman_correlation(xx,yy),
                    same_direction_fraction=float(((xx>0)&(yy>0)|(xx<0)&(yy<0)).mean()),interpretation='descriptive; dependent Heads; no significance or causality'))
    write_csv(paths['integrated']/'correlations.csv',pd.DataFrame(correlations))
    manifest=dict(created_at=utc_now(),run_id=run_id,primary_indices=primary,all_indices=indices,views=views,
        condition_order=['gcg','shuffled'],head_order='layer*32+head (zero-based)',sign_metrics=list(SIGN_METRICS),query_fields=list(QUERY_FIELDS),
        query_labels=list(QUERY_LABELS),integrated_metrics=names,score_references=refs,source_hashes=source_hashes,
        model_info=old['model_info'],numpy=np.__version__,pandas=pd.__version__,python=platform.python_version(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        formula='FP64 per sample/Head features; equal Head view mean then sample statistics; SD ddof=1',
        sign_policy='full all elements including future; causal Key<=Query including diagonal; N=all valid elements; quantiles linear',
        query_policy='each visible row all Keys denominator; suffix positions aligned, not token identities; mean20 after per-query calculation',
        undefined_policy='raw_other undefined where visible other count=0 (suffix queries/mean20); explicit NaN and valid count=0; no imputation',
        row_center_policy='subtract mean of all visible Keys; no future elements; centered metrics share query tables',
        scientific_limits='outcome-conditioned primary18; refusal-rule labels only; same-sample fixed Head selection; no independent Heads, causal or significance claims',
        model_loaded=False,GPU_used=False,independent100_used=False)
    for key,path in paths.items():write_json(path/'manifest.json',dict(stage=key,**manifest))
    validation=dict(status='passed',samples=20,primary_samples=18,sign_shape=list(energies.shape),query_shape=list(queries.shape),checks=validations,
                    zero_unexpected_nonfinite=True,old_inputs_and_head_groups_verified=True)
    for path in paths.values():write_json(path/'validation.json',validation)
    from utils.gcg_shuffled_additional_plots import make_figures
    make_figures(paths,manifest)
    write_reports(paths,manifest)
    for p,h in source_hashes.items():
        if sha256_file(p)!=h:raise ValueError('source changed during analysis')
    for ref in refs:
        if sha256_file(ref['path'])!=ref['sha256']:raise ValueError('Score changed during analysis')
    for path in paths.values():write_json(path/'output_hashes.json',{str(p.relative_to(path)):sha256_file(p) for p in path.rglob('*') if p.is_file()})
    print('Completed:',paths['integrated'],flush=True)


def write_reports(paths,m):
    lines=['# 追加分析の数値結果','', '主要18件と全20件。既存の固定Head群を維持。FP64でサンプル・Headごとに計算後、等重み集計。', '',
        'Headは独立標本ではない。43拒否表現ルールによる自動判定の探索的比較で、人手確認済み成功・因果・有意性・一般化は主張しない。', '',
        '## 主な全Head平均（元CSV: ../各stage/run/overview.csv）','', '| N | 指標 | GCG | Shuffled | 差 | 差SD | 増/減/同等サンプル |','|---|---|---:|---:|---:|---:|---|']
    overview=pd.read_csv(paths['integrated']/'overview.csv')
    selected=['E_pos_UU_full','E_neg_UU_full','E_pos_UO_full','E_neg_UO_full','mu_pos_UU_full','mu_neg_UU_full','mu_pos_UO_full','mu_neg_UO_full',
              'E_pos_UU_causal','E_neg_UU_causal','suffix_mean20_M_U','suffix_mean20_M_O','suffix_mean20_D','suffix_mean20_row_mean',
              'suffix_mean20_centered_U','suffix_mean20_centered_O','suffix_mean20_gap_U_minus_O','q4_M_U','q4_M_O','q4_D']
    for r in overview[overview.metric.isin(selected)].itertuples():
        lines.append(f'| {r.subset} | {r.metric} | {r.gcg_mean:.8g} | {r.shuffled_mean:.8g} | {r.delta_mean:+.8g} | {r.delta_std:.6g} | {r.gcg_larger_count}/{r.shuffled_larger_count}/{r.equal_count} |')
    lines+=['','## 定義と注意',
        'E_pos/E_negは各領域の全有効要素数で割った二乗エネルギー。差の和は平均RMS²の差であり、平均RMSの差や平均RMSの二乗差とは異なる。',
        'mu_pos+mu_neg=Signed Mean。正負割合・中央値・5/25/50/75/95%点は各サンプル・Headで計算。集約分位点は分位点の平均であり混合分布の分位点ではない。',
        'Suffix Queryは相対位置0-19で対応し、Token内容一致を意味しない。q1-q4は保存入力の終了4 Token。Suffix mean20は各QueryのM/D/中心化指標計算後の平均。',
        'Dは可視SuffixとOriginalのlogsumexp差。conditional_M_Uは両領域間だけの条件付き配分。全Key分母のM_Uとは異なる。',
        'Row中心化は可視Key平均を引く。SoftmaxとDは不変でも領域の中心化平均・相対差は比較できる。単一平均では特定Keyが原因と確定できず、row_profiles/にraw/centered/weightと位置・input_idsを保存。',
        'Suffix Queryのraw_otherは対象Key数0で未定義。NaNを異常値として隠さずvalid count=0で保存。重みM_otherは0。',
        'OO/UU/UOのRegion集計とq4のQueryは異なる。Scoreの符号や二乗エネルギーからAttention割合を断定しない。','',
        '## 再現', '```bash',f'/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_additional.py --output-root data/outputs/gcg_vs_shuffled/legacy_selection20 --run-id NEW_RUN','```',
        'NEW_RUNは未使用のrun ID。90_validation/に互換性監査、統一READMEと画像一覧から全stageを参照できる。詳細な解釈は最終interpretation.md/MTG_summary.md。']
    (paths['integrated']/'report.md').write_text('\n'.join(lines)+'\n')
    for key in ('sign','query','center'):
        (paths[key]/'README.md').write_text(f'# {key}\n\n定義・入力・列順はmanifest.json。指標CSVと18/20のHead・固定群・サンプル集計を保存。\n統合レポート: {paths["integrated"]}/report.md\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=DEFAULT_ROOT);p.add_argument('--run-id',required=True)
    a=p.parse_args();run(a.output_root,a.run_id)
