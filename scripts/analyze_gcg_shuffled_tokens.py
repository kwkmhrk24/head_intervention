#!/usr/bin/env python3
"""Explain existing suffix metrics using saved token-position scores; CPU only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd

from scripts.analyze_gcg_shuffled_four_metrics import inspect_inputs,write_csv,write_json
from utils.artifact_io import read_json,sha256_file,utc_now
from utils.gcg_shuffled_metrics import METRICS
from utils.gcg_shuffled_token_metrics import select_heads,reduce_views,closing_queries,matrix_diagnostics,sample_summary

PREVIOUS=ROOT/'data/outputs/gcg_shuffled_four_metrics/legacy_selection20_20261107T060324Z'
SCALARS=('M_U','M_O','D','F','F_sym','F_asym','R_sym','R_asym')


def summarize_fields(fields,indices):
    result={}
    for name,values in fields.items():
        stats=sample_summary(values,indices)
        for stat,array in stats.items():result[stat+'__'+name]=array
    for condition in ('gcg','shuffled'):
        result['square_of_mean_raw__'+condition]=result['mean__raw_'+condition]**2
        gap=result['mean__square_'+condition]-result['square_of_mean_raw__'+condition]
        if np.min(gap)<-1e-10:raise ValueError('mean(square) below square(mean)')
        result['jensen_gap__'+condition]=gap
    return result


def scalar_table(values,indices,views):
    rows=[]
    for n,positions in ((18,[i for i,index in enumerate(indices) if index not in (91,123)]),(20,list(range(20)))):
        for v,view in enumerate(views):
            for m,metric in enumerate(SCALARS):
                g=values[positions,v,0,m];s=values[positions,v,1,m];d=g-s
                rows.append(dict(subset=n,view_id=view['view_id'],metric=metric,sample_count=len(positions),
                    gcg_mean=g.mean(),shuffled_mean=s.mean(),delta_mean=d.mean(),delta_median=np.median(d),
                    delta_std=d.std(ddof=1),delta_mean_absolute=np.abs(d).mean(),
                    positive_count=int((d>1e-12).sum()),negative_count=int((d < -1e-12).sum())))
    return pd.DataFrame(rows)


def query_table(queries,indices,views):
    rows=[]
    for n,positions in ((18,[i for i,index in enumerate(indices) if index not in (91,123)]),(20,list(range(20)))):
        # Query metrics are evaluated before averaging over query/head/sample.
        x=queries[positions]
        x=np.concatenate([x,x.mean(axis=3,keepdims=True)],axis=3)
        for v,view in enumerate(views):
            for q in range(5):
                for m,metric in enumerate(('M_U','M_O','D')):
                    g=x[:,v,0,q,m];s=x[:,v,1,q,m];d=g-s
                    rows.append(dict(subset=n,view_id=view['view_id'],query_slot=str(q+1) if q<4 else 'mean4',
                        metric=metric,gcg_mean=g.mean(),shuffled_mean=s.mean(),delta_mean=d.mean(),
                        delta_median=np.median(d),delta_std=d.std(ddof=1),sample_count=len(positions),
                        positive_count=int((d>1e-12).sum()),negative_count=int((d < -1e-12).sum())))
    return pd.DataFrame(rows)


def row_aggregates(rows,primary_positions):
    result={}
    for n,positions in ((18,primary_positions),(20,list(range(20)))):
        fields={key:[] for key in rows[0] if key not in ('index','ranges','input_ids')}
        for i in positions:
            row=rows[i];u0,u1=row['ranges']['suffix'];c0,c1=row['ranges']['template_closing'];p0,p1=row['ranges']['template_prefix']
            for key in fields:
                value=row[key]
                if key.startswith('region_') or key.startswith('D_'):
                    fields[key].append(value)
                else:
                    # Only suffix / identical template-relative positions align
                    # across samples; Original positions are never averaged.
                    fields[key].append(value[...,u0:u1])
        subset={}
        for key,values in fields.items():
            stats=sample_summary(np.stack(values),list(range(len(values))))
            for stat,array in stats.items():subset[stat+'__'+key]=array
        for key in ('raw_gcg','raw_shuffled','delta_raw','abs_delta_raw','weight_gcg','weight_shuffled','delta_weight','abs_delta_weight'):
            ending=np.stack([rows[i][key][...,slice(*rows[i]['ranges']['template_closing'])] for i in positions])
            for stat,array in sample_summary(ending,list(range(len(ending)))).items():
                subset['closing_'+stat+'__'+key]=array
        result[n]=subset
    return result


def run(directory):
    if directory.exists():raise FileExistsError('new analysis directory required')
    dataset,primary,references=inspect_inputs()
    old_manifest=read_json(PREVIOUS/'manifest.json');old_hashes=read_json(PREVIOUS/'output_hashes.json')
    if old_manifest['primary_indices']!=primary:raise ValueError('primary18 differs from previous analysis')
    for name in ('metrics.csv','summary_18.csv','paired_differences.csv','manifest.json'):
        if sha256_file(PREVIOUS/name)!=old_hashes[name]:raise ValueError('previous analysis hash mismatch')
    summary=pd.read_csv(PREVIOUS/'summary_18.csv');selected,views=select_heads(summary)
    old_metrics=pd.read_csv(PREVIOUS/'metrics.csv').set_index(['dataset_index','condition']).sort_index()
    ids=[r['dataset_index'] for r in dataset];positions18=[i for i,index in enumerate(ids) if index in primary]
    refs={(r['dataset_index'],r['condition']):r for r in references}
    directory.mkdir(parents=True,exist_ok=False);(directory/'row_profiles').mkdir()
    write_csv(directory/'selected_heads.csv',selected)
    memberships=[]
    for view in views:
        if view['kind']!='individual':continue
        member=view['members'][0];part=selected[(selected.layer==member//32)&(selected['head']==member%32)]
        memberships.append(dict(view_id=view['view_id'],layer=member//32,head=member%32,
                                selection_count=len(part),groups=';'.join(part.group_id)))
    write_csv(directory/'head_memberships.csv',pd.DataFrame(memberships))
    source_files=[Path(__file__),ROOT/'utils/gcg_shuffled_common.py',ROOT/'utils/gcg_shuffled_token_metrics.py',ROOT/'utils/gcg_shuffled_token_plots.py',
        ROOT/'tests/test_gcg_shuffled_tokens.py',ROOT/'utils/gcg_shuffled_metrics.py',ROOT/'scripts/analyze_gcg_shuffled_four_metrics.py',
        PREVIOUS/'manifest.json',PREVIOUS/'summary_18.csv',PREVIOUS/'metrics.csv',PREVIOUS/'paired_differences.csv']
    source_hashes={str(path):sha256_file(path) for path in source_files}
    manifest=dict(created_at=utc_now(),role='exploratory_same_primary18_selection_and_consistency',
        primary_indices=primary,all_indices=ids,views=views,condition_order=['gcg','shuffled'],
        selection='positive descending top5 / negative ascending bottom5, primary18 signed mean; stable Layer/Head ties',
        selections=len(selected),unique_selected_heads=len(memberships),
        missing_direction_counts={f'{m}_{d}':int(len(selected[(selected.metric==m)&(selected.direction==d)])) for m in METRICS for d in ('increase','decrease')},
        alignment='suffix relative positions, not token-identity matching; different Original lengths never position-averaged',
        mean_order='derive/softmax/square/normalize per head and sample, mean heads within view, then mean samples; also save square of mean separately',
        consistency_sign_tolerance=1e-12,zero_norm_policy='undefined ratios trigger diagnostic stop; no zero norms in prior data',
        four_queries='actual closing range and IDs; causal visible keys <= query; mean4 averages calculated metrics',
        norm_attribution='delta square sums to delta norm squared; delta square/(normG+normS) sums to delta norm',
        raw_scores_contain_future=True,source_score_files_referenced_not_copied=True,
        source_hashes=source_hashes,score_references=references,previous_analysis=str(PREVIOUS),
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
        model_loaded=False,forward=False,independent100_used=False)
    write_json(directory/'manifest.json',manifest)
    matrix_lists=defaultdict(list);row_records=[];query_all=[];query_views=[];norm_all=[];scalar_views=[];checks=[]
    for row in dataset:
        index=row['dataset_index'];pair=[];blocks=[]
        for condition in ('gcg','shuffled'):
            reference=refs[index,condition];item=reference['input']
            scores=np.load(reference['path'],mmap_mode='r',allow_pickle=False)
            queries=closing_queries(scores,item);pair.append(queries)
            u0,u1=item['token_ranges']['suffix']
            blocks.append(np.asarray(scores[:,:,u0:u1,u0:u1],dtype=np.float64).reshape(1024,20,20))
        fields,norms=matrix_diagnostics(*blocks)
        for key,value in fields.items():matrix_lists[key].append(reduce_views(value,views))
        qm=np.stack([x['metrics'] for x in pair]);query_all.append(qm)
        qv=np.stack([reduce_views(x['metrics'],views) for x in pair],axis=1);query_views.append(qv)
        nv=np.stack([np.stack([x[k] for k in ('F','F_sym','F_asym','R_sym','R_asym')],axis=-1) for x in norms]);norm_all.append(nv)
        scalar=np.stack([np.concatenate([pair[c]['metrics'][:,3,:],nv[c]],axis=-1) for c in range(2)])
        # SCALARS matches M_U,M_O,D,F,F_sym,F_asym,R_sym,R_asym.
        scalar_views.append(np.stack([reduce_views(x,views) for x in scalar],axis=1))
        error=0.0
        for c,condition in enumerate(('gcg','shuffled')):
            old=old_metrics.loc[(index,condition)].sort_values(['layer','head'])
            for m,metric in enumerate(SCALARS):
                if metric=='R_sym':continue
                target=old[metric].to_numpy();np.testing.assert_allclose(scalar[c,:,m],target,rtol=1e-10,atol=1e-11)
                error=max(error,float(np.max(np.abs(scalar[c,:,m]-target))))
        profile=dict(index=index,ranges=row['conditions']['gcg']['token_ranges'],
            input_ids={c:row['conditions'][c]['input_ids'] for c in ('gcg','shuffled')})
        for feature in ('raw','weight'):
            g,s=[x[feature] for x in pair]
            for key,value in ((feature+'_gcg',g),(feature+'_shuffled',s),('delta_'+feature,g-s),('abs_delta_'+feature,np.abs(g-s))):
                profile[key]=reduce_views(value,views)
        for c,condition in enumerate(('gcg','shuffled')):
            profile['region_'+condition]=reduce_views(pair[c]['region_mass'],views)
        profile['region_delta']=profile['region_gcg']-profile['region_shuffled']
        profile['D_suffix_logmass_delta']=reduce_views(pair[0]['log_mass'][...,0]-pair[1]['log_mass'][...,0],views)
        profile['D_original_logmass_delta_negated']=reduce_views(pair[1]['log_mass'][...,1]-pair[0]['log_mass'][...,1],views)
        np.testing.assert_allclose(profile['D_suffix_logmass_delta']+profile['D_original_logmass_delta_negated'],qv[:,0,:,2]-qv[:,1,:,2],rtol=1e-10,atol=1e-11)
        u0,u1=profile['ranges']['suffix']
        np.testing.assert_allclose(profile['delta_weight'][...,u0:u1].sum(axis=-1),qv[:,0,:,0]-qv[:,1,:,0],rtol=1e-10,atol=1e-11)
        for name,value in profile.items():
            if isinstance(value,np.ndarray) and not np.isfinite(value).all():raise ValueError('nonfinite row profile')
        np.savez_compressed(directory/'row_profiles'/f'index_{index:04d}.npz',**{k:v for k,v in profile.items() if isinstance(v,np.ndarray)})
        row_records.append(profile);checks.append(dict(dataset_index=index,old_metric_max_absolute_error=error,identities_passed=True))
        print(f'processed index={index}; all1024 once; {len(views)} reusable views',flush=True)
    matrices={k:np.stack(v) for k,v in matrix_lists.items()};query_array=np.stack(query_all)
    norms_array=np.stack(norm_all);view_scalar=np.stack(scalar_views);view_queries=np.stack(query_views)
    for name,array in matrices.items():
        if not np.isfinite(array).all():raise ValueError(f'nonfinite/undefined matrix feature: {name}')
    np.savez_compressed(directory/'matrix_per_sample.npz',**matrices)
    np.savez_compressed(directory/'all_head_scalars.npz',query_metrics=query_array,norm_metrics=norms_array,
                        mean4_query_metrics=query_array.mean(axis=3),view_scalar=view_scalar,view_queries=view_queries)
    write_csv(directory/'scalar_consistency.csv',scalar_table(view_scalar,ids,views))
    write_csv(directory/'four_query_summary.csv',query_table(view_queries,ids,views))
    rows_aggregated=row_aggregates(row_records,positions18)
    for n,positions in ((18,positions18),(20,list(range(20)))):
        stats=summarize_fields(matrices,positions)
        np.savez_compressed(directory/f'matrix_summary_{n}.npz',**stats)
        np.savez_compressed(directory/f'row_summary_{n}.npz',**rows_aggregated[n])
    # Compact position/cell CSVs: aggregates remain available for every field in NPZ.
    positions=[];energy=[]
    for n in (18,20):
        rs=rows_aggregated[n]
        with np.load(directory/f'matrix_summary_{n}.npz',allow_pickle=False) as ms:
            for v,view in enumerate(views):
                for p in range(20):
                    positions.append(dict(subset=n,view_id=view['view_id'],suffix_position=p,
                        gcg_weight=rs['mean__weight_gcg'][v,3,p],shuffled_weight=rs['mean__weight_shuffled'][v,3,p],
                        delta_weight=rs['mean__delta_weight'][v,3,p],median_delta=rs['median__delta_weight'][v,3,p],
                        std_delta=rs['std__delta_weight'][v,3,p],positive_count=rs['positive_count__delta_weight'][v,3,p],
                        negative_count=rs['negative_count__delta_weight'][v,3,p],
                        mean_abs_head_sample_delta=rs['mean__abs_delta_weight'][v,3,p],
                        mean_abs_sample_view_delta=rs['mean_absolute__delta_weight'][v,3,p],
                        raw_score_delta=rs['mean__delta_raw'][v,3,p]))
                for p in range(20):
                    energy.append(dict(subset=n,view_id=view['view_id'],suffix_key_position=p,
                        delta_F2_key_contribution=ms['mean__delta_square'][v,:,p].sum(),
                        delta_F_key_contribution=ms['mean__F_delta_contribution'][v,:,p].sum(),
                        delta_Fsym_key_contribution=ms['mean__F_sym_delta_contribution'][v,:,p].sum(),
                        delta_Fasym_key_contribution=ms['mean__F_asym_delta_contribution'][v,:,p].sum(),
                        delta_Rasym_key_contribution=ms['mean__delta_normalized_asym_square'][v,:,p].sum()))
    write_csv(directory/'suffix_position_summary.csv',pd.DataFrame(positions))
    write_csv(directory/'energy_position_contributions.csv',pd.DataFrame(energy))
    write_json(directory/'validation.json',dict(status='passed',samples=20,primary_samples=18,heads_per_sample=1024,
        selected_unique_heads=len(memberships),selection_rows=len(selected),views=len(views),checks=checks,
        reconstruction=True,energy_identity=True,ratios_sum_one=True,causal_four_queries=True,
        final_query_matches_previous=True,delta_MU_token_sum=True,delta_D_logmass_sum=True,
        exact_norm_contributions=True,mean_square_order_verified=True))
    from utils.gcg_shuffled_token_plots import make_figures
    make_figures(directory)
    write_report(directory,manifest)
    if any(sha256_file(path)!=digest for path,digest in source_hashes.items()) or any(sha256_file(r['path'])!=r['sha256'] for r in references):
        raise ValueError('source changed during analysis')
    write_json(directory/'output_hashes.json',{str(p.relative_to(directory)):sha256_file(p) for p in directory.rglob('*') if p.is_file()})
    print(f'Completed {directory}',flush=True)


def write_report(directory,manifest):
    selected=pd.read_csv(directory/'selected_heads.csv');scalar=pd.read_csv(directory/'scalar_consistency.csv');queries=pd.read_csv(directory/'four_query_summary.csv')
    lines=['# GCG/Shuffled Token位置分析','',
        'CPU上で既存のScoreと既存指標を再利用。モデルロード・Forward・抽出・Suffix再生成・ASR再評価・介入・独立100件の使用なし。','',
        '## Head選択','',f'{manifest["selections"]}選択、重複を除く{manifest["unique_selected_heads"]} Head。{len(manifest["views"])} view。主18件で選択し、20件でも同じHeadを使用。5件未満の方向は存在するHeadだけを使用し、件数はmanifestのmissing_direction_countsに記録。','',
        '| 指標 | 増加上位5 | 減少下位5 |','|---|---|---|']
    for metric in METRICS:
        cells=[]
        for direction in ('increase','decrease'):
            rows=selected[(selected.metric==metric)&(selected.direction==direction)]
            cells.append(', '.join(f'L{int(r.layer)}H{int(r.head)}' for r in rows.itertuples()))
        lines.append(f'| {metric} | {cells[0]} | {cells[1]} |')
    lines+=['','順位・平均差・中央値・SD・増減サンプル数・重複はselected_heads.csv/head_memberships.csv。','',
        '## 寄与と平均の定義','',
        'Suffixは相対位置0〜19で比較し、同じ位置のToken IDが条件間・サンプル間で同一とは仮定しない。Originalの絶対位置はサンプル間平均せず、領域massと個別Index行分布を保存。',
        '各Head・各サンプルでSoftmax/二乗/正規化を計算後、view内Head平均、その後18/20サンプル平均。全Head平均は等重み記述で、代表的Headを仮定しない。',
        'ΣΔ(B²)=F_G²−F_S²であり、ΔFとは異なる。追加の要素寄与Δ(B²)/(F_G+F_S)はΣ=ΔFとして既存Fの変化に直接対応。Sym/Asymも同様。',
        '反対称二乗要素を各Head・サンプルのF²で割ってから平均し、Δ正規化寄与の和が平均ΔR_asymに一致する。R_sym+R_asym=1を検証。',
        'matrix_summaryのmean__squareとsquare_of_mean_rawは別保存。符号付き差、各Head差の絶対値平均、サンプルview差の絶対値平均を区別し、符号打消しを隠さない。','',
        '## 全Head・選択群の既存指標差','',
        '| view | ΔM_U | ΔD | ΔF | ΔR_asym |','|---|---:|---:|---:|---:|']
    for view in [v for v in manifest['views'] if v['kind']!='individual']:
        sub=scalar[(scalar.subset==18)&(scalar.view_id==view['view_id'])].set_index('metric')
        lines.append('| '+view['view_id']+' | '+' | '.join(f'{sub.loc[m,"delta_mean"]:+.6g}' for m in ('M_U','D','F','R_asym'))+' |')
    lines+=['','## 4 Query（全Head平均、主要18件）','',
        '実終了範囲とIDs=[518,29914,25580,29962]を確認。q1〜q3の未来終了TokenはSoftmax分母から除外。mean4は各Query指標を先に計算した平均で、既存q4指標は変更しない。','',
        '| Query | GCG M_U | Shuffled M_U | ΔM_U | ΔD |','|---|---:|---:|---:|---:|']
    for q in ('1','2','3','4','mean4'):
        sub=queries[(queries.subset==18)&(queries.view_id=='all_heads')&(queries.query_slot==q)].set_index('metric')
        lines.append(f'| {q} | {sub.loc["M_U","gcg_mean"]:.6g} | {sub.loc["M_U","shuffled_mean"]:.6g} | {sub.loc["M_U","delta_mean"]:+.6g} | {sub.loc["D","delta_mean"]:+.6g} |')
    lines+=['','## 一貫性と限界','',
        'selected scalar consistencyはscalar_consistency.csv、位置別mean/median/SD/増減数/絶対差はsuffix_position_summary.csvとsummary NPZに保存。選択した同じ18件での確認で、独立検証・有意差・因果を主張しない。',
        'Suffix行列にはMask前の未来要素を含む。反対称性を実際の情報流方向と同一視しない。平均の二乗は二乗平均より小さくなり得て、平均行列の見かけだけではノルム変化を説明できない。','',
        '## データと図','',
        'matrix_per_sample.npzは[20 sample,view,20 query,20 key]の派生部分行列・寄与で、元の全Scoreファイルは複製せず参照。row_profiles/index_*.npzは[view,4 query,L key]。Original位置の平均はなし。',
        'all_head_scalars.npzは[20,2,1024,4,3]のquery_metrics（M_U,M_O,D）、[20,2,1024,5]のnorm_metrics（F,F_sym,F_asym,R_sym,R_asym）。',
        'matrix_summary_18/20、row_summary_18/20は平均・中央値・標本SD(ddof=1)・絶対差・増減件数を保存。summaryのsuffix末軸は相対位置。',
        '画像は個別サンプルごとに大量生成せず、選択Headの比較、群平均、全Head平均、一貫性、4Queryを固定数で作成。','',
        '## 再現','',
        '```bash',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_gcg_shuffled_tokens -v',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_tokens.py --output-dir data/outputs/gcg_shuffled_token_analysis/NEW_RUN',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_tokens.py --analysis-dir RUN --render-index 68 --view L26H21',
        '```','',
        'Tokenの意味に関する仮説や因果解釈は、図の位置変化だけから断定しない。詳細な結果解釈は追加のinterpretation.mdを参照。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--analysis-dir',type=Path)
    parser.add_argument('--render-index',type=int)
    parser.add_argument('--view',default='all_heads')
    args=parser.parse_args()
    if args.render_index is not None:
        if args.analysis_dir is None:parser.error('--analysis-dir required')
        from utils.gcg_shuffled_token_plots import render_index
        render_index(args.analysis_dir,args.render_index,args.view)
    else:
        if args.output_dir is None:parser.error('--output-dir required')
        run(args.output_dir)


if __name__=='__main__':main()
