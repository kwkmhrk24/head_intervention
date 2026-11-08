#!/usr/bin/env python3
"""Attribute saved GCG/Shuffled negative-score energy on CPU; no inference."""
from pathlib import Path
import argparse
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from utils.artifact_io import read_json,sha256_file,utc_now
from utils.gcg_shuffled_common import inspect_inputs,region_layout,write_csv,write_json,reduce_views,summary_frame
from utils.gcg_negative_energy import REGIONS,CONDITION_FIELDS,PAIR_FIELDS,TAILS,paired_factors,concentration,head_distribution,paired_sample_summary,causal_query_factors,defined_stats
PRIOR=ROOT/'data/outputs/gcg_vs_shuffled/legacy_selection20/30_sign_decomposition/20261107T165012Z'
REGION_PRIOR=ROOT/'data/outputs/gcg_shuffled_region_analysis/legacy_selection20_20261107T094154Z'


def source_hashes():
    names=['scripts/analyze_gcg_negative_energy.py','utils/gcg_negative_energy.py','utils/gcg_negative_energy_plots.py',
           'utils/gcg_shuffled_common.py','utils/gcg_shuffled_metrics.py','utils/gcg_shuffled_token_metrics.py',
           'utils/gcg_shuffled_plot_common.py','utils/artifact_io.py','tests/test_gcg_negative_energy.py','utils/gcg_negative_energy_report.py']
    return {str(ROOT/name):sha256_file(ROOT/name) for name in names}


def blocks_for_pair(lookup,index):
    original=lookup[index,'original'];oa,ob=original['input']['token_ranges']['original']
    orig=np.load(original['path'],mmap_mode='r',allow_pickle=False)
    blocks=[];counts=None;masks=None
    for condition in ('gcg','shuffled'):
        ref=lookup[index,condition];scores=np.load(ref['path'],mmap_mode='r',allow_pickle=False)
        b,mask,n=region_layout(scores,ref['input']['token_ranges'])
        if counts is not None and (counts!=n or any(not np.array_equal(mask[k],masks[k]) for k in mask)):
            raise ValueError('paired region/mask mismatch')
        if not np.array_equal(b['OO'],orig[:,:,oa:ob,oa:ob]):raise ValueError('OO differs from Original-only input')
        blocks.append({k:x.reshape(1024,*x.shape[-2:]) for k,x in b.items()});masks=mask;counts=n
    if not np.array_equal(blocks[0]['OO'],blocks[1]['OO']):raise ValueError('OO pair differs')
    return blocks,masks,counts


def selected_elements(blocks,masks,name):
    if name=='UO':region,policy='UO','full'
    else:region,policy=name.split('_')
    mask=masks[region] if policy=='causal' else np.ones_like(masks[region])
    if region=='UO' and not masks['UO'].all():raise ValueError('UO has future keys')
    return [b[region][:,mask] for b in blocks],int(mask.sum())


def pair_summary_frame(values,positions,entities,names):
    stats=paired_sample_summary(values,positions)
    rows=pd.DataFrame([dict(**entity,metric=name,sample_count=len(positions)) for entity in entities for name in names])
    for key,array in stats.items():rows[key]=array.reshape(-1)
    return rows


def summarize_core(out,manifest,c,p,m2):
    tables=out/'tables';ids=manifest['all_indices'];primary=manifest['primary_indices'];views=manifest['views']
    rc=len(REGIONS);nr=len(views);hcount=1024
    vc=np.array([[[reduce_views(region,views) for region in condition] for condition in sample] for sample in c])
    vp=np.array([[reduce_views(region,views) for region in sample] for sample in p])
    entities=[dict(region=region,layer=h//32,head=h%32) for region in REGIONS for h in range(1024)]
    ve=[dict(region=region,view_id=v['view_id'],view_kind=v['kind'],head_count=len(v['members'])) for region in REGIONS for v in views]
    distributions=[];factor_overview=[];tails=[];sample_rows=[];leave=[];layer_rows=[]
    for n,pos in ((18,[ids.index(i) for i in primary]),(20,list(range(20)))):
        head=pair_summary_frame(p.reshape(20,rc*hcount,len(PAIR_FIELDS)),pos,entities,PAIR_FIELDS)
        view=pair_summary_frame(vp.reshape(20,rc*nr,len(PAIR_FIELDS)),pos,ve,PAIR_FIELDS)
        write_csv(tables/f'head_summary_{n}.csv',head);write_csv(tables/f'fixed_view_summary_{n}.csv',view)
        cond=summary_frame(c.reshape(20,2,rc*hcount,len(CONDITION_FIELDS)),pos,CONDITION_FIELDS,entities)
        write_csv(tables/f'condition_head_summary_{n}.csv',cond)
        mean_delta=p[pos,:,:,0].mean(axis=0)
        for ri,region in enumerate(REGIONS):
            for v in views:
                members=v['members'];hd=mean_delta[ri,members]
                distributions.append(dict(subset=n,region=region,view_id=v['view_id'],view_kind=v['kind'],**head_distribution(hd)))
                cp=np.take(c[pos,:,ri],members,axis=2);pp=np.take(p[pos,ri],members,axis=1)
                oldm=np.take(m2[pos,:,ri],members,axis=2)
                both=pp[:,:,PAIR_FIELDS.index('both_have_negative')].astype(bool);de=pp[:,:,0]
                row=dict(subset=n,region=region,view_id=v['view_id'],view_kind=v['kind'],head_count=len(members),sample_count=n,
                    gcg_E_neg=cp[:,0,:,0].mean(),shuffled_E_neg=cp[:,1,:,0].mean(),delta_E_neg=de.mean(),
                    delta_std=de.mean(axis=1).std(ddof=1),delta_median=float(np.median(de.mean(axis=1))),
                    positive_sample_count=int((de.mean(axis=1)>0).sum()),negative_sample_count=int((de.mean(axis=1)<0).sum()),
                    both_negative_pairs=int(both.sum()),zero_boundary_pairs=int((~both).sum()),total_pairs=int(both.size))
                for j,f in enumerate(CONDITION_FIELDS):
                    row['gcg_'+f]=cp[:,0,:,j].mean();row['shuffled_'+f]=cp[:,1,:,j].mean()
                for j,f in enumerate(PAIR_FIELDS):row['mean_'+f]=pp[:,:,j].mean()
                for f in ('frequency_contribution','magnitude_contribution','delta_E_neg'):
                    a=pp[:,:,PAIR_FIELDS.index(f)]
                    row['both_negative_'+f+'_per_original_pair']=np.where(both,a,0).mean()
                    row['zero_boundary_'+f+'_per_original_pair']=np.where(~both,a,0).mean()
                # Conditional moments remain undefined when no negative score exists.
                for condition,label in enumerate(('gcg','shuffled')):
                    x=oldm[:,condition];valid=np.isfinite(x)
                    row[label+'_conditional_m2_defined_cases']=int(valid.sum())
                    row[label+'_conditional_m2_defined_case_mean']=float(x[valid].mean()) if valid.any() else np.nan
                factor_overview.append(row)
            x=mean_delta[ri];positive=np.maximum(x,0).sum();order=np.argsort(-x,kind='stable')
            for frac in TAILS:
                k=int(np.ceil(frac*1024));remaining=x[order[k:]]
                tails.append(dict(subset=n,region=region,removed_fraction=frac,removed_heads=k,
                    original_mean=x.mean(),original_median=np.median(x),remaining_mean=remaining.mean(),
                    remaining_median=np.median(remaining),remaining_positive_fraction=(remaining>0).mean(),
                    retained_signed_sum_per_original_head=remaining.sum()/1024,
                    top_positive_delta_share=float(np.maximum(x[order[:k]],0).sum()/positive) if positive>0 else np.nan,
                    positive_head_delta_total=float(positive),top_positive_delta_share_defined=positive>0,
                    purpose='descriptive sensitivity; not a newly selected Head group'))
            for layer in range(32):
                a=x[layer*32:(layer+1)*32]
                layer_rows.append(dict(subset=n,region=region,layer=layer,**head_distribution(a)))
            sample_means=p[pos,ri,:,0].mean(axis=1)
            for i,index in enumerate([ids[j] for j in pos]):
                d=p[pos[i],ri,:,0];stats=head_distribution(d)
                sample_rows.append(dict(subset=n,dataset_index=index,region=region,**stats))
                leave.append(dict(subset=n,dataset_index_removed=index,region=region,
                    leave_one_sample_out_mean=float(np.delete(sample_means,i).mean()),original_mean=float(sample_means.mean())))
    for name,rows in [('head_distributions',distributions),('factor_overview',factor_overview),('head_tail_sensitivity',tails),
                      ('sample_summary',sample_rows),('leave_one_sample_out',leave),('layer_summary',layer_rows)]:write_csv(tables/(name+'.csv'),pd.DataFrame(rows))


def run_core(out):
    if any((out/name).exists() for name in ('core_manifest.json','core_values.npz','tables/condition_metrics.csv','tables/paired_metrics.csv')):raise FileExistsError('core analysis exists')
    dataset,primary,refs=inspect_inputs();prior=read_json(PRIOR/'manifest.json');old_hashes=read_json(PRIOR/'output_hashes.json')
    ids=[row['dataset_index'] for row in dataset]
    if ids!=prior['all_indices'] or primary!=prior['primary_indices']:raise ValueError('cohort mismatch')
    used=[PRIOR/'manifest.json',PRIOR/'values.npz',PRIOR/'element_counts.csv',REGION_PRIOR/'manifest.json']
    for f in used:
        if f.parent==PRIOR and sha256_file(f)!=old_hashes[f.name]:raise ValueError('prior sign result changed')
    oldcounts=pd.read_csv(PRIOR/'element_counts.csv').set_index('dataset_index')
    with np.load(PRIOR/'values.npz',allow_pickle=False) as z:old=z['values']
    lookup={(r['dataset_index'],r['condition']):r for r in refs};sources=source_hashes()
    out.mkdir(parents=True,exist_ok=True);(out/'tables').mkdir(exist_ok=True)
    cond=[];pairs=[];moments=[];element_counts=[];checks=[];frames=[];pair_frames=[]
    for i,index in enumerate(ids):
        blocks,masks,lengths=blocks_for_pair(lookup,index);cs=[];ps=[];ms=[];counts=[]
        for ri,name in enumerate(REGIONS):
            (g,s),count=selected_elements(blocks,masks,name);gg,ss,p,_,errors=paired_factors(g,s)
            region,policy=(('UO','full') if name=='UO' else name.split('_'))
            if count!=int(oldcounts.loc[index,f'{region}_{policy}']):raise ValueError('old element count mismatch')
            for c,condition in enumerate(('gcg','shuffled')):
                values=(gg,ss)[c]
                for new,existing in [('E_neg','E_neg'),('E_pos','E_pos'),('p_neg','negative_fraction')]:
                    ref=old[i,c,:,prior['sign_metrics'].index(f'{existing}_{region}_{policy}')]
                    np.testing.assert_allclose(values[new],ref,rtol=1e-12,atol=1e-12)
                if name=='UO':
                    for key in ('E_neg','E_pos','negative_fraction'):
                        np.testing.assert_array_equal(old[i,c,:,prior['sign_metrics'].index(f'{key}_UO_full')],old[i,c,:,prior['sign_metrics'].index(f'{key}_UO_causal')])
            if name.startswith('OO'):
                np.testing.assert_array_equal(p['delta_E_neg'],0)
                np.testing.assert_array_equal(p['frequency_contribution'],0);np.testing.assert_array_equal(p['magnitude_contribution'],0)
            cs.append(np.stack([[v[f] for f in CONDITION_FIELDS] for v in (gg,ss)]).transpose(0,2,1))
            ps.append(np.stack([p[f] for f in PAIR_FIELDS],axis=-1));ms.append(np.stack([gg['m2_neg'],ss['m2_neg']]))
            counts.append(count)
            checks.append(dict(dataset_index=index,region=name,valid_elements=count,**errors,
                old_E_neg_max_error=float(max(np.max(np.abs((gg,ss)[c]['E_neg']-old[i,c,:,prior['sign_metrics'].index(f'E_neg_{region}_{policy}')])) for c in range(2)))))
            for c,condition in enumerate(('gcg','shuffled')):
                v=(gg,ss)[c];frames.append(pd.DataFrame(dict(dataset_index=index,condition=condition,region=name,
                    layer=np.arange(1024)//32,head=np.arange(1024)%32,valid_elements=count,**v)))
            pair_frames.append(pd.DataFrame(dict(dataset_index=index,region=name,layer=np.arange(1024)//32,head=np.arange(1024)%32,
                                                valid_elements=count,**p)))
        cond.append(np.stack(cs,axis=1));pairs.append(np.stack(ps));moments.append(np.stack(ms,axis=1));element_counts.append(counts)
        print('Core:',index,flush=True)
    c=np.stack(cond);p=np.stack(pairs);m2=np.stack(moments)
    if c.shape!=(20,2,5,1024,len(CONDITION_FIELDS)) or p.shape!=(20,5,1024,len(PAIR_FIELDS)):
        raise ValueError('incomplete sample/region/Head/feature correspondence')
    np.savez_compressed(out/'core_values.npz',conditions=c,pairs=p,m2_neg=m2,element_counts=np.array(element_counts,dtype=int))
    write_csv(out/'tables/condition_metrics.csv',pd.concat(frames,ignore_index=True));write_csv(out/'tables/paired_metrics.csv',pd.concat(pair_frames,ignore_index=True))
    m=dict(created_at=utc_now(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        all_indices=ids,primary_indices=primary,views=prior['views'],regions=list(REGIONS),condition_order=['gcg','shuffled'],
        condition_fields=list(CONDITION_FIELDS),pair_fields=list(PAIR_FIELDS),score_references=refs,
        source_hashes=sources,input_hashes={str(p):sha256_file(p) for p in used},model_info=prior['model_info'],
        dtype='FP64 from immutable FP32',device='CPU',definition='E_neg=mean(min(S,0)^2) over all valid elements',
        element_totals='paired_metrics.csv stores exact positive/negative/equal element counts and signed contribution SUMS; contribution fields without _total are normalized by N',
        zero_negative_policy='conditional m2_neg undefined NaN; only decomposition uses explicit zero extension; zero-boundary contribution separately saved with original pair denominator',
        average_order='features per sample/Head/region first; equal Head mean then sample mean; paired SD ddof=1',
        sign_policy='literal >0/<0/==0; numerical_zero_count with abs<=1e-12 also reported; no significance tests',
        limits='same selected primary18 automatic refusal labels; same-sample fixed groups; Heads dependent; full UU includes future; no Attention/causal-success inference',
        numpy=np.__version__,pandas=pd.__version__,no_model_or_GPU=True,independent100_used=False)
    write_json(out/'core_manifest.json',m);summarize_core(out,m,c,p,m2)
    write_json(out/'core_validation.json',dict(status='passed',checks=checks,condition_shape=list(c.shape),pair_shape=list(p.shape),
        conditional_m2_undefined_count=int(np.isnan(m2).sum()),unexpected_nonfinite=0))
    if not np.isfinite(c).all() or not np.isfinite(p).all() or not np.array_equal(np.isnan(m2),c[...,CONDITION_FIELDS.index('zero_negative')].astype(bool)):
        raise ValueError('unexpected nonfinite or missing conditional moment')
    for path,h in sources.items():
        if sha256_file(path)!=h:raise ValueError('source changed during core run')
    print('Core completed:',out,flush=True)


def run_finish(out,tail_policy):
    if tail_policy!='both':raise ValueError('user specified energy=all, positive delta=active AND all')
    if (out/'tail_values.npz').exists() or (out/'finish_manifest.json').exists():raise FileExistsError('finish outputs already exist')
    m=read_json(out/'core_manifest.json');ids=m['all_indices'];primary=m['primary_indices'];views=m['views']
    refs=m['score_references'];lookup={(r['dataset_index'],r['condition']):r for r in refs};sources=source_hashes()
    with np.load(out/'core_values.npz',allow_pickle=False) as z:
        counts=z['element_counts'];corep=z['pairs'];corec=z['conditions']
    specs=[dict(kind=kind,basis=basis,fraction=f) for kind,bases in [('energy_gcg',('all',)),('energy_shuffled',('all',)),('positive_delta',('active','all'))] for basis in bases for f in TAILS]
    shares=[];ks=[];totals=[];tops=[];basis_counts=[];query_conditions=[];query_pairs=[];query_counts=[];checks=[]
    for i,index in enumerate(ids):
        blocks,masks,_=blocks_for_pair(lookup,index);ir=[];ik=[];it=[];ip=[];ib=[]
        for ri,region in enumerate(REGIONS):
            (g,s),n=selected_elements(blocks,masks,region)
            gg,ss,p,energies,_=paired_factors(g,s)
            np.testing.assert_allclose(p['delta_E_neg'],corep[i,ri,:,0],rtol=0,atol=0)
            data={'energy_gcg':energies['gcg'],'energy_shuffled':energies['shuffled'],'positive_delta':np.maximum(energies['delta'],0)}
            sr=[];kr=[];tr=[];pr=[];br=[]
            for spec in specs:
                a=data[spec['kind']];share,k,total=concentration(a,spec['fraction'],spec['basis'])
                top=np.where(total>0,share*total,0.)/n
                basis=np.full(1024,n,dtype=int) if spec['basis']=='all' else (a>0).sum(axis=-1)
                sr.append(share);kr.append(k);tr.append(total/n);pr.append(top);br.append(basis)
            ir.append(np.stack(sr,axis=-1));ik.append(np.stack(kr,axis=-1));it.append(np.stack(tr,axis=-1));ip.append(np.stack(pr,axis=-1));ib.append(np.stack(br,axis=-1))
        qc=[];qp=[];qn=[]
        for name in ('UU','UO'):
            c,p,n=causal_query_factors(blocks[0][name],blocks[1][name],name)
            ri=REGIONS.index('UU_causal' if name=='UU' else 'UO')
            for condition in range(2):
                weighted=np.average(c[condition,:,:,0],axis=1,weights=n)
                np.testing.assert_allclose(weighted,corec[i,condition,ri,:,0],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(np.average(p[:,:,0],axis=1,weights=n),corep[i,ri,:,0],rtol=1e-12,atol=1e-12)
            checks.append(dict(dataset_index=index,region=name,visible_counts=n.tolist(),
                weighted_region_max_error=float(np.max(np.abs(np.average(p[:,:,0],axis=1,weights=n)-corep[i,ri,:,0])))))
            qc.append(c);qp.append(p);qn.append(n)
        shares.append(ir);ks.append(ik);totals.append(it);tops.append(ip);basis_counts.append(ib)
        query_conditions.append(np.stack(qc,axis=1));query_pairs.append(qp);query_counts.append(qn)
        print('Tails/Queries:',index,flush=True)
    a=np.asarray(shares);k=np.asarray(ks);total=np.asarray(totals);top=np.asarray(tops);basis=np.asarray(basis_counts)
    qc=np.asarray(query_conditions);qp=np.asarray(query_pairs);qn=np.asarray(query_counts)
    if not np.array_equal(np.isnan(a),total==0) or np.isinf(a).any():raise ValueError('unexpected tail undefined case')
    if not np.isfinite(total).all() or not np.isfinite(top).all() or not np.isfinite(qc).all() or not np.isfinite(qp).all():raise ValueError('unexpected nonfinite result')
    np.savez_compressed(out/'tail_values.npz',shares=a,k=k,total_mean_energy=total,top_mean_energy=top,basis_counts=basis)
    np.savez_compressed(out/'query_values.npz',conditions=qc,pairs=qp,valid_key_counts=qn)
    summarize_tails(out,m,specs,a,k,total,top,basis,counts,corep)
    summarize_queries(out,m,qc,qp,qn)
    finish=dict(created_at=utc_now(),tail_policy=dict(energy_basis='all valid elements',positive_delta_bases=['positive delta element count','all valid elements'],
        top_counts='ceil(fraction*basis_count)',zero_policy='undefined share when total=0; active count=0 separately recorded; no epsilon'),
        tail_specs=specs,tail_shape=list(a.shape),query_shape=list(qp.shape),query_regions=['UU_causal','UO'],
        query_fields=list(CONDITION_FIELDS),query_pair_fields=list(PAIR_FIELDS),query_average_policy='per Query features then equal Query mean vs valid-key-count weighted mean; distinct outputs',
        source_hashes=sources,core_manifest_sha256=sha256_file(out/'core_manifest.json'),user_choice='energy concentration all elements; positive delta concentration both active and all',no_new_Head_selection=True)
    write_json(out/'finish_manifest.json',finish)
    write_json(out/'finish_validation.json',dict(status='passed',checks=checks,undefined_tail_share_count=int(np.isnan(a).sum()),unexpected_nonfinite=0))
    from utils.gcg_negative_energy_plots import make_figures
    make_figures(out)
    from utils.gcg_negative_energy_report import write_report
    write_report(out)
    for path,h in sources.items():
        if sha256_file(path)!=h:raise ValueError('source changed during finish')
    for ref in refs:
        if sha256_file(ref['path'])!=ref['sha256']:raise ValueError('Score source changed')
    write_json(out/'output_hashes.json',{str(p.relative_to(out)):sha256_file(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.log'})
    print('Completed attribution:',out,flush=True)


def summarize_tails(out,m,specs,a,k,total,top,basis,counts,corep):
    tables=out/'tables';ids=m['all_indices'];views=m['views'];overview=[];sample=[];undefined=[]
    for n,pos in ((18,[ids.index(i) for i in m['primary_indices']]),(20,list(range(20)))):
        head_rows=[]
        for ri,region in enumerate(REGIONS):
            for ti,spec in enumerate(specs):
                x=a[pos,ri,:,ti];stats=defined_stats(x,axis=0)
                for h in range(1024):
                    head_rows.append(dict(subset=n,region=region,layer=h//32,head=h%32,**spec,
                        **{key:array[h] for key,array in stats.items()},undefined_count=n-int(stats['defined_count'][h])))
                for v in views:
                    members=v['members'];ss=np.take(x,members,axis=1)
                    tt=np.take(total[pos,ri,:,ti],members,axis=1);tp=np.take(top[pos,ri,:,ti],members,axis=1)
                    kk=np.take(k[pos,ri,:,ti],members,axis=1);bb=np.take(basis[pos,ri,:,ti],members,axis=1)
                    finite=np.isfinite(ss);defined=int(finite.sum());den=tt.mean()
                    row=dict(subset=n,region=region,view_id=v['view_id'],view_kind=v['kind'],head_count=len(members),sample_count=n,**spec,
                        defined_pair_count=defined,undefined_pair_count=int(ss.size-defined),
                        defined_case_mean_share=float(ss[finite].mean()) if defined else np.nan,
                        defined_case_median_share=float(np.median(ss[finite])) if defined else np.nan,
                        equal_pair_mean_top_energy=tp.mean(),equal_pair_mean_total_energy=den,
                        energy_weighted_share=float(tp.mean()/den) if den>0 else np.nan,
                        mean_top_k=float(kk.mean()),mean_basis_count=float(bb.mean()),
                        mean_selected_fraction_of_valid=float((kk/counts[pos,ri,None]).mean()))
                    active=bb>0
                    row['defined_basis_mean_selected_fraction']=float((kk[active]/bb[active]).mean()) if active.any() else np.nan
                    if spec['kind']=='positive_delta':
                        d=np.take(corep[pos,ri,:,0],members,axis=1)
                        residual=d-tp;sample_delta=residual.mean(axis=1)
                        row.update(net_delta=residual.mean()+tp.mean(),net_delta_after_top_positive_removed=residual.mean(),
                                   residual_positive_samples=int((sample_delta>0).sum()),residual_negative_samples=int((sample_delta<0).sum()))
                    overview.append(row)
                # All-Head sample means are conditional on defined shares, with counts.
                for j,i in enumerate(pos):
                    ss=x[j];valid=np.isfinite(ss);t=total[i,ri,:,ti];p=top[i,ri,:,ti]
                    sample.append(dict(subset=n,dataset_index=ids[i],region=region,**spec,defined_heads=int(valid.sum()),undefined_heads=int((~valid).sum()),
                        defined_head_mean_share=float(ss[valid].mean()) if valid.any() else np.nan,
                        energy_weighted_share=float(p.mean()/t.mean()) if t.mean()>0 else np.nan,
                        top_mean_energy=p.mean(),total_mean_energy=t.mean()))
        write_csv(tables/f'tail_head_summary_{n}.csv',pd.DataFrame(head_rows))
    for ri,region in enumerate(REGIONS):
        for ti,spec in enumerate(specs):
            mask=total[:,ri,:,ti]==0
            reason='zero_negative_energy' if spec['kind'].startswith('energy_') else 'no_positive_delta_energy'
            undefined.append(dict(region=region,**spec,undefined_count=int(mask.sum()),zero_basis_count=int((basis[:,ri,:,ti]==0).sum()),reason=reason))
    write_csv(tables/'tail_overview.csv',pd.DataFrame(overview));write_csv(tables/'tail_sample_summary.csv',pd.DataFrame(sample));write_csv(tables/'undefined_tail_cases.csv',pd.DataFrame(undefined))


def summarize_queries(out,m,c,p,counts):
    tables=out/'tables';ids=m['all_indices'];views=m['views'];rows=[];sample=[];weights=[];layers=[]
    # c [sample, condition, region, Head, Query, field]; p [sample, region, Head, Query, field]
    for n,pos in ((18,[ids.index(i) for i in m['primary_indices']]),(20,list(range(20)))):
        head=[]
        for ri,region in enumerate(('UU_causal','UO')):
            delta=p[pos,ri,:,:,0];mean=delta.mean(axis=0)
            for h in range(1024):
                for q in range(20):head.append(dict(subset=n,region=region,layer=h//32,head=h%32,query=q,delta_mean=mean[h,q],
                    delta_std=delta[:,h,q].std(ddof=1),positive_sample_count=int((delta[:,h,q]>0).sum()),negative_sample_count=int((delta[:,h,q]<0).sum())))
            for v in views:
                members=v['members'];d=np.take(delta,members,axis=1).mean(axis=1)
                g=np.take(c[pos,0,ri,:,:,0],members,axis=1).mean(axis=1)
                s=np.take(c[pos,1,ri,:,:,0],members,axis=1).mean(axis=1)
                for q in range(20):
                    row=dict(subset=n,region=region,view_id=v['view_id'],view_kind=v['kind'],head_count=len(members),query=q,
                        gcg_mean=g[:,q].mean(),shuffled_mean=s[:,q].mean(),delta_mean=d[:,q].mean(),delta_std=d[:,q].std(ddof=1),
                        positive_sample_count=int((d[:,q]>0).sum()),negative_sample_count=int((d[:,q]<0).sum()),visible_keys_min=int(counts[pos,ri,q].min()),visible_keys_max=int(counts[pos,ri,q].max()))
                    rows.append(row)
                equal=d.mean(axis=1);weighted=np.array([np.average(d[j],weights=counts[i,ri]) for j,i in enumerate(pos)])
                weights.append(dict(subset=n,region=region,view_id=v['view_id'],equal_query_delta_mean=equal.mean(),
                    element_weighted_delta_mean=weighted.mean(),difference=equal.mean()-weighted.mean(),equal_query_positive_samples=int((equal>0).sum()),
                    weighted_positive_samples=int((weighted>0).sum()),query_count=20))
            for j,i in enumerate(pos):
                for q in range(20):sample.append(dict(subset=n,dataset_index=ids[i],region=region,query=q,delta_mean=delta[j,:,q].mean(),
                    positive_head_fraction=(delta[j,:,q]>0).mean(),contribution_to_region_mean=delta[j,:,q].mean()*counts[i,ri,q]/counts[i,ri].sum()))
            for layer in range(32):
                for q in range(20):layers.append(dict(subset=n,region=region,layer=layer,query=q,delta_mean=mean[layer*32:(layer+1)*32,q].mean()))
        write_csv(tables/f'query_head_summary_{n}.csv',pd.DataFrame(head))
    write_csv(tables/'query_overview.csv',pd.DataFrame(rows));write_csv(tables/'query_sample_summary.csv',pd.DataFrame(sample));write_csv(tables/'query_weighting.csv',pd.DataFrame(weights));write_csv(tables/'layer_query_summary.csv',pd.DataFrame(layers))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=('core','finish'),default='core')
    parser.add_argument('--tail-policy',choices=('both',))
    args=parser.parse_args()
    if args.stage=='core':run_core(args.output_dir)
    else:
        if args.tail_policy is None:parser.error('finish requires an explicit --tail-policy')
        run_finish(args.output_dir,args.tail_policy)


if __name__=='__main__':main()
