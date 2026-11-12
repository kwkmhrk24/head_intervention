"""CSV-only diagnostics of previously computed UU/OO RMS ratios."""
import numpy as np
import pandas as pd
from utils.gcg_shuffled_metrics import TIE_ATOL, TIE_RTOL
from utils.gcg_shuffled_reporting import require_columns, validate_summary

POLICIES=('full','causal')
METRICS=tuple('R_UU_OO_'+p for p in POLICIES)
KEYS=['dataset_index','layer','head']


def complete_heads(frame,label):
    require_columns(frame,('layer','head'),label)
    expected={(l,h) for l in range(32) for h in range(32)}
    actual=set(map(tuple,frame[['layer','head']].to_numpy()))
    if len(frame)!=1024 or frame.duplicated(['layer','head']).any() or actual!=expected:
        raise ValueError(f'{label}: missing/duplicate/invalid Layer-Head correspondence')


def direction_statistics(delta,gcg,shuffled):
    """Samples x Heads; mean direction agreement uses the original tie policy."""
    if delta.ndim!=2 or delta.shape!=gcg.shape or delta.shape!=shuffled.shape or not np.isfinite([delta,gcg,shuffled]).all():
        raise ValueError('invalid finite paired sample/Head arrays')
    np.testing.assert_allclose(delta,gcg-shuffled,rtol=1e-12,atol=1e-12)
    n=len(delta);tol=TIE_ATOL+TIE_RTOL*np.maximum(np.abs(gcg),np.abs(shuffled))
    pos=(delta>tol).sum(axis=0);neg=(delta < -tol).sum(axis=0);equal=n-pos-neg
    mean=delta.mean(axis=0);mtol=TIE_ATOL+TIE_RTOL*np.maximum(np.abs(gcg.mean(axis=0)),np.abs(shuffled.mean(axis=0)))
    sign=np.where(mean>mtol,1,np.where(mean < -mtol,-1,0))
    return dict(positive_samples=pos,negative_samples=neg,equal_samples=equal,
        positive_fraction=pos/n,negative_fraction=neg/n,equal_fraction=equal/n,
        majority_nonzero_fraction=np.maximum(pos,neg)/n,
        mean_direction=sign,mean_direction_agreement=np.where(sign>0,pos,np.where(sign<0,neg,equal))/n,
        strict_positive_samples=(delta>0).sum(axis=0),strict_negative_samples=(delta<0).sum(axis=0))


def align_pairs(metrics,paired,indices,policy):
    """Explicit index products avoid silently pairing different Heads/samples."""
    ratio='R_UU_OO_'+policy;delta='delta_'+ratio
    require_columns(metrics,KEYS+['condition',ratio,'primary18'],'metrics')
    require_columns(paired,KEYS+[delta],'paired differences')
    if metrics.duplicated(KEYS+['condition']).any() or paired.duplicated(KEYS).any():
        raise ValueError('duplicate sample/condition/Head keys')
    target=pd.MultiIndex.from_tuples([(i,l,h) for i in indices for l in range(32) for h in range(32)],names=KEYS)
    arrays=[]
    for c in ('gcg','shuffled'):
        part=metrics[(metrics.condition==c)&metrics.dataset_index.isin(indices)].set_index(KEYS)
        if len(part)!=len(target) or set(part.index)!=set(target):raise ValueError('missing paired '+c+' sample/Head')
        arrays.append(part.reindex(target)[ratio].to_numpy(float).reshape(len(indices),1024))
    part=paired[paired.dataset_index.isin(indices)].set_index(KEYS)
    if len(part)!=len(target) or set(part.index)!=set(target):raise ValueError('missing paired delta sample/Head')
    d=part.reindex(target)[delta].to_numpy(float).reshape(len(indices),1024)
    direction_statistics(d,*arrays)
    return arrays[0],arrays[1],d


def analyze_heads(metrics,paired,summary,maps,overview,indices,policy):
    n=len(indices);name='R_UU_OO_'+policy
    heads=summary[summary.metric==name].sort_values(['layer','head']).reset_index(drop=True).copy()
    complete_heads(heads,'summary '+name);validate_summary(heads,'summary '+name)
    if not heads.sample_count.eq(n).all() or not heads.pair_valid_count.eq(n).all():raise ValueError('incomplete ratio sample count')
    g,s,d=align_pairs(metrics,paired,indices,policy);stats=direction_statistics(d,g,s)
    comparisons={'gcg_mean':g.mean(axis=0),'shuffled_mean':s.mean(axis=0),'delta_mean':d.mean(axis=0),
        'delta_median':np.median(d,axis=0),'delta_std':d.std(axis=0,ddof=1)}
    errors={}
    for field,actual in comparisons.items():
        errors[field]=float(np.max(np.abs(actual-heads[field].to_numpy(float))))
        np.testing.assert_allclose(actual,heads[field],rtol=1e-12,atol=1e-12)
    for field,actual in [('gcg_larger_count',stats['positive_samples']),('shuffled_larger_count',stats['negative_samples']),('equal_count',stats['equal_samples'])]:
        np.testing.assert_array_equal(heads[field],actual)
    complete_heads(maps,'Head map')
    np.testing.assert_allclose(maps.sort_values(['layer','head'])['delta_'+name],heads.delta_mean,rtol=1e-12,atol=1e-12)
    row=overview[(overview.subset==n)&(overview.metric==name)]
    if len(row)!=1:raise ValueError('missing/duplicate all-Head overview')
    validate_summary(row,'overview')
    for field in ('gcg_mean','shuffled_mean','delta_mean'):
        np.testing.assert_allclose(heads[field].mean(),row.iloc[0][field],rtol=1e-12,atol=1e-12)
    heads.insert(0,'subset',n);heads.insert(1,'policy',policy)
    for key,value in stats.items():heads[key]=value
    heads['gcg_head_mean_gt1']=heads.gcg_mean>1
    heads['shuffled_head_mean_gt1']=heads.shuffled_mean>1
    heads['gcg_samples_gt1']=(g>1).sum(axis=0)
    heads['shuffled_samples_gt1']=(s>1).sum(axis=0)
    md=heads.mean_direction.to_numpy();dd=heads.delta_mean.to_numpy()
    positive=np.maximum(dd,0);top=int(np.ceil(.1*1024))
    result=dict(subset=n,policy=policy,gcg_mean=float(row.iloc[0].gcg_mean),shuffled_mean=float(row.iloc[0].shuffled_mean),
        delta_mean=float(row.iloc[0].delta_mean),head_delta_median=float(np.median(dd)),head_delta_std=float(dd.std(ddof=1)),
        head_delta_q05=float(np.quantile(dd,.05)),head_delta_q95=float(np.quantile(dd,.95)),
        positive_heads=int((md>0).sum()),negative_heads=int((md<0).sum()),equal_heads=int((md==0).sum()),
        strict_positive_heads=int((dd>0).sum()),strict_negative_heads=int((dd<0).sum()),
        gcg_head_mean_gt1=int(heads.gcg_head_mean_gt1.sum()),shuffled_head_mean_gt1=int(heads.shuffled_head_mean_gt1.sum()),
        upward_crossing_1=int(((s.mean(axis=0)<=1)&(g.mean(axis=0)>1)).sum()),
        downward_crossing_1=int(((s.mean(axis=0)>1)&(g.mean(axis=0)<=1)).sum()),
        positive_all_samples_heads=int((stats['positive_samples']==n).sum()),negative_all_samples_heads=int((stats['negative_samples']==n).sum()),
        mean_direction_agreement_mean=float(heads.mean_direction_agreement.mean()),
        mean_direction_agreement_median=float(heads.mean_direction_agreement.median()),
        at_least_80pct_positive_heads=int((stats['positive_fraction']>=.8).sum()),
        top10pct_positive_delta_share=float(np.sort(positive)[-top:].sum()/positive.sum()) if positive.sum()>0 else None,
        remaining_mean_after_top10pct=float(np.sort(dd)[:-top].mean()))
    layers=heads.groupby('layer',sort=True).agg(gcg_mean=('gcg_mean','mean'),shuffled_mean=('shuffled_mean','mean'),
        delta_mean=('delta_mean','mean'),positive_heads=('mean_direction',lambda x:int((x>0).sum())),
        negative_heads=('mean_direction',lambda x:int((x<0).sum())),gcg_head_mean_gt1=('gcg_head_mean_gt1','sum'),
        mean_direction_agreement=('mean_direction_agreement','mean')).reset_index()
    layers.insert(0,'subset',n);layers.insert(1,'policy',policy)
    samples=pd.DataFrame(dict(dataset_index=indices,gcg_mean=g.mean(axis=1),shuffled_mean=s.mean(axis=1),delta_mean=d.mean(axis=1),
        head_delta_median=np.median(d,axis=1),positive_head_fraction=(d>TIE_ATOL+TIE_RTOL*np.maximum(abs(g),abs(s))).mean(axis=1)))
    samples.insert(0,'subset',n);samples.insert(1,'policy',policy)
    return heads,result,layers,samples,errors


def fixed_group_tables(views,members,heads,n,policy):
    name='R_UU_OO_'+policy
    groups=views[(views.metric==name)&(views.view_kind=='selected_group')].copy()
    details=members[(members.subset==n)&(members.metric==name)].copy()
    if len(groups)!=12 or groups.view_id.duplicated().any() or not groups.head_count.eq(5).all():raise ValueError('existing fixed group definition mismatch')
    if set(groups.view_id)!=set(details.selected_group):raise ValueError('group membership names mismatch')
    validate_summary(groups,'fixed groups')
    if not groups.sample_count.eq(n).all():raise ValueError('fixed group cohort size mismatch')
    for group in groups.itertuples():
        part=details[details.selected_group==group.view_id]
        if len(part)!=5 or part.duplicated(['layer','head']).any():raise ValueError('fixed group duplicate/missing member: '+group.view_id)
        reference=heads.set_index(['layer','head']).reindex(pd.MultiIndex.from_frame(part[['layer','head']]))
        for field in ('gcg_mean','shuffled_mean','delta_mean'):
            np.testing.assert_allclose(reference[field],part[field],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(part[field].mean(),getattr(group,field),rtol=1e-12,atol=1e-12)
    groups.insert(0,'subset',n);groups.insert(1,'policy',policy)
    details.insert(1,'policy',policy)
    return groups,details
