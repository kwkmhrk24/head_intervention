"""Token-position diagnostics; every paired difference is GCG minus Shuffled."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.gcg_shuffled_metrics import METRICS,logsumexp

CLOSING_IDS=[518,29914,25580,29962]


def select_heads(summary):
    if summary.duplicated(['metric','layer','head']).any():
        raise ValueError('duplicate selection key')
    rows=[]
    for metric in METRICS:
        part=summary[summary.metric==metric]
        if len(part)!=1024 or not np.isfinite(part.delta_mean).all():
            raise ValueError('complete finite 1024-head summary required')
        for direction in ('increase','decrease'):
            selected=part[part.delta_mean>0] if direction=='increase' else part[part.delta_mean<0]
            selected=selected.sort_values(['delta_mean','layer','head'],ascending=[direction=='decrease',True,True]).head(5)
            for rank,row in enumerate(selected.to_dict('records'),1):
                rows.append(dict(**row,direction=direction,rank=rank,group_id=f'{metric}_{direction}',
                    reason=f'primary18 signed mean delta {"descending positive" if direction=="increase" else "ascending negative"}; Layer/Head tie break'))
    selected=pd.DataFrame(rows)
    unique=sorted({(int(r.layer),int(r.head)) for r in selected.itertuples()})
    views=[dict(view_id=f'L{layer:02d}H{head:02d}',kind='individual',members=[layer*32+head]) for layer,head in unique]
    for group,part in selected.groupby('group_id',sort=False):
        views.append(dict(view_id=group,kind='selected_group',members=[int(r.layer)*32+int(r.head) for r in part.itertuples()]))
    views.append(dict(view_id='all_heads',kind='all_heads',members=list(range(1024))))
    return selected,views


def reduce_views(values,views):
    """Features are computed per head before equal-weight aggregation."""
    return np.stack([values[v['members']].mean(axis=0) for v in views])


def closing_queries(scores,item):
    ids=item['input_ids'];length=len(ids);ranges=item['token_ranges']
    start,end=ranges['template_closing'];u0,u1=ranges['suffix'];o0,o1=ranges['original']
    if scores.shape!=(32,32,length,length) or start!=length-4 or end!=length or ids[start:end]!=CLOSING_IDS:
        raise ValueError('actual closing template/score shape mismatch')
    if u1-u0!=20 or not 0<=o0<o1<=u0<u1<=start:
        raise ValueError('invalid Original/Suffix boundary')
    if not np.isfinite(scores).all():raise ValueError('nonfinite score')
    row=np.asarray(scores[:,:,start:end,:],dtype=np.float64).reshape(1024,4,length)
    visible=np.arange(length)[None,None,:]<=np.arange(start,end)[None,:,None]
    masked=np.where(visible,row,-np.inf)
    denominator=logsumexp(masked)
    weights=np.exp(masked-denominator[...,None])
    np.testing.assert_allclose(weights.sum(axis=-1),1,rtol=1e-12,atol=1e-12)
    if np.any(np.where(visible,0,weights)!=0):raise ValueError('future template attention leaked')
    log_mu=logsumexp(row[...,u0:u1])-denominator
    log_mo=logsumexp(row[...,o0:o1])-denominator
    mu=weights[...,u0:u1].sum(axis=-1);mo=weights[...,o0:o1].sum(axis=-1)
    d=logsumexp(row[...,u0:u1])-logsumexp(row[...,o0:o1])
    np.testing.assert_allclose(d,log_mu-log_mo,rtol=1e-12,atol=1e-11)
    positive=(mu>0)&(mo>0)
    np.testing.assert_allclose(d[positive],np.log(mu[positive])-np.log(mo[positive]),rtol=1e-12,atol=1e-11)
    mass=np.stack([weights[...,a:b].sum(axis=-1) for a,b in [ranges[k] for k in ('template_prefix','original','suffix','template_closing')]],axis=-1)
    np.testing.assert_allclose(mass.sum(axis=-1),1,rtol=1e-12,atol=1e-12)
    return dict(raw=row,weight=weights,metrics=np.stack([mu,mo,d],axis=-1),region_mass=mass,
                log_mass=np.stack([log_mu,log_mo],axis=-1))


def matrix_diagnostics(gcg,shuffled):
    if gcg.shape!=shuffled.shape or gcg.ndim!=3 or gcg.shape[-2:]!=(20,20):
        raise ValueError('matched Head x 20 x 20 suffix blocks required')
    if not np.isfinite(gcg).all() or not np.isfinite(shuffled).all():raise ValueError('nonfinite suffix block')
    blocks=[np.asarray(gcg,dtype=np.float64),np.asarray(shuffled,dtype=np.float64)]
    components=[];scalar=[]
    for b in blocks:
        sym=(b+b.swapaxes(-1,-2))/2;asym=(b-b.swapaxes(-1,-2))/2
        sq=b*b;ss=sym*sym;aa=asym*asym
        energy=sq.sum(axis=(-2,-1));se=ss.sum(axis=(-2,-1));ae=aa.sum(axis=(-2,-1))
        np.testing.assert_allclose(b,sym+asym,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(energy,se+ae,rtol=1e-12,atol=1e-12)
        ps=np.full_like(ss,np.nan);pa=np.full_like(aa,np.nan)
        np.divide(ss,energy[:,None,None],out=ps,where=energy[:,None,None]>0)
        np.divide(aa,energy[:,None,None],out=pa,where=energy[:,None,None]>0)
        valid=energy>0
        np.testing.assert_allclose((ps+pa).sum(axis=(-2,-1))[valid],1,rtol=1e-12,atol=1e-12)
        components.append(dict(raw=b,square=sq,sym=sym,sym_square=ss,asym=asym,asym_square=aa,
                               normalized_sym_square=ps,normalized_asym_square=pa))
        scalar.append(dict(F=np.sqrt(energy),F_sym=np.sqrt(se),F_asym=np.sqrt(ae),
                           R_sym=ps.sum(axis=(-2,-1)),R_asym=pa.sum(axis=(-2,-1))))
    fields={}
    for name in components[0]:
        g,s=components[0][name],components[1][name]
        fields[name+'_gcg']=g;fields[name+'_shuffled']=s
        fields['delta_'+name]=g-s;fields['abs_delta_'+name]=np.abs(g-s)
    fields['delta_raw_squared']=(blocks[0]-blocks[1])**2
    for metric,base in [('F','square'),('F_sym','sym_square'),('F_asym','asym_square')]:
        delta=fields['delta_'+base];denom=scalar[0][metric]+scalar[1][metric]
        contribution=np.zeros_like(delta)
        np.divide(delta,denom[:,None,None],out=contribution,where=denom[:,None,None]>0)
        fields[metric+'_delta_contribution']=contribution
        np.testing.assert_allclose(delta.sum(axis=(-2,-1)),scalar[0][metric]**2-scalar[1][metric]**2,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(contribution.sum(axis=(-2,-1)),scalar[0][metric]-scalar[1][metric],rtol=1e-10,atol=1e-11)
    return fields,scalar


def sample_summary(values,indices):
    x=values[indices]
    if not np.isfinite(x).all():raise ValueError('undefined/nonfinite diagnostic; inspect zero block norm')
    return dict(mean=x.mean(axis=0),median=np.median(x,axis=0),std=x.std(axis=0,ddof=1),
                mean_absolute=np.abs(x).mean(axis=0),positive_count=(x>1e-12).sum(axis=0),
                negative_count=(x < -1e-12).sum(axis=0))
