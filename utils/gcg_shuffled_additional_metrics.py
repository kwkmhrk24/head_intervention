"""Sign decomposition and causal same-query diagnostics on saved raw scores."""
import numpy as np
from utils.gcg_shuffled_metrics import logsumexp

SIGN_FIELDS = ('E_pos','E_neg','mu_pos','mu_neg','positive_fraction','negative_fraction',
               'median','q05','q25','q50','q75','q95')
SIGN_METRICS = tuple(f'{field}_{region}_{policy}' for policy in ('full','causal')
                     for region in ('OO','UU','UO') for field in SIGN_FIELDS)
QUERY_FIELDS = ('M_U','M_O','D','conditional_M_U','M_prefix','M_other','raw_O','raw_U',
                'row_mean','centered_O','centered_U','gap_U_minus_O','lse_O','lse_U',
                'raw_prefix','raw_other')
QUERY_LABELS = tuple([f'suffix_{i:02d}' for i in range(20)]+[f'q{i}' for i in range(1,5)]+['suffix_mean20'])


from utils.gcg_shuffled_common import region_layout


def sign_metrics(scores,ranges):
    blocks,masks,counts=region_layout(scores,ranges);result={}
    for policy in ('full','causal'):
        for region,b in blocks.items():
            mask=np.ones_like(masks[region]) if policy=='full' else masks[region]
            x=b[...,mask];pos=np.maximum(x,0);neg=np.minimum(x,0)
            fields=dict(E_pos=np.mean(pos*pos,axis=-1),E_neg=np.mean(neg*neg,axis=-1),
                mu_pos=pos.mean(axis=-1),mu_neg=neg.mean(axis=-1),positive_fraction=(x>0).mean(axis=-1),
                negative_fraction=(x<0).mean(axis=-1),median=np.median(x,axis=-1))
            quantiles=np.quantile(x,[.05,.25,.5,.75,.95],axis=-1,method='linear')
            fields.update(zip(('q05','q25','q50','q75','q95'),quantiles))
            np.testing.assert_allclose(fields['E_pos']+fields['E_neg'],np.mean(x*x,axis=-1),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(fields['mu_pos']+fields['mu_neg'],x.mean(axis=-1),rtol=1e-12,atol=1e-12)
            result.update({f'{f}_{region}_{policy}':v for f,v in fields.items()})
            counts[f'{region}_{policy}']=int(mask.sum())
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite sign metric')
    return result,counts


def query_diagnostics(scores,item):
    """Suffix 20 and closing 4 queries; full visible-row normalization, FP64."""
    ranges=item['token_ranges'];ids=item['input_ids'];length=len(ids)
    a,b=ranges['original'];u,w=ranges['suffix'];t,e=ranges['template_closing'];p0,p1=ranges['template_prefix']
    if scores.shape!=(32,32,length,length) or w-u!=20 or t!=length-4 or e!=length:
        raise ValueError('score/query shape or boundaries mismatch')
    if not 0<=p0<p1<=a<b<=u<w<=t or ids[t:e]!=[518,29914,25580,29962]:raise ValueError('template/range mismatch')
    queries=np.r_[np.arange(u,w),np.arange(t,e)]
    row=np.asarray(scores[:,:,queries,:],dtype=np.float64).reshape(1024,24,length)
    if not np.isfinite(row).all():raise ValueError('nonfinite query score')
    visible=np.arange(length)[None,:]<=queries[:,None];count=visible.sum(axis=-1)
    masked=np.where(visible,row,-np.inf);den=logsumexp(masked)
    weights=np.exp(masked-den[...,None]);row_mean=np.where(visible,row,0).sum(axis=-1)/count
    centered=row-row_mean[...,None]
    cw=np.exp(np.where(visible,centered,-np.inf)-logsumexp(np.where(visible,centered,-np.inf))[...,None])
    np.testing.assert_allclose(weights.sum(axis=-1),1,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(weights[...,~visible],np.zeros_like(weights[...,~visible]))
    np.testing.assert_allclose(cw,weights,rtol=1e-12,atol=1e-12)
    center_error=np.max(np.abs(np.where(visible,centered,0).sum(axis=-1)/count))
    if center_error>1e-12:raise ValueError('centered visible row mean nonzero')
    omask=visible & ((np.arange(length)>=a)&(np.arange(length)<b))
    umask=visible & ((np.arange(length)>=u)&(np.arange(length)<w))
    pmask=visible & ((np.arange(length)>=p0)&(np.arange(length)<p1))
    other=visible & ~(omask|umask|pmask)
    masks={'O':omask,'U':umask,'prefix':pmask,'other':other}
    means={};mass={};counts={}
    for region,mask in masks.items():
        n=mask.sum(axis=-1);out=np.full(row.shape[:-1],np.nan)
        np.divide(np.where(mask,row,0).sum(axis=-1),n,out=out,where=n>0)
        means[region]=out;mass[region]=np.where(mask,weights,0).sum(axis=-1);counts[region]=n
    lo=logsumexp(np.where(omask,row,-np.inf));lu=logsumexp(np.where(umask,row,-np.inf));d=lu-lo
    dc=logsumexp(np.where(umask,centered,-np.inf))-logsumexp(np.where(omask,centered,-np.inf))
    np.testing.assert_allclose(d,dc,rtol=1e-12,atol=1e-11)
    if (mass['U']<=0).any() or (mass['O']<=0).any():raise ValueError('probability underflow: inspect explicitly')
    np.testing.assert_allclose(d,np.log(mass['U'])-np.log(mass['O']),rtol=1e-12,atol=1e-11)
    np.testing.assert_allclose(sum(mass.values()),1,rtol=1e-12,atol=1e-12)
    values=dict(M_U=mass['U'],M_O=mass['O'],D=d,conditional_M_U=mass['U']/(mass['U']+mass['O']),
                M_prefix=mass['prefix'],M_other=mass['other'],raw_O=means['O'],raw_U=means['U'],row_mean=row_mean,
                centered_O=means['O']-row_mean,centered_U=means['U']-row_mean,
                gap_U_minus_O=means['U']-means['O'],lse_O=lo,lse_U=lu,raw_prefix=means['prefix'],raw_other=means['other'])
    array=np.stack([values[f] for f in QUERY_FIELDS],axis=-1)
    # Only raw_other is undefined where no other visible keys exist (all suffix queries).
    if not np.isfinite(array[...,:-1]).all() or not np.isfinite(array[:,20:,-1]).all() or not np.isnan(array[:,:20,-1]).all():
        raise ValueError('unexpected undefined query metric')
    array=np.concatenate([array,array[:,:20].mean(axis=1,keepdims=True)],axis=1)
    return array,dict(raw=row,centered=centered,weight=weights,query_positions=queries,visible=visible,counts=counts,
                      centered_softmax_max_error=float(np.max(np.abs(cw-weights))),centered_mean_max_error=float(center_error),
                      centered_D_max_error=float(np.max(np.abs(d-dc))))
