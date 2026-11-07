"""CPU block norms/RMS and paired summaries; no attention recomputation."""

from __future__ import annotations

import warnings
import numpy as np

from utils.gcg_shuffled_metrics import TIE_ATOL,TIE_RTOL

METRICS=tuple(f'{metric}_{policy}' for policy in ('full','causal') for metric in
              ('F_OO','F_UU','F_UO','RMS_OO','RMS_UU','RMS_UO','R_UU_OO','R_UU_UO'))


def safe_ratio(numerator,denominator):
    result=np.full_like(numerator,np.nan,dtype=np.float64)
    np.divide(numerator,denominator,out=result,where=denominator>0)
    return result


def block_norm(block,visible):
    if block.shape[-2:]!=visible.shape or not visible.any() or not np.isfinite(block).all():
        raise ValueError('invalid block, visibility mask or values')
    values=np.asarray(block,dtype=np.float64)[...,visible]
    norm=np.sqrt(np.einsum('...i,...i->...',values,values))
    np.testing.assert_allclose(norm,np.linalg.norm(values,axis=-1),rtol=1e-12,atol=1e-12)
    return norm,norm/np.sqrt(int(visible.sum()))


def region_metrics(scores,ranges):
    if scores.ndim<2 or scores.shape[-1]!=scores.shape[-2]:raise ValueError('square scores required')
    a,b=ranges['original'];c,d=ranges['suffix'];length=scores.shape[-1]
    if not 0<=a<b<=c<d<=length:raise ValueError('invalid ordered Original/Suffix ranges')
    if d-c!=20:raise ValueError('suffix length must be twenty')
    op=np.arange(a,b);up=np.arange(c,d)
    blocks=dict(OO=np.asarray(scores[...,a:b,a:b],dtype=np.float64),
                UU=np.asarray(scores[...,c:d,c:d],dtype=np.float64),
                UO=np.asarray(scores[...,c:d,a:b],dtype=np.float64))
    masks=dict(OO=op[None,:]<=op[:,None],UU=up[None,:]<=up[:,None],UO=op[None,:]<=up[:,None])
    if not masks['UO'].all():raise ValueError('Suffix-to-Original contains future elements')
    m,n=b-a,d-c
    if masks['OO'].sum()!=m*(m+1)//2 or masks['UU'].sum()!=n*(n+1)//2 or masks['UO'].sum()!=m*n:
        raise ValueError('causal visible element count mismatch')
    values={};counts={}
    for policy in ('full','causal'):
        for region,block in blocks.items():
            mask=np.ones_like(masks[region]) if policy=='full' else masks[region]
            norm,rms=block_norm(block,mask)
            values[f'F_{region}_{policy}']=norm;values[f'RMS_{region}_{policy}']=rms
            counts[f'{region}_{policy}']=int(mask.sum())
        values[f'R_UU_OO_{policy}']=safe_ratio(values[f'RMS_UU_{policy}'],values[f'RMS_OO_{policy}'])
        values[f'R_UU_UO_{policy}']=safe_ratio(values[f'RMS_UU_{policy}'],values[f'RMS_UO_{policy}'])
    for name,value in values.items():
        if np.isinf(value).any() or (not name.startswith('R_UU_') and not np.isfinite(value).all()):
            raise ValueError(f'nonfinite regional metric: {name}')
    np.testing.assert_allclose(values['RMS_OO_full'],values['F_OO_full']/m,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(values['RMS_UU_full'],values['F_UU_full']/n,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(values['F_UO_full'],values['F_UO_causal'])
    return values,dict(original_length=m,suffix_length=n,**counts)


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
