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
    from utils.gcg_shuffled_common import region_layout
    blocks,masks,lengths=region_layout(scores,ranges)
    m,n=lengths['original_length'],lengths['suffix_length']
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


from utils.gcg_shuffled_common import paired_stats
