"""NumPy-only paired suffix metrics on saved, unmasked attention scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ('M_U', 'D', 'F', 'F_sym', 'F_asym', 'R_asym')
TIE_ATOL = 1e-12
TIE_RTOL = 1e-12


def logsumexp(values):
    maximum = np.max(values, axis=-1, keepdims=True)
    return maximum[..., 0] + np.log(np.exp(values-maximum).sum(axis=-1))


def calculate_metrics(scores, original, suffix, query):
    if scores.ndim != 4 or scores.shape[-1] != scores.shape[-2] or query != scores.shape[-1]-1:
        raise ValueError('square full scores and actual final query required')
    length = scores.shape[-1]
    o0, o1 = original
    u0, u1 = suffix
    if not (0 <= o0 < o1 <= u0 < u1 <= query) or u1-u0 != 20:
        raise ValueError('invalid Original/Suffix ranges; suffix must contain 20 tokens before final query')
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite input scores')
    row = np.asarray(scores[:, :, query, :], dtype=np.float64)
    shifted = row-row.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=-1, keepdims=True)
    mu = weights[..., u0:u1].sum(axis=-1)
    mo = weights[..., o0:o1].sum(axis=-1)
    d = logsumexp(row[..., u0:u1])-logsumexp(row[..., o0:o1])
    # Check the softmax identity independently, retaining a stable log-space
    # reference when a region's probability underflows to zero in float64.
    log_weights = shifted-np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    reference_d = logsumexp(log_weights[..., u0:u1])-logsumexp(log_weights[..., o0:o1])
    np.testing.assert_allclose(d, reference_d, rtol=1e-12, atol=1e-11)
    positive = (mu > 0) & (mo > 0)
    if np.any(positive):
        np.testing.assert_allclose(d[positive], np.log(mu[positive])-np.log(mo[positive]), rtol=1e-12, atol=1e-11)
    if not ((mu >= 0).all() and (mu <= 1+1e-14).all()):
        raise ValueError('suffix attention mass outside [0,1]')
    block = np.asarray(scores[:, :, u0:u1, u0:u1], dtype=np.float64)
    if block.shape[-2:] != (20,20):
        raise ValueError('suffix block is not 20 x 20')
    sym = (block+block.swapaxes(-1,-2))/2
    asym = (block-block.swapaxes(-1,-2))/2
    energy = np.einsum('...ij,...ij->...', block,block)
    sym_energy = np.einsum('...ij,...ij->...', sym,sym)
    asym_energy = np.einsum('...ij,...ij->...', asym,asym)
    norm, sym_norm, asym_norm = np.sqrt(energy), np.sqrt(sym_energy), np.sqrt(asym_energy)
    reference_norm = np.linalg.norm(block, ord='fro', axis=(-2,-1))
    np.testing.assert_allclose(norm,reference_norm,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(block,sym+asym,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(energy,sym_energy+asym_energy,rtol=1e-12,atol=1e-12)
    ratio = np.full_like(energy,np.nan)
    np.divide(asym_energy,energy,out=ratio,where=energy>0)
    if np.any((ratio[energy>0]<0) | (ratio[energy>0]>1+1e-12)):
        raise ValueError('invalid squared asymmetry energy ratio')
    values = dict(M_U=mu,M_O=mo,D=d,F=norm,F_sym=sym_norm,F_asym=asym_norm,R_asym=ratio)
    for name,value in values.items():
        if name != 'R_asym' and not np.isfinite(value).all():
            raise ValueError(f'nonfinite computed metric: {name}')
    validation = dict(zero_block_count=int((energy==0).sum()),
        softmax_region_underflow_count=int((~positive).sum()),
        log_mass_identity_max_error=float(np.max(np.abs(d-reference_d))),
        reconstruction_max_error=float(np.max(np.abs(block-sym-asym))),
        energy_identity_max_relative_error=float(np.max(np.abs(energy-sym_energy-asym_energy)/np.maximum(energy,np.finfo(float).tiny))),
        norm_reference_max_error=float(np.max(np.abs(norm-reference_norm))))
    return values,validation


def paired_differences(metrics):
    keys = ['dataset_index','layer','head']
    gcg = metrics[metrics.condition=='gcg']
    shuffled = metrics[metrics.condition=='shuffled']
    if gcg.duplicated(keys).any() or shuffled.duplicated(keys).any():
        raise ValueError('duplicate metric key')
    joined = gcg.merge(shuffled,on=keys,suffixes=('_gcg','_shuffled'),validate='one_to_one',how='outer',indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('unpaired condition rows')
    result = joined[keys].copy()
    if not joined['primary_18_gcg'].equals(joined['primary_18_shuffled']):
        raise ValueError('pair subset membership mismatch')
    result['primary_18'] = joined['primary_18_gcg']
    for metric in METRICS:
        result['delta_'+metric] = joined[metric+'_gcg']-joined[metric+'_shuffled']
    return result.sort_values(keys).reset_index(drop=True)


def summarize(metrics,paired,indices):
    subset = metrics[metrics.dataset_index.isin(indices)]
    differences = paired[paired.dataset_index.isin(indices)].set_index(['dataset_index','layer','head'])
    rows=[]
    for (layer,head),group in subset.groupby(['layer','head'],sort=True):
        gcg = group[group.condition=='gcg'].set_index('dataset_index').reindex(indices)
        shuffled = group[group.condition=='shuffled'].set_index('dataset_index').reindex(indices)
        for metric in METRICS:
            x=gcg[metric].to_numpy(float);y=shuffled[metric].to_numpy(float)
            delta=differences.xs((layer,head),level=['layer','head']).reindex(indices)['delta_'+metric].to_numpy(float)
            valid=np.isfinite(x)&np.isfinite(y)
            np.testing.assert_allclose(delta[valid],(x-y)[valid],rtol=0,atol=0)
            tolerance=TIE_ATOL+TIE_RTOL*np.maximum(np.abs(x),np.abs(y))
            rows.append(dict(layer=int(layer),head=int(head),metric=metric,sample_count=len(indices),
                gcg_valid_count=int(np.isfinite(x).sum()),shuffled_valid_count=int(np.isfinite(y).sum()),
                pair_valid_count=int(valid.sum()),
                gcg_mean=float(np.nanmean(x)) if np.isfinite(x).any() else np.nan,
                gcg_median=float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan,
                shuffled_mean=float(np.nanmean(y)) if np.isfinite(y).any() else np.nan,
                shuffled_median=float(np.nanmedian(y)) if np.isfinite(y).any() else np.nan,
                delta_mean=float(np.mean(delta[valid])) if valid.any() else np.nan,
                delta_median=float(np.median(delta[valid])) if valid.any() else np.nan,
                delta_std=float(np.std(delta[valid],ddof=1)) if valid.sum()>1 else np.nan,
                gcg_larger_count=int(((delta>tolerance)&valid).sum()),
                shuffled_larger_count=int(((delta < -tolerance)&valid).sum()),
                equal_within_tolerance_count=int(((np.abs(delta)<=tolerance)&valid).sum())))
    return pd.DataFrame(rows)
