"""CPU negative-energy attribution. Region masks/inputs are owned by prior code."""
import numpy as np
from utils.gcg_shuffled_token_metrics import sample_summary

REGIONS=('UU_full','UU_causal','UO','OO_full','OO_causal')
CONDITION_FIELDS=('E_neg','E_pos','p_neg','m2_neg_zero_extension','negative_count','zero_negative')
PAIR_FIELDS=('delta_E_neg','delta_p_neg','delta_m2_zero_extension','frequency_contribution','magnitude_contribution',
             'positive_element_fraction','negative_element_fraction','equal_element_fraction',
             'positive_contribution','negative_contribution','both_have_negative')
TAILS=(.01,.05,.10)


def factors(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim<1 or x.shape[-1]==0 or not np.isfinite(x).all():raise ValueError('finite nonempty elements required')
    neg=np.minimum(x,0);energy=neg*neg;n=x.shape[-1];count=(x<0).sum(axis=-1)
    total=energy.sum(axis=-1);m2=np.full(count.shape,np.nan,dtype=np.float64)
    np.divide(total,count,out=m2,where=count>0)
    extension=np.where(count>0,m2,0.)
    e=energy.mean(axis=-1);p=count/n;ep=np.mean(np.maximum(x,0)**2,axis=-1)
    np.testing.assert_allclose(e,p*extension,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(e+ep,np.mean(x*x,axis=-1),rtol=1e-12,atol=1e-12)
    return dict(E_neg=e,E_pos=ep,p_neg=p,m2_neg=m2,m2_neg_zero_extension=extension,
                negative_count=count,zero_negative=count==0),energy


def paired_factors(gcg,shuffled):
    if np.shape(gcg)!=np.shape(shuffled):raise ValueError('paired element shapes differ')
    g,eg=factors(gcg);s,es=factors(shuffled);de=eg-es
    delta=g['E_neg']-s['E_neg'];dp=g['p_neg']-s['p_neg']
    dm=g['m2_neg_zero_extension']-s['m2_neg_zero_extension']
    frequency=dp*(g['m2_neg_zero_extension']+s['m2_neg_zero_extension'])/2
    magnitude=dm*(g['p_neg']+s['p_neg'])/2
    pos=np.maximum(de,0).mean(axis=-1);neg=np.minimum(de,0).mean(axis=-1)
    np.testing.assert_allclose(frequency+magnitude,delta,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(de.mean(axis=-1),delta,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(pos+neg,delta,rtol=1e-12,atol=1e-12)
    values=dict(delta_E_neg=delta,delta_p_neg=dp,delta_m2_zero_extension=dm,
        frequency_contribution=frequency,magnitude_contribution=magnitude,
        positive_element_fraction=(de>0).mean(axis=-1),negative_element_fraction=(de<0).mean(axis=-1),
        equal_element_fraction=(de==0).mean(axis=-1),positive_contribution=pos,negative_contribution=neg,
        both_have_negative=(g['negative_count']>0)&(s['negative_count']>0),
        positive_element_count=(de>0).sum(axis=-1),negative_element_count=(de<0).sum(axis=-1),
        equal_element_count=(de==0).sum(axis=-1),positive_contribution_total=np.maximum(de,0).sum(axis=-1),
        negative_contribution_total=np.minimum(de,0).sum(axis=-1))
    errors=dict(factorization=float(np.max(np.abs(frequency+magnitude-delta))),
                element_identity=float(np.max(np.abs(de.mean(axis=-1)-delta))),
                signed_contribution_identity=float(np.max(np.abs(pos+neg-delta))))
    return g,s,values,dict(gcg=eg,shuffled=es,delta=de),errors


def concentration(nonnegative,fraction,basis='all'):
    """Sum largest ceil(fraction*N) terms / total; active basis counts >0 only.

    All and active bases are separately named. No interpolation/epsilon. A zero
    total has undefined share (NaN) and k=0 for active, ceil(f*N) for all.
    """
    x=np.asarray(nonnegative,dtype=np.float64)
    if x.ndim<1 or x.shape[-1]==0 or not np.isfinite(x).all() or (x<0).any():
        raise ValueError('finite nonnegative contributions required')
    if not 0<fraction<=1 or basis not in ('all','active'):raise ValueError('invalid fraction/basis')
    count=np.full(x.shape[:-1],x.shape[-1],dtype=int) if basis=='all' else (x>0).sum(axis=-1)
    k=np.ceil(fraction*count).astype(int);order=np.sort(x,axis=-1)[...,::-1]
    selected=np.arange(x.shape[-1])<k[...,None]
    numerator=np.where(selected,order,0).sum(axis=-1);total=x.sum(axis=-1)
    share=np.full(total.shape,np.nan,dtype=np.float64);np.divide(numerator,total,out=share,where=total>0)
    if ((share[np.isfinite(share)]<0)|(share[np.isfinite(share)]>1+1e-12)).any():raise ValueError('invalid concentration')
    return share,k,total


def head_distribution(delta):
    """Distribution of sample-mean deltas across Heads, not independent samples."""
    x=np.asarray(delta,dtype=np.float64)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all():raise ValueError('finite one-dimensional Head values required')
    q=np.quantile(x,[.05,.25,.50,.75,.95],method='linear')
    return dict(mean=float(x.mean()),median=float(np.median(x)),std=float(x.std(ddof=1)) if len(x)>1 else 0.,
                minimum=float(x.min()),maximum=float(x.max()),q05=float(q[0]),q25=float(q[1]),q50=float(q[2]),q75=float(q[3]),q95=float(q[4]),
                positive_count=int((x>0).sum()),negative_count=int((x<0).sum()),equal_count=int((x==0).sum()),
                positive_fraction=float((x>0).mean()),negative_fraction=float((x<0).mean()),equal_fraction=float((x==0).mean()),
                numerical_zero_count=int((np.abs(x)<=1e-12).sum()),head_count=len(x))


def paired_sample_summary(values,positions):
    stats=sample_summary(values,positions)
    # User definitions are literal >0/<0/=0; retain independent numerical-zero
    # diagnostics rather than changing any previous tolerance-based counts.
    x=values[positions]
    stats.update(positive_count=(x>0).sum(axis=0),negative_count=(x<0).sum(axis=0),
                 equal_count=(x==0).sum(axis=0),numerical_zero_count=(np.abs(x)<=1e-12).sum(axis=0))
    return stats


def causal_query_factors(gcg,shuffled,region):
    if gcg.shape!=shuffled.shape or gcg.ndim!=3 or gcg.shape[1]!=20 or region not in ('UU','UO'):
        raise ValueError('matched Head x20 xKey blocks with explicit region required')
    if region=='UU' and gcg.shape[-1]!=20:raise ValueError('UU must be 20x20')
    if not np.isfinite(gcg).all() or not np.isfinite(shuffled).all():raise ValueError('nonfinite Query block')
    g=[];s=[];pair=[];counts=[]
    for q in range(20):
        end=q+1 if region=='UU' else gcg.shape[-1]
        gg,ss,p,_,_=paired_factors(gcg[:,q,:end],shuffled[:,q,:end])
        g.append([gg[f] for f in CONDITION_FIELDS]);s.append([ss[f] for f in CONDITION_FIELDS])
        pair.append([p[f] for f in PAIR_FIELDS]);counts.append(end)
    # Arrays [condition, Head, Query, field], [Head, Query, field].
    conditions=np.stack([np.stack(g,axis=1).transpose(2,1,0),np.stack(s,axis=1).transpose(2,1,0)])
    pairs=np.stack(pair,axis=1).transpose(2,1,0)
    return conditions,pairs,np.asarray(counts,dtype=int)


def defined_stats(values,axis=0):
    """Conditional-on-defined summaries with explicit counts, never zero imputation."""
    import warnings
    x=np.asarray(values,dtype=float)
    if np.isinf(x).any():raise ValueError('infinite concentration')
    valid=np.isfinite(x);n=valid.sum(axis=axis);mean=np.full(n.shape,np.nan)
    np.divide(np.nansum(x,axis=axis),n,out=mean,where=n>0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(x,axis=axis)
    centered=x-np.expand_dims(mean,axis=axis);variance=np.full_like(mean,np.nan)
    np.divide(np.nansum(centered*centered,axis=axis),n-1,out=variance,where=n>1)
    return dict(defined_count=n,mean=mean,median=median,std=np.sqrt(variance))
