"""Figures from saved attribution tables only; no selection or model processing."""
import numpy as np
import pandas as pd
from utils.artifact_io import read_json
from utils.gcg_shuffled_plot_common import plt,finish

MAIN=('UU_full','UU_causal','UO')


def make_figures(out):
    f=out/'figures';f.mkdir(exist_ok=True);t=out/'tables'
    h=pd.read_csv(t/'head_summary_18.csv');h=h[h.metric=='delta_E_neg']
    dist=pd.read_csv(t/'head_distributions.csv');dist=dist[(dist.subset==18)&(dist.view_id=='all_heads')].set_index('region')
    fig,axes=plt.subplots(2,3,figsize=(16,9));limit=h[h.region.isin(MAIN)]['mean'].abs().max() or 1
    for c,region in enumerate(MAIN):
        part=h[h.region==region];grid=part.pivot(index='layer',columns='head',values='mean')
        im=axes[0,c].imshow(grid,cmap='RdBu_r',vmin=-limit,vmax=limit);fig.colorbar(im,ax=axes[0,c],label='GCG-Shuffled mean squared score')
        grid=part.pivot(index='layer',columns='head',values='positive_count')/18
        im=axes[1,c].imshow(grid,cmap='viridis',vmin=0,vmax=1);fig.colorbar(im,ax=axes[1,c],label='Fraction of 18 samples with delta > 0')
        for ax in axes[:,c]:ax.set(title=region,xlabel='Head (0-31)',ylabel='Layer (0-31)',xticks=[0,8,16,24,31],yticks=[0,8,16,24,31])
    finish(fig,f/'head_heatmaps.png','n=18 paired samples; features per Head first | Full includes future Keys; causal Key<=Query incl. diagonal; exact signs')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,region in zip(axes,MAIN):
        part=h[h.region==region];d=dist.loc[region]
        ax.hist(part['mean'],bins=60,color='slateblue',alpha=.75)
        ax.axvline(d['mean'],color='red',label=f'mean {d["mean"]:.3f}');ax.axvline(d['median'],color='orange',ls='--',label=f'median {d["median"]:.3f}')
        ax.axvline(0,color='black',lw=.7);ax.legend(fontsize=8);ax.set(title=f'{region}: {int(d.positive_count)}/1024 increasing',xlabel='Sample-mean delta E_neg (Score squared)',ylabel='Head count')
    finish(fig,f/'head_distributions.png','n=18 means across samples, distribution across 1024 dependent Heads; descriptive, not independent statistical samples')
    overview=pd.read_csv(t/'factor_overview.csv');a=overview[(overview.subset==18)&(overview.view_id=='all_heads')].set_index('region')
    with np.load(out/'core_values.npz') as z:p=z['pairs']
    m=read_json(out/'core_manifest.json');pos=[m['all_indices'].index(i) for i in m['primary_indices']]
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,region in zip(axes,MAIN):
        r=a.loc[region];ri=m['regions'].index(region)
        names=['delta_E_neg','frequency_contribution','magnitude_contribution']
        vals=[r['mean_'+name] for name in names]
        sd=[p[pos,ri,:,m['pair_fields'].index(name)].mean(axis=1).std(ddof=1) for name in names]
        ax.bar(range(3),vals,color=['grey','tab:orange','tab:blue']);ax.errorbar(range(3),vals,yerr=sd,fmt='none',color='black',capsize=3)
        ax.axhline(0,color='black',lw=.7);ax.set(xticks=range(3),xticklabels=['Total delta','Frequency term','Magnitude term'],title=region,ylabel='GCG-Shuffled E_neg (Score squared)')
    finish(fig,f/'factor_decomposition.png','n=18 equal Head then sample mean | midpoint identity; error bars=sample SD; zero-negative moment extension explicit in manifest')
    sample=pd.read_csv(t/'sample_summary.csv');sample=sample[sample.subset==18]
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for c,region in enumerate(MAIN):
        part=sample[sample.region==region]
        axes[0,c].bar(range(18),part['mean'],color=np.where(part['mean']>=0,'tab:red','tab:blue'))
        axes[0,c].plot(range(18),part['median'],color='black',marker='.',label='Head median');axes[0,c].legend(fontsize=8)
        axes[0,c].axhline(part['mean'].mean(),color='orange',ls='--');axes[0,c].set(title=region,ylabel='Delta E_neg (Score squared)')
        axes[1,c].bar(range(18),part.positive_fraction,color='purple');axes[1,c].set(ylabel='Fraction of Heads with delta >0',ylim=(0,1))
        for ax in axes[:,c]:ax.set_xticks(range(18),part.dataset_index,rotation=60,fontsize=7);ax.set_xlabel('Dataset Index')
    finish(fig,f/'sample_consistency.png','n=18 paired samples; equal 1024 Head mean / Head median; orange=sample mean; Full vs causal explicitly labeled')
    group_figure(out,m,h)
    tail_figures(out)
    query_figures(out)


def group_figure(out,m,heads):
    t=out/'tables';v=pd.read_csv(t/'fixed_view_summary_18.csv');v=v[v.metric=='delta_E_neg']
    order=['all_heads']+[x['view_id'] for x in m['views'] if x['kind']=='selected_group']
    views={x['view_id']:x for x in m['views']};fig,axes=plt.subplots(1,3,figsize=(18,9))
    for ax,region in zip(axes,MAIN):
        for i,name in enumerate(order):
            r=v[(v.region==region)&(v.view_id==name)].iloc[0]
            ax.errorbar(r['mean'],i,xerr=r['std'],fmt='D',color='black',markersize=4,capsize=2)
            if name!='all_heads':
                hh=heads[heads.region==region].set_index(['layer','head']);members=views[name]['members']
                values=[hh.loc[(h//32,h%32),'mean'] for h in members]
                ax.scatter(values,i+np.linspace(-.15,.15,len(values)),color='tab:blue',s=16)
        ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(len(order)),order,fontsize=8);ax.invert_yaxis();ax.set(title=region,xlabel='Delta E_neg (Score squared)')
    finish(fig,out/'figures/fixed_head_groups.png','n=18 | old fixed groups unchanged; diamond=group mean, bar=sample SD, dots=each of 5 member Head means; no new Head group')


def tail_figures(out):
    a=pd.read_csv(out/'tables/tail_overview.csv');a=a[(a.subset==18)&(a.view_id=='all_heads')]
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    for c,region in enumerate(MAIN):
        for kind,color in [('energy_gcg','tab:red'),('energy_shuffled','tab:blue')]:
            p=a[(a.region==region)&(a.kind==kind)].sort_values('fraction')
            axes[0,c].plot(100*p.fraction,p.energy_weighted_share,'o-',color=color,label=kind+' weighted')
            axes[0,c].plot(100*p.fraction,p.defined_case_mean_share,'--',color=color,label=kind+' defined Head mean')
        for basis,color in [('active','tab:green'),('all','tab:purple')]:
            p=a[(a.region==region)&(a.kind=='positive_delta')&(a.basis==basis)].sort_values('fraction')
            axes[1,c].plot(100*p.fraction,p.energy_weighted_share,'o-',color=color,label='positive delta / '+basis+' weighted')
            axes[1,c].plot(100*p.fraction,p.defined_case_mean_share,'--',color=color,label=basis+' defined Head mean')
        for r in range(2):
            axes[r,c].set(title=region,xlabel='Nominal top percentage; k=ceil(rate*basis count)',ylabel='Share of negative energy' if r==0 else 'Share of positive delta energy',ylim=(0,1.03),xticks=[1,5,10]);axes[r,c].legend(fontsize=6)
    finish(fig,out/'figures/element_concentration.png','n=18 | energy: ALL valid elements; positive delta: active/all bases distinct | solid=mean(top/N)/mean(total/N); dashed=defined-case mean share')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,region in zip(axes,MAIN):
        for basis,color in [('active','tab:green'),('all','tab:purple')]:
            p=a[(a.region==region)&(a.kind=='positive_delta')&(a.basis==basis)].sort_values('fraction')
            ax.plot(100*p.fraction,p.net_delta_after_top_positive_removed,'o-',color=color,label='basis: '+basis)
            ax.axhline(p.net_delta.iloc[0],color='orange',ls='--',label='original net delta' if basis=='active' else None)
        ax.axhline(0,color='black',lw=.7);ax.set(title=region,xlabel='Nominal top percentage (ceil)',ylabel='Residual net delta E_neg (Score squared)',xticks=[1,5,10]);ax.legend(fontsize=8)
    finish(fig,out/'figures/tail_sensitivity.png','n=18 equal Head/sample mean | subtract top positive contributions at original N; descriptive sensitivity, no Score/model intervention')


def query_figures(out):
    q=pd.read_csv(out/'tables/query_overview.csv');q=q[(q.subset==18)&(q.view_id=='all_heads')]
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for c,region in enumerate(('UU_causal','UO')):
        p=q[q.region==region].sort_values('query')
        axes[0,c].plot(p['query'],p.gcg_mean,label='GCG',color='tab:red');axes[0,c].plot(p['query'],p.shuffled_mean,label='Shuffled',color='tab:blue');axes[0,c].legend()
        axes[0,c].set(title=region,ylabel='E_neg per visible Query row (Score squared)')
        axes[1,c].plot(p['query'],p.delta_mean,color='purple');axes[1,c].fill_between(p['query'],p.delta_mean-p.delta_std,p.delta_mean+p.delta_std,alpha=.2)
        axes[1,c].axhline(0,color='black',lw=.7);axes[1,c].set(ylabel='Delta row E_neg; shade=sample SD')
        for r in range(2):axes[r,c].set_xlabel('Suffix Query relative position 0-19; position matched, token content differs')
    finish(fig,out/'figures/query_profiles.png','n=18 | Causal only, incl. diagonal; per Query/Head features then equal Head/sample mean; row vs element-weighted means saved separately')
    l=pd.read_csv(out/'tables/layer_query_summary.csv');l=l[l.subset==18]
    fig,axes=plt.subplots(1,2,figsize=(13,8));limit=l.delta_mean.abs().max() or 1
    for ax,region in zip(axes,('UU_causal','UO')):
        grid=l[l.region==region].pivot(index='layer',columns='query',values='delta_mean')
        im=ax.imshow(grid,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto');ax.set(title=region,xlabel='Suffix Query position',ylabel='Layer (0-31)');fig.colorbar(im,ax=ax,label='Delta row E_neg (Score squared)')
    finish(fig,out/'figures/layer_query_heatmaps.png','n=18 | equal 32 Head means per Layer, then sample mean; all entries causal, no significance claim')
    q=pd.read_csv(out/'tables/query_overview.csv');q=q[q.subset==18]
    m=read_json(out/'core_manifest.json');order=['all_heads']+[v['view_id'] for v in m['views'] if v['kind']=='selected_group']
    fig,axes=plt.subplots(1,2,figsize=(15,8));lim=q[q.view_id.isin(order)].delta_mean.abs().max() or 1
    for ax,region in zip(axes,('UU_causal','UO')):
        grid=q[q.region==region].pivot(index='view_id',columns='query',values='delta_mean').reindex(order)
        im=ax.imshow(grid,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_yticks(range(len(order)),order,fontsize=8);ax.set(title=region,xlabel='Suffix Query position');fig.colorbar(im,ax=ax,label='Delta row E_neg (Score squared)')
    finish(fig,out/'figures/fixed_group_query_profiles.png','n=18 | old fixed groups; per Query/Head metric before equal group/sample mean; causal only')
