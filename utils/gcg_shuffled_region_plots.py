"""Region-analysis figures with explicit masks, units and averaging levels."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from utils.gcg_shuffled_plot_common import plt,finish


def make_figures(root):
    figures=root/'figures';figures.mkdir()
    manifest=json.loads((root/'manifest.json').read_text());views=manifest['views']
    overview=pd.read_csv(root/'overview.csv');maps=pd.read_csv(root/'head_maps_18.csv')
    groups=pd.read_csv(root/'view_summary_18.csv');members=pd.read_csv(root/'fixed_group_member_effects.csv')
    metrics=manifest['metrics'];positions=[manifest['all_indices'].index(i) for i in manifest['primary_indices']]
    with np.load(root/'per_sample_values.npz',allow_pickle=False) as archive:
        values=archive['region_metrics'];attn=archive['attention_q4']
    n=18;primary=overview[overview.subset==n].set_index('metric')
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for r,policy in enumerate(('full','causal')):
        for c,(prefix,bases,unit) in enumerate([('F_',('OO','UU','UO'),'Frobenius norm (score)'),('RMS_',('OO','UU','UO'),'RMS (score)'),('R_',('UU_OO','UU_UO'),'RMS ratio (dimensionless)')]):
            names=[prefix+b+'_'+policy for b in bases];x=np.arange(len(names))
            axes[r,c].bar(x-.18,primary.loc[names].gcg_mean,width=.36,label='GCG',color='tab:red')
            axes[r,c].bar(x+.18,primary.loc[names].shuffled_mean,width=.36,label='Shuffled',color='tab:blue')
            axes[r,c].set(xticks=x,xticklabels=bases,ylabel=unit,title=f'{policy}: {prefix.rstrip("_")}');axes[r,c].legend(fontsize=8)
    for c in range(3):
        maximum=max(axes[0,c].get_ylim()[1],axes[1,c].get_ylim()[1])
        for r in range(2):axes[r,c].set_ylim(0,maximum)
    finish(fig,figures/'region_overview.png','n=18 | per-Head/sample metrics, then equal Head/sample mean | all L0-31 H0-31; causal includes diagonal')
    heatmaps(root,figures,maps)
    selected_names=['RMS_UU_full','RMS_UO_full','R_UU_OO_full','R_UU_UO_full','RMS_UU_causal','R_UU_UO_causal']
    fig,axes=plt.subplots(3,2,figsize=(15,12))
    for ax,name in zip(axes.flat,selected_names):
        j=metrics.index(name);d=(values[positions,0,:,j]-values[positions,1,:,j]).mean(axis=1)
        ax.bar(range(18),d,color=np.where(d>=0,'tab:red','tab:blue'),alpha=.7)
        ax.axhline(0,color='black',lw=.7);ax.axhline(d.mean(),color='orange',ls='--',label='sample mean')
        ax.set_xticks(range(18),manifest['primary_indices'],rotation=60,fontsize=7)
        ax.set(title=name,xlabel='Dataset Index',ylabel='GCG-Shuffled (ratio)' if name.startswith('R_') else 'GCG-Shuffled RMS (score)');ax.legend()
    finish(fig,figures/'sample_consistency.png','n=18 paired samples | equal mean of 1024 already-computed Head metrics; masks in metric names')
    group_figure(figures,groups,members,views,selected_names)
    mask_figure(figures,primary,maps)
    association_figure(figures,maps,root)
    query_figure(figures,root)
    sensitivity_figure(figures,overview)


def heatmaps(root,figures,maps):
    bases=['RMS_UU','RMS_UO','R_UU_OO','R_UU_UO'];grids=root/'heatmap_values';grids.mkdir()
    maxima={base:max(float(maps['delta_'+base+'_'+p].abs().max()) for p in ('full','causal')) for base in bases}
    for policy in ('full','causal'):
        fig,axes=plt.subplots(2,2,figsize=(13,11))
        for ax,base in zip(axes.flat,bases):
            field='delta_'+base+'_'+policy;grid=maps.pivot(index='layer',columns='head',values=field).reindex(index=range(32),columns=range(32))
            grid.to_csv(grids/f'{field}_18.csv',mode='x',float_format='%.17g')
            maximum=maxima[base] if maxima[base]>0 else 1
            cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('grey')
            image=ax.imshow(np.ma.masked_invalid(grid),origin='upper',cmap=cmap,vmin=-maximum,vmax=maximum)
            ax.set(xticks=[0,4,8,12,16,20,24,28,31],yticks=[0,4,8,12,16,20,24,28,31],xlabel='Head (0-based)',ylabel='Layer (0-based)',title=field)
            fig.colorbar(image,ax=ax,label='dimensionless' if base.startswith('R_') else 'score')
        finish(fig,figures/f'head_heatmaps_{policy}.png',f'n=18 | paired mean GCG-Shuffled | {policy}, per-sample metrics; shared full/causal color limits')


def group_figure(figures,groups,members,views,names):
    ordered=[v for v in views if v['kind']!='individual']
    ordered=[v for v in ordered if v['kind']=='all_heads']+[v for v in ordered if v['kind']!='all_heads']
    fig,axes=plt.subplots(3,2,figsize=(20,16))
    labels=[v['view_id']+'\n'+('L0-31 H0-31' if v['kind']=='all_heads' else ','.join(f'{h//32}:{h%32}' for h in v['members'])) for v in ordered]
    for ax,name in zip(axes.flat,names):
        for i,v in enumerate(ordered):
            row=groups[(groups.view_id==v['view_id'])&(groups.metric==name)].iloc[0]
            ax.errorbar(row.delta_mean,i,xerr=row.delta_std,fmt='D',color='black',capsize=2,markersize=4)
            if v['kind']!='all_heads':
                part=members[(members.subset==18)&(members.selected_group==v['view_id'])&(members.metric==name)].sort_values(['layer','head'])
                ax.scatter(part.delta_mean,i+np.linspace(-.16,.16,len(part)),s=18,color=np.where(part.delta_mean>=0,'tab:red','tab:blue'))
        ax.axvline(0,color='grey',lw=.8);ax.set_yticks(range(len(ordered)),labels,fontsize=6);ax.invert_yaxis()
        ax.set(title=name,xlabel='GCG-Shuffled (ratio)' if name.startswith('R_') else 'GCG-Shuffled RMS (score)')
    finish(fig,figures/'fixed_head_groups.png','Fixed old groups, n=18 | diamond: Head/sample mean; bars: sample SD; dots: each fixed member Head mean; masks in names')


def mask_figure(figures,overview,maps):
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for c,base in enumerate(['RMS_UU','R_UU_UO']):
        names=[base+'_full',base+'_causal'];x=np.arange(2)
        for condition,color,offset in [('gcg','tab:red',-.18),('shuffled','tab:blue',.18)]:
            axes[0,c].bar(x+offset,overview.loc[names][condition+'_mean'],width=.36,color=color,label=condition.upper())
        axes[0,c].set(xticks=x,xticklabels=['Full unmasked','Causal visible'],ylabel='ratio' if base.startswith('R_') else 'RMS score',title=base);axes[0,c].legend()
        a=maps['delta_'+base+'_full'];b=maps['delta_'+base+'_causal']
        axes[1,c].scatter(a,b,c=maps.layer,cmap='viridis',s=8,alpha=.6)
        maximum=max(float(a.abs().max()),float(b.abs().max()))
        axes[1,c].plot([-maximum,maximum],[-maximum,maximum],color='grey',ls='--')
        axes[1,c].axhline(0,color='grey',lw=.6);axes[1,c].axvline(0,color='grey',lw=.6)
        axes[1,c].set(title=base+' mean paired delta, all Layer/Head',xlabel='Full unmasked delta',ylabel='Causal-visible delta')
    finish(fig,figures/'mask_effect.png','n=18 | metrics before Head/sample averaging; UU counts 400 vs 210; UO all visible in both policies')


def association_figure(figures,maps,root):
    table=pd.read_csv(root/'associations.csv');fig,axes=plt.subplots(2,2,figsize=(14,11))
    maps=maps.copy();maps['negative_delta_M_O']=-maps.delta_M_O
    pairs=[('delta_M_U','negative_delta_M_O'),('delta_M_U','delta_RMS_UU_full'),('delta_M_O','delta_RMS_UO_full'),('delta_D','delta_R_UU_UO_full')]
    for ax,(x,y) in zip(axes.flat,pairs):
        image=ax.scatter(maps[x],maps[y],c=maps.layer,cmap='viridis',s=10,alpha=.65)
        relation=table[(table.subset==18)&(~table.layer_centered)&(table.x==x)&(table.y==y)].iloc[0]
        ax.axhline(0,color='grey',lw=.6);ax.axvline(0,color='grey',lw=.6)
        ax.set(xlabel=x,ylabel=y,title=f'Pearson {relation.pearson:.3f}; Spearman {relation.spearman:.3f}')
        fig.colorbar(image,ax=ax,label='Layer (0..31)')
    finish(fig,figures/'attention_vs_scores.png','n=18 mean paired changes, 1024 dependent Heads | mass fractions vs raw score RMS/ratios; no causal inference')


def query_figure(figures,root):
    table=pd.read_csv(root/'attention_existing_query_summary.csv');fig,axes=plt.subplots(1,3,figsize=(15,5))
    slots=['1','2','3','4','mean4']
    for ax,metric in zip(axes,('M_U','M_O','D')):
        part=table[(table.subset==18)&(table.view_id=='all_heads')&(table.metric==metric)].set_index('query_slot').loc[slots]
        for condition,color in [('gcg','tab:red'),('shuffled','tab:blue')]:
            ax.plot(range(5),part[condition+'_mean'],marker='o',label=condition.upper(),color=color)
        ax.set(xticks=range(5),xticklabels=['q1','q2','q3','q4','mean4'],ylabel='log mass ratio' if metric=='D' else 'attention fraction',title=metric);ax.legend()
    finish(fig,figures/'attention_existing_queries.png','Reused values, n=18, all L0-31/H0-31 | causal softmax per Query; average calculated metrics, not averaged scores')


def sensitivity_figure(figures,overview):
    names=['RMS_UU_full','RMS_UO_full','R_UU_OO_full','R_UU_UO_full','RMS_UU_causal','R_UU_UO_causal']
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,name in zip(axes.flat,names):
        part=overview[overview.metric==name].set_index('subset').loc[[18,20]]
        ax.bar([0,1],part.delta_mean,color=['tab:orange','tab:green']);ax.axhline(0,color='grey',lw=.7)
        ax.set(xticks=[0,1],xticklabels=['primary18','all20'],title=name,ylabel='mean paired delta (ratio)' if name.startswith('R_') else 'mean paired delta RMS (score)')
    finish(fig,figures/'sensitivity_18_20.png','Equal mean of per-Head/sample metrics; same fixed groups; masks explicit in metric names')
