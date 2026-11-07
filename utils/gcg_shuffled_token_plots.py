"""Bounded-size diagnostic sheets, with explicit positions and shared scales."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-gcg-shuffled-token-plots')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.artifact_io import utc_now
from utils.gcg_shuffled_metrics import METRICS


def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:return {k:archive[k] for k in archive.files}


def matrix_sheet(path,summary,rows,columns,title,sample_count=18):
    """columns: (field, title, nonnegative, units), same GCG/Shuffled scale."""
    nr,nc=len(rows),len(columns)
    fig,axes=plt.subplots(nr,nc,figsize=(3.5*nc,2.35*nr),squeeze=False)
    vmax=[max(float(np.max(np.abs(summary[field][[r[0] for r in rows]]))),1e-12) for field,_,_,_ in columns]
    if nc>=2:vmax[0]=vmax[1]=max(vmax[:2])
    for c,(field,label,nonnegative,unit) in enumerate(columns):
        for r,(view_index,row_label) in enumerate(rows):
            ax=axes[r,c]
            image=ax.imshow(summary[field][view_index],origin='upper',cmap='viridis' if nonnegative else 'RdBu_r',
                            vmin=0 if nonnegative else -vmax[c],vmax=vmax[c])
            ax.set_xticks([0,5,10,15,19]);ax.set_yticks([0,5,10,15,19]);ax.tick_params(labelsize=7)
            if r==0:ax.set_title(label,fontsize=10)
            if c==0:ax.set_ylabel(row_label+'\nQuery suffix position',fontsize=8)
            if r==nr-1:ax.set_xlabel('Key suffix position (0..19)')
        fig.colorbar(image,ax=axes[:,c].tolist(),location='bottom',shrink=.85,pad=.025,label=unit)
    fig.suptitle(title+f' | n={sample_count} | position-aligned, unmasked',fontsize=12,y=.995)
    fig.savefig(path,dpi=140,bbox_inches='tight');plt.close(fig)


def row_sheet(path,summary,rows,title):
    fig,axes=plt.subplots(len(rows),5,figsize=(20,2.15*len(rows)),squeeze=False)
    p=np.arange(20)
    for r,(v,label) in enumerate(rows):
        a=axes[r]
        for condition,color in [('gcg','tab:red'),('shuffled','tab:blue')]:
            a[0].plot(p,summary['mean__weight_'+condition][v,3],color=color,label=condition.upper())
            a[4].plot(range(4),summary['closing_mean__weight_'+condition][v,3],color=color,label=condition.upper())
        a[1].axhline(0,color='grey',lw=.7)
        a[1].plot(p,summary['mean__delta_weight'][v,3],color='black',label='signed mean')
        a[1].plot(p,summary['mean__abs_delta_weight'][v,3],color='orange',linestyle='--',label='mean |head delta|')
        a[2].axhline(0,color='grey',lw=.7);a[2].plot(p,summary['mean__delta_raw'][v,3],color='purple')
        g=summary['mean__region_gcg'][v,3];s=summary['mean__region_shuffled'][v,3]
        a[3].bar(np.arange(4)-.18,g,width=.36,color='tab:red',label='GCG')
        a[3].bar(np.arange(4)+.18,s,width=.36,color='tab:blue',label='Shuffled')
        a[3].set_xticks(range(4),['Start','Original','Suffix','Closing'],rotation=15)
        for c in (0,1,2):a[c].set_xticks([0,5,10,15,19])
        a[4].set_xticks(range(4))
        a[0].set_ylabel(label+'\nAttention fraction',fontsize=8)
        if r==0:
            for ax,t in zip(a,['Suffix: GCG / Shuffled','Suffix weight difference','Raw score GCG-Shuffled','Region attention fractions','Closing Key token weights']):ax.set_title(t,fontsize=9)
            for c in (0,1,3):a[c].legend(fontsize=6)
        if r==len(rows)-1:
            for c in (0,1,2):a[c].set_xlabel('Suffix relative Key position')
            a[4].set_xlabel('Closing relative Key position')
        for ax in a:ax.tick_params(labelsize=7)
    fig.suptitle(title+' | q=L-1, n=18 | Head-first then sample mean',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.985));fig.savefig(path,dpi=140);plt.close(fig)


def make_figures(directory):
    manifest=json.loads((directory/'manifest.json').read_text());views=manifest['views']
    ids={v['view_id']:i for i,v in enumerate(views)}
    selected=pd.read_csv(directory/'selected_heads.csv')
    matrix=load_npz(directory/'matrix_summary_18.npz');row=load_npz(directory/'row_summary_18.npz')
    figures=directory/'figures';figures.mkdir(exist_ok=False)
    for metric in METRICS:
        part=selected[selected.metric==metric]
        rows=[(ids[f'L{int(r.layer):02d}H{int(r.head):02d}'],f'L{int(r.layer)} H{int(r.head)} {r.direction}\n{metric}: {r.delta_mean:+.4g}') for r in part.itertuples()]
        if metric in ('M_U','D'):
            row_sheet(figures/f'heads_{metric}.png',row,rows,f'{metric}: signed top/bottom five individual Heads')
        else:
            if metric=='F':base,squared,contribution='raw','square','F_delta_contribution'
            elif metric=='F_sym':base,squared,contribution='sym','sym_square','F_sym_delta_contribution'
            elif metric=='F_asym':base,squared,contribution='asym','asym_square','F_asym_delta_contribution'
            else:base,squared,contribution='normalized_asym_square','normalized_asym_square','delta_normalized_asym_square'
            cols=[('mean__'+base+'_gcg','GCG mean',metric=='R_asym','fraction' if metric=='R_asym' else 'score'),
                  ('mean__'+base+'_shuffled','Shuffled mean',metric=='R_asym','fraction' if metric=='R_asym' else 'score'),
                  ('mean__delta_'+base,'Signed difference',False,'fraction' if metric=='R_asym' else 'score')]
            if metric=='R_asym':cols.append(('mean__delta_normalized_sym_square','Sym ratio contribution diff',False,'fraction'))
            else:cols.extend([('mean__delta_'+squared,'Squared-value difference',False,'score squared'),('mean__'+contribution,'Exact norm delta attribution',False,'score')])
            matrix_sheet(figures/f'heads_{metric}.png',matrix,rows,cols,f'{metric}: top/bottom five individual Heads')
    group_rows=[(ids['all_heads'],'All 1024 Heads')]+[(i,v['view_id']+' (5 Heads)') for i,v in enumerate(views) if v['kind']=='selected_group']
    row_sheet(figures/'group_query_profiles.png',row,group_rows,'Selected Head groups and all-Head mean')
    specs={
        'raw': [('mean__raw_gcg','GCG raw score',False,'score'),('mean__raw_shuffled','Shuffled raw score',False,'score'),('mean__delta_raw','Signed mean difference',False,'score'),('mean__abs_delta_raw','Mean absolute head difference',True,'absolute score')],
        'energy': [('mean__square_gcg','GCG mean B squared',True,'score squared'),('mean__square_shuffled','Shuffled mean B squared',True,'score squared'),('mean__delta_square','Squared-value difference',False,'score squared'),('mean__F_delta_contribution','Exact F delta attribution',False,'score')],
        'sym': [('mean__sym_gcg','GCG symmetric score',False,'score'),('mean__sym_shuffled','Shuffled symmetric score',False,'score'),('mean__delta_sym','Symmetric difference',False,'score'),('mean__delta_sym_square','Symmetric squared difference',False,'score squared')],
        'asym': [('mean__asym_gcg','GCG antisymmetric score',False,'score'),('mean__asym_shuffled','Shuffled antisymmetric score',False,'score'),('mean__delta_asym','Antisymmetric difference',False,'score'),('mean__delta_asym_square','Antisymmetric squared difference',False,'score squared')],
        'ratio': [('mean__normalized_asym_square_gcg','GCG asym ratio contribution',True,'fraction'),('mean__normalized_asym_square_shuffled','Shuffled asym ratio contribution',True,'fraction'),('mean__delta_normalized_asym_square','Asym ratio contribution delta',False,'fraction'),('mean__delta_normalized_sym_square','Sym ratio contribution delta',False,'fraction')]}
    for name,cols in specs.items():matrix_sheet(figures/f'group_{name}.png',matrix,group_rows,cols,f'Selected groups / all Heads: {name}')
    consistency_figure(directory,figures,selected,manifest)
    token_consistency_figure(figures,row,group_rows)
    query_figure(directory,figures)
    sensitivity_figure(directory,figures,ids['all_heads'])


def consistency_figure(directory,figures,selected,manifest):
    archive=load_npz(directory/'all_head_scalars.npz');values=archive['view_scalar']
    indices=[manifest['all_indices'].index(i) for i in manifest['primary_indices']]
    view_ids={v['view_id']:i for i,v in enumerate(manifest['views'])}
    names=('M_U','M_O','D','F','F_sym','F_asym','R_sym','R_asym')
    fig,axes=plt.subplots(3,2,figsize=(18,16))
    for ax,metric in zip(axes.flat,METRICS):
        heads=selected[selected.metric==metric]
        views=[view_ids[f'L{int(r.layer):02d}H{int(r.head):02d}'] for r in heads.itertuples()]
        data=(values[indices,:,0,names.index(metric)]-values[indices,:,1,names.index(metric)])[:,views].T
        maximum=max(float(np.max(np.abs(data))),1e-12)
        image=ax.imshow(data,cmap='RdBu_r',vmin=-maximum,vmax=maximum,aspect='auto')
        ax.set_xticks(range(18),manifest['primary_indices'],rotation=60,fontsize=7)
        ax.set_yticks(range(len(heads)),[f'L{int(r.layer)}H{int(r.head)} {r.direction[:3]}' for r in heads.itertuples()],fontsize=8)
        ax.set(title=f'{metric}: sample GCG - Shuffled',xlabel='Dataset Index');fig.colorbar(image,ax=ax,label='metric units')
    fig.suptitle('Consistency on the SAME primary18 used for Head selection (not independent validation)')
    fig.tight_layout();fig.savefig(figures/'sample_consistency.png',dpi=160);plt.close(fig)


def token_consistency_figure(figures,row,rows):
    ids=[r[0] for r in rows];labels=[r[1] for r in rows]
    fields=[('mean__delta_weight','Signed mean delta','fraction',False),('mean__abs_delta_weight','Mean |head/sample delta|','fraction',True),('positive_count__delta_weight','Positive sample count after Head mean','samples / 18',True)]
    fig,axes=plt.subplots(1,3,figsize=(17,8))
    for ax,(field,title,unit,nonnegative) in zip(axes,fields):
        data=row[field][ids,3,:];maximum=18 if field.startswith('positive') else max(float(np.max(np.abs(data))),1e-12)
        image=ax.imshow(data,aspect='auto',cmap='viridis' if nonnegative else 'RdBu_r',vmin=0 if nonnegative else -maximum,vmax=maximum)
        ax.set_xticks(range(20));ax.set_yticks(range(len(rows)),labels,fontsize=7);ax.tick_params(axis='x',labelsize=7)
        ax.set(title=title,xlabel='Suffix relative Key position');fig.colorbar(image,ax=ax,label=unit,location='bottom')
    fig.suptitle('Final Query token-position consistency | primary18; equal-weight Head views')
    fig.tight_layout();fig.savefig(figures/'token_position_consistency.png',dpi=180);plt.close(fig)


def query_figure(directory,figures):
    table=pd.read_csv(directory/'four_query_summary.csv');table=table[table.subset==18]
    fig,axes=plt.subplots(3,2,figsize=(14,12));slots=['1','2','3','4','mean4']
    for r,metric in enumerate(('M_U','M_O','D')):
        sub=table[(table.view_id=='all_heads')&(table.metric==metric)].set_index('query_slot').loc[slots]
        for condition,color in [('gcg','tab:red'),('shuffled','tab:blue')]:axes[r,0].plot(range(5),sub[condition+'_mean'],marker='o',color=color,label=condition.upper())
        for view in ('all_heads','M_U_increase','M_U_decrease','D_increase','D_decrease'):
            sub=table[(table.view_id==view)&(table.metric==metric)].set_index('query_slot').loc[slots]
            axes[r,1].plot(range(5),sub.delta_mean,marker='o',label=view)
        axes[r,1].axhline(0,color='grey',lw=.7)
        for c in range(2):
            axes[r,c].set_xticks(range(5),['q1 [518]','q2 [29914]','q3 [25580]','q4 [29962]','Mean 4'],rotation=15)
            axes[r,c].set_ylabel(metric+' (log ratio)' if metric=='D' else metric+' (fraction)');axes[r,c].legend(fontsize=8)
        axes[r,0].set_title(f'All 1024 Heads: {metric}');axes[r,1].set_title(f'Group mean GCG-Shuffled: {metric}')
    fig.suptitle('Closing-template Query diagnostics | primary18 | future keys excluded per Query')
    fig.tight_layout();fig.savefig(figures/'closing_queries.png',dpi=180);plt.close(fig)


def sensitivity_figure(directory,figures,all_view):
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for n,color in [(18,'tab:orange'),(20,'tab:green')]:
        row=load_npz(directory/f'row_summary_{n}.npz');matrix=load_npz(directory/f'matrix_summary_{n}.npz')
        axes[0,0].plot(range(20),row['mean__delta_weight'][all_view,3],label=f'n={n}',color=color)
        axes[0,1].plot(range(20),matrix['mean__delta_normalized_asym_square'][all_view].sum(axis=0),label=f'n={n}',color=color)
        axes[1,0].plot(range(20),matrix['mean__F_delta_contribution'][all_view].sum(axis=0),label=f'n={n}',color=color)
        q=pd.read_csv(directory/'four_query_summary.csv');sub=q[(q.subset==n)&(q.view_id=='all_heads')&(q.metric=='M_U')].set_index('query_slot').loc[['1','2','3','4']]
        axes[1,1].plot(range(1,5),sub.delta_mean,label=f'n={n}',color=color)
    titles=['Suffix weight delta (fraction)','Asym ratio Key contribution delta (fraction)','F norm Key attribution delta (score)','Closing Query suffix mass delta (fraction)']
    for i,(ax,title) in enumerate(zip(axes.flat,titles)):
        ax.set_title(title);ax.set_xlabel('Suffix relative Key position' if i<3 else 'Closing Query slot');ax.axhline(0,color='grey',lw=.7);ax.legend()
    fig.suptitle('All 1024 Heads: primary18 vs auxiliary20; same Head selection')
    fig.tight_layout();fig.savefig(figures/'sensitivity_18_20.png',dpi=180);plt.close(fig)


def render_index(directory,index,view):
    manifest=json.loads((directory/'manifest.json').read_text());names=[v['view_id'] for v in manifest['views']]
    if index not in manifest['all_indices'] or view not in names:raise ValueError('unknown saved Index/view')
    i=manifest['all_indices'].index(index);v=names.index(view)
    rows=load_npz(directory/'row_profiles'/f'index_{index:04d}.npz')
    data=load_npz(directory/'matrix_per_sample.npz')
    target=directory/'diagnostics'/f'index_{index}_{view}_{utc_now().replace(":", "")}'
    target.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(3,1,figsize=(13,9))
    for condition,color in [('gcg','tab:red'),('shuffled','tab:blue')]:axes[0].plot(rows['weight_'+condition][v,3],color=color,label=condition.upper())
    axes[1].plot(rows['delta_weight'][v,3],color='black');axes[2].plot(rows['delta_raw'][v,3],color='purple')
    for ax in axes:ax.axhline(0,color='grey',lw=.5);ax.legend() if ax is axes[0] else None;ax.set_xlabel('Absolute Key token position')
    reference=next(r for r in manifest['score_references'] if r['dataset_index']==index and r['condition']=='gcg')
    for name,color in [('template_prefix','grey'),('original','green'),('suffix','orange'),('template_closing','blue')]:
        a,b=reference['input']['token_ranges'][name]
        for ax in axes:ax.axvspan(a-.5,b-.5,color=color,alpha=.08)
        axes[0].text((a+b-1)/2,1.02,name,ha='center',fontsize=7,transform=axes[0].get_xaxis_transform())
    axes[0].set_ylabel('Attention fraction');axes[1].set_ylabel('GCG-Shuffled fraction');axes[2].set_ylabel('Raw score difference')
    fig.suptitle(f'Index {index}, {view}, final Query; variable Original positions NOT averaged')
    fig.tight_layout();fig.savefig(target/'query_profile.png',dpi=180);plt.close(fig)
    summary={'mean__'+k:value[i] for k,value in data.items()}
    matrix_sheet(target/'suffix_matrix.png',summary,[(v,view)],
        [('mean__raw_gcg','GCG',False,'score'),('mean__raw_shuffled','Shuffled',False,'score'),
         ('mean__delta_raw','Difference',False,'score'),('mean__F_delta_contribution','Exact F delta attribution',False,'score')],f'Index {index}, {view}',sample_count=1)
    print(target)
