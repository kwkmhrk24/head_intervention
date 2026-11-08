"""Compact publication-style panels; plotted values are saved stage tables."""
import numpy as np
import pandas as pd
from utils.gcg_shuffled_plot_common import plt,finish


def make_figures(paths,m):
    for p in paths.values():(p/'figures').mkdir()
    so=pd.read_csv(paths['sign']/'overview.csv');s=so[so.subset==18].set_index('metric')
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for r,policy in enumerate(('full','causal')):
        for c,region in enumerate(('OO','UU','UO')):
            names=[f'E_{sign}_{region}_{policy}' for sign in ('pos','neg')];x=np.arange(2);ax=axes[r,c]
            for cond,color,dx in [('gcg','tab:red',-.18),('shuffled','tab:blue',.18)]:ax.bar(x+dx,s.loc[names,cond+'_mean'],.36,label=cond,color=color)
            ax.set(xticks=x,xticklabels=['E_pos','E_neg'],title=region+' '+policy,ylabel='Mean squared score');ax.legend(fontsize=8)
    finish(fig,paths['sign']/'figures/energy_overview.png','n=18 | per sample/Head energy, equal Head/sample mean; full includes future, causal includes diagonal')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for r,policy in enumerate(('full','causal')):
        for c,base in enumerate(('E','mu')):
            ax=axes[r,c];x=np.arange(3)
            for sign,dx,color in [('pos',-.18,'tab:orange'),('neg',.18,'tab:blue')]:
                vals=[s.loc[f'{base}_{sign}_{region}_{policy}','delta_mean'] for region in ('OO','UU','UO')]
                ax.bar(x+dx,vals,.36,label=sign,color=color)
            ax.axhline(0,color='grey',lw=.7);ax.legend();ax.set(xticks=x,xticklabels=['OO','UU','UO'],title=policy+' '+base,
                ylabel='GCG-Shuffled '+('mean squared score' if base=='E' else 'signed score contribution'))
    finish(fig,paths['sign']/'figures/sign_contributions.png','n=18 paired means | delta E_pos+delta E_neg=delta mean(RMS squared); delta mu_pos+delta mu_neg=delta signed mean')
    hs=pd.read_csv(paths['sign']/'head_summary_18.csv');fig,axes=plt.subplots(2,4,figsize=(18,9))
    for c,(region,sign) in enumerate([('UU','pos'),('UU','neg'),('UO','pos'),('UO','neg')]):
        names=[f'E_{sign}_{region}_{p}' for p in ('full','causal')];limit=hs[hs.metric.isin(names)].delta_mean.abs().max() or 1
        for r,name in enumerate(names):
            ax=axes[r,c];grid=hs[hs.metric==name].pivot(index='layer',columns='head',values='delta_mean')
            im=ax.imshow(grid,cmap='RdBu_r',vmin=-limit,vmax=limit);ax.set(title=name,xlabel='Head 0-31',ylabel='Layer 0-31');fig.colorbar(im,ax=ax,label='delta mean squared score')
    finish(fig,paths['sign']/'figures/energy_head_heatmaps.png','n=18 paired sample mean | shared Full/Causal scale per metric; zero-centered')
    group_panels(paths['sign'],m,['E_pos_UU_full','E_neg_UU_full','E_pos_UO_full','E_neg_UO_full'],'energy_fixed_groups.png','mean squared score')
    sample_panels(paths['sign'],['E_pos_UU_full','E_neg_UU_full','E_pos_UO_full','E_neg_UO_full'],'energy_samples.png','mean squared score')
    q=pd.read_csv(paths['query']/'overview.csv');q=q[q.subset==18];suffix=list(m['query_labels'][:20])
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for c,name in enumerate(('M_U','M_O','D','conditional_M_U')):
        part=q[(q.metric==name)&q['query'].isin(suffix)].set_index('query').loc[suffix]
        for cond,color in [('gcg','tab:red'),('shuffled','tab:blue')]:axes[0,c].plot(range(20),part[cond+'_mean'],label=cond,color=color)
        axes[0,c].legend();axes[0,c].set(title=name,ylabel='log mass ratio' if name=='D' else 'fraction')
        axes[1,c].plot(range(20),part.delta_mean,color='purple');axes[1,c].fill_between(range(20),part.delta_mean-part.delta_std,part.delta_mean+part.delta_std,alpha=.15)
        axes[1,c].axhline(0,color='grey',lw=.7);axes[1,c].set(xlabel='Suffix query relative position (content differs)',ylabel='GCG-Shuffled; shade=sample SD')
    finish(fig,paths['query']/'figures/suffix_query_profiles.png','n=18 | per Head causal Softmax over ALL visible Keys, then equal Head/sample mean; no softmax of mean scores')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,name in zip(axes,['M_U','M_O','D']):
        part=q[(q.metric==name)&q['query'].isin(['q1','q2','q3','q4'])].set_index('query').loc[['q1','q2','q3','q4']]
        for cond,color in [('gcg','tab:red'),('shuffled','tab:blue')]:ax.plot(range(1,5),part[cond+'_mean'],marker='o',label=cond,color=color)
        ax.legend();ax.set(title=name,xlabel='Closing-template Query (q4=final)',ylabel='log ratio' if name=='D' else 'fraction')
    finish(fig,paths['query']/'figures/final_and_closing_queries.png','n=18 | causal visible Keys; all 4 closing Queries verified against old cached metrics')
    groups=pd.read_csv(paths['query']/'view_summary_18.csv');ordered=[v['view_id'] for v in m['views'] if v['kind']=='all_heads']+[v['view_id'] for v in m['views'] if v['kind']=='selected_group']
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    for ax,name in zip(axes,['M_U','D']):
        grid=groups[(groups.metric==name)&groups['query'].isin(suffix)].pivot(index='view_id',columns='query',values='delta_mean').reindex(index=ordered,columns=suffix)
        lim=grid.abs().max().max() or 1;im=ax.imshow(grid,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
        ax.set_yticks(range(len(ordered)),ordered,fontsize=8);ax.set(title=name,xlabel='Suffix Query relative position 0-19',ylabel='Fixed old Head groups');fig.colorbar(im,ax=ax,label='GCG-Shuffled '+('fraction' if name=='M_U' else 'log ratio'))
    finish(fig,paths['query']/'figures/fixed_group_query_profiles.png','n=18 | compute per Query/Head, then group mean and sample mean; all visible Keys in denominator')
    distributions(paths,m)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,names,title in [(axes[0,0],['raw_O','raw_U','row_mean'],'Raw score'),(axes[0,1],['centered_O','centered_U'],'Visible-row centered score')]:
        for name in names:
            part=q[(q.metric==name)&q['query'].isin(suffix)].set_index('query').loc[suffix]
            for cond,ls in [('gcg','-'),('shuffled','--')]:ax.plot(range(20),part[cond+'_mean'],ls,label=name+' '+cond)
        ax.legend(fontsize=7);ax.set(title=title,xlabel='Suffix Query position',ylabel='score')
    for ax,name in [(axes[1,0],'gap_U_minus_O'),(axes[1,1],'D')]:
        part=q[(q.metric==name)&q['query'].isin(suffix)].set_index('query').loc[suffix]
        ax.plot(range(20),part.delta_mean);ax.axhline(0,color='grey',lw=.7);ax.set(title=name,xlabel='Suffix Query position',ylabel='GCG-Shuffled '+('score' if name!='D' else 'log ratio'))
    finish(fig,paths['center']/'figures/row_centered_profiles.png','n=18 equal Head/sample means | row mean uses ONLY visible Keys; no new Head selection')
    integrated=pd.read_csv(paths['integrated']/'head_summary_18.csv');fig,axes=plt.subplots(2,2,figsize=(13,10))
    for ax,(x,y) in zip(axes.flat,[('RMS_UU_full','E_neg_UU_full'),('RMS_UU_full','mu_UU_full'),('suffix_mean20_gap_U_minus_O','suffix_mean20_D'),('suffix_mean20_centered_U','suffix_mean20_M_U')]):
        a=integrated[integrated.metric==x];b=integrated[integrated.metric==y];joined=a.merge(b,on=['layer','head'],suffixes=('_x','_y'),validate='one_to_one')
        im=ax.scatter(joined.delta_mean_x,joined.delta_mean_y,c=joined.layer,cmap='viridis',s=8,alpha=.6)
        ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7);ax.set(xlabel='Delta '+x,ylabel='Delta '+y);fig.colorbar(im,ax=ax,label='Layer')
    finish(fig,paths['integrated']/'figures/integrated_relationships.png','n=18 paired sample mean; 1024 dependent Heads | descriptive associations, distinct units in field names')
    io=pd.read_csv(paths['integrated']/'overview.csv');fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,name in zip(axes.flat,['RMS_UU_full','mu_UU_full','E_neg_UU_full','suffix_mean20_M_U','suffix_mean20_D','q4_M_U']):
        for n,dx,col in [(18,-.18,'tab:purple'),(20,.18,'tab:green')]:
            row=io[(io.subset==n)&(io.metric==name)].iloc[0];ax.bar(dx,row.delta_mean,width=.3,color=col,label=f'n={n}');ax.errorbar(dx,row.delta_mean,yerr=row.delta_std,fmt='none',color='black',capsize=2)
        ax.axhline(0,color='grey',lw=.7);ax.set(title=name,xticks=[],ylabel='GCG-Shuffled (metric units)');ax.legend()
    finish(fig,paths['integrated']/'figures/integrated_overview.png','Metrics first per sample/Head, then equal means | bars=sample SD; N18 primary vs N20 auxiliary, not independent validation')


def group_panels(root,m,names,filename,unit):
    groups=pd.read_csv(root/'view_summary_18.csv');members=pd.read_csv(root/'fixed_group_member_effects.csv');members=members[members.subset==18]
    ordered=[v['view_id'] for v in m['views'] if v['kind']=='all_heads']+[v['view_id'] for v in m['views'] if v['kind']=='selected_group']
    fig,axes=plt.subplots(2,2,figsize=(17,13))
    for ax,name in zip(axes.flat,names):
        for i,view in enumerate(ordered):
            row=groups[(groups.view_id==view)&(groups.metric==name)].iloc[0]
            ax.errorbar(row.delta_mean,i,xerr=row.delta_std,fmt='D',color='black',capsize=2,markersize=4)
            part=members[(members.selected_group==view)&(members.metric==name)]
            ax.scatter(part.delta_mean,i+np.linspace(-.15,.15,len(part)),s=15,color='tab:blue')
        ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(len(ordered)),ordered,fontsize=7);ax.invert_yaxis();ax.set(title=name,xlabel='GCG-Shuffled '+unit)
    finish(fig,root/'figures'/filename,'n=18 | fixed old groups; mean diamond, sample SD bar, individual member Head mean dots; masks in field names')


def sample_panels(root,names,filename,unit):
    sample=pd.read_csv(root/'sample_summary.csv');fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,name in zip(axes.flat,names):
        part=sample[(sample.primary18)&(sample.view_id=='all_heads')&(sample.metric==name)]
        ax.bar(range(18),part.delta,color=np.where(part.delta>=0,'tab:red','tab:blue'));ax.axhline(part.delta.mean(),color='orange',ls='--');ax.axhline(0,color='grey',lw=.7)
        ax.set_xticks(range(18),part.dataset_index,rotation=60,fontsize=8);ax.set(title=name,xlabel='Dataset Index',ylabel='GCG-Shuffled '+unit)
    finish(fig,root/'figures'/filename,'n=18 | per-Head features then equal 1024 Head mean; orange=sample mean; masks in field names')


def distributions(paths,m):
    # Deterministic existing group-first representatives and first primary sample;
    # never select using the new outcomes. All selected Heads remain in NPZ/CSVs.
    chosen=[]
    for v in m['views']:
        if v['kind']=='selected_group' and v['members'][0] not in chosen:chosen.append(v['members'][0])
        if len(chosen)==4:break
    index=m['primary_indices'][0];ref=next(r for r in m['score_references'] if r['dataset_index']==index and r['condition']=='gcg');ranges=ref['input']['token_ranges']
    with np.load(paths['query']/'row_profiles'/f'index_{index:03d}.npz') as z:
        fig,axes=plt.subplots(2,4,figsize=(19,8))
        for col,h in enumerate(chosen):
            vi=next(i for i,v in enumerate(m['views']) if v['members']==[h]);qslot=19
            keys=visible_query_keys(z,qslot,ranges['suffix'][0]+qslot)
            for row,name in enumerate(('raw','weight')):
                ax=axes[row,col]
                for cond,color in [('gcg','tab:red'),('shuffled','tab:blue')]:ax.plot(keys,z[cond+'_'+name][vi,qslot,keys],color=color,label=cond)
                for region in ('original','suffix'):ax.axvline(ranges[region][0]-.5,color='grey',lw=.5)
                ax.legend(fontsize=7);ax.set(title=f'L{h//32} H{h%32} suffix Query19',xlabel='Visible Key position (Key <= Query); O/U boundaries',ylabel='Raw score (pre-softmax)' if name=='raw' else 'Softmax Attention fraction (causal)')
        finish(fig,paths['query']/'figures/fixed_head_distributions.png',f'Index {index}, n=1 diagnostic | first members of old groups, no new selection; VISIBLE Keys only; raw pre-softmax vs causal Attention')


def visible_query_keys(profile,slot,expected_position):
    """Plot raw/weight data at the same verified causal Key positions."""
    q=int(profile['query_positions'][slot])
    keys=np.flatnonzero(profile['visible'][slot])
    if q!=expected_position or not np.array_equal(keys,np.arange(q+1)):
        raise ValueError('saved query position/visibility mismatch')
    return keys
