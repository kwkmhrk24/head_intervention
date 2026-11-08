#!/usr/bin/env python3
"""Build Japanese interpretation/MTG notes from saved tables, without inference."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from utils.artifact_io import read_json,sha256_file
from utils.gcg_shuffled_common import write_json,write_csv
from utils.gcg_shuffled_plot_common import plt,finish
from utils.gcg_shuffled_reporting import ReportData,render_reports,require_columns


def build(root,run_id,output_dir=None):
    p={k:root/v/run_id for k,v in [('sign','30_sign_decomposition'),('query','40_query_softmax'),('center','50_row_centered'),('integrated','60_integrated_analysis')]}
    source_integrated=p['integrated']
    m=read_json(source_integrated/'manifest.json');o=pd.read_csv(source_integrated/'overview.csv')
    q=pd.read_csv(p['query']/'overview.csv');q=q[q.subset==len(m['primary_indices'])]
    h=pd.read_csv(source_integrated/'head_summary_18.csv')
    groups={n:pd.read_csv(p['query']/f'view_summary_{n}.csv') for n in (18,20)}
    corr=pd.read_csv(source_integrated/'correlations.csv')
    data=ReportData(m,o,groups)
    interpretation,mtg=render_reports(data,run_id,corr)
    require_columns(h,('layer','head','metric','delta_mean'),'head summary')
    if h.duplicated(['layer','head','metric']).any():raise ValueError('duplicate Head metric')
    input_files=[source_integrated/name for name in ('manifest.json','overview.csv','head_summary_18.csv','correlations.csv')]
    input_files += [p['query']/name for name in ('overview.csv','view_summary_18.csv','view_summary_20.csv')]
    out=Path(output_dir) if output_dir is not None else source_integrated
    created=['interpretation.md','MTG_summary.md','energy_head_correspondence.csv','suffix_position_effects.csv','centered_key_diagnostics.csv','figures/centered_key_diagnostics.png']
    if any((out/name).exists() for name in created+['report_manifest.json','supplemental_hashes.json']):
        raise FileExistsError('report output files already exist; use --output-dir NEW_DIRECTORY')
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    # Keep exact numeric citations and supplementary direction quadrants.
    relations=[]
    for region in ('UU','UO'):
        for policy in ('full','causal'):
            x=h[h.metric==f'RMS_{region}_{policy}'][['layer','head','delta_mean']]
            y=h[h.metric==f'E_neg_{region}_{policy}'][['layer','head','delta_mean']]
            if len(x)!=1024 or len(y)!=1024:raise ValueError('missing Head rows for '+region+' '+policy)
            z=x.merge(y,on=['layer','head'],suffixes=('_rms','_neg'),validate='one_to_one')
            relations.append(dict(region=region,policy=policy,rms_up_negative_energy_up=int(((z.delta_mean_rms>0)&(z.delta_mean_neg>0)).sum()),head_count=len(z)))
    write_csv(out/'energy_head_correspondence.csv',pd.DataFrame(relations))
    # Export suffix-position deltas in a compact, easy-to-cite table.
    positions=q[q['query'].str.startswith('suffix_') & (q['query']!='suffix_mean20')]
    write_csv(out/'suffix_position_effects.csv',positions)
    # Additional centered-Key and paired Attention-delta diagnostic for the
    # same predetermined existing representative Heads as the original sheet.
    chosen=[]
    for v in m['views']:
        if v['kind']=='selected_group' and v['members'][0] not in chosen:chosen.append(v['members'][0])
        if len(chosen)==4:break
    index=m['primary_indices'][0];ref=next(r for r in m['score_references'] if r['dataset_index']==index and r['condition']=='gcg')
    ranges=ref['input']['token_ranges'];visible_end=ranges['suffix'][1];data=[]
    with np.load(p['query']/'row_profiles'/f'index_{index:03d}.npz') as z:
        fig,axes=plt.subplots(2,4,figsize=(18,8))
        for c,head in enumerate(chosen):
            vi=next(i for i,v in enumerate(m['views']) if v['members']==[head])
            for r,field in enumerate(('centered','weight')):
                g=z['gcg_'+field][vi,19,:visible_end];s=z['shuffled_'+field][vi,19,:visible_end]
                for k,(gg,ss) in enumerate(zip(g,s)):data.append(dict(dataset_index=index,layer=head//32,head=head%32,query='suffix_19',key=k,field=field,gcg=gg,shuffled=ss,delta=gg-ss))
                ax=axes[r,c]
                if field=='centered':
                    ax.plot(g,label='GCG',color='tab:red');ax.plot(s,label='Shuffled',color='tab:blue');ax.legend(fontsize=8)
                else:ax.plot(g-s,label='GCG-Shuffled',color='purple')
                for region in ('original','suffix'):ax.axvline(ranges[region][0]-.5,color='grey',lw=.6)
                ax.axhline(0,color='grey',lw=.6);ax.set(title=f'L{head//32} H{head%32}',xlabel='Visible Key position; prefix/O/U boundaries',ylabel='Centered score' if field=='centered' else 'Delta Attention fraction')
        finish(fig,out/'figures/centered_key_diagnostics.png',f'Index {index}, n=1; suffix Query19 | predetermined old group-first Heads; illustration, no selection from new metrics')
    write_csv(out/'centered_key_diagnostics.csv',pd.DataFrame(data))
    with (out/'interpretation.md').open('x') as handle:handle.write(interpretation)
    with (out/'MTG_summary.md').open('x') as handle:handle.write(mtg)
    input_files.append(p['query']/'row_profiles'/f'index_{index:03d}.npz')
    source_files=[Path(__file__),ROOT/'utils/gcg_shuffled_reporting.py',ROOT/'utils/gcg_shuffled_plot_common.py',
                  ROOT/'utils/gcg_shuffled_common.py',ROOT/'utils/gcg_shuffled_metrics.py',ROOT/'utils/artifact_io.py']
    write_json(out/'report_manifest.json',dict(input_root=str(root),run_id=run_id,output_dir=str(out),
        primary_indices=m['primary_indices'],all_indices=m['all_indices'],views=m['views'],
        delta='GCG - Shuffled',values='saved CSV only; fraction*100 => percent/percentage points; energy => Score squared',
        input_hashes={str(path.resolve()):sha256_file(path) for path in input_files},
        source_hashes={str(path):sha256_file(path) for path in source_files},
        correlation_provenance='saved correlations.csv (hashed input), not recomputed by report',
        input_analysis_source_hashes=m.get('source_hashes',{}),numpy=np.__version__,pandas=pd.__version__))
    write_json(out/'supplemental_hashes.json',{name:sha256_file(out/name) for name in created+['report_manifest.json']})
    print(out/'interpretation.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/outputs/gcg_vs_shuffled/legacy_selection20');p.add_argument('--run-id',required=True)
    p.add_argument('--output-dir',type=Path,help='New report directory; source analysis files remain unchanged')
    a=p.parse_args();build(a.root,a.run_id,a.output_dir)
