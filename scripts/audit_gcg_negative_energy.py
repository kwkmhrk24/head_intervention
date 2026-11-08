#!/usr/bin/env python3
"""Audit saved negative-energy attribution without modifying original artifacts."""
from pathlib import Path
import argparse
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from utils.artifact_io import read_json,sha256_file
from utils.gcg_shuffled_common import write_json
from utils.gcg_negative_energy import CONDITION_FIELDS,PAIR_FIELDS,REGIONS
from scripts.analyze_gcg_negative_energy import blocks_for_pair,selected_elements


def verify_hashed_files(root, expected, names=None):
    """Verify an explicit, nonempty set before consuming audited artifacts."""
    root=Path(root)
    names=list(expected if names is None else names)
    if not names:raise ValueError('empty hash inventory')
    result={}
    for name in names:
        if name not in expected:raise ValueError('missing audited hash: '+name)
        path=root/name
        if not path.is_file():raise ValueError('missing hashed file: '+str(path))
        actual=sha256_file(path)
        if actual!=expected[name]:raise ValueError('hash mismatch: '+str(path))
        result[name]=actual
    return result


def verify_report_bundle(directory):
    """Detached checksums cover reports AND manifests; the ledger excludes itself."""
    directory=Path(directory)
    hashes=read_json(directory/'report_hashes.json')
    required={'findings.md','MTG_findings.md','README.md','report_manifest.json','manifest.json'}
    if set(hashes)!=required:raise ValueError('report hash inventory mismatch')
    verify_hashed_files(directory,hashes)
    manifest=read_json(directory/'report_manifest.json')
    verify_hashed_files(Path('/'),manifest['input_hashes'])
    ledger=Path(manifest['audited_ledger'])
    verify_hashed_files(Path('/'),{str(ledger):manifest['audited_ledger_sha256']})
    verify_hashed_files(Path('/'),manifest['source_hashes'])
    return {'status':'passed','report_files_verified':len(hashes),
            'input_files_verified':len(manifest['input_hashes'])}


def snapshot(out):
    """Read-only baseline capture before a fresh run; no original file writes."""
    if (out/'baseline_hashes.json').exists():raise FileExistsError('baseline already exists')
    out.mkdir(parents=True,exist_ok=True)
    roots=('shuffled_gcg_pilot','shuffled_gcg_comparison','full_attention_scores','gcg_shuffled_four_metrics',
           'gcg_shuffled_token_analysis','gcg_shuffled_region_analysis','gcg_shuffled_signed_means','gcg_vs_shuffled')
    files=[p for name in roots for p in (ROOT/'data/outputs'/name).rglob('*') if p.is_file() and not p.resolve().is_relative_to(out.resolve())]
    write_json(out/'baseline_hashes.json',{str(p.relative_to(ROOT)):sha256_file(p) for p in files})
    (out/'README_before.md').write_bytes((ROOT/'data/outputs/gcg_vs_shuffled/legacy_selection20/README.md').read_bytes())
    write_json(out/'preflight.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),old_files=len(files),mode='read-only CPU snapshot'))
    print('Baseline captured:',len(files))


def audit(out,name='validation.json'):
    m=read_json(out/'core_manifest.json');f=read_json(out/'finish_manifest.json')
    for path,h in {**m['source_hashes'],**f['source_hashes']}.items():
        if sha256_file(path)!=h:raise ValueError('execution source changed: '+path)
    for path,h in m['input_hashes'].items():
        if sha256_file(path)!=h:raise ValueError('input changed: '+path)
    for path,h in read_json(out/'output_hashes.json').items():
        if sha256_file(out/path)!=h:raise ValueError('output hash mismatch: '+path)
    original_hashes=read_json(out/'baseline_hashes.json');navigation=[]
    for path,h in original_hashes.items():
        if sha256_file(path)==h:continue
        old=(out/'README_before.md').read_bytes();current=Path(path).read_bytes()
        if path=='data/outputs/gcg_vs_shuffled/legacy_selection20/README.md' and current.startswith(old) and out.name.encode() in current[len(old):]:
            navigation.append(path)
        else:raise ValueError('old artifact changed: '+path)
    with np.load(out/'core_values.npz') as z:c=z['conditions'];p=z['pairs'];m2=z['m2_neg'];counts=z['element_counts']
    with np.load(out/'tail_values.npz') as z:a=z['shares'];k=z['k'];total=z['total_mean_energy'];top=z['top_mean_energy'];basis=z['basis_counts']
    with np.load(out/'query_values.npz') as z:qc=z['conditions'];qp=z['pairs'];qn=z['valid_key_counts']
    assert c.shape==(20,2,5,1024,6) and p.shape==(20,5,1024,11)
    assert qc.shape==(20,2,2,1024,20,6) and qp.shape==(20,2,1024,20,11)
    assert a.shape==(20,5,1024,12) and k.dtype.kind in 'iu' and basis.dtype.kind in 'iu'
    for x in (c,p,qc,qp,top,total):assert np.isfinite(x).all()
    np.testing.assert_array_equal(np.isnan(m2),c[...,CONDITION_FIELDS.index('zero_negative')].astype(bool))
    np.testing.assert_array_equal(np.isnan(a),total==0)
    assert not np.isinf(m2).any() and not np.isinf(a).any()
    assert (a[np.isfinite(a)]>=0).all() and (a[np.isfinite(a)]<=1+1e-12).all()
    np.testing.assert_allclose(p[...,3]+p[...,4],p[...,0],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(p[...,8]+p[...,9],p[...,0],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(p[...,5:8].sum(axis=-1),1,rtol=1e-12,atol=1e-12)
    for region in ('OO_full','OO_causal'):
        ri=REGIONS.index(region);np.testing.assert_array_equal(p[:,ri,:,0],0)
    condition=pd.read_csv(out/'tables/condition_metrics.csv');pair=pd.read_csv(out/'tables/paired_metrics.csv')
    assert len(condition)==20*2*5*1024 and not condition.duplicated(['dataset_index','condition','region','layer','head']).any()
    assert len(pair)==20*5*1024 and not pair.duplicated(['dataset_index','region','layer','head']).any()
    np.testing.assert_array_equal(pair.positive_element_count+pair.negative_element_count+pair.equal_element_count,pair.valid_elements)
    np.testing.assert_allclose((pair.positive_contribution_total+pair.negative_contribution_total)/pair.valid_elements,pair.delta_E_neg,rtol=1e-12,atol=1e-12)
    head_reference_count=0
    lookup={(r['dataset_index'],r['condition']):r for r in m['score_references']}
    for i,index in enumerate(m['all_indices']):
        blocks,masks,_=blocks_for_pair(lookup,index)
        for ri,region in enumerate(REGIONS):
            (g,s),n=selected_elements(blocks,masks,region);assert n==counts[i,ri]
            cf=pair[(pair.dataset_index==index)&(pair.region==region)].sort_values(['layer','head'])
            np.testing.assert_allclose(cf[list(PAIR_FIELDS)].to_numpy(float),p[i,ri],rtol=1e-12,atol=1e-12)
            for ti,spec in enumerate(f['tail_specs']):
                np.testing.assert_array_equal(k[i,ri,:,ti],np.ceil(spec['fraction']*basis[i,ri,:,ti]).astype(int))
                if spec['kind']=='positive_delta':
                    np.testing.assert_allclose(total[i,ri,:,ti],p[i,ri,:,8],rtol=1e-12,atol=1e-12)
                    if spec['basis']=='active':np.testing.assert_array_equal(basis[i,ri,:,ti],cf.positive_element_count)
                else:
                    ci=0 if spec['kind']=='energy_gcg' else 1
                    np.testing.assert_array_equal(basis[i,ri,:,ti],n)
                    np.testing.assert_allclose(total[i,ri,:,ti],c[i,ci,ri,:,0],rtol=1e-12,atol=1e-12)
                # Independent Python-list reference for predetermined validation
                # Heads (not selected from results), all samples/regions/tails.
                for h in (0,31,511,1023):
                    eg=[min(float(x),0.)**2 for x in g[h]];es=[min(float(x),0.)**2 for x in s[h]]
                    values=eg if spec['kind']=='energy_gcg' else es if spec['kind']=='energy_shuffled' else [max(x-y,0.) for x,y in zip(eg,es)]
                    amount=int(k[i,ri,h,ti]);truth=sum(sorted(values,reverse=True)[:amount])/n
                    np.testing.assert_allclose(top[i,ri,h,ti],truth,rtol=1e-12,atol=1e-12);head_reference_count+=1
        for qri,region in enumerate(('UU_causal','UO')):
            ri=REGIONS.index(region)
            if qri==0:np.testing.assert_array_equal(qn[i,qri],np.arange(1,21))
            else:np.testing.assert_array_equal(qn[i,qri],counts[i,ri]//20)
            np.testing.assert_allclose(np.average(qp[i,qri,:,:,0],axis=1,weights=qn[i,qri]),p[i,ri,:,0],rtol=1e-12,atol=1e-12)
    # Full source-file hashes establish immutable Score identity as well as IDs.
    for ref in m['score_references']:
        if sha256_file(ref['path'])!=ref['sha256']:raise ValueError('Score hash changed')
    core_checks=read_json(out/'core_validation.json')['checks'];qchecks=read_json(out/'finish_validation.json')['checks']
    result=dict(status='passed',base_commit=m['base_commit'],old_artifact_hashes_verified=len(original_hashes),
        intentional_navigation_appends=navigation,old_scientific_artifacts_unchanged=True,
        condition_rows=len(condition),paired_rows=len(pair),undefined_conditional_m2=int(np.isnan(m2).sum()),
        undefined_tail_shares=int(np.isnan(a).sum()),unexpected_nonfinite=0,independent_tail_reference_cases=head_reference_count,
        old_E_neg_max_error=max(x['old_E_neg_max_error'] for x in core_checks),
        midpoint_decomposition_max_error=max(x['factorization'] for x in core_checks),
        element_mean_identity_max_error=max(x['element_identity'] for x in core_checks),
        query_weighted_identity_max_error=max(x['weighted_region_max_error'] for x in qchecks),
        oo_exactly_unchanged=True,uo_full_causal_identical=True,samples=20,primary_samples=18,
        sources={str(Path(__file__)):sha256_file(Path(__file__))},no_model_or_GPU=True)
    write_json(out/name,result)
    print(result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--audit-name',default='validation.json');p.add_argument('--verify-report',type=Path,help='Verify a detached report bundle without recomputation');p.add_argument('--snapshot',action='store_true',help='Capture immutable inputs before starting a fresh run')
    a=p.parse_args()
    if a.verify_report:print(verify_report_bundle(a.verify_report))
    elif a.snapshot:snapshot(a.output_dir)
    else:audit(a.output_dir,a.audit_name)
