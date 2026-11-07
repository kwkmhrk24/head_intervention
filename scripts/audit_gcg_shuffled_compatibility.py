#!/usr/bin/env python3
"""Strict scientific table/array compatibility, including order, dtype and NaNs."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from utils.artifact_io import sha256_file
from scripts.analyze_gcg_shuffled_four_metrics import write_json

RTOL=1e-12
ATOL=1e-12


def compare_array(before,after,label):
    a=np.asarray(before);b=np.asarray(after)
    if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError(f'{label}: shape/dtype differs: {a.shape}/{a.dtype}, {b.shape}/{b.dtype}')
    if np.issubdtype(a.dtype,np.inexact):
        np.testing.assert_array_equal(np.isnan(a),np.isnan(b),err_msg=label)
        np.testing.assert_array_equal(np.isposinf(a),np.isposinf(b),err_msg=label)
        np.testing.assert_array_equal(np.isneginf(a),np.isneginf(b),err_msg=label)
        np.testing.assert_allclose(a,b,rtol=RTOL,atol=ATOL,equal_nan=True,err_msg=label)
        finite=np.isfinite(a)&np.isfinite(b);error=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
    else:
        np.testing.assert_array_equal(a,b,err_msg=label);error=0.
    return dict(field=label,shape=list(a.shape),dtype=str(a.dtype),max_absolute_error=error)


def compare_trees(before,after):
    """CSV schema/row/column order and all NPZ keys/dtypes/values are mandatory."""
    names=lambda root:sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.suffix in ('.csv','.npz') and p.is_file())
    a,b=names(before),names(after)
    if a!=b:raise AssertionError('scientific file inventory differs')
    rows=[]
    for name in a:
        if name.endswith('.csv'):
            x=pd.read_csv(before/name);y=pd.read_csv(after/name)
            if list(x)!=list(y):raise AssertionError(name+': CSV column order differs')
            for column in x:
                if x[column].dtype==object:
                    # Text/IDs and missing text must match exactly, in row order.
                    pd.testing.assert_series_equal(x[column],y[column],check_names=True,check_dtype=True)
                    rows.append(dict(field=name+'::'+column,shape=[len(x)],dtype='object',max_absolute_error=0.))
                else:rows.append(compare_array(x[column].to_numpy(),y[column].to_numpy(),name+'::'+column))
        else:
            with np.load(before/name,allow_pickle=False) as x,np.load(after/name,allow_pickle=False) as y:
                if x.files!=y.files:raise AssertionError(name+': NPZ key order differs')
                for key in x.files:rows.append(compare_array(x[key],y[key],name+'::'+key))
    return dict(status='passed',before=str(before),after=str(after),rtol=RTOL,atol=ATOL,scientific_files=len(a),fields=rows,
                max_absolute_error=max((r['max_absolute_error'] for r in rows),default=0.),
                policy='IDs, order, dtype, Mask/counts/NaN exact; finite floats rtol=atol=1e-12; volatile provenance/PNG pixels not numeric comparison')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=compare_trees(a.before,a.after);write_json(a.output,r);print(f"Passed {r['scientific_files']} files, maximum error {r['max_absolute_error']}")


if __name__=='__main__':main()
