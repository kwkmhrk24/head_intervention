"""CPU-only review regressions: report mutation, dependency isolation, visibility."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from utils.gcg_shuffled_reporting import ReportData,render_reports,TABLE_METRICS,validate_summary
from utils.gcg_shuffled_additional_plots import visible_query_keys,distributions
from utils.gcg_shuffled_plot_common import plt
from utils.gcg_shuffled_statistics import pearson_correlation,spearman_correlation
ROOT=Path(__file__).resolve().parents[1]


def fixture():
    views=[dict(view_id=name,kind='selected_group',members=list(range(i*5,(i+1)*5)))
           for i,name in enumerate(('M_U_increase','D_increase','F_increase'))]
    m=dict(condition_order=['gcg','shuffled'],primary_indices=list(range(18)),all_indices=list(range(20)),views=views)
    o=[];groups={}
    for n in (18,20):
        base=dict(gcg_mean=2.,shuffled_mean=1.,delta_mean=1.,delta_std=.2,sample_count=n,
                  gcg_larger_count=1,shuffled_larger_count=n-1,equal_count=0)
        for metric in TABLE_METRICS:o.append(dict(subset=n,metric=metric,view_id='all_heads',**base))
        groups[n]=pd.DataFrame([dict(view_id=v['view_id'],view_kind=v['kind'],head_count=5,query=q,metric=f,**base)
            for v in views for q in ('suffix_mean20','q4') for f in ('M_U','M_O','D','centered_O','centered_U')])
    c=pd.DataFrame([dict(subset=n,unit='head_mean',x='x',y='y',pearson=.1,spearman=.2,same_direction_fraction=.5) for n in (18,20)])
    return m,pd.DataFrame(o),groups,c


def set_delta(frame,selected,value):
    frame.loc[selected,'delta_mean']=value
    frame.loc[selected,'gcg_mean']=frame.loc[selected,'shuffled_mean']+value


def render(m,o,g,c):return render_reports(ReportData(m,o,g),'test_run',c)


class ReportRegressionTests(unittest.TestCase):
    def test_every_result_family_updates_from_csv_values(self):
        for metric in ('RMS_UU_full','mu_UO_full','E_pos_UU_full','E_neg_UO_full','mu_neg_UU_full',
                       'suffix_mean20_M_U','q4_M_O','q4_D','q4_centered_U'):
            with self.subTest(metric=metric):
                m,o,g,c=fixture();set_delta(o,(o.subset==18)&(o.metric==metric),.123456)
                detail,mtg=render(m,o,g,c)
                self.assertIn('+0.1234560',detail)
                expected='+12.3456' if metric.endswith(('M_U','M_O')) else '+0.123456'
                self.assertIn(expected,mtg)

    def test_counts_and_cohorts_are_not_hardcoded(self):
        m,o,g,c=fixture();sel=(o.subset==18)&(o.metric=='suffix_mean20_M_U')
        o.loc[sel,['gcg_larger_count','shuffled_larger_count']]=[3,15]
        set_delta(o,(o.subset==20)&(o.metric=='suffix_mean20_M_U'),.7654321)
        detail,mtg=render(m,o,g,c)
        self.assertIn('3増/15減/0同等（18件）',mtg)
        self.assertIn('+0.7654321',detail)

    def test_fixed_groups_update_and_validate_manifest_identity(self):
        m,o,g,c=fixture();f=g[18];set_delta(f,(f.view_id=='D_increase')&(f['query']=='q4')&(f.metric=='M_U'),.123456)
        detail,mtg=render(m,o,g,c)
        self.assertIn('+12.3456',detail);self.assertIn('+12.35',mtg)
        m['views'][1]['view_id']='renamed'
        with self.assertRaises(ValueError):render(m,o,g,c)

    def test_missing_duplicate_nonfinite_and_counts_fail(self):
        m,o,g,c=fixture()
        for bad in [o.drop(columns='delta_mean'),pd.concat([o,o.iloc[:1]]),o[o.metric!='q4_D']]:
            with self.assertRaises(ValueError):render(m,bad,g,c)
        o.loc[0,'gcg_mean']=np.nan
        with self.assertRaises(ValueError):render(m,o,g,c)
        m,o,g,c=fixture();o.loc[0,'shuffled_larger_count']=0
        with self.assertRaises(ValueError):render(m,o,g,c)

    def test_condition_order_and_invalid_group_members_fail(self):
        m,o,g,c=fixture();m['condition_order'].reverse()
        with self.assertRaises(ValueError):render(m,o,g,c)
        m,o,g,c=fixture();m['views'][0]['members']=[0]*5
        with self.assertRaises(ValueError):render(m,o,g,c)

    def test_conclusions_follow_changed_data_not_old_story(self):
        m,o,g,c=fixture()
        set_delta(o,(o.subset==18)&o.metric.str.startswith('E_neg_'),0.)
        detail,mtg=render(m,o,g,c)
        self.assertIn('一括して要約できない',mtg)
        self.assertNotIn('変化は逆方向',mtg)
        set_delta(o,(o.subset==18)&(o.metric=='q4_M_U'),-.5)
        _,mtg=render(m,o,g,c);self.assertIn('変化は逆方向',mtg)

    def test_units_and_correlation_data_are_explicit(self):
        m,o,g,c=fixture();detail,mtg=render(m,o,g,c)
        self.assertIn('+100.0000パーセントポイント',detail);self.assertIn('単位Score²',detail)
        self.assertIn('平均RMSの二乗差',detail);self.assertIn('自然対数',detail)
        c.loc[0,'pearson']=.654321
        detail,_=render(m,o,g,c);self.assertIn('0.6543',detail)

    def test_build_rereads_changed_csv_and_preserves_existing_outputs(self):
        import json
        from scripts.report_gcg_shuffled_phase import build
        from utils.artifact_io import sha256_file
        m,o,g,c=fixture()
        first=[v['members'][0] for v in m['views']]
        m['views']=[dict(view_id=f'h{h}',kind='individual',members=[h]) for h in first]+m['views']
        m['score_references']=[dict(dataset_index=0,condition='gcg',input=dict(token_ranges=dict(original=[1,3],suffix=[3,23])))]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run='fixture';i=root/'60_integrated_analysis'/run;q=root/'40_query_softmax'/run
            i.mkdir(parents=True);q.mkdir(parents=True);(q/'row_profiles').mkdir()
            (i/'manifest.json').write_text(json.dumps(m));o.to_csv(i/'overview.csv',index=False);c.to_csv(i/'correlations.csv',index=False)
            rows=[dict(layer=h//32,head=h%32,metric=f'{f}_{region}_{policy}',delta_mean=.1)
                  for f in ('RMS','E_neg') for region in ('UU','UO') for policy in ('full','causal') for h in range(1024)]
            pd.DataFrame(rows).to_csv(i/'head_summary_18.csv',index=False)
            pd.DataFrame([dict(subset=18,query='suffix_00',metric='M_U',delta_mean=.1)]).to_csv(q/'overview.csv',index=False)
            for n in (18,20):g[n].to_csv(q/f'view_summary_{n}.csv',index=False)
            x=np.zeros((len(m['views']),24,27))
            np.savez(q/'row_profiles/index_000.npz',gcg_centered=x,shuffled_centered=x,gcg_weight=x,shuffled_weight=x)
            build(root,run,root/'first')
            old_hash=sha256_file(root/'first/MTG_summary.md')
            set_delta(o,(o.subset==18)&(o.metric=='q4_M_U'),.234567)
            o.to_csv(i/'overview.csv',index=False)
            build(root,run,root/'second')
            self.assertIn('+23.4567パーセントポイント',(root/'second/MTG_summary.md').read_text())
            self.assertEqual(old_hash,sha256_file(root/'first/MTG_summary.md'))
            meta=json.loads((root/'second/report_manifest.json').read_text())
            self.assertEqual(meta['input_hashes'][str((i/'overview.csv').resolve())],sha256_file(i/'overview.csv'))
            self.assertIn(str(ROOT/'utils/gcg_shuffled_reporting.py'),meta['source_hashes'])
            with self.assertRaises(FileExistsError):build(root,run,root/'second')


class DependencyTests(unittest.TestCase):
    def test_correlation_functions_match_legacy_ast_without_importing_torch(self):
        old=ast.parse((ROOT/'scripts/analyze_attention_head_metric_relationships.py').read_text())
        new=ast.parse((ROOT/'utils/gcg_shuffled_statistics.py').read_text())
        for name in ('pearson_correlation','spearman_correlation'):
            find=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(find(old)),ast.dump(find(new)))
        self.assertAlmostEqual(pearson_correlation([1,2,3],[3,2,1]),-1.)
        self.assertAlmostEqual(spearman_correlation([1,1,3],[2,2,1]),-1.)
        with self.assertRaises(ValueError):pearson_correlation([1,1],[2,3])

    def test_cpu_help_with_tensor_imports_blocked(self):
        blocker='''import sys,importlib.abc,runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','transformers'):raise ModuleNotFoundError('blocked: '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
'''
        for name in ('analyze_gcg_shuffled_additional','audit_gcg_shuffled_compatibility','report_gcg_shuffled_phase'):
            with self.subTest(script=name):
                p=subprocess.run([sys.executable,'-B','-c',blocker,str(ROOT/'scripts'/f'{name}.py')],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr);self.assertIn('usage:',p.stdout)


class VisibilityTests(unittest.TestCase):
    def test_visibility_uses_saved_query_and_rejects_disagreement(self):
        p=dict(query_positions=np.array([3]),visible=np.array([[True]*4+[False]*2]))
        np.testing.assert_array_equal(visible_query_keys(p,0,3),[0,1,2,3])
        with self.assertRaises(ValueError):visible_query_keys(p,0,4)
        p['visible'][0,-1]=True
        with self.assertRaises(ValueError):visible_query_keys(p,0,3)

    def test_actual_raw_and_weight_lines_exclude_future_keys(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'row_profiles').mkdir();views=[]
            for i in range(4):views.append(dict(view_id=f'h{i}',kind='individual',members=[i]))
            for i in range(4):views.append(dict(view_id=f'g{i}',kind='selected_group',members=[i]))
            raw=np.ones((8,24,27));weight=raw/27
            visible=np.arange(27)[None,:]<=np.r_[np.arange(3,23),np.arange(23,27)][:,None]
            weight=np.where(visible,weight,0)
            np.savez(root/'row_profiles/index_068.npz',gcg_raw=raw,shuffled_raw=raw,gcg_weight=weight,shuffled_weight=weight,
                     query_positions=np.r_[np.arange(3,23),np.arange(23,27)],visible=visible)
            m=dict(views=views,primary_indices=[68],score_references=[dict(dataset_index=68,condition='gcg',input=dict(token_ranges=dict(original=[1,3],suffix=[3,23],template_closing=[23,27])))])
            def inspect(fig,*args):
                for ax in fig.axes:
                    for line in ax.lines:
                        if len(line.get_xdata())>2:self.assertLessEqual(max(line.get_xdata()),22)
                    self.assertIn('Visible Key',ax.get_xlabel())
                self.assertIn('Raw score',fig.axes[0].get_ylabel())
                self.assertIn('Softmax Attention',fig.axes[4].get_ylabel());plt.close(fig)
            with patch('utils.gcg_shuffled_additional_plots.finish',side_effect=inspect):distributions({'query':root},m)


class SummaryConsistencyTests(unittest.TestCase):
    def test_rounding_error_and_zero_std_are_accepted(self):
        _,o,_,_=fixture();f=o.iloc[:1].copy();f['delta_mean']+=5e-13;f['delta_std']=0.
        validate_summary(f,'roundtrip')
        f['gcg_mean']=.1+.2;f['shuffled_mean']=.3;f['delta_mean']=0.
        validate_summary(f,'cancellation')

    def test_inconsistent_delta_names_metric_and_values(self):
        _,o,_,_=fixture();f=o.iloc[:1].copy();f['delta_mean']+=1e-5
        with self.assertRaisesRegex(ValueError,r'RMS_OO_full.*delta_mean|delta_mean.*RMS_OO_full'):
            validate_summary(f,'overview')

    def test_negative_std_names_metric_and_value(self):
        _,o,_,_=fixture();f=o.iloc[:1].copy();f['delta_std']=-1e-16
        with self.assertRaisesRegex(ValueError,r'RMS_OO_full.*delta_std|delta_std.*RMS_OO_full'):
            validate_summary(f,'overview')


class HeadCorrespondenceTests(unittest.TestCase):
    def build_case(self,change,expect=None):
        import json
        from scripts.report_gcg_shuffled_phase import build
        m,o,g,c=fixture();first=[v['members'][0] for v in m['views']]
        m['views']=[dict(view_id=f'h{h}',kind='individual',members=[h]) for h in first]+m['views']
        m['score_references']=[dict(dataset_index=0,condition='gcg',input=dict(token_ranges=dict(original=[1,3],suffix=[3,23])))]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);i=root/'60_integrated_analysis/fixture';q=root/'40_query_softmax/fixture'
            i.mkdir(parents=True);q.mkdir(parents=True);(q/'row_profiles').mkdir()
            (i/'manifest.json').write_text(json.dumps(m));o.to_csv(i/'overview.csv',index=False);c.to_csv(i/'correlations.csv',index=False)
            h=pd.DataFrame([dict(layer=h//32,head=h%32,metric=f'{f}_{r}_{p}',delta_mean=.1)
                for f in ('RMS','E_neg') for r in ('UU','UO') for p in ('full','causal') for h in range(1024)])
            change(h).to_csv(i/'head_summary_18.csv',index=False)
            pd.DataFrame([dict(subset=18,query='suffix_00',metric='M_U',delta_mean=.1)]).to_csv(q/'overview.csv',index=False)
            for n in (18,20):g[n].to_csv(q/f'view_summary_{n}.csv',index=False)
            x=np.zeros((len(m['views']),24,27))
            np.savez(q/'row_profiles/index_000.npz',gcg_centered=x,shuffled_centered=x,gcg_weight=x,shuffled_weight=x)
            if expect:
                with self.assertRaisesRegex(ValueError,expect):build(root,'fixture',root/'out')
                self.assertFalse((root/'out/interpretation.md').exists())
            else:
                build(root,'fixture',root/'out')
                result=pd.read_csv(root/'out/energy_head_correspondence.csv')
                np.testing.assert_array_equal(result.head_count,1024)
                np.testing.assert_array_equal(result.rms_up_negative_energy_up,1024)

    @staticmethod
    def replace_first_key(h,both=False):
        metric=h.metric.isin(['RMS_UU_full','E_neg_UU_full']) if both else h.metric.eq('E_neg_UU_full')
        h.loc[metric & h.layer.eq(0) & h['head'].eq(0),'layer']=32
        return h

    def test_equal_row_counts_with_different_keys_are_rejected(self):
        self.build_case(self.replace_first_key,r'UU full.*Head key mismatch')

    def test_matching_but_noncanonical_head_grid_is_rejected(self):
        self.build_case(lambda h:self.replace_first_key(h,True),r'UU full.*invalid Head keys')

    def test_missing_head_count_is_identified(self):
        self.build_case(lambda h:h.drop(h.index[0]),r'UU full RMS.*expected 1024.*1023')

    def test_duplicate_head_identifies_metric_and_key(self):
        self.build_case(lambda h:pd.concat([h,h.iloc[:1]],ignore_index=True),r'duplicate Head metric.*RMS_UU_full')

    def test_valid_permuted_head_rows_preserve_all_heads(self):
        self.build_case(lambda h:h.sample(frac=1,random_state=42))


if __name__=='__main__':unittest.main()
