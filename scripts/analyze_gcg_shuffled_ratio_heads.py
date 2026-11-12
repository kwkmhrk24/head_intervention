#!/usr/bin/env python3
"""CSV-only Head diagnostics of the existing Full/Causal UU/OO RMS ratios."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from utils.artifact_io import read_json,sha256_file,utc_now
from utils.gcg_shuffled_common import write_csv,write_json
from utils.gcg_shuffled_ratio_heads import POLICIES,analyze_heads,fixed_group_tables
from utils.gcg_shuffled_plot_common import plt
from matplotlib import font_manager

BASE=ROOT/'data/outputs/gcg_vs_shuffled/legacy_selection20'
SOURCE=BASE/'20_region_scores/rms_and_frobenius'


def draw_figures(out,heads,layers,groups,members):
    font=Path('/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf')
    if not font.is_file():raise ValueError('Japanese plotting font unavailable: '+str(font))
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=font).get_name(),'DejaVu Sans'],
                         'font.size':11,'axes.unicode_minus':False})
    records=[]
    def save(fig,name,title):
        fig.suptitle(title,fontsize=14)
        fig.tight_layout(rect=(0,0,1,.95));p=out/'figures'/name
        fig.savefig(p,dpi=300,bbox_inches='tight');plt.close(fig)
        records.append(dict(file='figures/'+name,dpi=300,subset=18,title=title))
    primary=heads[heads.subset==18]
    lower=min(float(primary[['gcg_mean','shuffled_mean']].min().min()),1.)
    upper=max(float(primary[['gcg_mean','shuffled_mean']].max().max()),1.)
    margin=.05*(upper-lower);limits=(max(0.,lower-margin),upper+margin)
    extent=max(abs(primary.delta_mean.min()),abs(primary.delta_mean.max()))
    bins=np.linspace(-extent,extent,51)
    def scatter(ax,p):
        h=primary[primary.policy==p]
        im=ax.scatter(h.shuffled_mean,h.gcg_mean,c=h.layer,cmap='viridis',vmin=0,vmax=31,s=12,alpha=.65)
        ax.plot(limits,limits,'k--',lw=1,label='GCG = Shuffled')
        ax.axvline(1,color='grey',ls=':',label='比率 = 1');ax.axhline(1,color='grey',ls=':')
        ax.set(xlim=limits,ylim=limits,xlabel='Shuffled：18サンプル平均比率（無次元）',ylabel='GCG：18サンプル平均比率（無次元）',
               title=p+'：各点 = 1 Head（全1,024 Head）')
        ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=8)
        ax.figure.colorbar(im,ax=ax,label='Layer（0始まり）')
    def histogram(ax,p):
        h=primary[primary.policy==p];d=h.delta_mean
        ax.hist(d,bins=bins,color='steelblue',edgecolor='white')
        ax.axvline(0,color='black',label='差 = 0')
        ax.axvline(d.mean(),color='crimson',ls='--',label=f'Head平均 {d.mean():+.5f}')
        ax.axvline(d.median(),color='darkorange',ls=':',label=f'Head中央値 {d.median():+.5f}')
        ax.set(xlabel='GCG - Shuffled：各Headの18件平均比率差（無次元）',ylabel='Head数',title=p+'：全1,024 Headの分布')
        ax.legend(fontsize=8)
    fig,axes=plt.subplots(2,2,figsize=(15,12))
    for row,p in enumerate(POLICIES):scatter(axes[row,0],p);histogram(axes[row,1],p)
    save(fig,'ratio_heads_overview.png','UU/OO RMS比：主要18ペア・全32 Layer × 32 Head\n各サンプル・Headで保存済み比率を算出後にサンプル平均／FullとCausalを区別')
    for p in POLICIES:
        for name,fn in [('scatter',scatter),('histogram',histogram)]:
            fig,ax=plt.subplots(figsize=(9,7));fn(ax,p)
            save(fig,f'{name}_{p}.png',f'UU/OO RMS比：{p}・主要18ペア\n比率・差は無次元／既存の各サンプル・Head比率の平均')
    def direction(ax,p):
        h=primary[primary.policy==p]
        counts=h.positive_samples.value_counts().reindex(range(19),fill_value=0)
        ax.bar(counts.index,counts.values,color='steelblue')
        ax.set(xticks=[0,3,6,9,12,15,18],xlabel='18ペア中でGCG > Shuffledとなったサンプル数（既存許容誤差で判定）',
               ylabel='Head数',title=p+'：サンプル間の増加方向の一致')
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,p in zip(axes,POLICIES):direction(ax,p)
    save(fig,'direction_consistency.png','UU/OO RMS比：主要18ペア・全1,024 Head\n18/18は全サンプルで増加。Headは独立した統計標本ではない')
    def layer(ax,p):
        l=layers[(layers.subset==18)&(layers.policy==p)]
        ax.bar(l.layer,l.delta_mean,color=np.where(l.delta_mean>=0,'firebrick','steelblue'),alpha=.7)
        ref=primary[primary.policy==p].delta_mean.mean()
        ax.axhline(0,color='black',lw=.8);ax.axhline(ref,color='darkorange',ls='--',label=f'全Head平均 {ref:+.5f}')
        ax.set(xticks=[0,4,8,12,16,20,24,28,31],xlabel='Layer（0始まり、各32 Head）',ylabel='GCG - Shuffled 平均比率差（無次元）',title=p+'：Layer別平均')
        ax.legend(fontsize=9)
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,p in zip(axes,POLICIES):layer(ax,p)
    save(fig,'layer_means.png','UU/OO RMS比：主要18ペア\n各Headのサンプル平均差をLayer内32 Headで等重み平均')
    for p in POLICIES:
        for name,fn in [('direction_consistency',direction),('layer_means',layer)]:
            fig,ax=plt.subplots(figsize=(10,6));fn(ax,p)
            save(fig,f'{name}_{p}.png',f'UU/OO RMS比：{p}・主要18ペア・全1,024 Head\n各サンプル・Headでの保存済み比率を使用')
    # Full fixed-group graph already exists; only the missing Causal counterpart is drawn.
    g=groups[(groups.subset==18)&(groups.policy=='causal')]
    fig,ax=plt.subplots(figsize=(12,8))
    for y,row in enumerate(g.itertuples()):
        part=members[(members.subset==18)&(members.policy=='causal')&(members.selected_group==row.view_id)]
        ax.errorbar(row.delta_mean,y,xerr=row.delta_std,fmt='D',color='black',capsize=3)
        ax.scatter(part.delta_mean,y+np.linspace(-.16,.16,len(part)),color='steelblue',s=22)
    ax.axvline(0,color='grey');ax.axvline(primary[primary.policy=='causal'].delta_mean.mean(),color='darkorange',ls='--',label='全1,024 Head平均')
    ax.set(yticks=range(len(g)),yticklabels=g.view_id,xlabel='GCG - Shuffled 平均比率差（無次元）',ylabel='既存固定5 Head群（名称・順番を維持）')
    ax.invert_yaxis();ax.legend()
    save(fig,'fixed_groups_causal.png','UU/OO RMS比：Causal・主要18ペア\n菱形：群平均、棒：サンプル差SD、点：各固定Headの18件平均／群を再選択していない')
    return records,str(font)


def write_report(out,overview,heads,layers,groups):
    lines=['# UU/OO RMS比：Head別の追加可視化','',
        '## 条件・定義','保存済みCSVのみをCPUで使用。Score・NPZ・モデル・応答は読み込んでいない。主分析18ペア、補助20ペア、全32×32 Head、既存固定群を維持。',
        'R = RMS_UU / RMS_OO。Fullは全要素、Causalは分子UU・分母OOともKey<=Query（対角含む）。各サンプル・Headで計算済みの比率をサンプル平均する。平均RMS同士の比ではない。差はGCG−Shuffled、無次元。',
        '増減は旧解析のatol=rtol=1e-12で判定。厳密な正負件数もCSVに別記。R>1はHeadの条件別サンプル平均について厳密に判定。',
        '## 観測：全体・全20件との比較','',
        '| n | 条件 | GCG平均 | Shuffled平均 | 平均差 | Head差中央値 | 増/減/同等Head | GCG平均比率>1 |',
        '|---|---|---:|---:|---:|---:|---|---:|']
    for r in overview.itertuples():
        lines.append(f'| {r.subset} | {r.policy} | {r.gcg_mean:.8f} | {r.shuffled_mean:.8f} | {r.delta_mean:+.8f} | {r.head_delta_median:+.8f} | {r.positive_heads}/{r.negative_heads}/{r.equal_heads} | {r.gcg_head_mean_gt1}/1024 |')
    lines+=['','出典：tables/overview.csv。GCG平均比率>1のHeadは、全18サンプルで比率>1とは限らない。各Headの超過サンプル数はheads.csv。Head差中央値はHead間の中央値であり、既存overviewのサンプル間中央値とは異なる。',
        '## 観測：広がりとサンプル間の一致','',
        '| n | 条件 | 全サンプルで増加するHead | 80%以上のサンプルで増加 | 平均方向一致率の中央値 | 上位10% Headが占める正の平均差の総量 | 上位10%除外後平均 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in overview.itertuples():
        lines.append(f'| {r.subset} | {r.policy} | {r.positive_all_samples_heads} | {r.at_least_80pct_positive_heads} | {100*r.mean_direction_agreement_median:.2f}% | {100*r.top10pct_positive_delta_share:.2f}% | {r.remaining_mean_after_top10pct:+.6f} |')
    lines+=['','主要18件の要点：']
    for p in POLICIES:
        r=overview[(overview.subset==18)&(overview.policy==p)].iloc[0]
        lines.append(f'{p}：平均差は{r.delta_mean:+.6f}、増加Headは{int(r.positive_heads)}/1024（{100*r.positive_heads/1024:.2f}%）。GCGの平均比率>1は{int(r.gcg_head_mean_gt1)}/1024（{100*r.gcg_head_mean_gt1/1024:.2f}%）、Shuffledは{int(r.shuffled_head_mean_gt1)}。1以下から1超へ移るHeadは{int(r.upward_crossing_1)}、逆方向は{int(r.downward_crossing_1)}。')
    lines+=['Head差中央値と上位10%除外後平均はいずれも正で、少数の大きなHeadだけが平均を正にした説明には合わない。一方、全サンプルで増加するHeadは一部であり、Head平均の広がりと個別入力での一貫性は区別する。']
    lines+=['','各Headの平均方向一致率＝そのHeadのサンプル平均差と同方向だったサンプル数/n。同等平均なら同等数/n。positive_fraction・negative_fraction・equal_fractionも別保存。多数派方向の非ゼロ一致率はmax(増加数,減少数)/nで、平均方向との一致率とは異なる。',
        '上位10%はサンプル平均差の大きい103 Headを除く記述的感度。新たな重要Head群・防御位置の選択ではなく、効果の集中を記述するための量。全Headが独立標本だとは扱わない。','',
        '## 観測：Layer・固定群','',
        '| 条件 | 平均差が正のLayer | Layer平均差の範囲 | 固定群の平均差の範囲 |',
        '|---|---:|---|---|']
    for p in ('full','causal'):
        l=layers[(layers.subset==18)&(layers.policy==p)];g=groups[(groups.subset==18)&(groups.policy==p)]
        lines.append(f'| {p} | {int((l.delta_mean>0).sum())}/32 | {l.delta_mean.min():+.6f}〜{l.delta_mean.max():+.6f} | {g.delta_mean.min():+.6f}〜{g.delta_mean.max():+.6f} |')
    lines+=['','| 既存固定5 Head群 | Full平均差 | Causal平均差 |',
            '|---|---:|---:|']
    for name in groups[(groups.subset==18)&(groups.policy=='full')].view_id:
        f=groups[(groups.subset==18)&(groups.policy=='full')&(groups.view_id==name)].iloc[0]
        c=groups[(groups.subset==18)&(groups.policy=='causal')&(groups.view_id==name)].iloc[0]
        # Group membership itself is taken from existing CSVs, never ranked here.
        lines.append(f'| {name} | {f.delta_mean:+.6f} | {c.delta_mean:+.6f} |')
    lines+=['','Layer平均と固定5 Head群は構成が異なるため、同一集団の推定値として比較しない。群名・順序・メンバーは既存CSVを再利用。tables/fixed_groups.csvに群平均・サンプルSD、fixed_group_members.csvに各メンバー結果、layer_vs_fixed_groups.csvに同じ全Head基準からの差を保存。',
        '## FullとCausalの対応','']
    for n in (18,20):
        a=heads[(heads.subset==n)&(heads.policy=='full')].sort_values(['layer','head'])
        b=heads[(heads.subset==n)&(heads.policy=='causal')].sort_values(['layer','head'])
        flipped=int(((a.mean_direction.to_numpy()*b.mean_direction.to_numpy())<0).sum())
        lines.append(f'{n}ペア：平均差の正負がFull/Causalで反転するHeadは{flipped}/1024。詳細はtables/full_causal_head_comparison.csv。')
    lines+=['','## 考察・言えないこと',
        '増加Head割合・Head差中央値・上位Head除外感度を併せて、平均増加が広く見られるかを評価する。平均が正でも全Head・全サンプルでの増加を意味しない。',
        'OOがGCG/Shuffledで不変であるため、比率差は各サンプル・HeadのΔRMS_UUをRMS_OOで正規化した量である。独立の機構的証拠ではない。R>1はUUのRMSがOOより大きいことを示すだけで、Attention配分の優勢や情報流・攻撃成功の原因を示さない。',
        '主要18件は既存43拒否表現ルールでOriginal失敗/GCG成功/Shuffled失敗とされた選択済み例。人手による成功確認ではない。固定群も同じデータから選択済みで、独立した検証・有意性・因果・一般化は主張しない。全20件も同じ選択集合の補助分析。',
        '## 図と既存図の参照',
        'まずfigures/ratio_heads_overview.png、direction_consistency.png、layer_means.pngを参照。散布図・ヒストグラム・方向一致・Layer平均は各Full/Causalの単独PNGも保存（300dpi、日本語）。新しいCausal固定群図はfixed_groups_causal.png。',
        '既存ヒートマップとFull固定群図は再生成していない。existing_figures.jsonのパス・SHA256から参照。既存Full固定群図ではR_UU_OO_fullのパネルを見ること。',
        '## 再現',
        '```bash','python -B scripts/analyze_gcg_shuffled_ratio_heads.py --output-dir NEW_DIRECTORY','python -B -m unittest tests.test_gcg_shuffled_ratio_heads','```',
        '入力hashはmanifest.json、数値整合性・旧CSV/図の不変性はvalidation.json。出力hash台帳output_hashes.jsonは自身を含めず、数値表・PNG・レポート・manifestを検証できる。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def run(source,out):
    source=source.resolve();out=out.resolve()
    if out.exists():raise FileExistsError('new output directory required')
    names=['metrics.csv','paired_differences.csv','summary_18.csv','summary_20.csv','head_maps_18.csv','head_maps_20.csv',
           'overview.csv','view_summary_18.csv','view_summary_20.csv','fixed_group_member_effects.csv']
    # Preserve all existing tabular/figure/report/manifest files, without reading Score or NPZ.
    originals=[p for p in source.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.md')]
    originals += [p for p in (source/'figures').iterdir() if p.is_file()]
    before={str(p):sha256_file(p) for p in originals}
    ledger=read_json(source/'output_hashes.json')
    for n in names:
        if sha256_file(source/n)!=ledger.get(n):raise ValueError('old CSV hash mismatch: '+n)
    tables={n:pd.read_csv(source/n,float_precision='round_trip') for n in names}
    metrics=tables['metrics.csv'];paired=tables['paired_differences.csv']
    if set(metrics.condition)!={'gcg','shuffled'} or metrics.primary18.isna().any():raise ValueError('invalid conditions or primary18 flags')
    flags=metrics.groupby('dataset_index').primary18.agg(['min','max'])
    if not flags['min'].equals(flags['max']):raise ValueError('inconsistent primary18 flags')
    all_indices=sorted(metrics.dataset_index.unique().tolist());primary=flags.index[flags['min']].tolist()
    if len(all_indices)!=20 or len(primary)!=18 or sorted(set(all_indices)-set(primary))!=[91,123] or any(i>=220 for i in all_indices):raise ValueError('not the existing legacy18/20 cohort')
    if len(metrics)!=20*2*1024 or len(paired)!=20*1024 or set(paired.dataset_index)!=set(all_indices):raise ValueError('sample count mismatch')
    out.mkdir(parents=True);(out/'tables').mkdir();(out/'figures').mkdir()
    heads=[];overviews=[];layers=[];samples=[];groups=[];members=[];errors=[]
    for n,indices in ((18,primary),(20,all_indices)):
        for policy in POLICIES:
            h,o,l,s,e=analyze_heads(metrics,paired,tables[f'summary_{n}.csv'],tables[f'head_maps_{n}.csv'],tables['overview.csv'],indices,policy)
            g,d=fixed_group_tables(tables[f'view_summary_{n}.csv'],tables['fixed_group_member_effects.csv'],h,n,policy)
            heads.append(h);overviews.append(o);layers.append(l);samples.append(s);groups.append(g);members.append(d);errors.append(dict(subset=n,policy=policy,**e))
    heads=pd.concat(heads,ignore_index=True);overview=pd.DataFrame(overviews);layers=pd.concat(layers,ignore_index=True)
    samples=pd.concat(samples,ignore_index=True);groups=pd.concat(groups,ignore_index=True);members=pd.concat(members,ignore_index=True)
    for n,f in [('heads',heads),('overview',overview),('layers',layers),('samples',samples),('fixed_groups',groups),('fixed_group_members',members)]:write_csv(out/'tables'/f'{n}.csv',f)
    comparisons=[]
    for n in (18,20):
        a=heads[(heads.subset==n)&(heads.policy=='full')];b=heads[(heads.subset==n)&(heads.policy=='causal')]
        comparisons.append(a[['subset','layer','head','gcg_mean','shuffled_mean','delta_mean','mean_direction']].merge(
            b[['subset','layer','head','gcg_mean','shuffled_mean','delta_mean','mean_direction']],on=['subset','layer','head'],suffixes=('_full','_causal'),validate='one_to_one'))
    write_csv(out/'tables/full_causal_head_comparison.csv',pd.concat(comparisons,ignore_index=True))
    # A descriptive comparison against the SAME all-Head reference; no selected groups added.
    viewrows=[]
    for n in (18,20):
        for p in POLICIES:
            ref=overview[(overview.subset==n)&(overview.policy==p)].iloc[0].delta_mean
            for r in layers[(layers.subset==n)&(layers.policy==p)].itertuples():viewrows.append(dict(subset=n,policy=p,kind='layer',name=str(r.layer),delta_mean=r.delta_mean,delta_from_all_heads=r.delta_mean-ref))
            for r in groups[(groups.subset==n)&(groups.policy==p)].itertuples():viewrows.append(dict(subset=n,policy=p,kind='existing_fixed_group',name=r.view_id,delta_mean=r.delta_mean,delta_from_all_heads=r.delta_mean-ref))
    write_csv(out/'tables/layer_vs_fixed_groups.csv',pd.DataFrame(viewrows))
    figure_records,font=draw_figures(out,heads,layers,groups,members)
    existing={n:dict(path=str(source/'figures'/n),sha256=sha256_file(source/'figures'/n)) for n in ['head_heatmaps_full.png','head_heatmaps_causal.png','fixed_head_groups.png','sample_consistency.png','sensitivity_18_20.png']}
    write_json(out/'existing_figures.json',existing)
    write_json(out/'figures/index.json',figure_records)
    write_report(out,overview,heads,layers,groups)
    code=[Path(__file__),ROOT/'utils/gcg_shuffled_ratio_heads.py',ROOT/'tests/test_gcg_shuffled_ratio_heads.py',ROOT/'utils/gcg_shuffled_common.py',ROOT/'utils/gcg_shuffled_reporting.py',ROOT/'utils/gcg_shuffled_metrics.py',ROOT/'utils/gcg_shuffled_plot_common.py',ROOT/'utils/artifact_io.py']
    write_json(out/'manifest.json',dict(created_at=utc_now(),source_directory=str(source),input_hashes={str(source/n):before[str(source/n)] for n in names},
        source_hashes={str(p):sha256_file(p) for p in code},git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        primary_indices=primary,all_indices=all_indices,comparison='GCG-Shuffled',ratio_definition='per sample/Head RMS_UU / RMS_OO; causal numerator AND denominator',
        mean_order='previous per-sample ratios; sample mean per Head; equal-weight Head/sample aggregation',
        direction_tolerance=dict(atol=1e-12,rtol=1e-12),strict_sign_counts_also_saved=True,threshold_1='condition-specific SAMPLE-MEAN ratio per Head > 1',
        direction_agreement='fraction of sample paired changes matching each Head sample-MEAN direction (ties included); positive/negative fractions separately',
        model_loaded=False,GPU_used=False,scores_loaded=False,npz_loaded=False,independent100_used=False,
        font=font,numpy=np.__version__,pandas=pd.__version__,python=sys.version,figure_dpi=300))
    for path,digest in before.items():
        if sha256_file(path)!=digest:raise ValueError('existing artifact changed: '+path)
    write_json(out/'validation.json',dict(status='passed',existing_artifact_hashes_unchanged=len(before),checks=errors,
        head_rows=len(heads),all_four_views_complete=len(heads)==4096,head_correspondence='complete 32x32 for both policies/cohorts',
        missing=0,unexpected_nonfinite=0,csv_values_and_old_direction_counts_verified=True,figure_count=len(figure_records)))
    write_json(out/'output_hashes.json',{str(p.relative_to(out)):sha256_file(p) for p in out.rglob('*') if p.is_file()})
    for p,h in read_json(out/'output_hashes.json').items():
        if sha256_file(out/p)!=h:raise ValueError('output hash mismatch: '+p)
    print(overview.to_string(index=False));print('Output:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=SOURCE)
    p.add_argument('--output-dir',type=Path,default=BASE/'20_region_scores/ratio_head_diagnostics'/utc_now().replace(':','').replace('.',''))
    a=p.parse_args();run(a.source_dir,a.output_dir)
