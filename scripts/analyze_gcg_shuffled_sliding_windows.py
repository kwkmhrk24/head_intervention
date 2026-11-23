"""Analyze diagonal GCG/Shuffled windows using only saved Full Scores."""
import argparse
import html
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.artifact_io import read_json, sha256_file, utc_now
from utils.gcg_shuffled_common import DATASET, SCORES, inspect_inputs, write_csv, write_json

BASE = ROOT / 'data/outputs/gcg_vs_shuffled/legacy_selection20/10_existing_attention'
FULL = BASE / 'full_norm_change_primary18/20261121'
PRIOR = ROOT / 'data/outputs/llama2/gcg_autodan_full_delta_R_comparison/20261123/common_six_heads_magnitude/six_heads.csv'
OUTPUT = BASE / 'sliding_window_primary18/20261123'
WIDTHS = (8, 16)
HEADS = 1024
FLOOR = 1e-12  # Diagnostic only; positive denominators are never floored.
EXPECTED = {8: (501, 141, 126, 234), 16: (357, 15, 252, 90)}
REGIONS = ('OO', 'Boundary', 'UU')
DELTA = 'delta_R_asym'


def metrics(window):
    """Component decomposition in FP64; exact-zero ratios stay undefined."""
    s = np.asarray(window, dtype=np.float64)
    st = s.swapaxes(-1, -2)
    full, sym, asym = (np.linalg.norm(m, axis=(-2, -1)) for m in (s, (s+st)/2, (s-st)/2))
    errors = np.abs(full**2 - sym**2 - asym**2)
    np.testing.assert_allclose(sym**2 + asym**2, full**2, atol=1e-12, rtol=1e-12)
    rs, ra = (np.divide(v, full, out=np.full_like(full, np.nan), where=full != 0) for v in (sym, asym))
    return {'fro_norm': full, 'R_sym': rs, 'R_asym': ra, 'N_asym': asym/s.shape[-1]}, float(errors.max())


def spans(start, boundary, stop, width):
    for t in range(start, stop-width+1):
        end = t+width
        yield t, end, 'OO' if end <= boundary else 'UU' if t >= boundary else 'Boundary'


def collect(primary, refs):
    idx = {(r['dataset_index'], r['condition']):r for r in refs}
    records, oo_max, max_error = [], 0., 0.
    for i in primary:
        g, f = (idx[i, c] for c in ('gcg', 'shuffled'))
        # inspect_inputs has checked all paired input IDs, matching ranges and hashes.
        positions = g['input']['token_ranges']
        start, boundary = positions['original']
        suffix_start, stop = positions['suffix']
        if boundary != suffix_start or stop-boundary != 20:
            raise ValueError(f'unexpected Original/Suffix partition for {i}')
        gs, fs = (np.load(r['path'], mmap_mode='r', allow_pickle=False) for r in (g, f))
        oo_max = max(oo_max, float(np.max(np.abs(gs[:, :, start:boundary, start:boundary] -
                                                 fs[:, :, start:boundary, start:boundary]))))
        for w in WIDTHS:
            for t, end, region in spans(start, boundary, stop, w):
                cond = {}
                for c, arr in [('gcg',gs), ('shuffled',fs)]:
                    values, error = metrics(arr[:, :, t:end, t:end])
                    max_error = max(max_error, error)
                    cond[c] = {k:v.reshape(HEADS) for k,v in values.items()}
                rec = pd.DataFrame({
                    'dataset_index':i, 'width':w, 'start':t, 'end_exclusive':end,
                    'relative_t':t-boundary, 'relative_center':t-boundary+(w-1)/2,
                    'region':region, 'original_tokens':max(0,min(end,boundary)-t),
                    'suffix_tokens':max(0,end-max(t,boundary)), 'suffix_start':boundary,
                    'layer':np.repeat(np.arange(32),32), 'head':np.tile(np.arange(32),32),
                })
                for k in cond['gcg']:
                    rec[f'{k}_gcg'] = cond['gcg'][k]
                    rec[f'{k}_shuffled'] = cond['shuffled'][k]
                    rec[f'delta_{k}'] = cond['gcg'][k]-cond['shuffled'][k]
                records.append(rec)
    result = pd.concat(records, ignore_index=True)
    keys = ['dataset_index','width','relative_t','layer','head']
    if result.duplicated(keys).any():
        raise ValueError('duplicate pair/window/head key')
    counts = result.drop_duplicates(keys[:3]).groupby(['width','region']).size()
    for w in WIDTHS:
        assert (int(counts.loc[w].sum()), *(int(counts[w,r]) for r in REGIONS)) == EXPECTED[w]
    norms = result[['fro_norm_gcg','fro_norm_shuffled']].to_numpy()
    zero = int((norms==0).sum())
    small = int(((norms>0)&(norms<=FLOOR)).sum())
    identity = max(float((result[f'R_sym_{c}']**2+result[f'R_asym_{c}']**2-1).abs().max())
                   for c in ('gcg','shuffled'))
    oo_delta = float(result.loc[result.region=='OO',DELTA].abs().max())
    if oo_max != 0 or oo_delta != 0 or identity > 1e-12:
        raise ValueError('OO or ratio identity differs from saved expectation')
    return result, {'oo_raw_score_max_abs_delta':oo_max,'oo_delta_R_asym_max_abs':oo_delta,
                    'max_energy_residual':max_error,'max_ratio_identity_error':identity,
                    'zero_norm':zero,'positive_norm_at_or_below_1e_12':small,
                    'min_fro_norm':float(norms.min()), 'window_counts':EXPECTED}


def aggregates(values, full):
    x = values.assign(abs_R=values[DELTA].abs(), abs_N=values.delta_N_asym.abs())
    pos_keys = ['width','relative_t','region']
    p = x.groupby(pos_keys+['dataset_index'],as_index=False).agg(
        mean_R=(DELTA,'mean'), mean_abs_R=('abs_R','mean'), mean_N=('delta_N_asym','mean'),
        mean_abs_N=('abs_N','mean'), median_R=(DELTA,'median'))
    pos = p.groupby(pos_keys,as_index=False).agg(
        pairs=('dataset_index','size'),mean_R=('mean_R','mean'),mean_abs_R=('mean_abs_R','mean'),
        mean_N=('mean_N','mean'),mean_abs_N=('mean_abs_N','mean'),
        pair_mean_q25=('mean_R',lambda s:s.quantile(.25)),pair_mean_q75=('mean_R',lambda s:s.quantile(.75)),
        n_positive_pairs=('mean_R',lambda s:(s>0).sum()),n_negative_pairs=('mean_R',lambda s:(s<0).sum()),
        n_zero_pairs=('mean_R',lambda s:(s==0).sum()))
    q = x.groupby(pos_keys)[DELTA].quantile([.1,.5,.9]).unstack()
    q.columns = ['head_pair_q10','head_pair_q50','head_pair_q90']
    pos = pos.join(q,on=pos_keys)
    region_pairs = p.groupby(['width','region','dataset_index'],as_index=False).agg(
        mean_R=('mean_R','mean'),mean_abs_R=('mean_abs_R','mean'),mean_N=('mean_N','mean'),
        mean_abs_N=('mean_abs_N','mean'),windows=('relative_t','size'))
    region = region_pairs.groupby(['width','region'],as_index=False).agg(
        pairs=('dataset_index','size'),windows=('windows','sum'),mean_R=('mean_R','mean'),
        mean_abs_R=('mean_abs_R','mean'),mean_N=('mean_N','mean'),mean_abs_N=('mean_abs_N','mean'),
        positive_pairs=('mean_R',lambda s:(s>0).sum()),negative_pairs=('mean_R',lambda s:(s<0).sum()))
    head_position = x.groupby(pos_keys+['layer','head'],as_index=False).agg(
        pairs=('dataset_index','size'),mean_R=(DELTA,'mean'),mean_abs_R=('abs_R','mean'),
        mean_N=('delta_N_asym','mean'),n_positive_pairs=(DELTA,lambda s:(s>0).sum()),
        n_negative_pairs=(DELTA,lambda s:(s<0).sum()))
    per_pair_head_region = x.groupby(['width','region','dataset_index','layer','head'],as_index=False)[
        [DELTA,'delta_N_asym','abs_R']].mean()
    head_region = per_pair_head_region.groupby(['width','region','layer','head'],as_index=False).agg(
        pairs=('dataset_index','size'),mean_R=(DELTA,'mean'),mean_abs_R=('abs_R','mean'),
        mean_N=('delta_N_asym','mean'),n_positive_pairs=(DELTA,lambda s:(s>0).sum()),
        n_negative_pairs=(DELTA,lambda s:(s<0).sum()))
    layer = head_region.groupby(['width','region','layer'],as_index=False)[['mean_R','mean_abs_R','mean_N']].mean()
    full_head = full.groupby(['layer','head'],as_index=False)[DELTA].mean().rename(columns={DELTA:'full_mean_R'})
    head_region = head_region.merge(full_head,on=['layer','head'],validate='many_to_one')
    full_stats = []
    for (w,r),group in head_region.groupby(['width','region']):
        scored=group.mean_R
        if r!='OO':
            full_stats.append({'width':w,'region':r,'full_pearson':scored.corr(group.full_mean_R),
                'full_spearman':scored.rank().corr(group.full_mean_R.rank()),
                'same_sign_heads':int((scored*group.full_mean_R>0).sum()),
                'region_mean_abs_head_delta':float(scored.abs().mean())})
    return pos,p,region,region_pairs,head_position,head_region,layer,pd.DataFrame(full_stats)


def compare_widths(values):
    match = []
    for coordinate in ('relative_t','relative_center'):
        key = ['dataset_index','layer','head',coordinate]
        columns = key+['region',DELTA,'delta_N_asym']
        a,b = (values[values.width==w][columns] for w in WIDTHS)
        both = a.merge(b,on=key,suffixes=('_8','_16'),validate='one_to_one')
        for (r,s), d in both.groupby(['region_8','region_16']):
            x,y = d[DELTA+'_8'],d[DELTA+'_16']
            valid=(x!=0)&(y!=0)&x.notna()&y.notna()
            match.append({'coordinate':coordinate,'region_W8':r,'region_W16':s,'rows':len(d),
                          'pairs':d.dataset_index.nunique(),'nonzero_comparisons':int(valid.sum()),
                          'same_sign_fraction':float((x[valid]*y[valid]>0).mean()) if valid.any() else np.nan,
                          'pearson':float(x[valid].corr(y[valid])) if valid.any() else np.nan})
    return pd.DataFrame(match)


def figs(out,pos,hpos,selected,pair_positions):
    colors={8:'#1b6093',16:'#ab5334'}
    fig,axes=plt.subplots(2,2,figsize=(11,6.6),layout='constrained',sharex=True,sharey='row')
    for col,w in enumerate(WIDTHS):
        d=pos[pos.width==w]
        ax=axes[0,col]
        ax.fill_between(d.relative_t,d.pair_mean_q25,d.pair_mean_q75,color=colors[w],alpha=.17,label='IQR across pairs')
        ax.plot(d.relative_t,d.mean_R,color=colors[w],label='Signed mean')
        ax.plot(d.relative_t,d.mean_abs_R,color='#354449',ls='--',label='Mean absolute delta')
        ax.axhline(0,c='#666',lw=.55)
        ax.legend(fontsize=8)
        ax.set(title=f'W={w}: Full unmasked windows',ylabel='ΔR_asym')
        sub=axes[1,col]
        sub.plot(d.relative_t,d.mean_N,c=colors[w],label='Signed mean')
        sub.plot(d.relative_t,d.mean_abs_N,c='#354449',ls='--',label='Mean absolute delta')
        sub.set(xlabel='Window start relative to suffix',ylabel='ΔN_asym')
        for panel in (ax,sub):
            panel.axvspan(-w+0.5,-0.5,color='#c8a55a',alpha=.17)
            panel.axvline(0,c='#555',lw=.7)
    fig.savefig(out/'position_profiles.png',dpi=165)
    plt.close(fig)

    lim=float(hpos.mean_R.abs().max())
    fig,axes=plt.subplots(1,2,figsize=(11,8),layout='constrained',sharey=True)
    for ax,w in zip(axes,WIDTHS,strict=True):
        d=hpos[hpos.width==w].assign(flat=lambda z:z.layer*32+z['head'])
        mat=d.pivot(index='flat',columns='relative_t',values='mean_R')
        xs=mat.columns.to_numpy()
        image=ax.imshow(mat.to_numpy(),origin='lower',aspect='auto',interpolation='nearest',cmap='RdBu_r',
              vmin=-lim,vmax=lim,extent=[xs[0]-.5,xs[-1]+.5,-.5,1023.5])
        ax.set(title=f'W={w}',xlabel='Window start relative to suffix',yticks=np.arange(16,1024,128),
               yticklabels=[f'L{i}' for i in range(0,32,4)])
        for off in (-w+.5,-.5):
            ax.axvline(off,c='#49505b',lw=.6)
    axes[0].set_ylabel('Layer × 32 Heads')
    fig.colorbar(image,ax=axes,label='Mean ΔR_asym across pairs',shrink=.8)
    fig.savefig(out/'head_position_heatmaps.png',dpi=165)
    plt.close(fig)

    keys = list(selected[['layer','head']].drop_duplicates().itertuples(index=False,name=None))
    fig,axes = plt.subplots(len(keys),2,figsize=(10.8,2.65*len(keys)),squeeze=False,layout='constrained',sharex=True)
    for row,(l,h) in enumerate(keys):
        for ax,w in zip(axes[row],WIDTHS,strict=True):
            d=pair_positions[(pair_positions.layer==l)&(pair_positions['head']==h)&(pair_positions.width==w)]
            for _,t in d.groupby('dataset_index'):
                ax.plot(t.relative_t,t[DELTA],c='#9aaab4',alpha=.55,lw=.6)
            avg=d.groupby('relative_t')[DELTA].mean()
            ax.plot(avg.index,avg.to_numpy(),c=colors[w],lw=2)
            ax.axhline(0,c='#555',lw=.6)
            ax.axvspan(-w+.5,-.5,color='#c8a55a',alpha=.17)
            ax.set(title=f'L{l}H{h} · W={w}',ylabel='ΔR_asym',xlabel='Window start relative to suffix')
    fig.savefig(out/'representative_heads.png',dpi=165)
    plt.close(fig)


def page_table(frame):
    return frame.to_html(index=False, float_format=lambda x:f'{x:+.6f}',escape=True,classes='data')


def report(out, pos, region, layer, cross, full_stats, chosen, prior_peaks, link_stats, findings):
    def f(value): return f'{value:+.6f}'
    peak_rows=[]
    for w in WIDTHS:
        d=pos[(pos.width==w)&(pos.pairs==18)]
        r=d.loc[d.mean_abs_R.idxmax()]
        s=d.loc[d.mean_R.abs().idxmax()]
        peak_rows.append(f'<li>W={w} の平均絶対ΔRピークは t={int(r.relative_t)}（{r.region}、{r.mean_abs_R:.6f}）。'
                         f'符号付き平均の絶対値ピークは t={int(s.relative_t)}（{s.region}、{f(s.mean_R)}、'
                         f'正のペア{int(s.n_positive_pairs)}／負のペア{int(s.n_negative_pairs)}）。</li>')
    peaks=''.join(peak_rows)
    leading_layers=(layer[layer.region!='OO'].sort_values(['mean_abs_R','layer'],ascending=[False,True])
                    .groupby(['width','region'],as_index=False).head(2)
                    [['width','region','layer','mean_abs_R']].sort_values(['width','region','mean_abs_R'],
                                                                        ascending=[True,True,False]))
    top_layers=leading_layers.groupby(['width','region']).first().layer.unique()
    layer_note=(f'境界・UUとも両窓幅でLayer {int(top_layers[0])}が最大だった。'
                if len(top_layers)==1 else '絶対差が最大のLayerは窓幅または領域で異なった。')
    uu=region.set_index(['width','region'])
    uu_note=(f'UU全体の符号付き平均は、W=8で {f(uu.loc[(8,"UU"),"mean_R"])}、'
             f'W=16で {f(uu.loc[(16,"UU"),"mean_R"])}。')
    (out/'report.html').write_text(f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>GCG − Shuffled | Sliding Window</title><style>
body{{font:16px/1.65 system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#18273b}}
h1,h2{{line-height:1.25}}figure{{margin:1.25rem 0}}img{{max-width:100%;height:auto}}figcaption{{color:#445d74}}
.data{{border-collapse:collapse;font-variant-numeric:tabular-nums;max-width:100%;font-size:13px}}
.data td,.data th{{border-bottom:1px solid #dae0e6;padding:.25rem .45rem;text-align:right}}
.scroll{{overflow-x:auto}}a{{color:#235d90}}</style></head><body>
<h1>GCG成功 − Shuffled失敗のSliding Window</h1>
<p>Llama-2-7b-chat-hfのprimary18、全1,024 Head。RoPE後・mask前・softmax前のFull Scoreから、
Original先頭〜Suffix終端だけを用いて主対角線上の正方窓をstride 1で切り出した。</p>
<h2>1. 位置ごとの観測</h2>
<p>OOの生ScoreとΔRは全比較で完全一致。W=8は501窓、W=16は357窓。うちW=16のOOは6ペアに限られる。
以下のピークは18ペアが揃う位置で比較した。絶対差は各Headで絶対値を取ってから平均している。</p>
<ul>{peaks}</ul>
<p>{uu_note} 幅ごとに窓内のToken構成と平滑化の範囲が変わる。</p>
<figure><img src="position_profiles.png" alt="W=8と16の位置別ΔR・ΔN"><figcaption>
金色の帯は境界Window。帯より左はOO、相対開始位置0以上はUU。上段の面はペア別Head平均の四分位範囲、破線はHead別の絶対差を平均。各位置の有効ペア数とHead×ペアの10/50/90分位点はCSVに収録。
</figcaption></figure>
<div class="scroll">{page_table(region)}</div>
<h2>2. ペア・Head間の共通性</h2>
<figure><img src="head_position_heatmaps.png" alt="全Head×Window位置のmean ΔR_asym"><figcaption>縦軸は Layer×32＋Head。両幅で色域を統一。</figcaption></figure>
<p>局所候補はW=8の18ペアが揃う非OO位置で、符号付き平均の絶対値が最大だった既存6 Head以外の1 Head。
下表のペア正負件数はその位置ごとの値。既存6 Headの詳細は別CSVにある。</p>
<div class="scroll">{page_table(chosen)}</div>
<div class="scroll">{page_table(prior_peaks)}</div>
<figure><img src="representative_heads.png" alt="L30H27と局所候補のペア別変化"><figcaption>灰線は各ペア、色線は符号付きペア平均。</figcaption></figure>
<p>Layer別に絶対差を集計すると、{layer_note}この表は変化の大きさであり、符号が揃ったことを意味しない。</p>
<div class="scroll">{page_table(leading_layers)}</div>
<h2>3. 窓幅・ΔN・Full Scoreとの比較</h2>
<p>同じ相対開始位置に加え、偶数幅のWindow中心が同じ比較を行った。幅が違うと同じ位置でも構成Tokenが異なる。
符号一致の分母は両方が非ゼロであるペア×Head×位置。依存する値に基づく記述集計である。</p>
<div class="scroll">{page_table(cross)}</div>
<p>ΔN_asymは反対称Normを各窓幅で除した量、ΔR_asymは全体Normを分母とした比の差。
これらの対応は非OOにおける符号とPearson相関で確認した。</p>
<div class="scroll">{page_table(link_stats)}</div>
<p>Full Scoreとの比較は領域平均を各ペア内で作り、ペア等重みでHeadごとに平均した順位との対応。
FullとWindowで範囲・分母が異なるため、局所差の数値とFull差は置換可能な尺度ではない。</p>
<div class="scroll">{page_table(full_stats)}</div>
<p>非OOの絶対ΔRをWindowごとに求め、各ペア×Headで最大を取った場合にFullの絶対ΔRを上回った割合は
W=8で {findings['local_max_gt_full_fraction'][8]:.1%}、W=16で {findings['local_max_gt_full_fraction'][16]:.1%}。
複数位置の最大値を取ること自体が差を大きく見せ得るため、記述値として扱う。</p>
<h2>解釈・未確認事項・次の仮説</h2>
<p>観測された位置依存性は、共通Originalの後からScore構造が変わることと整合する。
ΔNとΔRの一致しない部分には、窓全体のNormも関わり得る。R_sym²＋R_asym²＝1の制約があるため、
比率の変化だけから反対称成分自体の増減や独立した効果は判断できない。</p>
<p>保存したScoreは未来Token側も含む。したがって、この観測から実際の情報伝達やJailbreak成功への因果的寄与は判断できない。
次は、局所差が大きい位置が別ペア集合でも再現するか、因果的に参照できる位置に限定した指標でも残るかを検証する。
18ペアが観測単位であり、重複WindowとHeadを独立標本にした検定は行っていない。</p>
<h2>データ</h2><ul>{''.join(f'<li><a href="{html.escape(path.name)}">{html.escape(path.name)}</a></li>' for path in sorted(out.iterdir()) if path.suffix in ('.csv','.gz','.json'))}</ul>
</body></html>""",encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    out=parser.parse_args().output
    if out.exists():
        raise FileExistsError(out)
    _, primary, refs=inspect_inputs(conditions=('gcg','shuffled'))
    full=pd.read_csv(FULL/'pairs.csv',usecols=['dataset_index','layer','head',DELTA],float_precision='round_trip')
    if set(full.dataset_index)!=set(primary) or len(full)!=18*HEADS or full.duplicated(['dataset_index','layer','head']).any():
        raise ValueError('Full Score comparator does not match primary18')
    if sha256_file(FULL/'pairs.csv')!=read_json(FULL/'output_hashes.json')['pairs.csv']:
        raise ValueError('Full comparator has changed')
    prior=pd.read_csv(PRIOR).query("method == 'GCG'")[['layer','head']].drop_duplicates()
    assert len(prior)==6 and ((prior.layer==30)&(prior['head']==27)).any()
    values,validation=collect(primary,refs)
    pos,pp,region,rp,hpos,hr,layer,fstats=aggregates(values,full)
    print('Position summaries ready.',flush=True)
    # Identify the exploratory example only after producing the position summaries.
    qualified=hpos[(hpos.width==8)&(hpos.pairs==18)&(hpos.region!='OO')]
    new=qualified.merge(prior.assign(prior=True),on=['layer','head'],how='left')
    new=new[new.prior.isna()].copy()
    new['abs_mean']=new.mean_R.abs()
    new=new.sort_values(['abs_mean','layer','head','relative_t'],ascending=[False,True,True,True]).iloc[[0]]
    fixed=hpos.merge(prior,on=['layer','head'],how='inner')
    fixed=fixed[(fixed.pairs==18)&(fixed.region!='OO')].assign(abs_mean=lambda d:d.mean_R.abs())
    fixed=fixed.sort_values(['abs_mean','relative_t'],ascending=[False,True]).drop_duplicates(['width','layer','head'])
    shown_cols=['width','layer','head','region','relative_t','pairs','mean_R','mean_N','n_positive_pairs','n_negative_pairs']
    chosen=new[shown_cols].assign(candidate='exploratory window peak')
    prior_peaks=fixed[shown_cols]
    selected=pd.concat([prior.query("layer==30 and head==27"),new[['layer','head']]])
    representative=values.merge(selected,on=['layer','head'],validate='many_to_one')
    comparison=compare_widths(values)
    linked=[]
    for (w,r),d in values[values.region!='OO'].groupby(['width','region']):
        pair_corr=[]
        for _, p in d.groupby('dataset_index'):
            pair_corr.append(p[DELTA].corr(p.delta_N_asym))
        valid=d[DELTA].ne(0)&d.delta_N_asym.ne(0)&d[DELTA].notna()&d.delta_N_asym.notna()
        linked.append({'width':w,'region':r,'median_pair_pearson':float(np.nanmedian(pair_corr)),
                       'same_sign_fraction_nonzero':float((d.loc[valid,DELTA]*d.loc[valid,'delta_N_asym']>0).mean())})
    linked=pd.DataFrame(linked)
    junction=['dataset_index','layer','head']
    local=(values[values.region!='OO'].assign(local_abs=lambda d:d[DELTA].abs())
           .groupby(junction+['width'],as_index=False).local_abs.max().merge(full,on=junction,validate='many_to_one'))
    local['exceeds_full_abs']=local.local_abs>local[DELTA].abs()
    findings={'local_max_gt_full_fraction':local.groupby('width').exceeds_full_abs.mean().to_dict()}
    assert len(values)==(501+357)*HEADS
    out.mkdir(parents=True,exist_ok=False)
    values.to_csv(out/'window_heads.csv.gz',index=False,mode='x',compression='gzip',float_format='%.17g')
    for name,data in {'position_summary':pos,'pair_position_summary':pp,'region_summary':region,
        'pair_region_summary':rp,'head_position':hpos,'head_region':hr,'layer_region':layer,
        'full_head_comparison':fstats,'local_max_vs_full':local,'width_comparison':comparison,
        'R_N_comparison':linked,'exploratory_head':chosen,'prior_head_peaks':prior_peaks,
        'representative_pair_windows':representative}.items():
        write_csv(out/f'{name}.csv',data)
    figs(out,pos,hpos,selected,representative)
    files=[DATASET, SCORES/'manifest.jsonl', SCORES/'plan.json', SCORES/'validation_final.json',
           FULL/'pairs.csv', FULL/'output_hashes.json', PRIOR, Path(__file__).resolve()]
    write_json(out/'manifest.json', {'created_at':utc_now(),'git_head':subprocess.check_output(
        ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'primary_indices':primary,'widths':WIDTHS,
        'stride':1,'domain':'Original start through suffix end, half-open, excluding both template regions',
        'raw_score':read_json(SCORES/'plan.json')['computation'],'delta':'GCG - Shuffled',
        'denominator_warning_floor':FLOOR,'zero_policy':'zero denominator ratios undefined; no correction',
        'analysis_dtype':'float64','model_forward':False,'score_extraction':False,
        'files_sha256':{str(p):sha256_file(p) for p in files},
        'primary_score_sha256':{r['path']:r['sha256'] for r in refs if r['dataset_index'] in primary}})
    write_json(out/'validation.json',dict(validation,rows=len(values),pairs=len(primary),heads=HEADS,
                                          new_model_forward=False,new_score_extraction=False))
    report(out,pos,region,layer,comparison,fstats,chosen,prior_peaks,linked,findings)
    print(out, validation, flush=True)


if __name__=='__main__':
    main()
