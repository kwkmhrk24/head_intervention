#!/usr/bin/env python3
"""Publish a table-backed synthesis of negative-energy attribution (no recompute)."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd
from utils.artifact_io import read_json,sha256_file,utc_now
from utils.gcg_shuffled_common import write_json


def report(out):
    if (out/'findings.md').exists() or (out/'manifest.json').exists():raise FileExistsError('report exists')
    tables=out/'tables';m=read_json(out/'core_manifest.json');finish=read_json(out/'finish_manifest.json')
    validation=read_json(out/'validation.json')
    dist=pd.read_csv(tables/'head_distributions.csv');d=dist[(dist.subset==18)&(dist.view_id=='all_heads')].set_index('region')
    factor=pd.read_csv(tables/'factor_overview.csv');f=factor[(factor.subset==18)&(factor.view_id=='all_heads')].set_index('region')
    tail=pd.read_csv(tables/'tail_overview.csv');tail=tail[(tail.subset==18)&(tail.view_id=='all_heads')]
    trim=pd.read_csv(tables/'head_tail_sensitivity.csv');trim=trim[trim.subset==18]
    sample=pd.read_csv(tables/'sample_summary.csv');sample=sample[sample.subset==18]
    loo=pd.read_csv(tables/'leave_one_sample_out.csv');loo=loo[loo.subset==18]
    layers=pd.read_csv(tables/'layer_summary.csv');layers=layers[layers.subset==18]
    q=pd.read_csv(tables/'query_overview.csv');q=q[(q.subset==18)&(q.view_id=='all_heads')]
    w=pd.read_csv(tables/'query_weighting.csv');w=w[(w.subset==18)&(w.view_id=='all_heads')].set_index('region')
    main=['UU_full','UU_causal','UO']
    lines=['# 結論：Headには広い増加、要素差分には相殺と集中','',
        '主要18ペアの観測では、負側エネルギー増加は少数Headだけの現象ではない。一方、全Score要素が一様に変化したわけではなく、正負の要素差分の相殺後の純増分には上位数%の寄与が大きく効く。',
        '以下は保存済みCSVから取得した記述統計で、Headは独立標本ではない。人手確認済み成功やAttention増加の因果は主張しない。','',
        '## Q1：多数のHeadに共通するか','', '| 領域 | 平均ΔE_neg | 中央値 | 増加Head | 増加率 |','|---|---:|---:|---:|---:|']
    for r in main:
        x=d.loc[r];lines.append(f'| {r} | {x["mean"]:+.6f} | {x["median"]:+.6f} | {int(x.positive_count)}/1024 | {100*x.positive_fraction:.2f}% |')
    lines+=['','多数のHeadで増加し中央値も正、という観測は「一部の極端なHeadだけが平均を正にした」という説明と整合しない。ただし増加しないHeadもあり、効果量は一様ではない。出典：tables/head_distributions.csv、figures/head_distributions.png。','',
        '## Q2：少数Headの影響','', '| 領域 | 上位10%除外後の平均Δ | 上位10%が占める正のHead差分総量 |','|---|---:|---:|']
    for r in main:
        x=trim[(trim.region==r)&(trim.removed_fraction==.1)].iloc[0]
        lines.append(f'| {r} | {x.remaining_mean:+.6f} | {100*x.top_positive_delta_share:.2f}% |')
    lines+=['','上位除外は記述的感度で、別のHead群を用いた研究実験ではない。上位の寄与はあるが、除外後にも正の平均が残る。tables/head_tail_sensitivity.csv参照。','',
        '## Q3：負Scoreの個数か二乗平均か','', '| 領域 | Δp_neg（ポイント） | 割合項 | 二乗平均項 | ゼロ境界のΔE_neg寄与 |','|---|---:|---:|---:|---:|']
    for r in main:
        x=f.loc[r];lines.append(f'| {r} | {100*x.mean_delta_p_neg:+.6f} | {x.mean_frequency_contribution:+.6f} | {x.mean_magnitude_contribution:+.6f} | {x.zero_boundary_delta_E_neg_per_original_pair:+.8f} |')
    lines+=['','主に負Scoreの二乗平均の変化が寄与している。Causal UUとUOでは割合項が負で、二乗平均項が純増分を上回る。これは相殺であり100%超の帰属率は矛盾ではない。m2=0拡張に依存するゼロ境界の寄与は別記したとおり小さい。すべての負Scoreの絶対値が一様に増大したという意味ではない。tables/factor_overview.csv、figures/factor_decomposition.png。','',
        '## Q4：極端な要素が支配するか','',
        '| 領域 | 増加要素率 | 減少要素率 | 正寄与/N | 負寄与/N |','|---|---:|---:|---:|---:|']
    for r in main:
        x=f.loc[r];lines.append(f'| {r} | {100*x.mean_positive_element_fraction:.2f}% | {100*x.mean_negative_element_fraction:.2f}% | {x.mean_positive_contribution:+.6f} | {x.mean_negative_contribution:+.6f} |')
    lines+=['','約半分の位置に正の差分がある一方、負の差分も広く存在する。Head平均の増加と「全要素が増加する」は同義ではない。','',
        '| 領域 | 条件 | 全要素上位1%の負エネルギー割合 | 全要素上位10% |','|---|---|---:|---:|']
    for r in main:
        for kind in ('energy_gcg','energy_shuffled'):
            x=tail[(tail.region==r)&(tail.kind==kind)].set_index('fraction')
            lines.append(f'| {r} | {kind} | {100*x.loc[.01,"energy_weighted_share"]:.2f}% | {100*x.loc[.1,"energy_weighted_share"]:.2f}% |')
    lines+=['','これはmean(top/N)/mean(total/N)の割合で、定義される各Headの割合平均とは別。最も極端な1%だけがエネルギーの大半を占める状態ではない。Causalの名目1%は切り上げにより3/210要素である。','',
        '| 領域 | 上位数の基準 | 率 | 正寄与中の割合 | 元の純増分から差し引いた残差 |','|---|---|---:|---:|---:|']
    for x in tail[(tail.region.isin(main))&(tail.kind=='positive_delta')].itertuples():
        lines.append(f'| {x.region} | {x.basis} | {100*x.fraction:.0f}% | {100*x.energy_weighted_share:.2f}% | {x.net_delta_after_top_positive_removed:+.6f} |')
    lines+=['','純増分は正負相殺後の量なので、正寄与の一部を差し引くだけで符号が変わり得る。Full UU/UOは全要素上位5%で純増分が消える一方、Causal UUは同5%では正を保ち10%で負になる。したがって「ごく少数の異常値だけ」とは言えないが、「広く一様で極端要素の影響はない」とも言えない。これは差分の記述的感度で、モデルやScoreへの介入結果ではない。tables/tail_overview.csv、figures/element_concentration.png、tail_sensitivity.png参照。','',
        '## Q5：サンプル・Layer・Queryの一貫性','', '| 領域 | 平均増加サンプル | 1サンプル除外平均の範囲 | 増加Layer数 |','|---|---|---|---:|']
    for r in main:
        x=sample[sample.region==r];l=loo[loo.region==r];z=layers[layers.region==r]
        lines.append(f'| {r} | {int((x["mean"]>0).sum())}/{len(x)} | {l.leave_one_sample_out_mean.min():+.6f}〜{l.leave_one_sample_out_mean.max():+.6f} | {int((z["mean"]>0).sum())}/32 |')
    lines+=['','UUは全18サンプルで平均増加するが、UOは符号が混在する。いずれも1サンプルを除いて平均の符号は変わらない。Layer・固定群では減少もあり一様ではない。']
    for r in ('UU_causal','UO'):
        x=q[q.region==r];a=x[x['query']<10].delta_mean.mean();b=x[x['query']>=10].delta_mean.mean()
        lines.append(f'{r}：平均増加は20 Query中{int((x.delta_mean>0).sum())}位置。前半10位置平均Δ={a:+.6f}、後半={b:+.6f}。Query等重み平均={w.loc[r,"equal_query_delta_mean"]:+.6f}、全有効要素平均={w.loc[r,"element_weighted_delta_mean"]:+.6f}。')
    lines+=['後半の変化が大きく、Causal UUでは後半のQueryほど可視Suffix数も多いため、全要素平均の方が大きくなる。UOは各サンプル内で各Queryの要素数が同じなので2種類の平均が一致する。figures/query_profiles.png、layer_query_heatmaps.png、fixed_group_query_profiles.png。','',
        '## Q6：FullとCausal、主要18と全20','', '| N | 領域 | 平均Δ | 中央値 | 増加Head |','|---|---|---:|---:|---:|']
    for x in dist[(dist.view_id=='all_heads')&dist.region.isin(main)].itertuples():lines.append(f'| {x.subset} | {x.region} | {x.mean:+.6f} | {x.median:+.6f} | {int(x.positive_count)} |')
    lines+=['','広いHead増加・二乗平均項が主という主要な結論はCausalでも全20でも維持され、未来Scoreだけの特徴ではない。ただし上位寄与を差し引く感度やサンプル・位置別の強弱は変わる。OOの差は全て0、UOはFull/Causalで一致。','',
        '## 未確認と次の候補',
        '位置を揃えた比較は同じToken内容の比較ではない。Score規模からAttention配分・実際の有害応諾・成功因果は説明しない。少数成功例を選んだ同一18件と既存固定群の探索であり、Headを独立標本とした有意性も主張しない。',
        '今後の候補は、固定した分布・集中度の定義を独立な開発データで再現する確認と、位置とToken内容を区別する確認。今回の範囲を超える解析や独立100件は実施していない。','',
        '## 監査',f'従来E_negとの最大差={validation["old_E_neg_max_error"]:.3g}、要因分解誤差={validation["midpoint_decomposition_max_error"]:.3g}。詳細はvalidation.json、interpretation.md、docs/gcg_negative_energy_attribution.md。']
    with (out/'findings.md').open('x') as h:h.write('\n'.join(lines)+'\n')
    summary=['# MTG向け：最重要の発見','',
        '1. **Headには広範な増加。少数の極端Headだけではない。**',
        '2. **負Score数の増加より、負Scoreの二乗平均増加が主因。**',
        '3. **要素差分は正負に分かれ、相殺後の純増分には上位数%が大きく効く。**',
        '4. **Causalでも成立するが、Query後半・Layer・固定群で強弱がある。**','',
        '| 領域 | 平均ΔE_neg | 中央値 | 増加Head | 上位10% Head除外後平均 |','|---|---:|---:|---:|---:|']
    for r in main:
        z=d.loc[r];x=trim[(trim.region==r)&(trim.removed_fraction==.1)].iloc[0]
        summary.append(f'| {r} | {z["mean"]:+.4f} | {z["median"]:+.4f} | {int(z.positive_count)}/1024 | {x.remaining_mean:+.4f} |')
    summary+=['','条件別エネルギーの上位1%は約3〜5%のエネルギー。一方、正の差分寄与の全要素上位5%を引くとFull UU/UOの純増分は消え、Causal UUでは正を保つ。平均だけでは、広がりと集中の両面を区別できない。',
              '詳細値・Q1〜Q6の回答・制約はfindings.md、元数値はtables/。図はhead_distributions → factor_decomposition → element_concentration → tail_sensitivity → query_profilesの順を推奨。',
              '自動拒否判定の選択済み少数例の探索。因果・有意性・人手確認済み成功一般への結論ではない。']
    with (out/'MTG_findings.md').open('x') as h:h.write('\n'.join(summary)+'\n')
    with (out/'README.md').open('x') as h:h.write('# 負側エネルギー増加の要因\n\nまず[結論とQ1〜Q6](findings.md)、[MTG要約](MTG_findings.md)を参照。\n\n定義と全数表は[詳細レポート](interpretation.md)、[tables](tables/)、[図](figures/)、[検証](validation.json)。\n再現手順はdocs/gcg_negative_energy_attribution.md。core/tail/query_values.npzは配列順を各manifestで管理。\n')
    source_files=[Path(__file__),ROOT/'utils/artifact_io.py',ROOT/'utils/gcg_shuffled_common.py']
    inputs=[out/'tables'/name for name in ('head_distributions.csv','factor_overview.csv','tail_overview.csv','head_tail_sensitivity.csv','sample_summary.csv','leave_one_sample_out.csv','layer_summary.csv','query_overview.csv','query_weighting.csv')]
    write_json(out/'report_manifest.json',dict(created_at=utc_now(),input_hashes={str(x):sha256_file(x) for x in inputs},source_hashes={str(x):sha256_file(x) for x in source_files},role='saved-table synthesis, no new metrics'))
    write_json(out/'manifest.json',dict(**m,finish=finish,validation=validation,core_manifest_sha256=sha256_file(out/'core_manifest.json'),finish_manifest_sha256=sha256_file(out/'finish_manifest.json'),
        reporting_manifest_sha256=sha256_file(out/'report_manifest.json'),source_audit={str(ROOT/'scripts/audit_gcg_negative_energy.py'):sha256_file(ROOT/'scripts/audit_gcg_negative_energy.py')}))
    print(out/'findings.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    report(p.parse_args().output_dir)
