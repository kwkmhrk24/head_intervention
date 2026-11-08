"""Japanese interpretation of saved attribution results; numbers from tables."""
import numpy as np
import pandas as pd
from utils.artifact_io import read_json

MAIN=('UU_full','UU_causal','UO')


def write_report(out):
    t=out/'tables';m=read_json(out/'core_manifest.json');f=read_json(out/'finish_manifest.json')
    distributions=pd.read_csv(t/'head_distributions.csv');d=distributions[(distributions.subset==18)&(distributions.view_id=='all_heads')].set_index('region')
    factors=pd.read_csv(t/'factor_overview.csv');a=factors[(factors.subset==18)&(factors.view_id=='all_heads')].set_index('region')
    tails=pd.read_csv(t/'tail_overview.csv');tails=tails[(tails.subset==18)&(tails.view_id=='all_heads')]
    sensitivity=pd.read_csv(t/'head_tail_sensitivity.csv');sensitivity=sensitivity[sensitivity.subset==18]
    sample=pd.read_csv(t/'sample_summary.csv');sample=sample[sample.subset==18]
    weighting=pd.read_csv(t/'query_weighting.csv');weighting=weighting[(weighting.subset==18)&(weighting.view_id=='all_heads')]
    q=pd.read_csv(t/'query_overview.csv');q=q[(q.subset==18)&(q.view_id=='all_heads')]
    lines=['# 負側Scoreエネルギー増加の要因分析','',
        '## 条件・定義',
        '基準2d48477、旧legacy_selection20の同じペア。主分析primary18、補助20。全1024 Head、既存固定Head群、Token境界を維持。FP32保存ScoreをCPU/FP64で分析し、新規推論・GPU・Suffix・ASR・独立100件は使わない。',
        'E_neg=mean(min(S,0)^2)：分母は領域内の全有効要素数。Full UUは未来Keyを含む探索的Score、CausalはKey<=Queryで対角を含む。UOは全要素が過去方向。OO Full/Causalは不変確認用。',
        '特徴は各サンプル・Head・領域で計算した後に等重み集計。Headの分布は18サンプル平均差をHead間で比較し、Headを独立標本として有意性を主張しない。正負数は厳密な>0/<0/=0で、絶対1e-12以内の数値的同等数も別に保存。',
        '負Scoreなしの場合、条件付きm2_negは未定義NaN。要因分解だけは0へ拡張して恒等式を成立させる。この境界の帰属は規約依存なので、両条件に負Scoreがある場合とゼロ境界の場合を元のペア数を分母として分ける。',
        '要因分解は Δp×(m2_G+m2_S)/2 + Δm2×(p_G+p_S)/2 の和。再送式の第2項のマイナスは恒等式と整合しないため、最初の指示どおりプラスを使用。',
        '集中度：各条件は全有効要素数の上位1/5/10%。正のdelta_eでは正要素数基準(active)と全要素数基準(all)の両方。k=ceil(割合×基準要素数)。総寄与が0なら未定義で理由・件数を保存。',
        '総量で重み付けした集中度=mean(top energy/N)/mean(total energy/N)と、定義されるHead・サンプルでの集中度平均を別々に保存する。前者も先に各サンプル/HeadでN正規化し、Original長による重み付けをしない。',
        '位置の対応はToken内容の対応ではない。Suffixは同じ集合でも順序が異なる。集中度や二乗エネルギーからAttention割合・攻撃成功の因果を主張しない。',
        '', '## Q1 多数のHeadに共通するか（主要18件）','',
        '| 領域 | 平均ΔE_neg | 中央値 | Head間SD | 増/減/同等Head | 増加率 | q05 / q25 / q75 / q95 |',
        '|---|---:|---:|---:|---|---:|---|']
    for region in MAIN:
        r=d.loc[region];lines.append(f'| {region} | {r["mean"]:+.6f} | {r["median"]:+.6f} | {r["std"]:.6f} | {int(r.positive_count)}/{int(r.negative_count)}/{int(r.equal_count)} | {100*r.positive_fraction:.2f}% | {r.q05:+.4f} / {r.q25:+.4f} / {r.q75:+.4f} / {r.q95:+.4f} |')
    lines+=['','観測：head_distributions.csvに最小/最大・全分位点、head_summary_18/20.csvに各Headの分布を保存。過半数かどうかと変化量の集中は別問題である。','',
        '## Q2 一部のHeadが平均を押し上げるか','',
        '| 領域 | 除外上位Head率 | 除外数 | 元の平均 | 残りHead平均 | 上位が占める正のHead差分総量の割合 |','|---|---:|---:|---:|---:|---:|']
    for r in sensitivity[sensitivity.region.isin(MAIN)].itertuples():
        lines.append(f'| {r.region} | {100*r.removed_fraction:.0f}% | {r.removed_heads} | {r.original_mean:+.6f} | {r.remaining_mean:+.6f} | {100*r.top_positive_delta_share:.2f}% |')
    lines+=['','これは記述的な感度確認で、新しいHead群を選んだ実験ではない。残りの平均が正でも、全Headが一様に増えることや極端なHeadの影響がないことを意味しない。固定12群と45個別Headの結果はfixed_view_summary_18/20.csv。','',
        '## Q3 個数か絶対値か','',
        '| 領域 | Δp_neg | Δm2(0拡張) | 割合項 | 二乗平均項 | ゼロ境界ペア数 | ゼロ境界ΔE_neg（元ペア数で正規化） |','|---|---:|---:|---:|---:|---:|---:|']
    for region in MAIN:
        r=a.loc[region];lines.append(f'| {region} | {r.mean_delta_p_neg:+.8f} | {r.mean_delta_m2_zero_extension:+.6f} | {r.mean_frequency_contribution:+.6f} | {r.mean_magnitude_contribution:+.6f} | {int(r.zero_boundary_pairs)}/{int(r.total_pairs)} | {r.zero_boundary_delta_E_neg_per_original_pair:+.6f} |')
    lines+=['','factor_overview.csvには両条件のp_neg、m2の0拡張値、定義可能ケースだけの条件付きm2平均と件数、両条件負あり/ゼロ境界それぞれの寄与を保存。割合項と二乗平均項は符号付きで相殺し得る。寄与/純増分が100%を超えても誤りではない。単位はエネルギー・寄与がScore²、p_negは無次元。','',
        '## Q4 極端なScore要素に集中するか','',
        '| 領域 | 対象 | kの基準 | 名目上位率 | 定義可能ケース平均 | エネルギー重み付き割合 | 平均k | 実際の全要素選択率 | 純増分から上位正寄与を引いた残差 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in tails[tails.region.isin(MAIN)].itertuples():
        residual=getattr(r,'net_delta_after_top_positive_removed',np.nan)
        lines.append(f'| {r.region} | {r.kind} | {r.basis} | {100*r.fraction:.0f}% | {100*r.defined_case_mean_share:.2f}% | {100*r.energy_weighted_share:.2f}% | {r.mean_top_k:.2f} | {100*r.mean_selected_fraction_of_valid:.3f}% | {residual:+.6f} |')
    lines+=['','各条件のエネルギー集中と、差分の正寄与の集中は別の量である。後者では負の差分寄与も存在し、純増分は正負の相殺後の値となる。残差は元の要素数を保った記述的な差分感度であり、Scoreを書き換えた介入実験ではない。',
        '切り上げにより、Causal UUの名目1%は3/210=約1.43%の要素となる。positive_deltaのactive基準ではさらに実際の選択率が異なる。tail_values.npzにサンプル/Headごとのk・基準数・集中度・総量・上位量を保存。未定義はundefined_tail_cases.csv参照。','',
        '## Q5 サンプル・Layer・Queryの一貫性','',
        '| 領域 | 平均が増加したサンプル | サンプル間平均差の最小/最大 | 増加Head率の最小/最大 |','|---|---|---|---|']
    for region in MAIN:
        z=sample[sample.region==region]
        lines.append(f'| {region} | {int((z["mean"]>0).sum())}/{len(z)} | {z["mean"].min():+.6f} / {z["mean"].max():+.6f} | {100*z.positive_fraction.min():.2f}% / {100*z.positive_fraction.max():.2f}% |')
    lines+=['','leave_one_sample_out.csvは特定1サンプルへの依存を確認するための記述統計。Layer別はlayer_summary.csv、Head別Query変化はquery_head_summary_18/20.csv、固定群はquery_overview.csvに保存。',
        'Causal UUのQueryごとのE_negは可視Suffix数1〜20で平均する。各Query等重み平均と、可視要素数で重み付けした210要素平均を混同しない。UOでは各サンプル内の全20 Queryが同じOriginal数を参照するため一致する。','',
        '| 領域 | 20 Query等重みΔ | 全有効要素平均Δ | 差 |','|---|---:|---:|---:|']
    for r in weighting.itertuples():lines.append(f'| {r.region} | {r.equal_query_delta_mean:+.6f} | {r.element_weighted_delta_mean:+.6f} | {r.difference:+.6f} |')
    for region in ('UU_causal','UO'):
        z=q[q.region==region];lines.append(f'{region}：20位置中{int((z.delta_mean>0).sum())}位置で平均増加。前半10位置の平均差{z[z["query"]<10].delta_mean.mean():+.6f}、後半10位置{z[z["query"]>=10].delta_mean.mean():+.6f}。')
    lines+=['','## Q6 FullとCausal・全20件の補助確認','', '| N | 領域 | 平均Δ | 中央値 | 増加Head率 |','|---|---|---:|---:|---:|']
    for r in distributions[(distributions.view_id=='all_heads')&distributions.region.isin(MAIN)].itertuples():
        lines.append(f'| {r.subset} | {r.region} | {r.mean:+.6f} | {r.median:+.6f} | {100*r.positive_fraction:.2f}% |')
    lines+=['','## 観測・解釈・未検証の区別',
        '上の数値と恒等式は観測・計算上の事実。多数のHeadに増加があっても、効果量は偏り得る。中央値・Head上位除外感度・要素上位寄与・符号付き残差を組み合わせて、広がりと集中を別々に評価する。',
        '負側エネルギーは符号を落としたScore規模であり、Attention増加そのものではない。Fullの未来Scoreを情報流と解釈しない。今回はSoftmax分析や最終Queryの説明を追加していない。',
        '自動拒否表現判定による選択済み18件と同じ固定群を用いた探索で、人手確認済み有害応諾、因果、有意性、別攻撃・モデルへの普遍性は主張しない。',
        '次の候補は固定条件で独立な開発データへ分布/集中の再現性を確認すること。独立100件は今回使用しない。','',
        '## 保存と再現',
        '各サンプル/Headの因子はtables/condition_metrics.csvとpaired_metrics.csv。条件付きm2の未定義理由はzero_negativeフラグ。巨大な要素別CSVを複製せず、元Scoreの参照・ハッシュとcore/tail/query_values.npzから再現する。',
        '```bash',f'python -B scripts/analyze_gcg_negative_energy.py --output-dir NEW_RUN --stage core',
        'python -B scripts/analyze_gcg_negative_energy.py --output-dir NEW_RUN --stage finish --tail-policy both','```',
        '新規runのみ。core_manifest/finish_manifestにsource/input hash・モデル情報・ID・Head群・定義を保存。validation.json、各stage検証、output_hashes.jsonを参照。']
    with (out/'interpretation.md').open('x') as h:h.write('\n'.join(lines)+'\n')
    mtg=['# MTG要約：負側エネルギーの広がりと集中','',
         '同じ主要18ペア・全1024 Head、既存固定群。CPU/保存Scoreのみ。','',
         '| 領域 | 平均Δ | 中央値 | 増加Head数 | 割合項 | 二乗平均項 |','|---|---:|---:|---:|---:|---:|']
    for region in MAIN:
        z=d.loc[region];r=a.loc[region]
        mtg.append(f'| {region} | {z["mean"]:+.4f} | {z["median"]:+.4f} | {int(z.positive_count)}/1024 | {r.mean_frequency_contribution:+.4f} | {r.mean_magnitude_contribution:+.4f} |')
    mtg+=['','集中度と上位除外感度はinterpretation.mdのQ2/Q4表、図element_concentration/tail_sensitivity参照。条件別エネルギーの集中と正の差分寄与の集中は分ける。',
          '主要18と全20、FullとCausal、各Query等重みと全要素平均の違いを保持。ゼロ負Scoreの条件付き二乗平均は未定義で、分解だけ0拡張する境界規約を明示。',
          '図の推奨順：head_distributions → factor_decomposition → element_concentration → tail_sensitivity → query_profiles。',
          'Headは独立標本ではなく、Score規模からAttentionや成功因果を断定しない。詳細値・定義・制約・再現方法はinterpretation.mdとmanifest。']
    with (out/'MTG_summary.md').open('x') as h:h.write('\n'.join(mtg)+'\n')
