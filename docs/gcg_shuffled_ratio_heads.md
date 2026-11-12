# 保存済みUU/OO RMS比のHead別可視化

対象はlegacy_selection20。主要18ペア・補助20ペアの全1024 Head。
既存`20_region_scores/rms_and_frobenius`のCSVだけを数値入力にする。
Score/NPZ/モデル/応答/独立100件は読み込まない。既存コード・成果物を上書きしない。

```bash
python -B scripts/analyze_gcg_shuffled_ratio_heads.py --output-dir NEW_DIRECTORY
python -B -m unittest tests.test_gcg_shuffled_ratio_heads
```

入力：metrics.csv、paired_differences.csv、summary_18/20.csv、head_maps_18/20.csv、
overview.csv、view_summary_18/20.csv、fixed_group_member_effects.csv。
既存output_hashes.jsonで入力CSVを事前検証し、入力および旧CSV/PNG/レポート/manifestの
実行前後のハッシュも比較する。生Scoreはハッシュ確認のためにも読み込まない。

各サンプル・Headの保存済み`R_UU_OO_full/causal`を使用する。CausalはUUとOOの両方を
可視要素に限定した指標。比率は無次元で、平均RMS同士の比ではない。差はGCG−Shuffled。
既存summaryの条件別平均・差分平均/中央値/SD・増減件数・Head map・overviewとの
一致をrtol=atol=1e-12で検証する。Head/サンプルの対応は完全なキー集合で確認する。

増減判定は既存のatol+rtol*max(abs(G),abs(S))、atol=rtol=1e-12。
全Headの増加件数は各Headのサンプル平均差の符号。厳密な正負件数も別保存する。
`gcg_head_mean_gt1`は各HeadのGCGサンプル平均比率が厳密に1を超えること。
全サンプルで1を超えることではない。各Headの超過サンプル数も保存する。

方向一致率は以下を区別する。

- positive_fraction / negative_fraction / equal_fraction：各方向のサンプル数/n。
- mean_direction_agreement：Headのサンプル平均差と同じ方向のサンプル数/n。
  平均が同等なら同等サンプル数/n。
- majority_nonzero_fraction：max(増加数,減少数)/n。平均方向と多数派方向は一致するとは限らない。

Layerは32 Headの等重み平均。既存固定12群は名称・順序・各5メンバーを維持。
Layerと固定群は構成が異なる集団なので、同一集団の推定とは解釈しない。
上位10% Head除外は103 Headを除く記述的感度であり、新しい重要Head群の選択ではない。

出力はtables/heads.csv、overview.csv、layers.csv、samples.csv、fixed_groups.csv、
fixed_group_members.csv、full_causal_head_comparison.csv、layer_vs_fixed_groups.csv。
既存にない散布図・ヒストグラム・方向一致分布・Layer平均を300dpi日本語PNGで保存。
複合図と各条件の単独図を用意し、Causal固定群図も保存する。既存のFull固定群図と
Full/Causal Headヒートマップ・サンプル図は再生成せずexisting_figures.jsonから参照する。
manifest・validation・report・output_hashesを保存する。ハッシュ台帳自体は台帳に含めない。

Headを独立した1024標本として検定しない。同じ選択済み18件・旧固定群での探索であり、
実際の有害応諾、人手確認済み成功、因果・一般化を主張しない。OO不変時の比率差は
各サンプル・HeadのΔRMS_UU/RMS_OOという正規化で、独立の機構的証拠ではない。
