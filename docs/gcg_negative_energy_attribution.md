# 負側Scoreエネルギーの広がり・集中分析

legacy_selection20の既存primary18/全20ペア、全1024 Head、従来固定群を維持。
基準2d48477。CPUで保存FP32 ScoreをFP64分析する。モデル/GPU/ASR/介入/独立100件は対象外。
入力対応・ハッシュ・領域・Mask・固定view集計は既存gcg_shuffled_common等を再利用する。

## 定義

E_neg=mean(min(S,0)^2)、分母は全有効要素。UU Full=400、Causal=210、
UO=20*mで全て過去、OO Full/Causalは不変確認用。HeadはLayer*32+Headの0始まり。
同位置比較はToken内容の対応ではない。Fullの未来要素を情報流と解釈しない。

- A：サンプル平均差のHead分布、厳密な>0/<0/=0、Layer/固定群/サンプル集計。
  数値的同等abs<=1e-12も別に保存。上位Headを除く感度は記述診断であり新規Head群ではない。
- B：p_neg×m2_neg。負Scoreなしの条件付きm2はNaN、分解に限り明示的0拡張。
  Δp×平均m2 + Δm2×平均p = ΔE。ゼロ境界と両条件負ありの寄与を元のペア数で分ける。
  m2条件付き平均は定義ケース数も記録し、NaNを無言で0補完しない。
- C：条件別集中度は全有効要素数を基準。正のdelta_e集中度は正要素数(active)と
  全有効要素数(all)を別に保存。k=ceil(rate*basis_count)。総量0なら未定義。
  全体集計は定義ケース平均とmean(top/N)/mean(total/N)の2種類を区別。
  残差は元の分母Nを保って上位正寄与を差し引く記述的感度であり、モデル介入ではない。
- D：Suffix20 QueryのCausal UU/UO。Queryごとの特徴計算後の等重み平均と、
  有効Key数で重み付けした領域平均を分ける。Layer・固定群・全Head別の値も保存。

寄与率は正負相殺で100%超になり得る。ScoreエネルギーはAttention割合ではない。
同じ18件で選択された固定群と自動判定ラベルの探索で、独立標本・有意性・因果は主張しない。

## 実行

```bash
python -B scripts/audit_gcg_negative_energy.py --output-dir NEW_RUN --snapshot
python -B scripts/analyze_gcg_negative_energy.py --output-dir NEW_RUN --stage core
python -B scripts/analyze_gcg_negative_energy.py --output-dir NEW_RUN --stage finish --tail-policy both
python -B scripts/audit_gcg_negative_energy.py --output-dir NEW_RUN
python -B scripts/report_gcg_negative_energy.py --output-dir NEW_RUN
python -B -m unittest tests.test_gcg_negative_energy
```

NEW_RUNは統一階層70_negative_energy_attribution/<run_id>の新規場所。
各stageは既存の主要出力を上書きしない。実行ログは終了時まで変化するため、
output_hashesから除外する。最終監査・追補報告には別のhash/manifestを保存する。

## 保存構造

- condition_metrics.csv：サンプル/条件/領域/Head、p_neg、E_neg、m2、負数個数。
- paired_metrics.csv：対応差、2項分解、正負/同等要素数と割合、正負寄与の総量とN正規化値。
- head_summary_18/20、condition_head_summary、fixed_view_summary：各Head/固定群の集計。
- head_distributions、sample_summary、layer_summary、head_tail_sensitivity、leave_one_sample_out。
- factor_overview：両条件の因子と寄与、ゼロ境界の規約依存性を含む。
- tail_values.npz：sample,region,Head,tail_spec。集中度、k、基準数、総量/N、上位量/N。
- query_values.npz：条件はsample,condition,region,Head,Query,field、差分はcondition軸なし。
- tail_head_summary、tail_overview、undefined_tail_cases、query_head_summary、query_overview等。
- core_manifest/finish_manifest：フィールド順・ID・入力/source hash・モデル条件。
- interpretation/MTG_summary、figures、validation、最終manifest/output hash。

巨大な要素別CSVや元Scoreの複製を避ける。要素ごとの値は元の不変ScoreとMaskから再現可能。
分母が0の集中度、負Scoreなしの条件付きm2、定義ケース数<2のSDは未定義として保存する。
Head方向のSDは1024 Headの記述であり、1024独立標本の推論ではない。

旧READMEは元内容を保存したうえで末尾に新runへの案内だけを追記する。
数値・画像・旧runのmanifestは変更しない。README追記のハッシュ差だけを明示的に監査する。

## 解釈の修正版とレポート監査

上位1/5/10%は**各サンプル・各Head・各領域内**で選び、その後に集計する。
全サンプル・全Headをプールした上位要素ではない。条件別エネルギーはall、
正の差分寄与はall（全有効要素数）とactive（正寄与要素数）の両方を保持する。
正負の要素差分が相殺されるため、上位正寄与を除くと純増分の符号が変わり得る。
これは純増分の感度であり、その事実だけで要素自体の集中を断定しない。
集中度は既存tables/tail_overview.csvの割合と別に評価する。

既存runを上書きせず、レポートのみ別ディレクトリへ生成する：

```bash
python -B scripts/report_gcg_negative_energy.py --output-dir EXISTING_RUN --report-dir NEW_REPORT_DIR
python -B scripts/audit_gcg_negative_energy.py --output-dir EXISTING_RUN --verify-report NEW_REPORT_DIR
python -B -m unittest tests.test_gcg_negative_energy_reporting tests.test_gcg_negative_energy
```

生成前に全参照CSVとcore/finish manifestを、既存の監査済みoutput hashと照合する。
final_output_hashes.jsonがあれば優先し、なければoutput_hashes.jsonを使う。
元runのハッシュ台帳・参照入力・実行ソースのハッシュと元レポートの対応先は
新しいreport_manifest.jsonに保存する。新ディレクトリのREADMEから元の表・図へ参照する。
生成後はreport_hashes.jsonが最終Markdown3件とmanifest2件を検証する。
台帳自身は含めず、manifestから台帳への逆参照もしないため循環しない。
検証は入力・ソース・監査元台帳も再照合し、不一致ならValueErrorとする。
ハッシュは変更検出用であり、台帳と全ファイルの同時改ざんを防ぐ電子署名ではない。
既存の数値・図・レポートは不変で、追加開発や新規分析は実施しない。
