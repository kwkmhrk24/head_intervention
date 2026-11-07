# Signed Mean Score追加分析

既存Region run `legacy_selection20_20261107T094154Z` のprimary18、all20、
Token範囲、固定Head群を再利用するCPU分析。既存Scoreを変更せず、モデル処理はしない。
各サンプル・Head・条件でOO/UU/UOの生ScoreをFP64に変換し、有効要素の
符号付き算術平均を取る。二乗・絶対値化・クリッピングはしない。
Fullは全要素、Causalは絶対Key位置<=Query位置（対角含む）。
マスク位置に-infを挿入せず、有効要素だけを平均する。

集計順序は領域内平均→固定Head群内の等重み平均→サンプル統計。
ペア差はGCG−Shuffled、SDはサンプル間の差のddof=1。
Head別および群内メンバー別集計、全1024 Head平均と各サンプルの結果を保存する。
既存paired_statsの数値同等判定を利用する。Head平均差の方向数は絶対1e-12で判定する。
OOは全Headで完全一致を要求。Full/Causal要素数は既存Region CSVと照合する。
既存RMSは再計算せずNPZの対応値を参照する。

出力：metrics.csv（40,960行）、paired_differences.csv（20,480行）、
head_summary_18/20.csv、view_summary_18/20.csv、overview.csv、
sample_summary.csv、fixed_group_member_effects.csv、rms_comparison.csv、
element_counts.csv、signed_means.npz（sample, condition G/S, layer*32+head, metric）、
5枚の図、manifest/validation/output_hashes.json、report.md。
manifestには入力ID・範囲・Scoreハッシュと固定Headの定義を保存する。

```bash
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_gcg_shuffled_signed_means tests.test_gcg_shuffled_regions tests.test_gcg_shuffled_tokens tests.test_gcg_shuffled_four_metrics
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_signed_means.py --output-dir data/outputs/gcg_shuffled_signed_means/NEW_RUN
```

NEW_RUNは新規パス。既存データ・成果物を上書きしない。
RMSとSigned Meanは異なる指標である。いずれからも正負分布全体、Softmax変化の理由、
攻撃成功の因果・一般性は確定しない。Region集計と最終Queryの割合を同一視しない。
Headは独立標本でなく、既存Head群の選択・確認に同じ18件を使う探索的分析。
正負別エネルギー・新規Softmax分析は実装しない。コミット・pushはユーザー確認後。
