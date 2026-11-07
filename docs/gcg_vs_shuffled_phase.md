# GCG vs Shuffled実験フェーズの案内

統一入口は `data/outputs/gcg_vs_shuffled/legacy_selection20/README.md`。
機械可読一覧はindex.json、run一覧はruns.csv、コード依存はcode_inventory.csv、
画像一覧はfigures.mdとローカル静的figures.html。成果物はGit対象外。

## 範囲と既存入口

| 段階 | CLI / 主なユーティリティ | 旧成果物（そのまま残す） |
|---|---|---|
| pilot/比較入力 | evaluate_shuffled_gcg_pilot.py / build_shuffled_gcg_comparison.py | shuffled_gcg_pilot / shuffled_gcg_comparison |
| Score抽出 | extract_full_attention_scores.py / full_attention_scores.py | full_attention_scores |
| 4指標 | analyze_gcg_shuffled_four_metrics.py / gcg_shuffled_metrics.py | gcg_shuffled_four_metrics |
| Token/固定Head/終了4 Query | analyze_gcg_shuffled_tokens.py / gcg_shuffled_token_metrics.py / gcg_shuffled_token_plots.py | gcg_shuffled_token_analysis |
| Region Norm/RMS | analyze_gcg_shuffled_regions.py / gcg_shuffled_region_metrics.py / gcg_shuffled_region_plots.py | gcg_shuffled_region_analysis |
| Signed Mean | analyze_gcg_shuffled_signed_means.py / gcg_shuffled_signed_means.py | gcg_shuffled_signed_means |
| 正負分解/同一Query/中心化/統合 | analyze_gcg_shuffled_additional.py / gcg_shuffled_additional_metrics.py / gcg_shuffled_additional_plots.py | 統一階層30/40/50/60の新規run |

scripts/、utils/、tests/test_*.py、各docs/gcg_shuffled_*等を棚卸しする。
pilot・生成・抽出のコードは履歴・依存の確認のみで変更・実行しない。
他の攻撃/モデル/介入/防御フェーズや、共有import先は対象外としてindex.jsonに記録。

## 統一出力階層

```
gcg_vs_shuffled/legacy_selection20/
  00_inputs/{comparison_dataset,raw_attention_scores,pilot5_reference}  # 旧runへの相対リンク
  10_existing_attention/{four_metrics,token_position}                  # 相対リンク
  20_region_scores/{rms_and_frobenius,signed_mean}                      # 相対リンク
  30_sign_decomposition/<run_id>/
  40_query_softmax/<run_id>/
  50_row_centered/<run_id>/
  60_integrated_analysis/<run_id>/
  90_validation/<run_id>/
```

元Scoreは複製しない。50は40と同じQueryの表・NPZを相対リンクし、中心化図・説明を別置する。
旧パスは全て存続。カタログは既存ディレクトリ/リンクを上書きしない初回作成専用。
将来のカタログ改訂は別パスに版管理して作る。各runのmanifestとhashで対応を確認する。

## 追加指標

主分析は既存primary18、補助はselection20全20ペア。差はGCG−Shuffled。
32 Layer×32 Head、固定群・Token境界・入力IDを維持し、新規Head選択や推論はしない。
FP32保存ScoreをFP64でCPU計算する。

- 正負分解：E_pos=mean(max(S,0)^2)、E_neg=mean(min(S,0)^2)。分母は全有効要素数。
  mu_pos/mu_negの和がSigned Mean、E_pos/E_negの和が各Head/sampleのRMS²。
  正負割合、中央値、5/25/50/75/95%点（linear補間）も保存する。平均分位点は混合分布分位点ではない。
- Fullは未来要素を含む全要素。Causalは絶対Key<=Queryで対角含む。
  OOはm²/m(m+1)/2、UUは400/210、UOは20*mで両policy同じ。
- 同一Query：Suffix20位置と既存終了4 Query。各Query/Headの可視Key全体を分母とする安定Softmax。
  M_U/M_O/D/条件付きM_U=M_U/(M_U+M_O)、prefix/other割合・Scoreを保存する。
  Suffix位置の一致はToken内容の一致ではない。suffix_mean20は各Queryの指標を計算後の平均。
- Row中心化：全可視Keyの平均を引き、SoftmaxとDの不変性を検証する。
  raw/centeredのOriginal/Suffix平均と差、row_mean、領域logsumexpを保存する。
- suffix Queryにはother可視Keyがなくraw_otherはNaN、valid count=0と明示。Attention割合は0。
  存在しない領域の値を補完しない。

計算順序はサンプル/Head/Queryの特徴計算→等重みHead群平均→サンプル集計。
差SDはddof=1。Headは独立標本として扱わない。RMS、Signed Mean、正負エネルギー、
Region Queryと最終Query、全Key割合と条件付き割合を区別する。
自動拒否判定で条件づけられた同じ18件による探索で、人手確認済み成否・因果・有意性・一般化は主張しない。

## 共通化と互換性

`utils/gcg_shuffled_common.py`は旧関数本体を保持して入力検証/IO/領域/Mask/Head対応/
等重みview/ペア集計を共有する。旧moduleのimport名は再exportされる。
`utils/gcg_shuffled_plot_common.py`は既存Region図の設定・finishを共有し、旧exportも保持する。
既存4 CLIの引数と出力CSVの列/並び/意味は維持。共通依存のハッシュをmanifestに追加する。
数値演算の順序・許容誤差・Head群は変更しない。

90_validationに変更前コード・ハッシュ・テスト、旧4 CLIと新規分析の変更前後再実行を保存。
既存CSV/NPZとの高精度比較、変更前後のschema/dtype/shape/行順/ID/NaN/Mask/要素数/
全20件・18/20集計・固定群・個別Headを監査する。
整数・文字列・選択・Maskは完全一致、浮動小数はrtol=atol=1e-12（最大誤差を全field保存）。
日時・Git状態・新規source hash等のprovenance差と、科学的な値の差を区別する。

## 再現コマンド

```bash
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_additional.py --run-id NEW_RUN
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/audit_gcg_shuffled_compatibility.py --before BEFORE --after AFTER --output NEW_AUDIT.json
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/index_gcg_shuffled_phase.py --root NEW_CATALOG_ROOT --run-id RUN
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_shuffled_gcg_pilot tests.test_shuffled_gcg_comparison tests.test_full_attention_scores tests.test_gcg_shuffled_four_metrics tests.test_gcg_shuffled_tokens tests.test_gcg_shuffled_regions tests.test_gcg_shuffled_signed_means tests.test_gcg_shuffled_additional tests.test_gcg_shuffled_compatibility
```

全出力は未使用パスに作る。新規分析は--output-rootで保存階層を明示可能。
indexスクリプトは指定rootに新規stage/runが既に存在することを要求する。
モデルロード、Forward、ASR再評価、Suffix生成、GPU、独立100件は実施しない。
