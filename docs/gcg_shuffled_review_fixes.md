# GCG vs Shuffled：5a5fe74レビュー修正

研究条件・入力・Head・Mask・集計・数値演算は変更しない品質修正。

## CPU依存

相関関数を取得するための過去スクリプトimportがtorchを連鎖importしていた。
`utils/gcg_shuffled_statistics.py`は元のPearson/Spearman関数を演算順序も含めて保持する。
過去の`analyze_attention_head_metric_relationships.py`（tensorロード機能を含む）は変更しない。
CPU入口のIO/集計は`gcg_shuffled_common.py`を直接利用する。
CPU分析はNumPy/Pandas/Matplotlibで動作し、torch/transformersを要求しない。

旧CPUテストの「モデルロード禁止」mockもtransformers自体を要求していたため、
torch/transformersのimportを失敗させる強い検証へ置き換えた。入力検証のassertは維持する。

## レポート

`utils/gcg_shuffled_reporting.py`はCSV/manifestを検証し、数値・方向件数・固定群・
18/20ペアの結果と単位を読み取る。群はmanifestの名前・種類・5 Head対応を確認する。
必要列・値・キーが欠損/重複/非有限の場合、また条件順や件数が不正な場合はエラーにする。
結論文も保存値に応じて変化し、過去の正負・件数をハードコードしない。

既存`--root`/`--run-id`とbuild(root, run_id)は維持し、新規の`--output-dir`を追加。
既存レポートは上書きせず、入力元と保存先を分ける。
report_manifest.jsonに入力CSV/manifest/使用profileとレポート/描画/IOソースのSHA256を記録する。
相関CSVを再計算したと偽らず、入力相関CSVのハッシュと元分析のsource_hashesを保存する。
分析runのmanifestには実際に使用するCPU相関/描画コードのハッシュを記録する。
互換性監査CLIも監査自身とIOコードのハッシュを出力する。

```
python -B scripts/report_gcg_shuffled_phase.py \
  --root data/outputs/gcg_vs_shuffled/legacy_selection20 \
  --run-id 20261107T165012Z --output-dir NEW_REPORT_DIRECTORY
```

## 描画

fixed_head_distributions.pngのraw/weight両方で、保存済みquery_positions/visibleを検証し、
Key<=Queryの可視位置だけを描く。未来KeyのRaw値やSoftmax値を変更せず描画対象だけを制限する。
凡例・軸でRaw pre-softmaxとcausal Softmax Attention fractionを区別する。

## 検証

新規tests/test_gcg_shuffled_review.py：値/件数/固定群変更が本文とMTG要約へ反映されること、
不正schema/対応の検出、方向に応じた結論、単位、相関関数AST一致、tensor import禁止でのCLI、
保存Maskと実際の描画線の未来Key除外を検証する。

```
python -B -m unittest tests.test_gcg_shuffled_four_metrics tests.test_gcg_shuffled_tokens \
  tests.test_gcg_shuffled_regions tests.test_gcg_shuffled_signed_means \
  tests.test_gcg_shuffled_additional tests.test_gcg_shuffled_compatibility \
  tests.test_gcg_shuffled_review
```

CPU用のNumPy/Pandas/Matplotlibと依存パッケージだけを置き、
`importlib.util.find_spec('torch') is None`、transformersもNoneの隔離venvで、
上記CPUテストとadditional/audit/reportの--helpを実行する。
通常環境では既存pilot/comparison/full_attention_scoresのテストも併せて実行する。

検証成果物は`data/outputs/gcg_shuffled_review/review_5a5fe74/`（Git対象外）。
baseline_hashes.json、no_torch_environment.json、no_torch_tests_fixed.log、full_tests_final.log、
compatibility_*.json、report_compatibility.json、figure_compatibility.json、final_audit.jsonを参照。
変更前レポートはbefore_report、修正後はreport_after_final。
csv_mutation_input/report_csv_mutationは意図的にCSVを書き換えたレポート専用テストfixtureで、研究結果ではない。

全20件を旧4 CLI・追加分析で再実行し、既存の各サンプル/Head、Full/Causal OO/UU/UO、
Suffix20/Closing4、18/20集計、固定群/個別/全HeadのCSV/NPZを厳密比較する。
描画範囲を変更した1図だけは意図的差分として数値互換性から分離する。
元のレポート表30行・派生CSV3個・中心化診断図の一致も確認する。
入力や既存成果物は移動・削除・上書きしない。GPU/推論/独立100件は使用しない。
