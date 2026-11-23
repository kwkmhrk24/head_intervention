# AutoDAN FastChatペアデータセット

旧AutoDANのFastChat成功49件から、Original未置換のID 10を除いた48件を成功側に固定する。入力の元資料を更新する必要はない。構築処理はローカルtokenizerと保存済み応答だけを使用する。

## 実行

リポジトリのルートから実行する。出力には存在しないディレクトリを指定する。`--additional` を省略すると48件の固定データを照合し、現在の成立状況を出力する。複数ファイルは `--additional` を繰り返し指定する。

```bash
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/build_autodan_fastchat_pairs.py \
  --additional path/to/saved_candidates.jsonl \
  --output data/outputs/llama2/autodan/fastchat_pair_dataset_next
```

## 追加JSONL

1行に候補1件を記録する。必須フィールドは `candidate_id`、`original_id`、`prompt_text`、`source`、`generation_method`、`provenance.artifact`。IDは固定48件と照合する。`prompt_text` は評価テンプレートを適用する前の本文。空文字、型不正、範囲外のIDは `rejected_records.jsonl` に理由とともに保存する。

評価済みとして判定するには、次のフィールドを指定する。

| フィールド | 内容 |
|---|---|
| `evaluation_format` | `FastChat` |
| `evaluation_version`、`model_revision`、`evaluation_conditions` | 基準出力の `manifest.json` と一致する値。外部の評価記録が各条件を満たすことを確かめて記録する |
| `response`、`stored_label` | 保存済み応答と真偽値ラベル。どちらかが欠ける候補は未評価として扱う |
| `input_token_ids` | 実際の評価入力の完全なToken ID配列。FastChat形式での再トークン化結果と照合する |
| `provenance.evaluation_artifact` | 保存済み応答・入力・条件の参照先 |

`input_length` と `original_token_span` を提出した場合は再構成した値と照合する。条件の一致判定は保存されたメタデータに基づく。外部の評価記録そのものの真正性は参照元で確認する。

## ペア適格性と出力

旧候補では元のOriginal本文と大文字・小文字が異なる場合があるため、旧候補内に残っている実際の表記を固定し、追加候補にも一意に存在することを求める。Tokenスパンは0始まりの半開区間。Originalに重なるTokenを含め、文字境界をまたぐ場合はフラグを立てる。Original文字列の一致とToken ID列の一致は別に扱う。

保存応答に既存の43拒否文字列ルールを適用し、保存ラベルと照合する。ラベルはこのルール上の結果を表す。空応答、別条件、同じ入力での矛盾ラベル、重複は、理由を付けて記録する。追加の成功候補も保存し、ペアの成功側には固定48候補を使う。

各Originalの適格ペアを、Original Token ID列の一致、完全入力の絶対長さ差、開始・終了位置の絶対差の和、Token編集距離、候補IDの順で並べ、先頭を `selected=true` にする。適格な非選定ペアもCSVとJSONLに残す。

後続のAttention Score分析では、`pairs.jsonl` で選定済みの行を `(original_id, candidate_id)` で `candidates.jsonl` の `reference` または `evaluated` 行に結合し、固定済みの `input_token_ids` と `original_token_span` を入力として使える。来歴は `manifest.json`、不足状況は `original_status.csv`、代表ペアの長さ差・位置差は `pair_statistics.json` を参照する。
