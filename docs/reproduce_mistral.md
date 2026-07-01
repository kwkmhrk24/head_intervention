# Mistral Reproduction: Phase 2 Preparation

This phase only checks the tokenizer and the availability of Mistral-specific datasets. It does not run AIE computation, intervention-vector extraction, ASR evaluation, or attention-score decomposition.

## Model path

```bash
MODEL=/mnt/mp_nas_mks/labmember/h.kawakami/models/Mistral-7B-Instruct-v0.2/
```

## Inspect the target token

The current Mistral target token ID in `utils/intervention_utils.py` is `12875`. Inspect its decoded form and compare the tokenization of `Sure`, space-prefixed `Sure`, and newline-prefixed `Sure` without loading the model itself:

```bash
python scripts/inspect_target_token.py \
  --model "$MODEL" \
  --strings "Sure" " Sure" "\nSure"
```

The script expands the CLI sequence `\n` to an actual newline. It reports tokenization both with and without special tokens, decodes each token separately, and compares the chat-template suffix before and after appending one ASCII space.

Do not decide whether to pass `--append_target_space` until these results have been inspected.

## Check for Mistral datasets

List likely Mistral, attack, original, jailbreak, and refusal files:

```bash
find data -maxdepth 5 -type f | grep -Ei "mistral|autodan|gcg|jailbreak|original|refusal"
```

Check whether these candidate paths exist:

```text
data/derived/mistral/autodan/original_prompts.jsonl
data/derived/mistral/autodan/jailbreak_prompts.jsonl
data/derived/mistral/refusal/refusal_prompts.jsonl
data/derived/mistral/gcg/original_prompts.jsonl
data/derived/mistral/gcg/jailbreak_prompts.jsonl
```

## Current cautions

- Using Llama2 AutoDAN/GCG prompts with Mistral is a reference experiment, not a strict Mistral reproduction.
- Confirm that paired Mistral AutoDAN/GCG original and jailbreak prompts exist before starting experiments.
- If Mistral-specific data is unavailable, decide the data-preparation policy in the next phase.
- Decide whether to use `--append_target_space` only after reviewing the target-token diagnostics.

## Attention-score decomposition smoke test

The analysis uses the post-RoPE, pre-mask, pre-softmax score matrix
`S = QK^T / sqrt(d_head)`. For Mistral GQA, key heads are repeated to match
query heads in the same way as the Transformers Mistral attention implementation.

The existing argument for a tail window is `--tail_tokens`, and `--max_prompts 3`
avoids creating separate three-row input files.

```bash
MODEL=/mnt/mp_nas_mks/labmember/h.kawakami/models/Mistral-7B-Instruct-v0.2/
mkdir -p logs

python scripts/analyze_attention_scores.py \
  --model "$MODEL" \
  --prompts_file data/derived/mistral/autodan/original_prompts.jsonl \
  --heads all \
  --output_csv data/outputs/mistral/attention_scores/smoke/original_tail20.csv \
  --max_prompts 3 \
  --tail_tokens 20 \
  --debug \
  2>&1 | tee logs/mistral_attention_scores_smoke_original_tail20.log

python scripts/analyze_attention_scores.py \
  --model "$MODEL" \
  --prompts_file data/derived/mistral/autodan/jailbreak_prompts.jsonl \
  --heads all \
  --output_csv data/outputs/mistral/attention_scores/smoke/jailbreak_tail20.csv \
  --max_prompts 3 \
  --tail_tokens 20 \
  --debug \
  2>&1 | tee logs/mistral_attention_scores_smoke_jailbreak_tail20.log

python scripts/analyze_attention_scores.py \
  --model "$MODEL" \
  --prompts_file data/derived/mistral/refusal/xstest_prompts.jsonl \
  --heads all \
  --output_csv data/outputs/mistral/attention_scores/smoke/refusal_tail20.csv \
  --max_prompts 3 \
  --tail_tokens 20 \
  --debug \
  2>&1 | tee logs/mistral_attention_scores_smoke_refusal_tail20.log
```

After tail20 succeeds, the same commands accept `--tail_tokens 10`, `16`, or
`32`. Change the output and log names accordingly.

## Summarize paper heads and AIE top heads

The existing summarizer accepts one selected-head set per run. Run it once for
the Mistral paper heads and once for the Mistral AIE top 10. Each output includes
`all_heads`, `selected_heads_mean`, individual heads, original, jailbreak,
safe/refusal, and `delta_jailbreak_minus_original` rows.

```bash
PAPER_HEADS="16,14;31,14;15,14;30,3;16,31;31,13;11,9;14,20;30,9;14,27"
AIE_TOP10="30,3;16,15;14,27;24,9;18,11;16,1;16,13;25,18;14,1;16,12"

python scripts/summarize_attention_score_metrics.py \
  --original_csv data/outputs/mistral/attention_scores/smoke/original_tail20.csv \
  --jailbreak_csv data/outputs/mistral/attention_scores/smoke/jailbreak_tail20.csv \
  --safe_csv data/outputs/mistral/attention_scores/smoke/refusal_tail20.csv \
  --heads "$PAPER_HEADS" \
  --output_csv data/outputs/mistral/attention_scores/smoke/summary_paper_heads_tail20.csv

python scripts/summarize_attention_score_metrics.py \
  --original_csv data/outputs/mistral/attention_scores/smoke/original_tail20.csv \
  --jailbreak_csv data/outputs/mistral/attention_scores/smoke/jailbreak_tail20.csv \
  --safe_csv data/outputs/mistral/attention_scores/smoke/refusal_tail20.csv \
  --heads "$AIE_TOP10" \
  --output_csv data/outputs/mistral/attention_scores/smoke/summary_aie_top10_tail20.csv
```

The plotter currently visualizes jailbreak-minus-original summaries and raw
original/jailbreak heatmaps. It does not plot the safe/refusal condition.
Generate comparable plots for each selected-head set as follows:

```bash
python scripts/plot_attention_score_metrics.py \
  --summary_csv data/outputs/mistral/attention_scores/smoke/summary_paper_heads_tail20.csv \
  --original_csv data/outputs/mistral/attention_scores/smoke/original_tail20.csv \
  --jailbreak_csv data/outputs/mistral/attention_scores/smoke/jailbreak_tail20.csv \
  --output_dir data/outputs/mistral/attention_scores/smoke/figures_paper_heads \
  --groups all_heads selected_heads_mean \
  --paper_heads "$PAPER_HEADS"

python scripts/plot_attention_score_metrics.py \
  --summary_csv data/outputs/mistral/attention_scores/smoke/summary_aie_top10_tail20.csv \
  --output_dir data/outputs/mistral/attention_scores/smoke/figures_aie_top10 \
  --groups all_heads selected_heads_mean
```
