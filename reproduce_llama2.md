# Llama2 Reproduction Notes

## Purpose

This branch is for reproducing the original Llama2 setting from "Jailbreak Defense in LLM via Attention Head Analysis and Selective Intervention". It is not for continuing Qwen2.5 extension experiments.

## Branch Policy

- Start from `main`.
- Keep `support-qwen2` as reference only.
- Do not import Qwen-specific scripts, docs, temporary head settings, or experiment notes.
- Commit only minimal code fixes, small utility scripts, and reproduction notes.

## Do Not Commit

- Data files under `data/`
- `.pt`, `.pth`, `.bin`, `.safetensors`
- Model weights
- Logs and large intermediate outputs

## Data Layout

The local data tree is organized as:

- `data/raw/`: original downloaded or received files
- `data/derived/`: prompt-only JSONL files prepared for this repository's scripts
- `data/outputs/`: local `.pt` outputs and other experiment artifacts

## Data Summary

Observed local data formats:

- `data/raw/advbench.jsonl`: 520 records, keys `index`, `prompt`, `target`
- `data/raw/llama2/GCG_llama2.jsonl`: 160 records, keys `index`, `prompt`
- `data/raw/llama2/AutoDAN_llama2.jsonl`: 130 records, keys `index`, `prompt`
- `data/raw/xstest.jsonl`: 250 records, keys `index`, `prompt`
- `data/derived/llama2/autodan/original_prompts.jsonl`: AutoDAN original prompts, key `prompt`
- `data/derived/llama2/autodan/jailbreak_prompts.jsonl`: AutoDAN jailbreak prompts, key `prompt`
- `data/derived/llama2/refusal/xstest_prompts.jsonl`: refusal prompts from XSTest, key `prompt`

`compute_AIE.py`, `extract_intervention_vector.py`, and `evaluate.py` read prompt JSONL files through `get_prompts_list()`, which expects a `prompt` key.

## Dataset Pairing

AutoDAN is the primary prepared attack set for reproduction. `data/raw/llama2/AutoDAN_llama2.jsonl` has 130 records, and its indices correspond to `data/raw/advbench.jsonl`. The local `data/raw/llama2/llama2_AutoDAN.json` also matches the JSONL by count, index, and prompt.

GCG is tentative. `data/raw/llama2/GCG_llama2.jsonl` indices exist in `data/raw/advbench.jsonl`, but its mapping to `data/raw/llama2/llama2_GCG.json` is not fully aligned. Do not treat generated GCG pairs as confirmed until this mapping is checked.

For refusal activations, use `data/raw/xstest.jsonl` as the first candidate and derive `data/derived/llama2/refusal/xstest_prompts.jsonl`.

## Prepare Data

Current canonical derived files:

- `data/derived/llama2/autodan/original_prompts.jsonl`
- `data/derived/llama2/autodan/jailbreak_prompts.jsonl`
- `data/derived/llama2/refusal/xstest_prompts.jsonl`

The helper script was originally written for flat output names. If regenerating data, dry-run against the raw inputs first, then normalize the generated filenames into the canonical paths above.

```bash
python scripts/prepare_llama2_repro_data.py \
  --advbench data/raw/advbench.jsonl \
  --autodan data/raw/llama2/AutoDAN_llama2.jsonl \
  --xstest data/raw/xstest.jsonl \
  --output-dir data/derived/llama2/autodan \
  --dry-run
```

Regenerate only when needed:

```bash
python scripts/prepare_llama2_repro_data.py \
  --advbench data/raw/advbench.jsonl \
  --autodan data/raw/llama2/AutoDAN_llama2.jsonl \
  --xstest data/raw/xstest.jsonl \
  --output-dir data/derived/llama2/autodan
```

After regeneration, keep the experiment-facing files at:

- `data/derived/llama2/autodan/original_prompts.jsonl`
- `data/derived/llama2/autodan/jailbreak_prompts.jsonl`
- `data/derived/llama2/refusal/xstest_prompts.jsonl`

Optional tentative GCG dry-run:

```bash
python scripts/prepare_llama2_repro_data.py \
  --advbench data/raw/advbench.jsonl \
  --autodan data/raw/llama2/AutoDAN_llama2.jsonl \
  --gcg data/raw/llama2/GCG_llama2.jsonl \
  --xstest data/raw/xstest.jsonl \
  --output-dir data/derived/llama2/gcg \
  --include-gcg \
  --dry-run
```

Expected derived files:

- `data/derived/llama2/autodan/original_prompts.jsonl`
- `data/derived/llama2/autodan/jailbreak_prompts.jsonl`
- `data/derived/llama2/refusal/xstest_prompts.jsonl`

Optional tentative GCG files:

- `data/derived/llama2/gcg/original_prompts.jsonl`
- `data/derived/llama2/gcg/jailbreak_prompts.jsonl`

## Model Load Check

Use the real local model path when available. Placeholder:

```bash
python - <<'PY'
from utils.model_utils import load_model_and_tokenizer

path = "/path/to/Llama-2-7b-chat-hf"
model, tokenizer, config = load_model_and_tokenizer(path)

print(config)
print("device:", model.device)
PY
```

Expected config is approximately:

```text
model_name: Llama
n_heads: 32
n_layers: 32
resid_dim: 4096
```

## Extract Intervention Vector

The XSTest-derived refusal vector is useful for provisional experiments, but the closer paper-reproduction condition is to use 20 AdvBench-derived prompts that are not in the evaluation set and that Llama2 safely refuses.

Select AdvBench safe rejection prompts on the GPU machine:

```bash
python scripts/select_advbench_refusal_prompts.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --advbench_file data/raw/advbench.jsonl \
  --exclude_file data/raw/llama2/AutoDAN_llama2.jsonl \
  --output_file data/derived/llama2/refusal/advbench_safe20_prompts.jsonl \
  --num_prompts 20 \
  --max_new_tokens 32
```

The script saves to `./data/{output_file}.pt`, so create the output directory first:

```bash
mkdir -p data/outputs/llama2/autodan
```

Recommended paper-reproduction intervention vector from AdvBench safe20:

```bash
python extract_intervention_vector.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --prompts_file data/derived/llama2/refusal/advbench_safe20_prompts.jsonl \
  --output_file outputs/llama2/autodan/intervention_activation_advbench_safe20
```

Provisional XSTest-derived intervention vector:

```bash
python extract_intervention_vector.py \
  --model /path/to/Llama-2-7b-chat-hf \
  --prompts_file data/derived/llama2/refusal/xstest_prompts.jsonl \
  --output_file outputs/llama2/autodan/intervention_activation
```

Expected local output:

```text
data/outputs/llama2/autodan/intervention_activation_advbench_safe20.pt
data/outputs/llama2/autodan/intervention_activation.pt
```

Do not commit this file.

## Compute AutoDAN AIE

The script saves to `./data/{output_file}.pt`, so create the output directory first if it does not exist:

```bash
mkdir -p data/outputs/llama2/autodan
mkdir -p data/outputs/llama2/smoke
```

For Llama2-chat, diagnostics showed that the token immediately after `[/INST]` is often a leading-space token before the target token `Sure`. The default AIE behavior is still available, but first try the target-space correction for Llama2-chat by passing `--append_target_space`. This appends the same literal space to both original and jailbreak templated prompts so the intervention activation position and target-token position stay aligned.

Recommended AutoDAN AIE command:

```bash
python compute_AIE.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --jailbreak_prompts data/derived/llama2/autodan/jailbreak_prompts.jsonl \
  --original_prompts data/derived/llama2/autodan/original_prompts.jsonl \
  --output_file outputs/llama2/autodan/aie_target_space \
  --append_target_space
```

Recommended AutoDAN10 smoke AIE command:

```bash
python compute_AIE.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --jailbreak_prompts data/derived/llama2/autodan/jailbreak_prompts_10.jsonl \
  --original_prompts data/derived/llama2/autodan/original_prompts_10.jsonl \
  --output_file outputs/llama2/smoke/autodan10_aie_target_space \
  --append_target_space
```

Default AIE command without the target-space correction:

```bash
python compute_AIE.py \
  --model /path/to/Llama-2-7b-chat-hf \
  --jailbreak_prompts data/derived/llama2/autodan/jailbreak_prompts.jsonl \
  --original_prompts data/derived/llama2/autodan/original_prompts.jsonl \
  --output_file outputs/llama2/autodan/aie
```

Expected local output:

```text
data/outputs/llama2/autodan/aie.pt
data/outputs/llama2/autodan/aie_target_space.pt
data/outputs/llama2/smoke/autodan10_aie_target_space.pt
```

Do not commit this file.

## Inspect AIE

```bash
python scripts/inspect_aie.py \
  --aie data/outputs/llama2/autodan/aie.pt \
  --topk 10
```

## Evaluate AutoDAN ASR

True no-intervention baseline. This is the pure No Defense setting with no intervention hooks and no intervention activation loaded:

```bash
python evaluate.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input_file data/derived/llama2/autodan/jailbreak_prompts.jsonl \
  --no_intervention
```

Default fixed-head reproduction setting:

```bash
python evaluate.py \
  --model /path/to/Llama-2-7b-chat-hf \
  --input_file data/derived/llama2/autodan/jailbreak_prompts.jsonl \
  --intervention_activation data/outputs/llama2/autodan/intervention_activation.pt \
  --intervened_head_num 5 \
  --interpolation_rate 0.75
```

The existing Llama head setting in `evaluate.py` is:

```python
"Llama": [(3,0),(7,16),(9,9),(13,4),(3,5)]
```

Use this first for paper reproduction before trying AIE-derived alternatives.

To evaluate custom heads from the AutoDAN AIE top 5, pass `--intervention_heads`. This overrides `--intervened_head_num`.

AutoDAN10 smoke with AutoDAN AIE top 5:

```bash
python evaluate.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input_file data/derived/llama2/autodan/jailbreak_prompts_10.jsonl \
  --intervention_activation data/outputs/llama2/autodan/intervention_activation.pt \
  --interpolation_rate 0.75 \
  --intervention_heads "13,4;16,15;15,30;12,1;21,14"
```

AutoDAN130 with AutoDAN AIE top 5:

```bash
python evaluate.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input_file data/derived/llama2/autodan/jailbreak_prompts.jsonl \
  --intervention_activation data/outputs/llama2/autodan/intervention_activation.pt \
  --interpolation_rate 0.75 \
  --intervention_heads "13,4;16,15;15,30;12,1;21,14"
```

For a hook-through baseline-like comparison with the same selected heads, keep the same `--intervention_heads` value and set `--interpolation_rate 0.0`. Compare this with `--no_intervention`: `--no_intervention` is the pure hook-free No Defense evaluation, while `--interpolation_rate 0.0` still installs the intervention hooks.

## Smoke-Test Candidates

These existing prompt-only files can be used for small checks before full runs:

- `data/derived/llama2/autodan/original_prompts_10.jsonl`
- `data/derived/llama2/autodan/jailbreak_prompts_10.jsonl`
- `data/derived/legacy_smoke/refusal_prompts_10.jsonl`

Suggested smoke output location:

- `data/outputs/llama2/smoke/intervention_activation.pt`
- `data/outputs/llama2/smoke/autodan10_aie.pt`

## Attention Score Decomposition Analysis

The reproduction experiments above intervene on attention head outputs. As a follow-up analysis, inspect the upstream attention score matrix:

```text
S = QK^T / sqrt(d_head)
```

Use the post-RoPE, pre-mask, pre-softmax score matrix so the analysis includes positional information but avoids the causal mask triangle and softmax normalization. The analysis decomposes each selected head's score matrix into:

```text
S_sym = (S + S^T) / 2
S_skew = (S - S^T) / 2
```

The script saves scalar metrics such as Frobenius norms, symmetry/skew ratios, reconstruction error, and energy error to CSV. It does not save the score matrices themselves.

Start with the five paper heads:

```bash
python scripts/analyze_attention_scores.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --prompts_file data/derived/llama2/autodan/jailbreak_prompts_10.jsonl \
  --heads "3,0;7,16;9,9;13,4;3,5" \
  --output_csv data/outputs/llama2/analysis/attention_score_metrics_jailbreak10.csv \
  --max_prompts 10 \
  --debug
```

Full-score analysis can be affected by prompt length because AutoDAN jailbreak prompts are much longer than the original prompts. Since the intervention rewrites the selected head output at the final token position, also compare a tail window around the end of the sequence:

```text
S_tail = S[-N:, -N:]
```

Start with `--tail_tokens 20` to inspect the final-token neighborhood while keeping the existing full-score analysis available:

```bash
python scripts/analyze_attention_scores.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --prompts_file data/derived/llama2/autodan/jailbreak_prompts_10.jsonl \
  --heads "3,0;7,16;9,9;13,4;3,5" \
  --output_csv data/outputs/llama2/analysis/attention_score_metrics_jailbreak10_tail20.csv \
  --max_prompts 10 \
  --tail_tokens 20 \
  --debug
```

Suggested tail-window output names:

- `attention_score_metrics_original130_tail20.csv`
- `attention_score_metrics_jailbreak130_tail20.csv`
- `attention_score_metrics_safe20_tail20.csv`

Use `--heads all` to analyze every layer/head pair. This is useful for comparing the all-head average against the paper heads and checking whether a score-structure change is broad or concentrated in important heads. Start with `--max_prompts 1`, then move to 10 prompts, and finally run the full 130-prompt set.

```bash
python scripts/analyze_attention_scores.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --prompts_file data/derived/llama2/autodan/jailbreak_prompts_10.jsonl \
  --heads all \
  --output_csv data/outputs/llama2/analysis/attention_score_metrics_jailbreak1_all_heads_tail20_debug.csv \
  --max_prompts 1 \
  --tail_tokens 20 \
  --debug
```

Suggested all-head tail-window output names:

- `attention_score_metrics_original130_all_heads_tail20.csv`
- `attention_score_metrics_jailbreak130_all_heads_tail20.csv`
- `attention_score_metrics_safe20_all_heads_tail20.csv`

Summarize all-head averages against selected paper heads:

```bash
python scripts/summarize_attention_score_metrics.py \
  --original_csv data/outputs/llama2/analysis/attention_score_metrics_original130_all_heads_tail20.csv \
  --jailbreak_csv data/outputs/llama2/analysis/attention_score_metrics_jailbreak130_all_heads_tail20.csv \
  --safe_csv data/outputs/llama2/analysis/attention_score_metrics_safe20_all_heads_tail20.csv \
  --heads "3,0;7,16;9,9;13,4;3,5" \
  --output_csv data/outputs/llama2/analysis/summary_all_heads_vs_paper_heads_tail20.csv
```

In the summary CSV, `all_heads` is the average over every available head, `selected_heads_mean` is the average over the specified paper-head set, and rows like `layer13_head4` are individual selected heads. The `delta_jailbreak_minus_original` condition reports paired jailbreak - original differences, which helps check whether the selected heads change more strongly than the all-head average.

Useful comparison inputs:

- `data/derived/llama2/autodan/original_prompts_10.jsonl`
- `data/derived/llama2/autodan/jailbreak_prompts_10.jsonl`
- `data/derived/llama2/refusal/advbench_safe20_prompts.jsonl`

## Not Yet Run

- Llama2 model loading
- Data preparation without `--dry-run`
- Intervention vector extraction
- AIE computation
- AIE inspection on a real `.pt`
- ASR evaluation

Needed next:

- Real Llama2 model path
- Confirmation of GCG index mapping policy
- Approval to run data preparation without `--dry-run`
