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

## Data Summary

Observed local data formats:

- `data/advbench.jsonl`: 520 records, keys `index`, `prompt`, `target`
- `data/GCG_llama2.jsonl`: 160 records, keys `index`, `prompt`
- `data/AutoDAN_llama2.jsonl`: 130 records, keys `index`, `prompt`
- `data/xstest.jsonl`: 250 records, keys `index`, `prompt`
- `data/original_prompts_10.jsonl`: 10 records, key `prompt`
- `data/jailbreak_prompts_10_smoke.jsonl`: 10 records, key `prompt`
- `data/refusal_prompts_10.jsonl`: 10 records, key `prompt`

`compute_AIE.py`, `extract_intervention_vector.py`, and `evaluate.py` read prompt JSONL files through `get_prompts_list()`, which expects a `prompt` key.

## Dataset Pairing

AutoDAN is the primary prepared attack set for reproduction. `AutoDAN_llama2.jsonl` has 130 records, and its indices correspond to `advbench.jsonl`. The local `llama2_AutoDAN.json` also matches the JSONL by count, index, and prompt.

GCG is tentative. `GCG_llama2.jsonl` indices exist in `advbench.jsonl`, but its mapping to `llama2_GCG.json` is not fully aligned. Do not treat generated GCG pairs as confirmed until this mapping is checked.

For refusal activations, use `xstest.jsonl` as the first candidate and derive `data/refusal_prompts_llama2.jsonl`.

## Prepare Data

Dry-run first:

```bash
python scripts/prepare_llama2_repro_data.py --dry-run
```

Write AutoDAN and refusal derived files:

```bash
python scripts/prepare_llama2_repro_data.py
```

Optional tentative GCG dry-run:

```bash
python scripts/prepare_llama2_repro_data.py --include-gcg --dry-run
```

Expected derived files:

- `data/original_prompts_autodan_llama2.jsonl`
- `data/jailbreak_prompts_autodan_llama2.jsonl`
- `data/refusal_prompts_llama2.jsonl`

Optional tentative GCG files:

- `data/original_prompts_gcg_llama2.jsonl`
- `data/jailbreak_prompts_gcg_llama2.jsonl`

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

```bash
python extract_intervention_vector.py \
  --model /path/to/Llama-2-7b-chat-hf \
  --prompts_file data/refusal_prompts_llama2.jsonl \
  --output_file llama2_intervention_activation
```

Expected local output:

```text
data/llama2_intervention_activation.pt
```

Do not commit this file.

## Compute AutoDAN AIE

```bash
python compute_AIE.py \
  --model /path/to/Llama-2-7b-chat-hf \
  --jailbreak_prompts data/jailbreak_prompts_autodan_llama2.jsonl \
  --original_prompts data/original_prompts_autodan_llama2.jsonl \
  --output_file llama2_autodan_aie
```

Expected local output:

```text
data/llama2_autodan_aie.pt
```

Do not commit this file.

## Inspect AIE

```bash
python scripts/inspect_aie.py \
  --aie data/llama2_autodan_aie.pt \
  --topk 10
```

## Evaluate AutoDAN ASR

```bash
python evaluate.py \
  --model /path/to/Llama-2-7b-chat-hf \
  --input_file data/jailbreak_prompts_autodan_llama2.jsonl \
  --intervention_activation data/llama2_intervention_activation.pt \
  --intervened_head_num 5 \
  --interpolation_rate 0.75
```

The existing Llama head setting in `evaluate.py` is:

```python
"Llama": [(3,0),(7,16),(9,9),(13,4),(3,5)]
```

Use this first for paper reproduction before trying AIE-derived alternatives.

## Smoke-Test Candidates

These existing prompt-only files can be used for small checks before full runs:

- `data/original_prompts_10.jsonl`
- `data/jailbreak_prompts_10_smoke.jsonl`
- `data/refusal_prompts_10.jsonl`

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
