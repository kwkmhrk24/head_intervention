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

The script saves to `./data/{output_file}.pt`, so create the output directory first:

```bash
mkdir -p data/outputs/llama2/autodan
```

```bash
python extract_intervention_vector.py \
  --model /path/to/Llama-2-7b-chat-hf \
  --prompts_file data/derived/llama2/refusal/xstest_prompts.jsonl \
  --output_file outputs/llama2/autodan/intervention_activation
```

Expected local output:

```text
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

## Smoke-Test Candidates

These existing prompt-only files can be used for small checks before full runs:

- `data/derived/llama2/autodan/original_prompts_10.jsonl`
- `data/derived/llama2/autodan/jailbreak_prompts_10.jsonl`
- `data/derived/legacy_smoke/refusal_prompts_10.jsonl`

Suggested smoke output location:

- `data/outputs/llama2/smoke/intervention_activation.pt`
- `data/outputs/llama2/smoke/autodan10_aie.pt`

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
