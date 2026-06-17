# Qwen2.5 Dataset Schema

This dataset layer is for reproducing the existing head intervention defense workflow on Qwen2.5-7B-Instruct. It is not a place to create, improve, or propose new jailbreak prompts. GCG and AutoDAN prompts should come from existing authorized outputs and be imported as data.

All dataset files under `data/` are local experiment artifacts and must not be committed.

## Master JSONL Schema

Recommended row format:

```json
{
  "id": "qwen2_advbench_000_gcg",
  "source": "advbench",
  "original_index": 0,
  "attack_method": "gcg",
  "target_model": "Qwen2.5-7B-Instruct",
  "original_prompt": "...",
  "jailbreak_prompt": "...",
  "direct_refused": true,
  "direct_success_by_current_rule": false,
  "attack_refused": false,
  "attack_success": true,
  "split": "aie_train",
  "notes": ""
}
```

Fields:

- `id`: stable dataset row id, usually including model family, AdvBench index, and attack method.
- `source`: source dataset, currently `advbench`.
- `original_index`: original AdvBench `index` value when present.
- `attack_method`: `direct`, `gcg`, or `autodan`.
- `target_model`: model the labels or attack outputs correspond to.
- `original_prompt`: AdvBench original harmful prompt loaded from `data/advbench.jsonl`.
- `jailbreak_prompt`: existing imported attack prompt. This is `null` for direct seed rows.
- `direct_refused`: whether Qwen2.5 refused the original direct prompt under the current string rule.
- `direct_success_by_current_rule`: `check_success(output)` for direct generation.
- `attack_refused`: whether Qwen2.5 refused the imported attack prompt.
- `attack_success`: `check_success(output)` for the imported attack prompt.
- `split`: intended use, for example `aie_train`, `eval`, or `smoke`.
- `notes`: short free-form notes. Avoid storing large generated outputs here.

For AutoDAN rows, use `"attack_method": "autodan"`. For GCG rows, use `"attack_method": "gcg"`.

## Prompt Roles

- Direct prompt: the original AdvBench prompt. Use it to measure Qwen2.5's baseline refusal behavior.
- GCG prompt: an existing GCG output targeting Qwen2.5, aligned row-by-row with the original prompt.
- AutoDAN prompt: an existing AutoDAN output targeting Qwen2.5, aligned row-by-row with the original prompt.
- Refusal prompt: an original prompt that Qwen2.5 refused directly. These prompts are used by the existing extraction workflow to collect refusal-side head activations.

## Derived Files For Existing Code

The existing scripts expect prompt-only JSONL:

```json
{"prompt": "..."}
```

Use `scripts/export_for_head_intervention.py` to create:

- `original_prompts.jsonl`: `original_prompt`, aligned row-by-row with jailbreak prompts.
- `jailbreak_prompts.jsonl`: `jailbreak_prompt`, aligned row-by-row with original prompts.
- `refusal_prompts.jsonl`: `original_prompt` rows where `direct_refused=true`.

Keep AIE and evaluation files separate. `compute_AIE.py` needs paired `original_prompts.jsonl` and `jailbreak_prompts.jsonl`; `evaluate.py` evaluates prompts against selected intervention heads. Do not treat direct prompt checks as jailbreak ASR.

## Small-Scale Workflow

Start with about 20 rows:

```bash
python scripts/build_qwen_seed_dataset.py \
  --advbench data/advbench.jsonl \
  --output data/qwen2_dataset/seed_20.jsonl \
  --count 20
```

Label direct refusal:

```bash
python scripts/label_qwen_direct_refusal.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Qwen2.5-7B-Instruct \
  --input data/qwen2_dataset/seed_20.jsonl \
  --output data/qwen2_dataset/direct_labeled_20.jsonl
```

Import existing authorized GCG or AutoDAN outputs:

```bash
python scripts/import_attack_outputs.py \
  --seed data/qwen2_dataset/direct_labeled_20.jsonl \
  --attack-file path/to/attack_output.jsonl \
  --attack-method gcg \
  --output data/qwen2_dataset/qwen2_gcg_master_20.jsonl
```

Label attack success:

```bash
python scripts/label_qwen_attack_success.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Qwen2.5-7B-Instruct \
  --input data/qwen2_dataset/qwen2_gcg_master_20.jsonl \
  --output data/qwen2_dataset/qwen2_gcg_labeled_20.jsonl
```

Export for the head intervention scripts:

```bash
python scripts/export_for_head_intervention.py \
  --input data/qwen2_dataset/qwen2_gcg_labeled_20.jsonl \
  --output-dir data/qwen2_dataset/gcg_export \
  --only-successful-attacks
```

Then run the existing workflow:

```bash
python compute_AIE.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Qwen2.5-7B-Instruct \
  --jailbreak_prompts data/qwen2_dataset/gcg_export/jailbreak_prompts.jsonl \
  --original_prompts data/qwen2_dataset/gcg_export/original_prompts.jsonl \
  --output_file qwen25_gcg_aie_20

python scripts/inspect_aie.py --aie data/qwen25_gcg_aie_20.pt --topk 10

python evaluate.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Qwen2.5-7B-Instruct \
  --input_file data/qwen2_dataset/gcg_export/jailbreak_prompts.jsonl \
  --intervention_activation data/qwen25_intervention_activation_20.pt \
  --intervened_head_num 1 \
  --interpolation_rate 0.75
```

Inspect any master/labeled dataset:

```bash
python scripts/inspect_qwen_dataset.py \
  --input data/qwen2_dataset/direct_labeled_20.jsonl
```

## Git Policy

Commit scripts and documentation only. Do not commit:

- `data/*.jsonl`
- `data/*.pt`
- `logs/`
- model weights
- caches
- `__pycache__/`
