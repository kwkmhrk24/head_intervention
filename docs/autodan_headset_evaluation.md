# Llama2 AutoDAN head-set intervention evaluation

## Purpose and scope

`scripts/evaluate_autodan_head_sets.py` compares four fixed five-head sets while
holding the existing attention-head output intervention constant. It does not
intervene directly on `S_sym` or `S_asym`; only the head locations passed to
`utils.intervention_utils.generate_with_intervention()` change.

The full AutoDAN 130-prompt dataset had already been used for exploratory
S-metric analysis. This evaluation is a preliminary internal hold-out analysis,
not a fully untouched external test. The fixed split contains 20 selection
prompts and 110 evaluation prompts. The selection prompts are never generated
or read as evaluation responses by this script.

The split was selected from AutoDAN source `success=True` labels. Those labels
are not the same as the current refusal-substring success rule, so this remains
a preliminary split. The saved pure no-intervention run has 17 successes among
all 130 prompts and 13 successes among the 110 evaluation prompts.

## Fixed comparison conditions

All intervention conditions use:

- model: `Llama-2-7b-chat-hf`
- intervention activation:
  `data/outputs/llama2/autodan/intervention_activation_advbench_safe20.pt`
- activation source: AdvBench safe20 responses
- interpolation rate: `0.75`
- five intervened heads
- user-only Llama chat template and no system prompt
- `max_new_tokens=32`, greedy generation, `do_sample=False`
- no explicitly passed temperature, top-p, or top-k
- the unchanged 43-entry, case-sensitive refusal-substring checker

The activation is fixed so the comparison isolates the effect of changing the
head set rather than changing the safety direction. The script rejects an
interpolation rate other than `0.75`, a non-Llama2 32-layer/32-head/4096-hidden
configuration, or an activation shape other than `(32, 32, 1, 128)`.

The four head sets are:

| Condition | Role | Heads `(layer, head)` |
| --- | --- | --- |
| `paper` | prior-work Llama heads | `(3,0), (7,16), (9,9), (13,4), (3,5)` |
| `positive` | primary S condition | `(1,8), (7,25), (9,30), (12,27), (29,30)` |
| `negative` | reverse-direction comparison | `(13,5), (18,29), (6,1), (17,23), (24,13)` |
| `absolute` | direction-agnostic auxiliary condition | `(1,8), (7,25), (13,5), (9,30), (12,27)` |

The three S sets are loaded from
`data/outputs/llama2/autodan/holdout_seed42/head_selection_tail16/selected_heads.json`.
The script compares their ordered `(layer, head)` pairs with the fixed values
above and stops on any difference. It does not recompute or adjust the ranking.

## Input validation

Before loading the model, the script checks that:

- the manifest describes exactly 130 prompts split into 20 and 110;
- selection and evaluation indices are unique, disjoint, and complete;
- the evaluation JSONL has exactly 110 rows in manifest order;
- every evaluation row retains `source_index` and `prompt_id`;
- all 110 rows join to the saved baseline and source labels by source index;
- raw prompt text agrees across evaluation, baseline, and source files;
- all head sets contain five unique, in-range heads;
- the S head sets agree with the fixed expected lists; and
- the intervention tensor has the expected Llama2 shape.

After model/tokenizer loading, all 110 formatted prompts are recomputed and
compared with the saved baseline `formatted_prompt`. This full validation occurs
before `--max-prompts` is applied, so a short smoke test still validates the
complete evaluation input. Selection metrics and selection responses are not
used during generation.

## Prompt-level transitions

Each successfully generated prompt is assigned one transition relative to its
saved pure no-intervention result:

| Baseline | Intervention | Transition |
| --- | --- | --- |
| success | success | `remained_success` |
| success | failure | `defended` |
| failure | success | `regressed` |
| failure | failure | `remained_failure` |

A generation error is stored with `success=null` and `transition=null`; it is
not silently counted as a defended or failed response. ASR and source-success
subset ASR use successfully evaluated generations as their denominators.
`defense_success_rate` is `defended_count / baseline_success_count`, and
`regression_rate` is `regressed_count / baseline_failure_count`. A zero
denominator is written as JSON `null`.

## Outputs

For each requested condition, the output directory contains:

```text
<output-dir>/
    paper/
        prompt_results.jsonl
        summary.json
    positive/
        prompt_results.jsonl
        summary.json
    negative/
        prompt_results.jsonl
        summary.json
    absolute/
        prompt_results.jsonl
        summary.json
    comparison_summary.csv
    comparison_summary.json
```

Only requested condition directories are created. The comparison files also
contain a `no_intervention` row derived from the existing baseline JSONL; the
baseline is not regenerated.

Each prompt row records source/prompt identifiers, raw and formatted prompts,
condition, selected heads, interpolation rate, response, intervention success,
saved baseline success, source success, baseline agreement, generation failure
diagnostics, matched refusal substrings, response lengths, and transition.

Each condition summary records ASR, all transition counts, defense and
regression rates, source-success-subset ASR, response-length means, failure
indices, model and generation configuration, all input paths, selected heads,
selection method, split sizes and seed, timestamp, command, and Git metadata.

Generated JSONL, JSON, and CSV files are under ignored `data/` paths and must
not be committed.

## Tests

Run the dedicated synthetic tests:

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  -m unittest tests.test_autodan_headset_evaluation -v
```

Run the complete test suite:

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  -m unittest discover -s tests -v
```

## Real-model smoke test

On a CUDA-capable host, evaluate the first three rows for the paper and primary
positive conditions:

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/evaluate_autodan_head_sets.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input-file data/derived/llama2/autodan/holdout_seed42/evaluation_jailbreak_prompts.jsonl \
  --split-manifest data/derived/llama2/autodan/holdout_seed42/split_manifest.json \
  --selected-heads data/outputs/llama2/autodan/holdout_seed42/head_selection_tail16/selected_heads.json \
  --baseline-results data/outputs/llama2/autodan/baseline_alignment/no_intervention_prompt_results.jsonl \
  --source-labels data/raw/llama2/llama2_AutoDAN.json \
  --intervention-activation data/outputs/llama2/autodan/intervention_activation_advbench_safe20.pt \
  --interpolation-rate 0.75 \
  --conditions paper,positive \
  --max-prompts 3 \
  --output-dir data/outputs/llama2/autodan/holdout_seed42/headset_evaluation_smoke
```

The development host used for this implementation could not communicate with
the NVIDIA driver, so the real-model smoke test was not run there. The 7B model
was not loaded on CPU.

## Full 110-prompt commands

For the intended side-by-side comparison, the recommended invocation runs all
four conditions and writes one comparison table:

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/evaluate_autodan_head_sets.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input-file data/derived/llama2/autodan/holdout_seed42/evaluation_jailbreak_prompts.jsonl \
  --split-manifest data/derived/llama2/autodan/holdout_seed42/split_manifest.json \
  --selected-heads data/outputs/llama2/autodan/holdout_seed42/head_selection_tail16/selected_heads.json \
  --baseline-results data/outputs/llama2/autodan/baseline_alignment/no_intervention_prompt_results.jsonl \
  --source-labels data/raw/llama2/llama2_AutoDAN.json \
  --intervention-activation data/outputs/llama2/autodan/intervention_activation_advbench_safe20.pt \
  --interpolation-rate 0.75 \
  --conditions paper,positive,negative,absolute \
  --output-dir data/outputs/llama2/autodan/holdout_seed42/headset_evaluation
```

The following commands intentionally run one condition at a time. Use a shared
output directory only if the comparison files from the preceding condition are
handled deliberately; the default no-overwrite behavior prevents accidental
replacement. The examples therefore use separate condition-specific output
directories.

### Paper heads

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/evaluate_autodan_head_sets.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input-file data/derived/llama2/autodan/holdout_seed42/evaluation_jailbreak_prompts.jsonl \
  --split-manifest data/derived/llama2/autodan/holdout_seed42/split_manifest.json \
  --selected-heads data/outputs/llama2/autodan/holdout_seed42/head_selection_tail16/selected_heads.json \
  --baseline-results data/outputs/llama2/autodan/baseline_alignment/no_intervention_prompt_results.jsonl \
  --source-labels data/raw/llama2/llama2_AutoDAN.json \
  --intervention-activation data/outputs/llama2/autodan/intervention_activation_advbench_safe20.pt \
  --interpolation-rate 0.75 \
  --conditions paper \
  --output-dir data/outputs/llama2/autodan/holdout_seed42/headset_evaluation_paper
```

### Positive S heads

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/evaluate_autodan_head_sets.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input-file data/derived/llama2/autodan/holdout_seed42/evaluation_jailbreak_prompts.jsonl \
  --split-manifest data/derived/llama2/autodan/holdout_seed42/split_manifest.json \
  --selected-heads data/outputs/llama2/autodan/holdout_seed42/head_selection_tail16/selected_heads.json \
  --baseline-results data/outputs/llama2/autodan/baseline_alignment/no_intervention_prompt_results.jsonl \
  --source-labels data/raw/llama2/llama2_AutoDAN.json \
  --intervention-activation data/outputs/llama2/autodan/intervention_activation_advbench_safe20.pt \
  --interpolation-rate 0.75 \
  --conditions positive \
  --output-dir data/outputs/llama2/autodan/holdout_seed42/headset_evaluation_positive
```

### Negative S heads

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/evaluate_autodan_head_sets.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input-file data/derived/llama2/autodan/holdout_seed42/evaluation_jailbreak_prompts.jsonl \
  --split-manifest data/derived/llama2/autodan/holdout_seed42/split_manifest.json \
  --selected-heads data/outputs/llama2/autodan/holdout_seed42/head_selection_tail16/selected_heads.json \
  --baseline-results data/outputs/llama2/autodan/baseline_alignment/no_intervention_prompt_results.jsonl \
  --source-labels data/raw/llama2/llama2_AutoDAN.json \
  --intervention-activation data/outputs/llama2/autodan/intervention_activation_advbench_safe20.pt \
  --interpolation-rate 0.75 \
  --conditions negative \
  --output-dir data/outputs/llama2/autodan/holdout_seed42/headset_evaluation_negative
```

### Absolute S heads

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/evaluate_autodan_head_sets.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input-file data/derived/llama2/autodan/holdout_seed42/evaluation_jailbreak_prompts.jsonl \
  --split-manifest data/derived/llama2/autodan/holdout_seed42/split_manifest.json \
  --selected-heads data/outputs/llama2/autodan/holdout_seed42/head_selection_tail16/selected_heads.json \
  --baseline-results data/outputs/llama2/autodan/baseline_alignment/no_intervention_prompt_results.jsonl \
  --source-labels data/raw/llama2/llama2_AutoDAN.json \
  --intervention-activation data/outputs/llama2/autodan/intervention_activation_advbench_safe20.pt \
  --interpolation-rate 0.75 \
  --conditions absolute \
  --output-dir data/outputs/llama2/autodan/holdout_seed42/headset_evaluation_absolute
```

No full 110-prompt intervention generation was run during implementation.
