# AutoDAN direct antisymmetric attention-score intervention

## Purpose

This experiment directly suppresses the antisymmetric component of selected
Llama2 attention-score matrices. It does not load a safety activation, replace
head outputs, call `replace_activation_w_avg`, or combine with the existing
activation intervention.

For a selected layer/head, let the post-RoPE, repeated-KV, pre-mask,
pre-softmax score matrix be

```text
S = QK^T / sqrt(d_head)
S_sym  = (S + S^T) / 2
S_asym = (S - S^T) / 2
S_beta = S_sym + (1 - beta) * S_asym
       = S - beta * S_asym
```

`beta` is not the existing activation-interpolation alpha:

- `beta=0`: no score change;
- `beta=0.5`: halve the antisymmetric component; and
- `beta=1`: remove the antisymmetric component.

The implementation rejects values outside `0 <= beta <= 1`.

## Audited Transformers implementation

The installed version is Transformers `4.42.3`. The local
`Llama-2-7b-chat-hf` config selects eager attention through
`LLAMA_ATTENTION_CLASSES`, and the configured class is
`transformers.models.llama.modeling_llama.LlamaAttention`.

The audited forward signature is:

```text
(self, hidden_states, attention_mask=None, position_ids=None,
 past_key_value=None, output_attentions=False, use_cache=False,
 cache_position=None, **kwargs)
```

`utils/asym_score_intervention.py` follows that exact 4.42.3 eager forward and
adds one operation immediately after

```text
attn_weights = query_states @ key_states.transpose(-2, -1) / sqrt(head_dim)
```

and before the causal-mask addition. It preserves projection behavior,
RoPE, `repeat_kv`, attention-mask slicing, float32 softmax upcasting, dropout,
value aggregation, output projection, return values, and `output_attentions`.
The context manager refuses a different Transformers version or any SDPA/Flash
attention class and restores every patched `forward` after normal completion or
an exception.

Masking must occur after the score edit because the causal mask contains
directional `-inf`/large-negative values and is not part of the raw QK score to
be decomposed. Softmax attention probabilities are not expected to be symmetric
after causal masking, even when `beta=1` makes the raw selected score symmetric.

## No-cache requirement

Symmetric/antisymmetric decomposition requires a square score matrix. Cached
generation normally produces `query_length=1, key_length>1` after the first
step, so this implementation requires all of the following:

```text
model.config.use_cache = False
model.generate(..., use_cache=False)
patched forward: use_cache=False and past_key_value=None
query_length == key_length
```

There is an additional Transformers 4.42.3 behavior to account for. Its
`GenerationMixin.generate()` sets `model_kwargs["use_cache"]` from the
generation config, but can subsequently create an empty `DynamicCache()` when
`model._supports_default_dynamic_cache()` is true. Consequently, the first
real-model equivalence attempt reached the patched attention with a non-null
`past_key_value` even though `use_cache=False`, and correctly failed with
`UnsupportedAttentionCacheError`.

The patched attention's cache rejection was not relaxed. Instead, every call
to `model.generate()` in this pipeline—including regenerated no-intervention,
beta-zero control, and all positive-beta conditions—is wrapped in
`temporarily_disable_default_dynamic_cache(model)`. The context temporarily
sets the model instance's private `_supports_cache_class` flag to `False`, then
restores both its original value and instance/class attribute state after
normal or exceptional exit. This prevents `generate()` from creating the empty
default cache; an explicit cache, `use_cache=True`, or a non-square score still
fails immediately in the patched forward.

Transformers 4.42.3 `LlamaForCausalLM.prepare_inputs_for_generation` was also
inspected: with no past cache, it passes the full growing `input_ids` sequence.
The context uses a private Transformers attribute and is therefore deliberately
version-pinned to 4.42.3. It modifies only the model instance during generation,
not the model class or installed package. The code does not broadcast, add a
query cache, or provide a q-length-one shortcut.

Disabling the KV cache recomputes the full sequence at every decoding step and
is substantially slower and more memory/computation intensive than ordinary
cached generation. This is an accuracy-first reference implementation.

## Intervention scope and head sets

The heads were selected from tail16 analysis, but intervention applies to the
entire square score matrix at every forward call, not only the final 16×16
window.

The default comparison uses the paper heads and three fixed
`delta_asym_ratio` head sets selected from the 20-prompt seed-42 selection
split:

```text
paper:
  (3,0), (7,16), (9,9), (13,4), (3,5)

positive:
  (1,8), (7,25), (9,30), (12,27), (29,30)

negative:
  (13,5), (18,29), (6,1), (17,23), (24,13)

absolute:
  (1,8), (7,25), (13,5), (9,30), (12,27)
```

The paper set comes from the prior intervention setup. `positive` ranks heads
by decreasing `mean_delta_asym_ratio`, `negative` ranks by increasing
`mean_delta_asym_ratio`, and `absolute` ranks by decreasing
`abs(mean_delta_asym_ratio)`. Positive and negative describe the direction of
the metric change; they do not mean that a head is intrinsically safe or
dangerous. The official condition name is `absolute`; `ab` is accepted only as
a CLI alias and is normalized to `absolute` in condition names and outputs.

These sets are fixed. They are not reranked using the 110-prompt evaluation
split. Metadata identifies their family as `delta_asym_ratio`, their metric as
`mean_delta_asym_ratio`, and their split as `selection_20_seed42`.

The elementwise-consistency sets remain implemented for later experiments.
They are loaded only when an `element_*` condition is explicitly selected,
from:

```text
data/outputs/llama2/autodan/holdout_seed42/
  elementwise_head_metrics_tail16/top_heads_by_metric.json
```

and fail-fast checked against:

```text
element_positive:
  (14,11), (11,24), (8,7), (7,22), (9,10)

element_negative:
  (31,22), (10,30), (8,29), (10,31), (9,15)

element_absolute:
  (31,22), (8,29), (10,30), (11,12), (9,15)
```

`--elementwise-heads` is therefore optional for the default
`paper,positive,negative,absolute` run and required for any `element_*` run.
The ratio and elementwise names refer to different fixed head sets and are not
interchangeable.

## Numerical diagnostics

No score matrix is saved. Each patched call records scalar diagnostics only:

```text
selected_head_count
query_length / key_length
pre_asym_fro_norm / post_asym_fro_norm
observed_asym_scale / expected_asym_scale
sym_component_max_abs_error
intervention_identity_max_abs_error
unselected_head_max_abs_error
```

Prompt and condition summaries contain call counts, call-weighted mean observed
scale, aggregate pre/post Frobenius norms, the aggregate observed scale,
selected-head count, and maximum errors. The expected scale is `1-beta`.

## Regenerated baseline and beta-zero control

The old saved 13/110 baseline is not used as the transition reference. Direct
score patching requires eager attention and `use_cache=False`, so the dedicated
pipeline regenerates:

1. `no_intervention`, without a custom forward and with `use_cache=False`;
2. `beta_zero_control`, through the custom paper-head forward with `beta=0`; and
3. each requested head-set/beta condition.

Transitions for every later condition use the newly generated
`no_intervention` response from the same run. Historical baseline agreement is
not treated as an experimental baseline. The optional
`--historical-baseline-results` input only adds an auxiliary agreement table to
the regenerated no-intervention summary and the comparison JSON; it never
changes prompt transitions.

Before any intervention condition, the script compares one prompt's unpatched
and beta-zero-patched logits, argmax token, and greedy response. Failure of the
equivalence tolerance, argmax equality, response equality, or score identity
stops the run before beta-positive conditions.

## Output layout

```text
<output-dir>/
  beta_zero_equivalence.json
  comparison_summary.csv
  comparison_summary.json
  no_intervention/{prompt_results.jsonl,summary.json}
  beta_zero_control/{prompt_results.jsonl,summary.json}
  <headset>_beta_<beta>/{prompt_results.jsonl,summary.json}
```

Prompt rows contain raw/formatted prompts, response and success diagnostics,
new-baseline transition, selected heads, beta, `use_cache=false`, response
lengths, score call count, observed scale, and maximum identity/symmetry errors.

## One-prompt beta-zero equivalence command

The preflight is automatic. This command runs only the regenerated baseline and
beta-zero control after it passes:

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/evaluate_autodan_asym_score_intervention.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input-file data/derived/llama2/autodan/holdout_seed42/evaluation_jailbreak_prompts.jsonl \
  --split-manifest data/derived/llama2/autodan/holdout_seed42/split_manifest.json \
  --source-labels data/raw/llama2/llama2_AutoDAN.json \
  --historical-baseline-results data/outputs/llama2/autodan/baseline_alignment/no_intervention_prompt_results.jsonl \
  --conditions paper \
  --betas 0 \
  --max-prompts 1 \
  --max-new-tokens 32 \
  --seed 42 \
  --output-dir data/outputs/llama2/autodan/holdout_seed42/asym_score_equivalence_smoke_v2
```

## Three-prompt ratio-headset smoke command

Run this later in the user GPU environment. The automatic one-prompt beta-zero
equivalence preflight still runs before any positive-beta condition.

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main

mkdir -p logs

/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/evaluate_autodan_asym_score_intervention.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input-file data/derived/llama2/autodan/holdout_seed42/evaluation_jailbreak_prompts.jsonl \
  --split-manifest data/derived/llama2/autodan/holdout_seed42/split_manifest.json \
  --source-labels data/raw/llama2/llama2_AutoDAN.json \
  --historical-baseline-results data/outputs/llama2/autodan/baseline_alignment/no_intervention_prompt_results.jsonl \
  --conditions paper,positive,negative,absolute \
  --betas 0.5,1.0 \
  --max-prompts 3 \
  --max-new-tokens 32 \
  --seed 42 \
  --output-dir data/outputs/llama2/autodan/holdout_seed42/asym_score_ratio_headsets_smoke \
  2>&1 | tee logs/asym_score_ratio_headsets_smoke.log
```

After the GPU run, validate structural and numerical smoke criteria with:

```bash
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python - <<'PY'
import json
from pathlib import Path

root = Path(
    "data/outputs/llama2/autodan/holdout_seed42/"
    "asym_score_ratio_headsets_smoke"
)
conditions = [
    "no_intervention",
    "beta_zero_control",
    "paper_beta_0.5",
    "paper_beta_1",
    "positive_beta_0.5",
    "positive_beta_1",
    "negative_beta_0.5",
    "negative_beta_1",
    "absolute_beta_0.5",
    "absolute_beta_1",
]

with (root / "beta_zero_equivalence.json").open() as handle:
    equivalence = json.load(handle)
assert equivalence["passed"] is True
assert equivalence["generation_score_diagnostics"][
    "all_score_matrices_square"
] is True

for condition in conditions:
    with (root / condition / "summary.json").open() as handle:
        summary = json.load(handle)
    with (root / condition / "prompt_results.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    assert summary["generation_failure_count"] == 0, condition
    assert summary["max_intervention_identity_error"] <= 1e-5, condition
    assert summary["max_symmetric_component_error"] <= 1e-5, condition
    assert summary["max_unselected_head_error"] <= 1e-5, condition
    assert all(
        isinstance(row["response"], str) and row["response"].strip()
        for row in rows
    )

    if condition != "no_intervention":
        assert all(row["all_score_matrices_square"] is True for row in rows)
    if summary["beta"] == 0.5:
        assert abs(summary["mean_observed_asym_scale"] - 0.5) <= 1e-4
    if summary["beta"] == 1.0:
        assert abs(summary["mean_observed_asym_scale"] - 0.0) <= 1e-4

print("ratio-headset smoke checks passed")
PY
```

This validates execution only. Three prompts are insufficient for any ASR or
defense-performance conclusion.

## Three-prompt elementwise intervention smoke command

Run only after beta-zero equivalence passes:

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/evaluate_autodan_asym_score_intervention.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --input-file data/derived/llama2/autodan/holdout_seed42/evaluation_jailbreak_prompts.jsonl \
  --split-manifest data/derived/llama2/autodan/holdout_seed42/split_manifest.json \
  --elementwise-heads data/outputs/llama2/autodan/holdout_seed42/elementwise_head_metrics_tail16/top_heads_by_metric.json \
  --source-labels data/raw/llama2/llama2_AutoDAN.json \
  --historical-baseline-results data/outputs/llama2/autodan/baseline_alignment/no_intervention_prompt_results.jsonl \
  --conditions paper,element_positive \
  --betas 0.5,1.0 \
  --max-prompts 3 \
  --max-new-tokens 32 \
  --seed 42 \
  --output-dir data/outputs/llama2/autodan/holdout_seed42/asym_score_intervention_smoke
```

The user-side GPU beta-zero equivalence run passed. The Codex implementation
host cannot communicate with the NVIDIA driver, so it did not load the 7B model
or run either three-prompt smoke command. The model was not loaded on CPU.

## Research status

No 110-prompt full generation or beta performance sweep has been run. The full
AutoDAN data informed exploratory head selection, so a later evaluation on the
110-prompt internal split remains post-hoc and must not be presented as a fully
untouched external test.
