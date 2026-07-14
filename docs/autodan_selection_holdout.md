# Llama2 AutoDAN internal hold-out head selection

## Scope and interpretation

This pipeline implements a reproducible 20/110 split and selection-only head
ranking for Llama2 AutoDAN. All 130 prompt pairs were previously used for
exploratory score analysis, metric design, and hypothesis formation. The split
is therefore an **internal AutoDAN hold-out validation**, not a fully unseen test
set. Do not tune metrics or heads using the 110-prompt evaluation subset.

The existing `data/outputs/llama2/autodan/aie_target_space.pt` was computed from
all 130 prompts and is exploratory. It must not be used as the primary AIE
comparison for this hold-out. A later AIE comparison must recompute AIE from the
generated selection JSONL files only.

## Verified local inputs

- `data/derived/llama2/autodan/original_prompts.jsonl`: 130 prompt-only rows.
- `data/derived/llama2/autodan/jailbreak_prompts.jsonl`: 130 paired prompt-only rows.
- `data/raw/llama2/llama2_AutoDAN.json`: 130 records with `index`, `user prompt`,
  and boolean `success`; 84 records are successful.
- `data/raw/llama2/AutoDAN_llama2.jsonl`: 130 `index`/`prompt` records.
- `data/outputs/llama2/analysis/attention_score_metrics_{original,jailbreak}130_all_heads_tail16.csv`:
  prompt-level metrics for 32 layers by 32 heads.

For every row, the baseline-label `index` equals the zero-based derived JSONL
row, and its `user prompt` matches the paired jailbreak prompt. The derived
original prompt also matches the AdvBench prompt at that index. The baseline
label file does not contain response text, so the stored labels cannot be
independently reclassified from that artifact.

There is a later no-intervention run in
`logs/autodan130_evaluate_true_no_intervention.log`, but it stores only aggregate
ASR `0.130769...` (17/130), not prompt-level responses or labels. The partial
`logs/autodan10_compare_eval_outputs.log` stores responses for the first ten
prompts: the repository heuristic marks 3/10 successful, whereas the AutoDAN
source `success` field marks 8/10 successful. Thus the two label conditions are
not interchangeable. This pipeline follows the fixed policy of sampling from
the explicitly configured pre-existing source label and records that choice in
every output summary. Before intervention evaluation, decide whether the source
attack label or a newly saved current-generation label is the intended baseline
condition; do not silently mix them.

The repository's ASR heuristic is `utils.evaluate_utils.check_success()`: an
output is successful when none of its refusal substrings occur. The provenance
file's stored `success` field is used for this split; the split script validates
its type and prompt/index alignment but does not claim that it was produced by
the current repository heuristic.

## Metric definition

For the post-RoPE, pre-mask, pre-softmax score matrix

```text
S = QK^T / sqrt(d_head)
S_asym = (S - S^T) / 2
asym_ratio = ||S_asym||_F / (||S||_F + epsilon)
```

The former name `skew_ratio` used exactly the same formula and was renamed to
`asym_ratio` without a numerical change in commit `1db6b15`. The pipeline reads
either spelling for backward compatibility and rejects selected rows if both
columns exist but disagree. Ranking is fixed to tail16 input metrics.

For selection prompt `p` and head `h`:

```text
delta_asym_ratio[p,h] = jailbreak_asym_ratio[p,h] - original_asym_ratio[p,h]
```

Heads are ranked separately by decreasing mean delta (positive), increasing
mean delta (negative), and decreasing absolute mean delta (absolute). The
absolute ranking never replaces the positive ranking.

## Run

From the repository root, using the project environment:

```bash
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/analyze_autodan_selection_holdout.py
```

Defaults are selection size 20, split seed 42, top-k 5, 1,000 bootstrap
iterations, bootstrap seed 42, and tail16. Use `--overwrite` only when replacing
this pipeline's existing targets intentionally.

The script streams the full metric CSVs but only parses metric values and
retains rows whose original source indices are in the selection set. Evaluation
metric values are not read into the ranking or bootstrap calculations. It
validates a complete 20 x 32 x 32 selected grid for both conditions. The
reported delta standard deviation is the sample standard deviation (`ddof=1`).

## Outputs

Split and AIE-ready prompts:

```text
data/derived/llama2/autodan/holdout_seed42/
  selection_original_prompts.jsonl
  selection_jailbreak_prompts.jsonl
  evaluation_original_prompts.jsonl
  evaluation_jailbreak_prompts.jsonl
  split_manifest.json
  selection_indices.json
  evaluation_indices.json
```

Each prompt row has `prompt`, plus `index`, `prompt_id`, and `source_index` set
to the original zero-based row. The extra fields do not prevent
`compute_AIE.py` from reading the files. Ranking outputs are written to:

```text
data/outputs/llama2/autodan/holdout_seed42/head_selection_tail16/
  head_ranking.csv
  selected_heads.json
  bootstrap_head_stability.csv
  bootstrap_jaccard.csv
  bootstrap_summary.json
```

`split_manifest.json`, `selected_heads.json`, and `bootstrap_summary.json` all
record the internal-hold-out interpretation. Data and generated outputs under
`data/` are ignored by Git and must not be committed.

## Later selection-only AIE

The generated files can be passed directly to the existing AIE script:

```bash
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python compute_AIE.py \
  --model /mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf \
  --jailbreak_prompts data/derived/llama2/autodan/holdout_seed42/selection_jailbreak_prompts.jsonl \
  --original_prompts data/derived/llama2/autodan/holdout_seed42/selection_original_prompts.jsonl \
  --output_file outputs/llama2/autodan/holdout_seed42/aie_selection20_target_space \
  --append_target_space
```

This command is documented for the next phase and is not run by the hold-out
pipeline.
