# AutoDAN elementwise antisymmetric head metrics

## Purpose and research scope

The existing `asym_ratio` compresses each attention-score matrix to

```text
||S_asym||_F / (||S||_F + eps)
```

and therefore discards element position, signed local changes, and whether the
same position changes consistently across prompts. The offline analysis in
`scripts/analyze_autodan_elementwise_head_metrics.py` retains all 120 strict
lower-triangle elements from each tail16 `S_asym` matrix.

The full AutoDAN dataset had already been used for exploratory metric analysis.
The new elementwise metrics were computed only from the fixed 20-prompt
selection subset. Any later reuse of the 110-prompt evaluation subset is
post-hoc internal analysis rather than a fully untouched test.

The script does not load a model or Transformers, use a GPU, regenerate
attention scores, read intervention responses, calculate ASR, or run an
intervention. The existing 110-prompt intervention results are background
motivation only and are not inputs to the ranking.

## Audited raw input

The raw arrays are:

```text
data/outputs/attention_score_element_distributions/llama2/autodan/
  all_heads_raw_tail16_full/
    raw_original_values.npy
    raw_jailbreak_values.npy
    raw_delta_values.npy
    metadata.json
```

Their confirmed properties are:

- shape: `(130, 32, 32, 120)`
- axes: `prompt, layer, head, strict_lower_position`
- dtype: `float32`
- component: strict-lower elements of `S_asym = (S - S.T) / 2`
- score: `S = QK.T / sqrt(d_head)`
- post-RoPE, pre-mask, pre-softmax
- tail window: 16 tokens
- positions: `row_pos > col_pos` in NumPy/PyTorch row-major order
- prompt/source index: zero-based paired JSONL row number
- delta: jailbreak minus original at the same prompt/layer/head/position

The arrays are opened with `numpy.load(..., mmap_mode="r")`. Only the 20 rows
listed in the split manifest are copied for metric calculation. The fixed
selection indices are:

```text
[4, 5, 15, 18, 19, 26, 36, 39, 40, 41,
 43, 48, 80, 82, 96, 108, 110, 113, 126, 129]
```

The script validates the NPY headers, metadata, prompt-file paths and row-index
definitions, split completeness, selected values, and the selected delta
identity before computing metrics.

## Elementwise metrics

Let `d[p,h,r]` be the jailbreak-minus-original element delta for selection
prompt `p`, head `h`, and strict-lower position `r`.

Across all prompt-position values, the script computes:

```text
signed_mean       = mean(d)
signed_median     = median(d)
positive_mass     = mean(max(d, 0))
negative_mass     = mean(max(-d, 0))
absolute_mass     = mean(abs(d))
positive_rate     = mean(d > 0)
negative_rate     = mean(d < 0)
sign_imbalance    = positive_rate - negative_rate
```

It verifies `absolute_mass = positive_mass + negative_mass` within numerical
tolerance. Negative mass is non-negative and is ranked in descending order.
`signed_mean` has separate positive-top and negative-bottom rankings.

## Main consistent-position metrics

For each head and position, first average over selection prompts:

```text
mean_delta_by_position[h,r] = mean_p(d[p,h,r])
```

The main candidates are:

```text
consistent_positive = mean_r(max(mean_delta_by_position, 0))
consistent_negative = mean_r(max(-mean_delta_by_position, 0))
consistent_absolute = mean_r(abs(mean_delta_by_position))
```

These metrics suppress positions whose direction varies across prompts and
retain positions that repeatedly move in the same direction. The implementation
verifies:

```text
consistent_absolute = consistent_positive + consistent_negative
```

It also computes the fraction of positions whose delta is positive or negative
for at least 75% of the selection prompts. The threshold is configurable with
`--consistency-threshold` and defaults to `0.75`, corresponding to at least 15
of 20 prompts.

## Last-row and lag metrics

The last-row metrics use only the 15 strict-lower positions with `row_pos == 15`:

```text
last_row_signed_mean
last_row_positive_mass
last_row_negative_mass
last_row_absolute_mass
last_row_consistent_positive
last_row_consistent_negative
last_row_consistent_absolute
```

Row 15 is an index inside the tail16 window, not an absolute token index in the
original sequence.

For each position, `lag = row_pos - col_pos`, giving lags 1 through 15. The
script stores every head's 15-dimensional signed and absolute profile:

```text
lag_signed_mean[h,k]
lag_absolute_mean[h,k]
lag_profile_absolute = mean_k(abs(lag_signed_mean[h,k]))
lag_profile_mass     = mean_k(lag_absolute_mean[h,k])
```

## Structural diagnostics

For each prompt/head pair, original and jailbreak are compared as 120-element
vectors.

- Cosine similarity: both zero vectors are assigned similarity 1; exactly one
  zero vector is assigned 0. The head mean and `1 - mean` distance are stored.
- Sign flip: direct `np.sign(original) != np.sign(jailbreak)`, so a change
  between zero and a nonzero value is a flip.
- Pearson correlation: a prompt/head correlation is null when either vector is
  constant. Head means use valid prompts only, and the valid-prompt count is
  saved.

These diagnostics are not primary ranking criteria.

## Bootstrap stability

The default bootstrap draws 20 selection prompt indices with replacement for
each of 1,000 iterations using seed 42. Positions are never resampled. Each
iteration ranks `consistent_positive`, `consistent_negative`, and
`consistent_absolute` and records top-5 inclusion.

Outputs include per-head inclusion counts/probabilities and the mean, standard
deviation, 5th percentile, median, and 95th percentile of Jaccard similarity
between the original top-5 and each bootstrap top-5.

## Reference head sets

The new top-5/top-10 sets are compared with:

- paper heads from `evaluate.py`;
- existing selection-20 asym-ratio positive, negative, and absolute sets from
  `selected_heads.json`; and
- AIE top heads from the existing AIE ranking CSV.

The audited AIE positive top-5 is:

```text
[(13,4), (16,15), (15,30), (12,1), (21,14)]
```

The script stops if the AIE ranking does not reproduce this set. It writes
overlap counts, Jaccard similarities, common heads, and a long-format table of
every reference head's value and rank under every new ranking.

Spearman correlations across all 1,024 heads include the major elementwise
metrics and the existing selection-20 `mean_delta_asym_ratio` loaded from
`head_ranking.csv`.

## Run command

```bash
cd /home/h.kawakami/ulab/m/head_intervention_defense-main

/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  scripts/analyze_autodan_elementwise_head_metrics.py \
  --raw-original data/outputs/attention_score_element_distributions/llama2/autodan/all_heads_raw_tail16_full/raw_original_values.npy \
  --raw-jailbreak data/outputs/attention_score_element_distributions/llama2/autodan/all_heads_raw_tail16_full/raw_jailbreak_values.npy \
  --raw-delta data/outputs/attention_score_element_distributions/llama2/autodan/all_heads_raw_tail16_full/raw_delta_values.npy \
  --metadata data/outputs/attention_score_element_distributions/llama2/autodan/all_heads_raw_tail16_full/metadata.json \
  --split-manifest data/derived/llama2/autodan/holdout_seed42/split_manifest.json \
  --selected-heads data/outputs/llama2/autodan/holdout_seed42/head_selection_tail16/selected_heads.json \
  --aie-ranking data/outputs/attention_score_element_distributions/llama2/autodan/head_metric_relationships_tail16_full/ranking_aie_highest.csv \
  --ratio-ranking data/outputs/llama2/autodan/holdout_seed42/head_selection_tail16/head_ranking.csv \
  --output-dir data/outputs/llama2/autodan/holdout_seed42/elementwise_head_metrics_tail16 \
  --top-k 5 \
  --bootstrap-samples 1000 \
  --seed 42
```

Existing outputs are not overwritten unless `--overwrite` is supplied.

## Outputs

```text
per_head_elementwise_metrics.csv
top_heads_by_metric.json
top_heads_by_metric.csv
metric_rank_correlations.csv
headset_overlap_summary.csv
headset_overlap_summary.json
reference_head_metric_ranks.csv
lag_profiles.csv
position_mean_delta.npy
bootstrap_head_inclusion.csv
bootstrap_stability_summary.json
metadata.json
```

Every scalar metric has a top-20 ranking. `signed_mean` has separate positive
and negative rankings. Generated NPY/CSV/JSON outputs are ignored data artifacts
and are not committed.

## Offline smoke result

The selection-20 analysis completed for all 1,024 heads with 1,000 bootstrap
iterations. The primary top-5 sets were:

```text
consistent_positive: [(14,11), (11,24), (8,7), (7,22), (9,10)]
consistent_negative: [(31,22), (10,30), (8,29), (10,31), (9,15)]
consistent_absolute: [(31,22), (8,29), (10,30), (11,12), (9,15)]
```

Each set had zero top-5 overlap with the paper, AIE, and all three existing
asym-ratio top-5 sets. This is a selection-set ranking result only. No conclusion
about intervention effectiveness is drawn, and no intervention evaluation was
performed.

## Tests

```bash
/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  -m unittest tests.test_autodan_elementwise_head_metrics -v

/home/h.kawakami/miniconda3/envs/head-intervention/bin/python \
  -m unittest discover -s tests -v
```
