# Original/Suffix region score comparison

CPU-only analysis of the saved selection20 scores, using primary18 and all20.
No model loading/forward, suffix generation, score extraction, response/ASR
evaluation, intervention or independent100 access occurs. Existing inputs and
outputs stay immutable. Head groups are the previous token analysis's fixed
forty-five individuals and twelve groups; no new ranking is used.

## Reuse and definitions

`inspect_inputs` from the four-metric analysis verifies Index/condition/ranges,
source hashes, raw-score policy and 32x32 Layer/Head structure. Original IDs are
checked against Original-only input IDs. All three conditions' Original blocks
are compared per sample/Head, with rtol=atol=1e-5 and actual maximum errors saved.
The existing F_UU is checked against the token-analysis scalar archive.

Analysis 1 does not recompute softmax/logsumexp. The query summary CSV and D
decomposition are reused. M_U/D Head summaries are copied from the previous
four-metric CSVs; only missing M_O Head summaries are aggregated from stored
probabilities. Four-Query and per-sample arrays also reuse existing results.

For O before U, define OO=S[O,O], UU=S[U,U], UO=S[U,O].

- `full`: include every element, including future entries.
- `causal`: include absolute Key index <= Query index, including the diagonal.
  Valid counts: OO m(m+1)/2, UU 210, UO 20*m. All UO elements are past-facing.
- F is the square root of summed squared selected raw scores.
- RMS is F divided by sqrt(valid count). Full square OO/UU reduce to F/m and
  F/20; rectangular UO reduces to F/sqrt(20*m). Do not zero-pad masked elements
  and then divide by the original full count.
- `R_UU_OO_full` is the requested original-baseline ratio R_UO.
- `R_UU_UO_full` is the requested internal/external RMS ratio.
- Auxiliary causal ratios use causal numerator **and denominator**. OO causal
  is additionally calculated for a consistent causal OO comparison. This does
  not redefine the primary full OO ratio. UO full/causal are exactly equal.

No epsilon is added to denominators. Zero denominators yield undefined ratios
(blank CSV), explicit zero records and valid counts. Norms remain defined when
zero. An undefined member propagates to its view mean rather than being silently
replaced or used to change Head weights. The supplied real data is audited for
these events. Pair summaries retain explicit per-condition/pair valid counts.

## Aggregation and interpretation

Calculate F/RMS/ratios per sample and Head first, then equal Head averages within
the fixed views, then sample statistics. Never use a norm/ratio of mean matrices.
Paired SD uses ddof=1. Larger/equal counts use the existing absolute 1e-12 plus
relative 1e-12 comparison tolerance. Both condition mean/median, paired mean,
median, SD, mean absolute difference and directional counts are saved.

When OO is invariant, delta(RMS_UU/RMS_OO) is algebraically just the UU change
rescaled by its fixed OO baseline. Its identity is checked; it is not independent
mechanistic evidence. Across Heads the normalization changes weights, so global
means of ratios need not be ratios of global means.

M_O is a normalized final-Query Attention mass. OO uses Original Queries and
Original Keys; UO uses Suffix Queries and Original Keys. Their raw RMS values are
unsigned magnitudes, including large negative scores, and are not Attention
probabilities. Lower M_O must not be relabeled weaker Original-internal scores.
Full-block future entries likewise must not be interpreted as actual causal
attention or information flow.

Head-map Pearson/Spearman correlations and sign quadrants are descriptive, with
raw and layer-centered versions. Heads are dependent. Existing successful-case
selection, outcome-conditioned primary18, and reuse of the same samples for Head
selection/consistency prevent causal, significance or universal-success claims.

## Outputs

- `metrics.csv`, `paired_differences.csv`: 40,960 condition rows / 20,480 pairs,
  with sixteen full/causal regional metrics.
- `element_counts.csv`, `oo_invariance.csv`, `zero_denominators.csv`: explicit
  m/n, visible counts, per-Head invariance and undefined events.
- `summary_18/20.csv`, `view_summary_18/20.csv`: Head and fixed-view summaries.
- `fixed_group_member_effects.csv`: effects for each original selected member,
  allowing comparison of all five members without reselection.
- `attention_existing_query_summary.csv`, `attention_existing_D_decomposition.csv`,
  `attention_head_summary_18/20.csv`: reused/arranged Attention results.
- `head_maps_18/20.csv`, `associations.csv`, `overview.csv`: aligned comparisons.
- `per_sample_values.npz`: `[20,2,1024,16]` regional metrics and reused q4 masses,
  plus fixed-view arrays. Orders are recorded in the manifest.
- `figures/`: overview, eight Head maps in two sheets, sample consistency,
  fixed-group member comparison, causal sensitivity, mass-vs-score associations,
  reused four-Query overview and 18/20 sensitivity. Numerical grids are saved.
- `manifest.json`, `validation.json`, `output_hashes.json`, Japanese `report.md`
  and delivery `interpretation.md`. Post-run audit/delivery additions have their
  own hashes; initial artifacts are not overwritten.

## Reproduction

```bash
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_gcg_shuffled_regions tests.test_gcg_shuffled_tokens tests.test_gcg_shuffled_four_metrics -v
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_regions.py --output-dir data/outputs/gcg_shuffled_region_analysis/NEW_RUN
```

Choose a nonexistent NEW_RUN. Tests include hand-computable rectangular/square
norms, causal counts and RMS denominator, future entries, zero ratios, malformed
ranges/nonfinite scores, invariance failure, metric-before-mean order, paired
SD/counts and subset membership. Any mismatch aborts without fabricating results.
