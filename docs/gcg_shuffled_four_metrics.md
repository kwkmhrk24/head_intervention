# GCG vs Shuffled: four metric families

CPU-only analysis of the existing full-score capture and comparison dataset.
No model loading/forward, tokenization, suffix generation, ASR re-evaluation,
intervention, score extraction, or independent100 access occurs.

## Frozen definitions

Use source metadata for every input, region and final Query; never infer token
boundaries from text. Compare GCG and Shuffled at the same Index/Layer/Head.
Original scores have no suffix metrics. Original metadata and historical
automatic labels are used to validate correspondence and define the subset.

The primary subset is the eighteen records with historical automatic
Original=false, GCG=true, Shuffled=false. Auxiliary analysis retains all twenty,
including 91 and 123. These are refusal-substring decisions, not human labels.

Float32 saved scores are read-only. Calculations use float64:

1. `M_U`: stable softmax of the final Query over **all** keys, then suffix sum.
   Save `M_O` as an auxiliary check.
2. `D`: stable `logsumexp(S[q,U]) - logsumexp(S[q,O])`. Check against the
   log softmax mass ratio. If a mass underflows, retain the log-domain identity
   check and record the underflow count rather than misclassifying it as infinity.
3. `F`: Frobenius norm of the complete raw 20x20 suffix block.
4. `F_sym`, `F_asym`, `R_asym`: norms of `(B+B.T)/2`, `(B-B.T)/2`, and the
   **squared** asymmetry norm divided by squared full norm. No mask or softmax is
   applied to the block. A zero full norm makes only the ratio undefined; it is
   saved as a blank CSV field, without epsilon substitution.

Every difference is GCG minus Shuffled. Report per-head mean, median and paired
sample SD (`ddof=1`), with explicit valid counts. Larger/equal counts use numerical
equality `atol=1e-12 + rtol=1e-12 * max(abs(GCG),abs(Shuffled))`. Means and
differences themselves are not clipped or rounded to zero.

The formulas agree with the repository's decomposition conventions, but its
older metrics function reports a norm ratio instead of the requested squared
energy ratio and includes unrelated tail metrics. It is intentionally not used
as a drop-in calculator. Existing strict JSON/hash helpers and Pearson/average-
rank Spearman functions are reused. The correlation module imports Torch, but
no model/GPU execution is invoked by this analysis.

## Outputs and interpretation

- `metrics.csv`: 40,960 Index/condition/Layer/Head rows, GCG and Shuffled only.
- `paired_differences.csv`: 20,480 paired rows; all twenty retained.
- `summary_18.csv`, `summary_20.csv`: long-form six-metric summaries, 6,144 rows
  each, including both condition means/medians and paired delta statistics.
- `sym_asym_changes_18.csv`, `sym_asym_changes_20.csv`: convenient wide tables.
- `overview.csv`: equal-weight head averages, sign counts, and variation of
  sample-level global mean differences. Heads are not independent replicates.
- `descriptive_head_extremes.csv`: absolute mean difference top five per metric
  and subset, ties broken by Layer then Head. This is descriptive reporting,
  not an automatic defense-location decision.
- `associations.csv`, `mass_asym_head_maps_*.csv`: descriptive associations of
  mean delta M_U with mean delta R_asym, including layer-centered comparisons.
- `subset_comparison.csv`: eighteen-versus-twenty head-map agreement.
- `heatmap_values/`: exact 32x32 CSV grids used in the figures.
- `figures/`: all six differences for each subset, a primary four-panel view,
  and mass/asymmetry scatter plots. Corresponding subsets share symmetric
  zero-centered color limits; undefined cells, if any, are grey.
- `manifest.json`, `validation.json`, `output_hashes.json`, `report.md`:
  input hashes, formulas, subsets, runtime/code provenance, validation and report.

These are exploratory summaries of a small, success-selected dataset, with an
additional outcome-conditioned primary subset. No significance tests are
performed and no generalization or causality is inferred from head correlations.
Full-block Frobenius norms do not uniquely identify structure. Raw antisymmetry
includes future scores and must not be called a direction of actual information
flow. A larger absolute asymmetry norm need not imply a larger asymmetry ratio.

## Reproduction

From the repository root, use a new, nonexistent output directory:

```bash
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_gcg_shuffled_four_metrics tests.test_attention_head_metric_relationships -v
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_four_metrics.py --output-dir data/outputs/gcg_shuffled_four_metrics/NEW_RUN
```

Source hashes and score hashes are verified before and after the analysis.
Tests cover uniform attention including template keys, extreme logits,
zero-norm undefined ratios, squared-ratio semantics, Frobenius/decomposition
identities, simultaneous-permutation invariance, incorrect Query/ranges,
nonfinite inputs, keyed pairing, summary SD/counts, and missing pairs. Final CSV
audits also check complete grids, identities, finite results and count totals.
