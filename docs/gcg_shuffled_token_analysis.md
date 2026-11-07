# Token-position explanation of the GCG/Shuffled metrics

This is a CPU-only diagnostic of existing scores and outcomes. No model loading,
forward pass, tokenization, suffix generation, score extraction, ASR evaluation,
intervention, or independent100 access occurs. Source score files stay read-only
and are referenced rather than copied. Derived suffix slices and contributions
are saved to explain the original scalar metrics.

## Selection and alignment

Use the immutable prior `summary_18.csv`. For each of M_U, D, F, F_sym, F_asym,
R_asym, choose five positive mean deltas in descending order and five negative
mean deltas in ascending order. Break ties by Layer then Head. If fewer than five
exist, retain only existing Heads and record counts. The fixed choices also apply
to the supplementary twenty samples; do not rerank on twenty.

`selected_heads.csv` stores rank, reason, mean/median/sample SD and directional
sample counts. `head_memberships.csv` records duplicates. The real data has sixty
memberships and forty-five unique Heads, computed once per input. Twelve group
views and one equal-weight all-1024 view reuse those results. Groups overlap and
are not independent comparisons.

Suffix positions are relative 0–19, not aligned Token identities. Permutation
changes which Token ID occupies each position. Never interpret a position mean
as the effect of the same lexical Token. Original lengths vary, so only its region
mass is averaged across samples; full Original position profiles remain per Index.
Closing-Key positions can be averaged by their common four-token relative range.

## Calculation and exact connections to previous metrics

- For each actual closing Query, use stable softmax over keys `j <= q`. Stored
  row scores stay raw; future closing weights are exactly zero for q1–q3.
- Sum final-Query suffix weight deltas to recover delta M_U. D is the difference
  of log masses; its change splits into a suffix log-mass change and the negative
  Original log-mass change. Both sum to the original delta D.
- Derive raw B, B squared, symmetric/antisymmetric components and their squares
  per Head/sample. Save GCG, Shuffled, signed differences, mean absolute
  differences, and squared raw differences.
- `sum(delta_square) = F_G^2 - F_S^2`, **not** F_G-F_S. The additional exact
  contribution `delta_square/(F_G+F_S)` sums to delta F. Use the same construction
  for the symmetric and antisymmetric norms; both-zero norms have zero norm change.
- Divide each component's squared cells by that Head/sample's full squared norm
  before averaging. Their sums recover R_sym and R_asym, which sum to one for
  nonzero B. A zero B has undefined ratios and stops this diagnostic rather than
  silently substituting epsilon/zero. There are no zero blocks in the source data.
- Never causal-mask B. Its future elements are part of the saved pre-mask score,
  and signed antisymmetry is not actual directional information flow.

The averaging order is feature calculation per Head/sample, equal-weight Head
mean within each view, then sample statistics. In particular, softmax is not
applied to averaged scores, and normalized ratios are not calculated from an
averaged B. Save `mean__square_*` and `square_of_mean_raw__*` separately.
`mean__abs_delta_*` averages absolute differences **before** Head averaging;
`mean_absolute__delta_*` takes the absolute sample-view difference **after** Head
averaging. These distinguish cancellation within Heads from sample cancellation.
Sample SD uses ddof=1; directional counts use absolute tolerance 1e-12.

## Artifact schema

- `manifest.json`: frozen selection, all Index/view orders, member flat Head
  indices (`Layer*32+Head`), source references/hashes and averaging definitions.
- `matrix_per_sample.npz`: derived fields `[20 samples,58 views,20 Query,20 Key]`.
- `matrix_summary_18/20.npz`: `mean__FIELD`, `median__FIELD`, `std__FIELD`,
  `mean_absolute__FIELD`, `positive_count__FIELD`, `negative_count__FIELD`.
- `row_profiles/index_NNNN.npz`: variable-L `[view,4 closing Query,L Key]` raw
  score/weight profiles and paired changes; region masses and D components too.
- `row_summary_18/20.npz`: suffix-relative row fields `[view,4 Query,20 Key]`,
  closing-relative fields `closing_*` `[view,4 Query,4 Key]`, and region quantities.
- `all_head_scalars.npz`: query_metrics `[20,2,1024,4,3]` in M_U/M_O/D order;
  norm_metrics `[20,2,1024,5]` in F/F_sym/F_asym/R_sym/R_asym order; view-level
  scalar/query arrays and mean4 (mean of metrics already calculated per Query).
- `scalar_consistency.csv`, `suffix_position_summary.csv`,
  `energy_position_contributions.csv`, `four_query_summary.csv`: readable results.
- `figures/`: sixteen combined sheets; no unconditional per-sample image dump.
  Numerical data is stored for both eighteen and twenty samples.
- `validation.json`, `validation_postrun.json`, `report.md`, `interpretation.md`:
  checks, Japanese findings and limitations. The last two audit/interpretation
  artifacts are delivery additions, separate from initial `output_hashes.json`.

## Reproduction and optional single-Index rendering

```bash
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_gcg_shuffled_tokens tests.test_gcg_shuffled_four_metrics -v
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_tokens.py --output-dir data/outputs/gcg_shuffled_token_analysis/NEW_RUN
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_tokens.py --analysis-dir RUN --render-index 68 --view L26H21
```

Use a nonexistent NEW_RUN. Saved individual/group/all view IDs can be rendered
later without model execution. Each rendering creates a new diagnostic directory.
All source hashes are checked and no source artifacts are overwritten. The same
eighteen samples select the Heads and assess consistency, so this is not an
independent validation or a causal/statistical-significance result.
