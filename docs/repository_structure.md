# Current repository structure

## Branches

`research/jailbreak-defense` is the canonical branch for current experiments.
It contains the complete Llama/Mistral research history through GCG reference
factorization plus the previously divergent cross-attack candidate evaluation.

`main` retains the initial import. `support-qwen2` is intentionally separate:
its five Qwen-specific commits are not part of the canonical Llama/Mistral
lineage and include model, dataset-preparation, and labeling changes.

Historical feature, integration, and reproduction branch names were removed
locally after their tips were verified reachable from the canonical branch or
the safety tag `pre-repository-cleanup-20260823`. The AutoDAN baseline branch
and `reproduce-original` contained patch-equivalent changes, while the
cross-attack candidate commit was cherry-picked into the canonical branch.
Removing these names does not remove their commit history.

## Maintained entrypoint groups

| Group | Entrypoints | Responsibility |
| --- | --- | --- |
| Attention capture and analysis | `analyze_attention_scores.py`, `analyze_attention_score_all_heads_distributions.py`, `analyze_attention_score_element_distributions.py`, `analyze_matched_attention_head_metrics.py` | Capture tail16 scores and analyze strict-lower-triangle, `S_sym`, `S_asym`, head, and element distributions. |
| Direct score intervention | `evaluate_asym_score_intervention.py` | Shared Llama/Mistral and AutoDAN/GCG evaluation with condition/beta options. |
| AutoDAN activation intervention | `evaluate_autodan_head_sets.py`, `export_intervention_head_configs.py` | Evaluate legacy or provenance-bearing candidate head sets. |
| GCG reference experiment | `prepare_gcg_wasserstein_reference.py`, `evaluate_gcg_wasserstein_reference_intervention.py` | Build fixed Wasserstein/reference artifacts and run the reference intervention. |
| First-token and factorization | `evaluate_gcg_first_token_reference_ablation.py`, `evaluate_gcg_first_token_holdout.py`, `diagnose_gcg_first_token_generalization.py`, `diagnose_gcg_reference_factorization.py` | Preserve the current `alpha=0.5`, first-token-only, holdout, and factorization workflow. |
| Offline audit | `audit_gcg_reference.py`, `analyze_cross_run_intervention_transitions.py` | Validate saved artifacts and compare prompt-level transitions without model loading. |

Other preparation, plotting, and reproduction scripts remain because they have
distinct inputs or outputs. Similar filenames alone were not treated as a
reason to combine them.

## Reusable modules

| Module | Responsibility |
| --- | --- |
| `utils/artifact_io.py` | Strict JSON/JSONL reads, exclusive artifact writes, timestamps, and SHA-256 hashing. |
| `utils/experiment_runtime.py` | Shared random seeding and Git provenance capture. |
| `utils/evaluate_utils.py` | Historical refusal heuristic and standard four-way ASR transitions. |
| `utils/asym_score_dataset.py` | Dataset normalization and split validation. |
| `utils/asym_score_headsets.py` | Model-specific paper and consensus head-set loading. |
| `utils/asym_score_intervention.py` | Direct `S_asym` score intervention. |
| `utils/reference_asym_score_intervention.py` | Reference reconstruction, Mean/Pair loading, and first-token timing. |
| `utils/gcg_wasserstein_reference.py` | Fixed split, tail16 extraction, Wasserstein ranking, and reference artifacts. |
| `utils/gcg_reference_audit.py` | Reusable audits behind `audit_gcg_reference.py`. |
| `utils/intervention_transitions.py` | Cross-run pairing, transition summaries, overlaps, and statistical tests. |
| `utils/gcg_*_diagnostics.py` | Method-specific logit, KL, margin, head, holdout, and factorization summaries. |

The old `summarize_gcg_*` and `verify_gcg_wasserstein_reference_artifacts.py`
entrypoints were removed because their behavior is covered by the four
`audit_gcg_reference.py` subcommands. Their source remains recoverable from Git
history. No file under `data/raw`, `data/derived`, or `data/outputs` is part of
this cleanup.
