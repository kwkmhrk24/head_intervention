# Three-condition Shuffled GCG comparison dataset

Development/diagnostic data: all twenty GCG successes in the existing legacy
selection manifest. No independent100 data, GCG optimization, attention
extraction, or intervention is used. No outcome-based exclusions or seed search.

## Construction and response provenance

`scripts/build_shuffled_gcg_comparison.py` reuses the pilot's validated token
boundary and permutation helpers. The pilot's default five-example selection is
unchanged; `use_all_selection=True` exposes its existing selection20 records.

The three conditions are Original, GCG, and Shuffled. Shuffled is a permutation
of only the actual input ID suffix interval, using `random.Random(20261107 +
dataset_index).shuffle`. An identical order is rejected, without checking any
response. No decoding/re-encoding is used to construct shuffled inputs.

The five existing pilot permutations and their ten GCG/Shuffled responses are
reused only after verifying exact input IDs, seeds, evaluation settings, source
data/tokenizer/checker/generation helper hashes, software versions, and actual
local model weight hashes. The expanded preparation helper's historical version
is recorded by the pilot plan; its default behavior remains covered by tests.

Generate Original responses for all twenty and GCG/Shuffled responses for the
other fifteen: fifty new generations. Older Original results lack saved input
IDs and are used as correspondence/reproduction references rather than copied.
All twenty final GCG responses must match historical responses and be heuristic
successes. Discrepancies fail validation; outputs are retained, never filtered.

Generation: local Llama-2-7b-chat-hf, FP16, eager attention, batch size 1,
system prompt absent, greedy, maximum 32 new tokens, cache disabled, seed 42.
Reuse the existing generation helper and 43 case-sensitive refusal substrings.
Heuristic success does not establish harmful compliance. All human labels remain
null, human review is `not_reviewed`, and actual jailbreak success is `unknown`.

## Schema and analysis boundaries

`dataset.jsonl` contains twenty rows keyed by `dataset_index`. Each row includes
Original text, all three input ID lists, both suffix ID lists, suffix length,
shuffle seed/attempts, source references, and a `conditions` mapping.

Each condition contains its authoritative input IDs, `token_ranges`, zero-based
`final_input_token_position`, `final_input_token_id`, and `evaluation` with the
response, response IDs, heuristic decision, human review state, and response
provenance. Generation/model/tokenizer settings are referenced in `plan.json`,
`runtime.json`, `model_weights.json`, and `loaded_model.json`.

Ranges are half-open and partition the input into `template_prefix`, `original`,
`suffix`, `template_closing`. The Original condition has an empty suffix range.
The character interval for Original includes the whitespace attached to its first
SentencePiece token. Prefix, Original tokens, and closing-template tokens are
identical across all conditions; deleting the GCG suffix exactly reproduces the
Original condition. The closing template remains after the suffix, and its last
token is the final input token used for a future Query.

The Original condition's final Query is earlier by the suffix length. Future
analysis should use the saved ranges, feed exact saved IDs (especially Shuffled),
capture actual position IDs, and extract post-RoPE/pre-mask/pre-softmax QK scores.
No such score extraction is implemented or executed here. Avoid replacing these
IDs with tokenized decoded text, or using existing antisymmetric tail statistics
as raw Query-to-Key scores.

## Reproduction and validation

From the repository root, choose a nonexistent output directory:

```bash
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_shuffled_gcg_comparison tests.test_shuffled_gcg_pilot tests.test_gcg_first_token_reference_ablation -v
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/build_shuffled_gcg_comparison.py --stage prepare --output-dir data/outputs/shuffled_gcg_comparison/NEW_RUN
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/build_shuffled_gcg_comparison.py --stage run --output-dir data/outputs/shuffled_gcg_comparison/NEW_RUN
```

Follow AGENTS.md GPU checks before the run stage. If sandbox-exempt execution is
required, record it with `--execution-context`. No CPU or model fallback.
Files are exclusively created; partial runs must be inspected and retained.

`plan.json` freezes sources and inputs. `results.jsonl` records all sixty responses
(including reused results); `dataset.jsonl` joins them to the twenty input triplets.
`summary.json` records counts, validation status and artifact hashes. The helper
`validate_sample` checks ranges, token multisets, unchanged flanks and seed replay;
`validate_results` checks response coverage, inputs, automatic labels, response
IDs and GCG reproduction. Tests also reject changed Original tokens, malformed
boundaries, wrong Query positions, mismatched reuse seeds/labels and fake human
review labels. Research artifacts remain ignored by Git.
