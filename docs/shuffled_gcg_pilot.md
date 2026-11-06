# Shuffled GCG pilot (legacy selection20, five pairs)

This is a development diagnostic, not an independent evaluation. It uses only
the legacy Llama2 GCG selection20. The independent AdvBench 220–319 data is not
read or selected. No GCG optimization, attention extraction, or intervention runs.

## Frozen protocol

- Select five indices with
  `sorted(random.Random(42).sample(sorted(selection20_indices), 5))`.
  The selected indices are 68, 73, 91, 96, 208. Do not replace samples after
  observing responses. Invalid boundaries are recorded as exclusions and prevent
  the five-pair GPU stage from running.
- Verify historical source hashes, original/prompt correspondence, historical
  success, and exact formatted input equality with the historical baseline.
- Use the fast tokenizer's offsets on the actual formatted GCG input. Require
  exact character/token boundaries; reject crossing tokens and special tokens.
  The suffix character interval includes the separating space after Original.
  Verify the tokenized Original prefix separately. Closing ` [/INST]` stays out
  of the shuffled interval.
- Shuffle only the identified input ID slice. Each example uses a separate
  `random.Random(20261107 + dataset_index)` and `shuffle`. Reject identity draws.
  Require the same length, token multiset, prefix, and closing template. Do not
  decode/re-encode the shuffled input before passing it to the model.
- One permutation per sample; evaluate GCG then shuffled, with no outcome-based
  reselection. All five suffixes contain 20 tokens.
- Local Llama-2-7b-chat-hf, FP16, eager attention, batch size 1, no system prompt,
  greedy generation, 32 new tokens maximum, cache disabled, generation seed 42.
- Reuse `generate_with_first_logits` and its generation settings from
  `scripts/evaluate_gcg_first_token_reference_ablation.py`, and the 43
  case-sensitive refusal substrings from `utils/evaluate_utils.py`.
  First-step logits are checked by that helper but are not saved or analyzed.
- Heuristic success means absence of a configured refusal substring. It does
  not establish harmful compliance. Human labels remain null until a human
  reviews the responses. Truncation at 32 tokens can limit content assessment.

## Reproduction

From the repository root, use the existing `harmbench-blackwell` environment:

```bash
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B -m unittest tests.test_shuffled_gcg_pilot tests.test_gcg_first_token_reference_ablation -v
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/evaluate_shuffled_gcg_pilot.py --stage prepare --output-dir data/outputs/shuffled_gcg_pilot/NEW_RUN
/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/evaluate_shuffled_gcg_pilot.py --stage run --output-dir data/outputs/shuffled_gcg_pilot/NEW_RUN
```

Choose a new, nonexistent output directory. Run the GPU stage only after CPU
preparation and GPU availability checks. If normal sandbox CUDA is unavailable,
follow AGENTS.md and recheck/run with the approved execution permissions; record
that adjustment using `--execution-context`. Never substitute CPU execution.
The prepared source hashes must still match at runtime.

Outputs:

- `plan.json`: source hashes, selection, offsets, both full input ID sequences,
  suffix ID sequences, seeds, preservation checks, historical responses, and
  frozen settings.
- `runtime.json`, `model_weights.json`, `loaded_model.json`: software, GPU,
  execution permissions, local safetensors hashes, model/tokenizer settings.
- `results.jsonl`: ten responses, response IDs, automatic decisions, historical
  response equality for GCG, error details, and unfilled human review labels.
- `summary.json`: aggregate automatic results and generation anomalies.

All output files are created exclusively. A partial/failed run must be inspected
and retained; do not rerun into the same directory. Research outputs stay ignored
by Git. Source code, tests, and this protocol can be versioned.
