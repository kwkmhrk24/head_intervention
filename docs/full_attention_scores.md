# Full Llama attention score extraction

This pipeline saves only raw scores for the existing legacy selection20
Original/GCG/Shuffled comparison dataset: 60 saved input ID sequences. It does
not tokenize input text, generate responses, evaluate ASR, optimize suffixes,
decompose scores, rank heads, plot research data, or use independent100.

## Definition and precision

`S = Q K^T / sqrt(head_dim)` uses FP16 post-RoPE Q/K, cast to FP32 **before**
the matrix product and scaling. Score storage is FP32. TF32 is disabled and
PyTorch float32 matmul precision is `highest`. Model weights and ordinary eager
attention remain FP16. Each forward is batch one, unpadded, in inference mode,
with `model.eval()`, cache disabled, and no output attentions/hidden-state history.

The original eager backbone still performs its native causal attention to
produce each subsequent layer's hidden states. A read-only self-attention
pre-hook reuses `compute_post_rope_qk` from `scripts/analyze_attention_scores.py`
with the **actual** hidden states and position IDs passed to that layer. It
recomputes Q/K and saves the entire matrix before adding any mask or softmax.
The native forward is not replaced and saved scores are never fed back into it.

This follows the existing FP32 score recomputation method. It differs from
`utils/gcg_wasserstein_reference.py`, which calculates FP16 score products before
converting its tail/component results. No tail selection or Sym/Asym conversion
is reused here. The installed model code/versions and source hashes are frozen.

Future-direction elements are intentionally saved. Metadata records the actual
layer position IDs and cache positions, the observed causal mask shape/dtype,
and its rule `key_index <= query_index`. It verifies the actual mask obeys that
rule. Position IDs are captured, not supplied as an assumed range by the script.

## Stages and restart

From the repository root, using the existing environment and a new output path:

```bash
PYTHON=/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python
$PYTHON -B -m unittest tests.test_full_attention_scores tests.test_all_head_attention_distributions tests.test_shuffled_gcg_comparison tests.test_shuffled_gcg_pilot -v
$PYTHON -B scripts/extract_full_attention_scores.py --stage prepare --output-dir data/outputs/full_attention_scores/NEW_RUN
$PYTHON -B scripts/extract_full_attention_scores.py --stage smoke --output-dir data/outputs/full_attention_scores/NEW_RUN
$PYTHON -B scripts/extract_full_attention_scores.py --stage run --output-dir data/outputs/full_attention_scores/NEW_RUN
```

Follow AGENTS.md GPU availability checks. If normal sandbox CUDA is unavailable,
recheck and execute in the approved sandbox-exempt environment, recording the
adjustment with `--execution-context`. No model/precision/CPU fallback occurs.

Preparation checks the dataset hash, all ID/range mappings, model configuration,
total NPY capacity and disk space. It freezes every input and source hash. GPU
execution checks free VRAM against weight size plus a 2 GiB workspace margin,
verifies model shard hashes against the comparison experiment, and records the
GPU/software/runtime state. One layer of one input is written at a time to a CPU
memory-mapped NPY; full score collections are never retained on GPU.

Smoke extracts Index 68's three conditions first. A temporary observer of native
RoPE outputs returns those tensors unchanged, comparing saved matrices against
the existing per-head FP32 method on the native Q/K (rtol 1e-5, atol 1e-4).
It also checks their common-prefix score square (rtol .02, atol .1), allowing for
FP16 backbone sequence-shape rounding. These are numerical validation checks,
not research metrics. Only after smoke passes can the full stage proceed.

Completed inputs are hash/shape/dtype/finiteness checked and skipped on resume.
Repeat the same `smoke` or `run` command to resume. Failed partial attempts are
retained and retried in a new attempt directory; completed files are not replaced.
A process lock prevents simultaneous writers. Changed sources, software or
corrupted completed files cause failure rather than silent overwrite.

## Artifacts and reading

- `plan.json`: input list, model revision/configuration, tokenizer configuration,
  source/dataset hashes, capacity, precision, conditions, Git state.
- `runtimes/*.json`, `model_weights.json`, `loaded_model.json`: runtime metadata.
- `inputs/index_NNNN/CONDITION/complete.json`: pointer/hash of completed metadata.
- `inputs/.../attempt_.../scores.npy`: FP32 `[layer, head, query, key]` array.
- Same attempt's `metadata.json`: exact IDs/ranges/final token, captured positions,
  numerical method, mask rule, source-plan/runtime references and score hash.
- `smoke_validation.json`: native-reference and prefix numerical checks.
- `manifest.jsonl`: 60 rows mapping index/condition to NPY and completion pointer.
- `summaries/*.json`: immutable completion/capacity summaries.

```python
import json
from pathlib import Path
import numpy as np

root = Path('data/outputs/full_attention_scores/NEW_RUN')
record = json.loads((root / 'manifest.jsonl').read_text().splitlines()[0])
S = np.load(root / record['score_file'], mmap_mode='r', allow_pickle=False)
pointer = json.loads((root / record['completion_file']).read_text())
metadata = json.loads((root / pointer['metadata_file']).read_text())
positions = metadata['extraction']['position_ids']
ranges = metadata['input']['token_ranges']
# S[layer, head] is the complete square needed for future Sym/Asym decomposition.
# The transpose axes for such future work are the final two (query, key) axes.
```

Do not re-tokenize decoded IDs for subsequent work. The saved position IDs and
input ranges provide the mapping to original/suffix/template regions. All raw
experiment artifacts stay under the ignored `data/outputs/` directory.
