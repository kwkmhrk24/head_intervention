# Mistral Reproduction: Phase 2 Preparation

This phase only checks the tokenizer and the availability of Mistral-specific datasets. It does not run AIE computation, intervention-vector extraction, ASR evaluation, or attention-score decomposition.

## Model path

```bash
MODEL=/mnt/mp_nas_mks/labmember/h.kawakami/models/Mistral-7B-Instruct-v0.2/
```

## Inspect the target token

The current Mistral target token ID in `utils/intervention_utils.py` is `12875`. Inspect its decoded form and compare the tokenization of `Sure`, space-prefixed `Sure`, and newline-prefixed `Sure` without loading the model itself:

```bash
python scripts/inspect_target_token.py \
  --model "$MODEL" \
  --strings "Sure" " Sure" "\nSure"
```

The script expands the CLI sequence `\n` to an actual newline. It reports tokenization both with and without special tokens, decodes each token separately, and compares the chat-template suffix before and after appending one ASCII space.

Do not decide whether to pass `--append_target_space` until these results have been inspected.

## Check for Mistral datasets

List likely Mistral, attack, original, jailbreak, and refusal files:

```bash
find data -maxdepth 5 -type f | grep -Ei "mistral|autodan|gcg|jailbreak|original|refusal"
```

Check whether these candidate paths exist:

```text
data/derived/mistral/autodan/original_prompts.jsonl
data/derived/mistral/autodan/jailbreak_prompts.jsonl
data/derived/mistral/refusal/refusal_prompts.jsonl
data/derived/mistral/gcg/original_prompts.jsonl
data/derived/mistral/gcg/jailbreak_prompts.jsonl
```

## Current cautions

- Using Llama2 AutoDAN/GCG prompts with Mistral is a reference experiment, not a strict Mistral reproduction.
- Confirm that paired Mistral AutoDAN/GCG original and jailbreak prompts exist before starting experiments.
- If Mistral-specific data is unavailable, decide the data-preparation policy in the next phase.
- Decide whether to use `--append_target_space` only after reviewing the target-token diagnostics.
