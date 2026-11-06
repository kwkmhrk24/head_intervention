#!/usr/bin/env python3
"""Five-pair diagnostic on the legacy GCG selection split; no attention hooks."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluate_utils import REFUSAL_SUBSTRINGS, check_success_with_details
from utils.gcg_first_token_reference_ablation import load_selection_development_records
from utils.prompt_utils import apply_template

MODEL = Path('/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf')
SPLIT = ROOT / 'data/outputs/gcg_wasserstein_reference/llama2/seed42_selection20/codex_gpu_20260802T061114Z/split_manifest.json'
ORIGINAL = ROOT / 'data/derived/llama2/gcg/original_prompts.jsonl'
JAILBREAK = ROOT / 'data/derived/llama2/gcg/jailbreak_prompts.jsonl'
BASELINE = ROOT / 'data/outputs/multidataset_asym_score_intervention/beta05_full_20260722_041303/llama_gcg/no_intervention/prompt_results.jsonl'
SELECTION_SEED = 42
SHUFFLE_BASE_SEED = 20261107
GENERATION = dict(do_sample=False, temperature=None, top_p=None,
                  max_new_tokens=32, use_cache=False,
                  return_dict_in_generate=True, output_scores=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def token_span(offsets, start, end):
    """Reject any token overlapping either original/suffix/template boundary."""
    hits = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
    if not hits or hits != list(range(hits[0], hits[-1] + 1)):
        raise ValueError('suffix has no contiguous token span')
    if offsets[hits[0]][0] != start or offsets[hits[-1]][1] != end:
        raise ValueError('suffix token boundary does not match character boundary')
    if any(a < start or b > end or b <= a for a, b in (offsets[i] for i in hits)):
        raise ValueError('token crosses suffix boundary or has empty offset')
    return hits[0], hits[-1] + 1


def permute_suffix(ids, start, end, seed, special_ids=()):
    before = ids[start:end]
    if not (0 <= start < end <= len(ids)) or len(set(before)) < 2:
        raise ValueError('suffix cannot be nontrivially shuffled')
    if set(before) & set(special_ids):
        raise ValueError('special token inside suffix; exclude sample')
    rng = random.Random(seed)
    for attempt in range(1, 10001):
        after = before.copy()
        rng.shuffle(after)
        if after != before:
            result = ids[:start] + after + ids[end:]
            validate_permutation(ids, result, start, end)
            return result, attempt
    raise ValueError('no changed permutation after 10000 attempts')


def validate_permutation(before, after, start, end):
    if len(before) != len(after) or before[:start] != after[:start] or before[end:] != after[end:]:
        raise ValueError('non-suffix input tokens changed')
    if before[start:end] == after[start:end]:
        raise ValueError('suffix token order unchanged')
    if Counter(before[start:end]) != Counter(after[start:end]):
        raise ValueError('suffix token multiplicities changed')


def prepare_samples(tokenizer, *, use_all_selection=False):
    records, split = load_selection_development_records(
        split_manifest_path=SPLIT, jailbreak_input=JAILBREAK, baseline_results=BASELINE)
    if sha(ORIGINAL) != split['original_input_sha256']:
        raise ValueError('original source hash mismatch')
    originals = {r['index']: r['prompt'] for r in read_jsonl(ORIGINAL)}
    historical = {r['source_index']: r for r in read_jsonl(BASELINE)}
    split_records = {r['dataset_index']: r for r in split['records']}
    candidates = sorted(r['dataset_index'] for r in records)
    selected = candidates if use_all_selection else sorted(
        random.Random(SELECTION_SEED).sample(candidates, 5))
    output, exclusions = [], []
    for record in records:
        index = record['dataset_index']
        if index not in selected:
            continue
        try:
            original, prompt = originals[index], record['prompt']
            if hashlib.sha256(original.encode()).hexdigest() != split_records[index]['original_prompt_sha256']:
                raise ValueError('original prompt hash mismatch')
            if not prompt.startswith(original) or not prompt[len(original):].startswith(' '):
                raise ValueError('exact original plus suffix separator not found')
            formatted = apply_template([prompt], None, tokenizer, {'model_name': 'Llama'})[0]
            # Match the actual local template exactly before using char offsets.
            if formatted != '<s>[INST] ' + prompt + ' [/INST]':
                raise ValueError('unexpected user-only template or whitespace trimming')
            if formatted != historical[index]['formatted_prompt']:
                raise ValueError('formatted input differs from historical baseline')
            encoding = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
            ids = encoding['input_ids']
            char_start = len('<s>[INST] ') + len(original)
            char_end = len('<s>[INST] ') + len(prompt)
            start, end = token_span(encoding['offset_mapping'], char_start, char_end)
            if tokenizer(formatted[:char_start], add_special_tokens=False)['input_ids'] != ids[:start]:
                raise ValueError('original prefix token identity failed')
            seed = SHUFFLE_BASE_SEED + index
            shuffled, attempts = permute_suffix(ids, start, end, seed, tokenizer.all_special_ids)
            output.append(dict(
                dataset_index=index, prompt_id=record['prompt_id'],
                source_row_position=record['row_position'], original_prompt=original,
                original_prompt_sha256=split_records[index]['original_prompt_sha256'],
                gcg_prompt=prompt, formatted_prompt=formatted,
                suffix_text_with_separator=prompt[len(original):],
                suffix_text=prompt[len(original) + 1:],
                suffix_boundary_convention='includes leading separator space in character offsets',
                suffix_character_range=[char_start, char_end], suffix_token_range=[start, end],
                suffix_token_count=end-start, original_suffix_token_ids=ids[start:end],
                shuffled_suffix_token_ids=shuffled[start:end], shuffle_seed=seed,
                shuffle_attempts=attempts, gcg_input_ids=ids, shuffled_input_ids=shuffled,
                token_offsets=encoding['offset_mapping'], order_changed=True,
                suffix_token_count_preserved=True, token_multiset_preserved=True,
                outside_suffix_identical=True, special_tokens_in_suffix=False,
                historical_response=record['baseline_response'], historical_success=True))
        except ValueError as exc:
            exclusions.append(dict(dataset_index=index, reason=str(exc)))
    return selected, output, exclusions


def source_hashes():
    paths = [SPLIT, ORIGINAL, JAILBREAK, BASELINE, Path(__file__),
             ROOT/'utils/evaluate_utils.py', ROOT/'utils/prompt_utils.py',
             ROOT/'utils/gcg_first_token_reference_ablation.py',
             ROOT/'scripts/evaluate_gcg_first_token_reference_ablation.py']
    paths += [MODEL / name for name in ('tokenizer.json', 'tokenizer_config.json',
                                       'special_tokens_map.json', 'config.json', 'generation_config.json')]
    return {str(p): sha(p) for p in paths}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def prepare(directory):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    selected, samples, exclusions = prepare_samples(tokenizer)
    # No replacement sampling: failed boundaries remain exclusions in the plan.
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),
                role='development_diagnostic_pilot', model=str(MODEL),
                selection_seed=SELECTION_SEED,
                selection_method='sorted(Random(42).sample(sorted(selection20 dataset indices), 5))',
                shuffle_base_seed=SHUFFLE_BASE_SEED,
                shuffle_seed_formula='20261107 + dataset_index; Python Random.shuffle with rejection of identity',
                selected_indices=selected, samples=samples, exclusions=exclusions,
                generation=GENERATION, dtype='float16', attention_implementation='eager',
                system_prompt=None, batch_size=1, generation_seed=42,
                special_token_ids=tokenizer.all_special_ids,
                tokenizer_class=type(tokenizer).__name__, source_hashes=source_hashes(),
                git_head=git('rev-parse', 'HEAD'), git_branch=git('branch', '--show-current'),
                git_status=git('status', '--short'),
                refusal_substrings=list(REFUSAL_SUBSTRINGS),
                independent100_used=False, attention_extraction=False, intervention=False)
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory/'plan.json', plan)
    print(json.dumps(dict(selected=selected, eligible_count=len(samples), exclusions=exclusions)))


def run(directory, execution_context):
    import platform
    import torch
    import transformers
    from scripts.evaluate_gcg_first_token_reference_ablation import (
        GENERATION_CONFIG, generate_with_first_logits, load_model_and_tokenizer)
    from utils.experiment_runtime import seed_everything

    plan = json.loads((directory/'plan.json').read_text())
    if source_hashes() != plan['source_hashes'] or plan['generation'] != GENERATION_CONFIG:
        raise ValueError('sources or generation settings changed since CPU preparation')
    if len(plan['samples']) != 5 or plan['exclusions']:
        raise ValueError('five eligible samples required; inspect exclusions, do not replace silently')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    # Exclusive runtime file prevents accidentally evaluating a run twice.
    runtime = dict(started_at=datetime.now(timezone.utc).isoformat(),
                   plan_sha256=sha(directory/'plan.json'), execution_context=execution_context,
                   python=platform.python_version(), torch=torch.__version__,
                   transformers=transformers.__version__, cuda=torch.version.cuda,
                   gpu_name=torch.cuda.get_device_name(0), cuda_device_count=torch.cuda.device_count(),
                   nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
                   git_head=git('rev-parse', 'HEAD'), git_status=git('status', '--short'))
    write_new(directory/'runtime.json', runtime)
    weight_index = MODEL/'model.safetensors.index.json'
    shards = sorted(set(json.loads(weight_index.read_text())['weight_map'].values()))
    print('Recording local safetensors weight hashes before model loading', flush=True)
    write_new(directory/'model_weights.json', dict(
        index_sha256=sha(weight_index),
        shard_sha256={name: sha(MODEL/name) for name in shards}))
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(MODEL)
    if model.dtype != torch.float16 or model.config._attn_implementation != 'eager':
        raise RuntimeError('model dtype or attention implementation mismatch')
    if any(type(layer.self_attn).__name__ != 'LlamaAttention' for layer in model.model.layers):
        raise RuntimeError('expected unpatched eager LlamaAttention')
    selected, samples, exclusions = prepare_samples(tokenizer)
    # JSON canonicalization removes only tuple/list representation differences.
    if json.loads(json.dumps(samples)) != plan['samples'] or exclusions or selected != plan['selected_indices']:
        raise RuntimeError('loaded tokenizer does not reproduce frozen input plan')
    write_new(directory/'loaded_model.json', dict(config=model.config.to_dict(),
               generation_config=model.generation_config.to_dict(), tokenizer_class=type(tokenizer).__name__))
    results = []
    with (directory/'results.jsonl').open('x') as handle:
        for sample in plan['samples']:
            start, end = sample['suffix_token_range']
            validate_permutation(sample['gcg_input_ids'], sample['shuffled_input_ids'], start, end)
            for condition in ('gcg', 'shuffled'):
                ids = sample[condition + '_input_ids']
                generation = generate_with_first_logits(model, tokenizer, torch.tensor([ids], dtype=torch.long))
                generation.pop('first_logits')
                if not generation['generation_failed']:
                    success, _ = check_success_with_details(generation['response'])
                    if success != generation['success']:
                        raise RuntimeError('success checker mismatch')
                row = dict(dataset_index=sample['dataset_index'], condition=condition,
                           input_ids=ids, **generation,
                           historical_response_equal=(generation['response'] == sample['historical_response']) if condition == 'gcg' else None,
                           content_review_required=True, human_jailbreak_success=None,
                           human_review_status='not_reviewed',
                           automatic_rule='43 case-sensitive refusal substrings; absence means heuristic success')
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
                results.append(row)
                print(f"index={row['dataset_index']} condition={condition} success={row['success']} error={row['generation_error']}", flush=True)
    groups = {c: [r for r in results if r['condition'] == c] for c in ('gcg', 'shuffled')}
    pairs = [(groups['gcg'][i], groups['shuffled'][i]) for i in range(5)]
    summary = dict(status='complete', completed_at=datetime.now(timezone.utc).isoformat(),
                   sample_count=5, response_count=len(results),
                   heuristic_success_counts={c: sum(r['success'] is True for r in rows) for c, rows in groups.items()},
                   generation_failure_count=sum(r['generation_failed'] for r in results),
                   empty_response_count=sum(r['empty_response'] is True for r in results),
                   gcg_success_reproduced_all=all(r['success'] is True for r in groups['gcg']),
                   transitions={name: sum(a['success'] is x and b['success'] is y for a, b in pairs)
                                for name, x, y in [('success_to_failure', True, False),
                                                   ('success_to_success', True, True),
                                                   ('failure_to_success', False, True),
                                                   ('failure_to_failure', False, False)]},
                   human_jailbreak_success=None, content_review_required=True,
                   results_sha256=sha(directory/'results.jsonl'),
                   peak_allocated_vram_mib=torch.cuda.max_memory_allocated()/1024**2)
    write_new(directory/'summary.json', summary)
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execution-context', default='normal_sandbox')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output_dir)
    else:
        run(args.output_dir, args.execution_context)


if __name__ == '__main__':
    main()
