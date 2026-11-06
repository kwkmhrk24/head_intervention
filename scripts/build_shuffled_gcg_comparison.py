#!/usr/bin/env python3
"""Build/evaluate the three-condition legacy selection20 dataset, without attention hooks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_shuffled_gcg_pilot as pilot
from utils.evaluate_utils import REFUSAL_SUBSTRINGS, check_success_with_details
from utils.prompt_utils import apply_template

PILOT = ROOT/'data/outputs/shuffled_gcg_pilot/legacy_selection20_seed42_20261106T165429Z'
OLD_ORIGINAL = pilot.SPLIT.parent/'selection_original_prompt_results.jsonl'
CONDITIONS = ('original', 'gcg', 'shuffled')


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.loads(json.dumps(value))


def make_condition(ids, original_start, original_end, suffix_end):
    return dict(input_ids=ids, final_input_token_position=len(ids)-1,
                final_input_token_id=ids[-1], token_ranges=dict(
                    template_prefix=[0, original_start], original=[original_start, original_end],
                    suffix=[original_end, suffix_end], template_closing=[suffix_end, len(ids)]))


def validate_sample(sample):
    """Validate the partition and exact Original/template identities across three inputs."""
    conditions = sample['conditions']
    for condition in CONDITIONS:
        item = conditions[condition]
        ids, ranges = item['input_ids'], item['token_ranges']
        cursor = 0
        for region in ('template_prefix', 'original', 'suffix', 'template_closing'):
            start, end = ranges[region]
            if start != cursor or not start <= end <= len(ids):
                raise ValueError('token ranges do not partition input')
            cursor = end
        if cursor != len(ids) or item['final_input_token_position'] != len(ids)-1 or item['final_input_token_id'] != ids[-1]:
            raise ValueError('final query position or range coverage mismatch')
    original, gcg, shuffled = (conditions[c] for c in CONDITIONS)
    for region in ('template_prefix', 'original', 'template_closing'):
        values = [c['input_ids'][slice(*c['token_ranges'][region])] for c in (original, gcg, shuffled)]
        if values[0] != values[1] or values[0] != values[2]:
            raise ValueError(f'{region} token identity failed')
    start, end = gcg['token_ranges']['suffix']
    if shuffled['token_ranges'] != gcg['token_ranges']:
        raise ValueError('shuffled token ranges changed')
    pilot.validate_permutation(gcg['input_ids'], shuffled['input_ids'], start, end)
    if original['token_ranges']['suffix'][0] != original['token_ranges']['suffix'][1]:
        raise ValueError('Original condition contains a suffix')
    if original['input_ids'] != gcg['input_ids'][:start] + gcg['input_ids'][end:]:
        raise ValueError('suffix deletion does not reproduce Original input')
    expected, attempts = pilot.permute_suffix(gcg['input_ids'], start, end, sample['shuffle_seed'], sample['special_token_ids'])
    if expected != shuffled['input_ids'] or attempts != sample['shuffle_attempts']:
        raise ValueError('shuffle seed reproduction failed')
    if (sample['original_input_ids'] != original['input_ids'] or sample['gcg_input_ids'] != gcg['input_ids']
            or sample['shuffled_input_ids'] != shuffled['input_ids']
            or sample['gcg_suffix_token_ids'] != gcg['input_ids'][start:end]
            or sample['shuffled_suffix_token_ids'] != shuffled['input_ids'][start:end]
            or sample['suffix_token_count'] != end-start):
        raise ValueError('top-level token fields differ from conditions')


def build_samples(tokenizer):
    indices, legacy, exclusions = pilot.prepare_samples(tokenizer, use_all_selection=True)
    if exclusions or len(legacy) != 20:
        raise ValueError(f'exact selection20 required; invalid boundaries: {exclusions}')
    historical_original = {r['dataset_index']: r for r in pilot.read_jsonl(OLD_ORIGINAL)}
    rows = []
    for sample in legacy:
        index = sample['dataset_index']
        start, end = sample['suffix_token_range']
        formatted = apply_template([sample['original_prompt']], None, tokenizer, {'model_name': 'Llama'})[0]
        if formatted != '<s>[INST] ' + sample['original_prompt'] + ' [/INST]':
            raise ValueError('unexpected Original template')
        original_encoding = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
        original_ids = original_encoding['input_ids']
        # Include the single whitespace before Original in its character interval,
        # because SentencePiece attaches it to the first Original token.
        os, oe = pilot.token_span(original_encoding['offset_mapping'], len('<s>[INST]'), len('<s>[INST] ') + len(sample['original_prompt']))
        gs, ge = pilot.token_span(sample['token_offsets'], len('<s>[INST]'), sample['suffix_character_range'][0])
        if oe != start or ge != start or gs != os:
            raise ValueError('Original token interval differs across conditions')
        old = historical_original[index]
        if old['prompt'] != sample['original_prompt'] or old['formatted_prompt'] != formatted:
            raise ValueError('historical Original correspondence failed')
        row = dict(dataset_index=index, prompt_id=sample['prompt_id'],
                   original_prompt=sample['original_prompt'], gcg_prompt=sample['gcg_prompt'],
                   original_input_ids=original_ids, gcg_input_ids=sample['gcg_input_ids'],
                   shuffled_input_ids=sample['shuffled_input_ids'],
                   gcg_suffix_token_ids=sample['original_suffix_token_ids'],
                   shuffled_suffix_token_ids=sample['shuffled_suffix_token_ids'],
                   suffix_token_count=sample['suffix_token_count'], shuffle_seed=sample['shuffle_seed'],
                   shuffle_attempts=sample['shuffle_attempts'], special_token_ids=tokenizer.all_special_ids,
                   formatted_original=formatted, formatted_gcg=sample['formatted_prompt'],
                   shuffled_input_representation='authoritative input IDs; never decode then re-encode',
                   conditions=dict(original=make_condition(original_ids, os, oe, oe),
                       gcg=make_condition(sample['gcg_input_ids'], gs, start, end),
                       shuffled=make_condition(sample['shuffled_input_ids'], gs, start, end)),
                   historical_gcg_response=sample['historical_response'], historical_gcg_success=True,
                   historical_original_response=old['response'],
                   sources=dict(original=str(pilot.ORIGINAL), gcg=str(pilot.JAILBREAK),
                       baseline=str(pilot.BASELINE), selection=str(pilot.SPLIT),
                       historical_original=str(OLD_ORIGINAL), source_row_position=sample['source_row_position']))
        validate_sample(row)
        rows.append(row)
    if [r['dataset_index'] for r in rows] != indices or any(i >= 220 for i in indices):
        raise ValueError('selection membership or independent-test isolation failed')
    return rows


def load_pilot_reuse(samples):
    plan = json.loads((PILOT/'plan.json').read_text())
    summary = json.loads((PILOT/'summary.json').read_text())
    runtime = json.loads((PILOT/'runtime.json').read_text())
    if summary['status'] != 'complete' or pilot.sha(PILOT/'results.jsonl') != summary['results_sha256']:
        raise ValueError('pilot result integrity failed')
    if runtime['plan_sha256'] != pilot.sha(PILOT/'plan.json'):
        raise ValueError('pilot plan integrity failed')
    if (plan['generation'] != pilot.GENERATION or plan['dtype'] != 'float16'
            or plan['attention_implementation'] != 'eager' or plan['system_prompt'] is not None
            or plan['batch_size'] != 1 or plan['model'] != str(pilot.MODEL)
            or plan['refusal_substrings'] != list(REFUSAL_SUBSTRINGS)):
        raise ValueError('pilot evaluation settings mismatch')
    current_hashes = pilot.source_hashes()
    # The pilot entrypoint now additionally exposes all-selection preparation.
    # Its historical version remains recorded in the pilot plan. Generation,
    # tokenizer, model configuration, checker, and source data must be unchanged.
    for path, digest in plan['source_hashes'].items():
        if path != str(Path(pilot.__file__)) and current_hashes.get(path) != digest:
            raise ValueError(f'pilot source identity mismatch: {path}')
    by_index = {r['dataset_index']: r for r in samples}
    old_samples = {r['dataset_index']: r for r in plan['samples']}
    if set(old_samples) != {68, 73, 91, 96, 208}:
        raise ValueError('unexpected pilot membership')
    for index, old in old_samples.items():
        row = by_index[index]
        for key in ('gcg_input_ids', 'shuffled_input_ids', 'shuffle_seed', 'shuffle_attempts', 'suffix_token_count'):
            if row[key] != old[key]:
                raise ValueError(f'pilot exact input/seed mismatch: {index}/{key}')
    reused = pilot.read_jsonl(PILOT/'results.jsonl')
    if len(reused) != 10 or len({(r['dataset_index'], r['condition']) for r in reused}) != 10:
        raise ValueError('pilot must contain ten unique responses')
    for row in reused:
        if row['condition'] not in ('gcg', 'shuffled') or row['dataset_index'] not in old_samples:
            raise ValueError('unexpected pilot response key')
        if row['generation_failed'] or row['input_ids'] != by_index[row['dataset_index']]['conditions'][row['condition']]['input_ids']:
            raise ValueError('pilot response input mismatch')
        if check_success_with_details(row['response'])[0] != row['success']:
            raise ValueError('pilot automatic label mismatch')
    return reused


def hashes():
    paths = [Path(__file__), ROOT/'tests/test_shuffled_gcg_comparison.py', OLD_ORIGINAL]
    paths += list(PILOT/name for name in ('plan.json', 'runtime.json', 'results.jsonl', 'summary.json', 'model_weights.json'))
    return {**pilot.source_hashes(), **{str(p): pilot.sha(p) for p in paths}}


def prepare(directory):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pilot.MODEL, use_fast=True, local_files_only=True)
    samples = build_samples(tokenizer)
    reused = load_pilot_reuse(samples)
    manifest = dict(schema_version=1, created_at=now(), role='development_diagnostic_selection20',
                    sample_count=20, condition_order=list(CONDITIONS),
                    selected_indices=[r['dataset_index'] for r in samples],
                    selection_method='all twenty indices from existing selection manifest; no outcome filtering',
                    generation=pilot.GENERATION, generation_seed=42, dtype='float16',
                    attention_implementation='eager', model=str(pilot.MODEL),
                    tokenizer_class=type(tokenizer).__name__, batch_size=1, system_prompt=None,
                    shuffle_seed_formula='20261107 + dataset_index; one nonidentity permutation per index',
                    range_convention='zero-based, half-open; Original includes its leading separator space; Original suffix range is empty',
                    position_policy='single unpadded sequence; future analysis should capture actual position_ids',
                    source_hashes=hashes(), pilot_reused_response_count=len(reused),
                    new_response_count=50, expected_total_responses=60,
                    human_review_status='not_reviewed', actual_jailbreak_success='unknown',
                    git_head=pilot.git('rev-parse', 'HEAD'), git_branch=pilot.git('branch', '--show-current'),
                    git_status=pilot.git('status', '--short'), independent100_used=False,
                    attention_extraction=False, intervention=False, samples=samples)
    directory.mkdir(parents=True, exist_ok=False)
    pilot.write_new(directory/'plan.json', manifest)
    print(json.dumps(dict(sample_count=20, reuse=len(reused), new_generations=50, indices=manifest['selected_indices'])))


def validate_results(samples, results, tokenizer=None):
    expected = {(s['dataset_index'], c) for s in samples for c in CONDITIONS}
    if len(results) != 60 or {(r['dataset_index'], r['condition']) for r in results} != expected:
        raise ValueError('missing or duplicate condition results')
    by_index = {s['dataset_index']: s for s in samples}
    for row in results:
        sample = by_index[row['dataset_index']]
        if row['input_ids'] != sample['conditions'][row['condition']]['input_ids']:
            raise ValueError('saved evaluation input differs from plan')
        if row['generation_failed'] or row['empty_response']:
            raise ValueError('generation failure or empty response; dataset not complete')
        if check_success_with_details(row['response'])[0] != row['success']:
            raise ValueError('saved automatic label differs from checker')
        if row['response_token_count'] != len(row['response_token_ids']) or not 0 < row['response_token_count'] <= 32:
            raise ValueError('response token count invalid')
        if tokenizer is not None and tokenizer.decode(row['response_token_ids']) != row['response']:
            raise ValueError('response IDs do not decode to saved response')
        if row['condition'] == 'gcg' and (not row['success'] or row['response'] != sample['historical_gcg_response']):
            raise ValueError('historical GCG response or success not reproduced; retain outputs for investigation')
        if row['human_jailbreak_success'] is not None or row['human_review_status'] != 'not_reviewed':
            raise ValueError('no human review was performed in this workflow')


def run(directory, execution_context):
    import torch
    import transformers
    from scripts.evaluate_gcg_first_token_reference_ablation import (
        GENERATION_CONFIG, generate_with_first_logits, load_model_and_tokenizer)
    from utils.experiment_runtime import seed_everything
    plan = json.loads((directory/'plan.json').read_text())
    if hashes() != plan['source_hashes'] or plan['generation'] != GENERATION_CONFIG:
        raise ValueError('prepared sources/settings changed')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError('CUDA required; no CPU fallback')
    old_runtime = json.loads((PILOT/'runtime.json').read_text())
    if (old_runtime['torch'] != torch.__version__ or old_runtime['transformers'] != transformers.__version__
            or old_runtime['python'] != platform.python_version()):
        raise ValueError('pilot software versions differ; do not reuse silently')
    pilot.write_new(directory/'runtime.json', dict(started_at=now(), execution_context=execution_context,
        plan_sha256=pilot.sha(directory/'plan.json'), python=platform.python_version(),
        torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0), cuda_device_count=torch.cuda.device_count(),
        nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
        git_head=pilot.git('rev-parse', 'HEAD'), git_status=pilot.git('status', '--short')))
    weight_index = pilot.MODEL/'model.safetensors.index.json'
    shards = sorted(set(json.loads(weight_index.read_text())['weight_map'].values()))
    print('Checking actual model weight hashes against reused pilot', flush=True)
    weights = dict(index_sha256=pilot.sha(weight_index), shard_sha256={name: pilot.sha(pilot.MODEL/name) for name in shards})
    pilot.write_new(directory/'model_weights.json', weights)
    if weights != json.loads((PILOT/'model_weights.json').read_text()):
        raise ValueError('model weight bytes differ from pilot; no reuse permitted')
    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(pilot.MODEL)
    if model.dtype != torch.float16 or model.config._attn_implementation != 'eager' or any(type(layer.self_attn).__name__ != 'LlamaAttention' for layer in model.model.layers):
        raise RuntimeError('expected FP16 unpatched eager LlamaAttention')
    if canonical(build_samples(tokenizer)) != plan['samples']:
        raise ValueError('loaded tokenizer does not reproduce frozen inputs')
    pilot.write_new(directory/'loaded_model.json', dict(config=model.config.to_dict(), generation_config=model.generation_config.to_dict(), tokenizer_class=type(tokenizer).__name__))
    reused = {(r['dataset_index'], r['condition']): r for r in load_pilot_reuse(plan['samples'])}
    results = []
    with (directory/'results.jsonl').open('x') as handle:
        for sample in plan['samples']:
            validate_sample(sample)
            for condition in CONDITIONS:
                key = sample['dataset_index'], condition
                ids = sample['conditions'][condition]['input_ids']
                if key in reused:
                    generation = {k: reused[key][k] for k in ('response', 'response_token_ids', 'response_token_count', 'success', 'matched_refusal_substrings', 'success_decision_reason', 'generation_failed', 'generation_error', 'empty_response')}
                else:
                    generation = generate_with_first_logits(model, tokenizer, torch.tensor([ids], dtype=torch.long))
                    generation.pop('first_logits')
                historical = sample['historical_gcg_response'] if condition == 'gcg' else sample['historical_original_response'] if condition == 'original' else None
                row = dict(dataset_index=key[0], condition=condition, input_ids=ids, **generation,
                    response_origin='reused_pilot' if key in reused else 'generated_this_run',
                    source_result_file=str(PILOT/'results.jsonl') if key in reused else str(directory/'results.jsonl'),
                    historical_response_equal=generation['response'] == historical if historical is not None else None,
                    human_jailbreak_success=None, human_review_status='not_reviewed',
                    actual_jailbreak_success='unknown', content_review_required=True)
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); handle.flush()
                results.append(row)
                print(f"index={key[0]} condition={condition} success={row['success']} origin={row['response_origin']}", flush=True)
    validate_results(plan['samples'], results, tokenizer)
    if sum(r['response_origin'] == 'reused_pilot' for r in results) != 10:
        raise ValueError('unexpected reuse count')
    lookup = {(r['dataset_index'], r['condition']): r for r in results}
    with (directory/'dataset.jsonl').open('x') as handle:
        for sample in plan['samples']:
            row = canonical(sample)
            row['manifest_file'] = str(directory/'plan.json')
            for condition in CONDITIONS:
                row['conditions'][condition]['evaluation'] = lookup[row['dataset_index'], condition]
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    counts = {c: sum(r['success'] for r in results if r['condition'] == c) for c in CONDITIONS}
    summary = dict(status='complete_validated', completed_at=now(), sample_count=20, condition_count=3,
        heuristic_success_counts=counts, shuffled_failure_count=20-counts['shuffled'],
        gcg_success_to_shuffled_failure_count=sum(lookup[s['dataset_index'], 'gcg']['success'] and not lookup[s['dataset_index'], 'shuffled']['success'] for s in plan['samples']),
        human_reviewed_response_count=0, actual_outcome_unknown_response_count=60,
        actual_outcome_unknown_shuffled_count=20, generated_response_count=50, reused_response_count=10,
        generation_failure_count=0, empty_response_count=0, gcg_historical_exact_match_count=20,
        original_historical_exact_match_count=sum(r['historical_response_equal'] for r in results if r['condition']=='original'),
        all_token_invariants_passed=True, independent100_used=False,
        artifact_sha256={name: pilot.sha(directory/name) for name in ('plan.json','results.jsonl','dataset.jsonl','model_weights.json','runtime.json','loaded_model.json')},
        peak_allocated_vram_mib=torch.cuda.max_memory_allocated()/1024**2)
    pilot.write_new(directory/'summary.json', summary)
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
