#!/usr/bin/env python3
"""Prepare, smoke-check, and resume full FP32 attention-score extraction."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers.models.llama import modeling_llama

from utils.artifact_io import read_json, read_jsonl, sha256_file, utc_now
from utils.full_attention_scores import SCORE_METHOD, extract_to_npy, prefix_validation

DATASET = ROOT/'data/outputs/shuffled_gcg_comparison/legacy_selection20_20261106T171044Z/dataset.jsonl'
CONDITIONS = ('original', 'gcg', 'shuffled')


def exclusive_json(path, payload):
    with Path(path).open('x') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def unique_name():
    return utc_now().replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:8]


def software_versions():
    import transformers
    return dict(python=platform.python_version(), torch=torch.__version__,
                transformers=transformers.__version__, numpy=np.__version__, cuda=torch.version.cuda)


def input_tasks(dataset=DATASET):
    rows = read_jsonl(dataset)
    plan = read_json(dataset.parent/'plan.json')
    summary = read_json(dataset.parent/'summary.json')
    if sha256_file(dataset) != summary['artifact_sha256']['dataset.jsonl']:
        raise ValueError('source dataset hash mismatch')
    if len(rows) != 20 or [row['dataset_index'] for row in rows] != plan['selected_indices']:
        raise ValueError('exact selection20 dataset required')
    if len({r['dataset_index'] for r in rows}) != 20 or any(r['dataset_index'] >= 220 for r in rows):
        raise ValueError('duplicate index or non-legacy input')
    tasks = []
    for row in rows:
        for condition in CONDITIONS:
            item = row['conditions'][condition]
            ids = item['input_ids']
            if not ids or any(type(i) is not int or i < 0 for i in ids) or row[condition+'_input_ids'] != ids:
                raise ValueError('saved input IDs invalid or inconsistent')
            cursor = 0
            for region in ('template_prefix', 'original', 'suffix', 'template_closing'):
                start, end = item['token_ranges'][region]
                if start != cursor or not start <= end <= len(ids):
                    raise ValueError('token ranges do not partition saved input')
                cursor = end
            if cursor != len(ids) or item['final_input_token_position'] != len(ids)-1 or item['final_input_token_id'] != ids[-1]:
                raise ValueError('saved final input token or ranges invalid')
            tasks.append(dict(dataset_index=row['dataset_index'], condition=condition,
                input_ids=ids, sequence_length=len(ids), token_ranges=item['token_ranges'],
                final_input_token_position=item['final_input_token_position'],
                final_input_token_id=item['final_input_token_id']))
    return tasks, plan


def array_bytes(shape):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, dict(descr=np.dtype('float32').str, fortran_order=False, shape=tuple(shape)))
    return int(np.prod(shape))*4 + len(header.getvalue())


def sources(model):
    paths = [DATASET, DATASET.parent/'plan.json', DATASET.parent/'summary.json',
        DATASET.parent/'model_weights.json', DATASET.parent/'loaded_model.json',
        Path(__file__), ROOT/'utils/full_attention_scores.py', ROOT/'tests/test_full_attention_scores.py',
        ROOT/'scripts/analyze_attention_scores.py', ROOT/'utils/artifact_io.py',
        ROOT/'scripts/evaluate_gcg_first_token_reference_ablation.py',
        Path(modeling_llama.__file__)]
    paths += [model/name for name in ('config.json', 'generation_config.json', 'tokenizer.json',
                                    'tokenizer_config.json', 'special_tokens_map.json', 'model.safetensors.index.json')]
    return {str(path): sha256_file(path) for path in paths}


def prepare(directory):
    tasks, dataset_plan = input_tasks()
    model = Path(dataset_plan['model'])
    config = read_json(model/'config.json')
    if config['model_type'] != 'llama' or config['pretraining_tp'] != 1:
        raise ValueError('unsupported model configuration')
    layers, heads = config['num_hidden_layers'], config['num_attention_heads']
    for task in tasks:
        if max(task['input_ids']) >= config['vocab_size'] or task['sequence_length'] > config['max_position_embeddings']:
            raise ValueError('saved input exceeds configured vocabulary/context')
        task['shape'] = [layers, heads, task['sequence_length'], task['sequence_length']]
        task['expected_npy_bytes'] = array_bytes(task['shape'])
    payload_bytes = sum(int(np.prod(t['shape']))*4 for t in tasks)
    score_bytes = sum(t['expected_npy_bytes'] for t in tasks)
    free = shutil.disk_usage(ROOT).free
    reserve = score_bytes*2 + 256*1024**2
    if free < reserve:
        raise RuntimeError(f'insufficient disk: free={free}, required with margin={reserve}')
    weights = read_json(model/'model.safetensors.index.json')['metadata']['total_size']
    metadata = dict(created_at=utc_now(), role='development_diagnostic_raw_score_capture',
        source_dataset=str(DATASET), dataset_sha256=sha256_file(DATASET), model=str(model),
        model_name=config.get('_name_or_path', 'meta-llama/Llama-2-7b-chat-hf'),
        model_revision=subprocess.check_output(['git', '-c', f'safe.directory={model}',
            '-C', str(model), 'rev-parse', 'HEAD'], text=True).strip(),
        model_config=config, tokenizer_info=read_json(model/'tokenizer_config.json'),
        input_count=len(tasks), tasks=tasks, score_dtype='float32', model_dtype='float16',
        computation=SCORE_METHOD, attention_implementation='eager', batch_size=1,
        use_cache=False, eval_mode=True, system_prompt=None, retokenization=False,
        software_versions=software_versions(),
        generate_responses=False, source_hashes=sources(model),
        payload_bytes=payload_bytes, expected_score_file_bytes=score_bytes,
        free_disk_bytes_at_preflight=free, disk_required_with_margin=reserve,
        estimated_gpu_required_bytes=weights+2*1024**3,
        gpu_estimate_note='weight bytes plus 2 GiB workspace/allocator margin; one layer score at a time',
        native_check_rtol=1e-5, native_check_atol=1e-4,
        prefix_check_rtol=0.02, prefix_check_atol=0.1,
        smoke_tasks=[(t['dataset_index'], t['condition']) for t in tasks[:3]],
        independent100_used=False, sym_asym_decomposition=False,
        git_commit=git('rev-parse', 'HEAD'), git_branch=git('branch', '--show-current'),
        git_status=git('status', '--short'))
    directory.mkdir(parents=True, exist_ok=False)
    exclusive_json(directory/'plan.json', metadata)
    print(json.dumps({k: metadata[k] for k in ('input_count','payload_bytes','expected_score_file_bytes','estimated_gpu_required_bytes')}))


def task_directory(directory, task):
    return directory/'inputs'/f"index_{task['dataset_index']:04d}"/task['condition']


def completed(directory, task):
    marker = task_directory(directory, task)/'complete.json'
    if not marker.exists():
        return None
    pointer = read_json(marker)
    metadata_path = directory/pointer['metadata_file']
    if sha256_file(metadata_path) != pointer['metadata_sha256']:
        raise ValueError('completed metadata hash mismatch; do not overwrite')
    metadata = read_json(metadata_path)
    if metadata['input'] != task or metadata['plan_sha256'] != sha256_file(directory/'plan.json'):
        raise ValueError('completed input/plan mismatch')
    score_path = directory/metadata['score_file']
    if sha256_file(score_path) != metadata['score_sha256']:
        raise ValueError('completed score hash mismatch; do not overwrite')
    values = np.load(score_path, mmap_mode='r', allow_pickle=False)
    if list(values.shape) != task['shape'] or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('completed score shape/dtype/finiteness mismatch')
    if score_path.stat().st_size != task['expected_npy_bytes']:
        raise ValueError('completed NPY size mismatch')
    return metadata


def capture_one(directory, task, model, runtime_file, native):
    prior = completed(directory, task)
    if prior:
        print(f"skip verified index={task['dataset_index']} condition={task['condition']}", flush=True)
        return prior
    attempt = task_directory(directory, task)/('attempt_'+unique_name())
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        info = extract_to_npy(model.model, torch.tensor([task['input_ids']], dtype=torch.long, device=model.device),
            attempt/'scores.npy', validate_native=native)
        if info['shape'] != task['shape'] or (attempt/'scores.npy').stat().st_size != task['expected_npy_bytes']:
            raise ValueError('saved shape/file size differs from capacity plan')
        metadata = dict(input=task, extraction=info, score_file=str((attempt/'scores.npy').relative_to(directory)),
            score_sha256=sha256_file(attempt/'scores.npy'), plan_sha256=sha256_file(directory/'plan.json'),
            runtime_file=str(runtime_file.relative_to(directory)), saved_at=utc_now())
        exclusive_json(attempt/'metadata.json', metadata)
        # Only a complete, validated attempt gets a completion marker. Failed
        # attempts remain intact; a resume uses a new attempt directory.
        exclusive_json(task_directory(directory, task)/'complete.json', dict(
            metadata_file=str((attempt/'metadata.json').relative_to(directory)),
            metadata_sha256=sha256_file(attempt/'metadata.json')))
        print(f"saved index={task['dataset_index']} condition={task['condition']} shape={info['shape']}", flush=True)
        return metadata
    except Exception as error:
        exclusive_json(attempt/'failure.json', dict(at=utc_now(), error=f'{type(error).__name__}: {error}'))
        raise


@contextmanager
def execution_lock(directory):
    with (directory/'.execution.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def execute(directory, stage, execution_context):
    import transformers
    from scripts.evaluate_gcg_first_token_reference_ablation import load_model_and_tokenizer
    with execution_lock(directory):
        plan = read_json(directory/'plan.json')
        model_path = Path(plan['model'])
        if sources(model_path) != plan['source_hashes']:
            raise ValueError('frozen source/model config bytes changed')
        if software_versions() != plan['software_versions']:
            raise ValueError('software versions changed since preflight')
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError('CUDA required; no fallback')
        if stage == 'run':
            smoke_path = directory/'smoke_validation.json'
            if not smoke_path.exists():
                raise ValueError('run requires completed smoke validation')
            smoke = read_json(smoke_path)
            if smoke['status'] != 'passed' or smoke['plan_sha256'] != sha256_file(directory/'plan.json'):
                raise ValueError('smoke validation status/plan mismatch')
        selected = plan['tasks'][:3] if stage == 'smoke' else plan['tasks']
        pending = [task for task in selected if completed(directory, task) is None]
        required = sum(t['expected_npy_bytes'] for t in pending) + 256*1024**2
        if shutil.disk_usage(directory).free < required:
            raise RuntimeError(f'insufficient disk for remaining inputs: required={required}')
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        if pending and free_gpu < plan['estimated_gpu_required_bytes']:
            raise RuntimeError(f'insufficient free GPU memory: {free_gpu} < {plan["estimated_gpu_required_bytes"]}')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        runtimes = directory/'runtimes'
        runtimes.mkdir(exist_ok=True)
        runtime_path = runtimes/(unique_name()+'.json')
        exclusive_json(runtime_path, dict(started_at=utc_now(), stage=stage,
            plan_sha256=sha256_file(directory/'plan.json'), execution_context=execution_context,
            python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
            numpy=np.__version__, cuda=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0),
            cuda_device_count=torch.cuda.device_count(), gpu_free_bytes=free_gpu, gpu_total_bytes=total_gpu,
            nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
            fp32_matmul_precision=torch.get_float32_matmul_precision(), allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--short')))
        weight_index = model_path/'model.safetensors.index.json'
        weight_names = sorted(set(read_json(weight_index)['weight_map'].values()))
        print('Verifying local model weight bytes', flush=True)
        weights = dict(index_sha256=sha256_file(weight_index), shard_sha256={name:sha256_file(model_path/name) for name in weight_names})
        if weights != read_json(DATASET.parent/'model_weights.json'):
            raise ValueError('actual model weights differ from comparison dataset model')
        if (directory/'model_weights.json').exists():
            if read_json(directory/'model_weights.json') != weights:
                raise ValueError('model bytes changed across resume')
        else:
            exclusive_json(directory/'model_weights.json', weights)
        if pending:
            model, tokenizer = load_model_and_tokenizer(model_path)
            if not (directory/'loaded_model.json').exists():
                exclusive_json(directory/'loaded_model.json', dict(config=model.config.to_dict(),
                    tokenizer_class=type(tokenizer).__name__, tokenizer_name_or_path=tokenizer.name_or_path))
            for task in pending:
                capture_one(directory, task, model, runtime_path, native=stage=='smoke')
            del model, tokenizer
            torch.cuda.empty_cache()
        records = [completed(directory, task) for task in selected]
        if stage == 'smoke':
            if any(len(r['extraction']['native_qk_reference_checks']) != plan['model_config']['num_hidden_layers'] for r in records):
                raise ValueError('smoke native-RoPE reference checks missing')
            prefix = selected[1]['token_ranges']['suffix'][0]
            if any(t['input_ids'][:prefix] != selected[1]['input_ids'][:prefix] for t in selected):
                raise ValueError('smoke inputs do not share the checked prefix')
            check = prefix_validation({r['input']['condition']:directory/r['score_file'] for r in records}, prefix)
            if not all(v['close'] for v in check['comparisons_to_gcg'].values()):
                raise ValueError(f'prefix numerical validation failed: {check}')
            smoke = dict(status='passed', completed_at=utc_now(), plan_sha256=sha256_file(directory/'plan.json'),
                prefix_validation=check, native_reference_max_absolute_error=max(v['max_absolute_error'] for r in records for v in r['extraction']['native_qk_reference_checks']),
                inputs=[r['score_file'] for r in records])
            if not (directory/'smoke_validation.json').exists():
                exclusive_json(directory/'smoke_validation.json', smoke)
            print(json.dumps(smoke))
        else:
            manifest = directory/'manifest.jsonl'
            manifest_rows = [dict(dataset_index=r['input']['dataset_index'], condition=r['input']['condition'],
                score_file=r['score_file'], shape=r['extraction']['shape'], dtype='float32',
                completion_file=str((task_directory(directory,r['input'])/'complete.json').relative_to(directory))) for r in records]
            if manifest.exists():
                if read_jsonl(manifest) != manifest_rows:
                    raise ValueError('existing final manifest differs')
            else:
                with manifest.open('x') as handle:
                    for row in manifest_rows:
                        handle.write(json.dumps(row)+'\n')
            summary = dict(status='complete_validated', completed_at=utc_now(), input_count=len(records),
                expected_input_count=60, score_dtype='float32', all_square_and_finite=True,
                score_file_bytes=sum((directory/r['score_file']).stat().st_size for r in records),
                payload_bytes=plan['payload_bytes'], dataset_sha256=plan['dataset_sha256'],
                plan_sha256=sha256_file(directory/'plan.json'), manifest_sha256=sha256_file(manifest),
                reused_completed_inputs=len(selected)-len(pending), newly_extracted_inputs=len(pending),
                peak_allocated_vram_mib=torch.cuda.max_memory_allocated()/1024**2)
            summaries = directory/'summaries'
            summaries.mkdir(exist_ok=True)
            exclusive_json(summaries/(unique_name()+'.json'),summary)
            print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare','smoke','run'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execution-context', default='normal_sandbox')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output_dir)
    else:
        execute(args.output_dir,args.stage,args.execution_context)


if __name__ == '__main__':
    main()
