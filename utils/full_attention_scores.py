"""Stream full post-RoPE, pre-mask Llama scores to a single input's FP32 NPY."""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers.models.llama import modeling_llama

from scripts.analyze_attention_scores import compute_post_rope_qk, compute_score_matrix


SCORE_METHOD = 'post-RoPE FP16 Q/K cast to FP32; FP32 Q @ K.transpose(-1,-2) / sqrt(head_dim); no mask/softmax/window/decomposition'


def full_fp32_scores(query, key):
    if query.dtype != torch.float16 or key.dtype != torch.float16:
        raise ValueError('post-RoPE Q and K must both be FP16')
    if query.shape != key.shape or query.ndim != 4 or query.shape[0] != 1:
        raise ValueError('expected matched, batch-one post-repeat Q/K tensors')
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(query.shape[-1])
    if scores.dtype != torch.float32 or not bool(torch.isfinite(scores).all()):
        raise ValueError('nonfinite or non-FP32 raw attention scores')
    return scores[0]


def check_model(model):
    config = model.config
    if model.training or model.dtype != torch.float16:
        raise ValueError('model.eval() and FP16 weights required')
    if config.model_type != 'llama' or config._attn_implementation != 'eager' or config.pretraining_tp != 1:
        raise ValueError('requires eager Llama with pretraining_tp=1')
    if len(model.layers) != config.num_hidden_layers:
        raise ValueError('configured layer count differs from actual layers')
    for layer in model.layers:
        attention = layer.self_attn
        if type(attention) is not modeling_llama.LlamaAttention:
            raise TypeError('unmodified eager LlamaAttention required')
        if attention.num_heads != config.num_attention_heads:
            raise ValueError('head count differs from actual model configuration')
    return config.num_hidden_layers, config.num_attention_heads


def extract_to_npy(model, input_ids, path, *, validate_native=False, forward_kwargs=None):
    """Hook actual layer arguments; write/release one layer before native attention.

    The optional smoke spy observes native post-RoPE tensors and returns them
    unchanged. It checks the saved score against per-head native-Q/K references.
    forward_kwargs exists for tests of nondefault actual position IDs; production
    invokes this function without supplying or constructing position IDs.
    """
    layers, heads = check_model(model)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
        raise ValueError('one unpadded torch.long input required')
    path = Path(path)
    # Exclusive creation: even a partial NPY from an interrupted attempt is immutable.
    with path.open('xb'):
        pass
    length = int(input_ids.shape[-1])
    if length == 0:
        raise ValueError('empty input')
    shape = (layers, heads, length, length)
    destination = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
    captured, handles, native_checks, pending = {}, [], [], {}
    original_rope = modeling_llama.apply_rotary_pos_emb

    # Keep the exact unexpanded native K, including GQA models.
    def native_rope_spy(*args, **kwargs):
        query, native_key = original_rope(*args, **kwargs)
        if not pending:
            raise RuntimeError('unexpected native RoPE call')
        index, saved = pending.pop('current')
        key = modeling_llama.repeat_kv(native_key, model.layers[index].self_attn.num_key_value_groups)
        reference = torch.stack([compute_score_matrix(query, key, head) for head in range(heads)])
        maximum = float((reference - saved).abs().max())
        torch.testing.assert_close(saved, reference, rtol=1e-5, atol=1e-4)
        native_checks.append(dict(layer=index, max_absolute_error=maximum))
        return query, native_key

    def make_hook(index):
        def hook(attention, args, kwargs):
            if index in captured:
                raise RuntimeError('attention layer called more than once')
            hidden = args[0] if args else kwargs.get('hidden_states')
            positions = kwargs.get('position_ids')
            cache_positions = kwargs.get('cache_position')
            mask = kwargs.get('attention_mask')
            if hidden is None or positions is None or cache_positions is None:
                raise RuntimeError('actual hidden/position/cache-position arguments missing')
            if hidden.shape[:2] != (1, length) or positions.shape != (1, length):
                raise ValueError('unexpected batch/position shape')
            if kwargs.get('past_key_value') is not None or kwargs.get('use_cache', False):
                raise ValueError('cache must be disabled')
            if mask is None or mask.shape[-2] != length or mask.shape[-1] < length:
                raise ValueError('actual eager causal mask missing or malformed')
            actual_mask = mask[0, 0, :, :length]
            future = torch.triu(torch.ones((length, length), dtype=torch.bool, device=hidden.device), diagonal=1)
            if not bool((actual_mask[~future] == 0).all()) or not bool((actual_mask[future] < -1e4).all()):
                raise ValueError('actual mask is not the unpadded causal rule key_index <= query_index')
            captured[index] = dict(position_ids=positions.detach().cpu().tolist(),
                cache_position=cache_positions.detach().cpu().tolist(),
                mask_shape=list(mask.shape), mask_dtype=str(mask.dtype),
                future_mask_value=float(actual_mask[future][0]) if bool(future.any()) else None)
            query, key = compute_post_rope_qk(attention, hidden, positions,
                original_rope, modeling_llama.repeat_kv, 'Llama')
            scores = full_fp32_scores(query, key)
            if scores.shape != (heads, length, length):
                raise ValueError('raw score shape mismatch')
            destination[index] = scores.cpu().numpy()
            if validate_native:
                if pending:
                    raise RuntimeError('unverified previous native attention')
                pending['current'] = index, scores
        return hook

    try:
        for index, layer in enumerate(model.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(make_hook(index), with_kwargs=True))
        observer = patch.object(modeling_llama, 'apply_rotary_pos_emb', native_rope_spy) if validate_native else nullcontext()
        with torch.inference_mode(), observer:
            options = dict(forward_kwargs or {})
            if set(options) - {'position_ids'}:
                raise ValueError('only explicit test position_ids supported')
            result = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                use_cache=False, output_attentions=False, output_hidden_states=False,
                return_dict=True, **options)
            if result.past_key_values is not None or not bool(torch.isfinite(result.last_hidden_state).all()):
                raise RuntimeError('unexpected cache or nonfinite model output')
        if set(captured) != set(range(layers)) or pending or (validate_native and len(native_checks) != layers):
            raise RuntimeError('incomplete layer/native capture')
        position_ids = captured[0]['position_ids']
        if any(value['position_ids'] != position_ids for value in captured.values()):
            raise ValueError('position_ids differ across layers')
        destination.flush()
    finally:
        for handle in handles:
            handle.remove()
        destination.flush()
        del destination
    saved = np.load(path, mmap_mode='r', allow_pickle=False)
    if saved.shape != shape or saved.dtype != np.float32 or not np.isfinite(saved).all():
        raise ValueError('saved NPY validation failed')
    del saved
    return dict(shape=list(shape), dtype='float32', computation=SCORE_METHOD,
        position_ids=position_ids, position_ids_source='actual self_attn forward kwargs; equality verified across all layers',
        layer_arguments={str(k): v for k, v in captured.items()},
        native_qk_reference_checks=native_checks, all_finite=True,
        causal_rule='key token index <= query token index; all input tokens unpadded',
        causal_mask_applied_to_saved_scores=False, softmax_applied=False,
        future_direction_entries_preserved=True)


def prefix_validation(artifacts, prefix_length):
    """Numerical extraction check only; no score decomposition or statistics."""
    arrays = {name: np.load(path, mmap_mode='r', allow_pickle=False) for name, path in artifacts.items()}
    reference = arrays['gcg'][:, :, :prefix_length, :prefix_length]
    checks = {}
    for name in ('original', 'shuffled'):
        candidate = arrays[name][:, :, :prefix_length, :prefix_length]
        checks[name] = dict(max_absolute_error=float(np.max(np.abs(reference-candidate))),
            close=bool(np.allclose(reference, candidate, rtol=0.02, atol=0.1)), rtol=0.02, atol=0.1)
    return dict(prefix_length=prefix_length, comparisons_to_gcg=checks,
                interpretation='FP16 backbone may vary with GEMM sequence shape; validation only')
