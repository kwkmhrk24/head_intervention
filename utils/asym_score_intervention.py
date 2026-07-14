"""Direct antisymmetric attention-score suppression for Transformers 4.42.3 Llama."""

import math
import types
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama import modeling_llama


SUPPORTED_TRANSFORMERS_VERSION = "4.42.3"


class NonSquareAttentionScoreError(ValueError):
    """Raised when cached/non-square scores cannot be decomposed safely."""


class UnsupportedAttentionCacheError(ValueError):
    """Raised when cache state reaches the primary no-cache implementation."""


@contextmanager
def temporarily_disable_default_dynamic_cache(model):
    """Prevent Transformers 4.42.3 ``generate`` from creating DynamicCache.

    Only the model instance is changed. The original effective value and
    whether it was stored directly on the instance are both restored, including
    on exceptional exit. Nested use is safe because each level restores the
    state it observed on entry.
    """

    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "default DynamicCache suppression supports Transformers "
            f"{SUPPORTED_TRANSFORMERS_VERSION}, found {transformers.__version__}"
        )
    if not hasattr(model, "_supports_cache_class"):
        raise AttributeError(
            "model lacks the Transformers _supports_cache_class attribute"
        )
    original = model._supports_cache_class
    if type(original) is not bool:
        raise TypeError("model._supports_cache_class must be a bool")

    instance_attributes = vars(model)
    had_instance_attribute = "_supports_cache_class" in instance_attributes
    previous_instance_value = instance_attributes.get("_supports_cache_class")
    state = {
        "generation_default_dynamic_cache_disabled": True,
        "generation_model_supports_cache_class_original": original,
        "generation_model_supports_cache_class_during_context": False,
        "generation_model_supports_cache_class_restored": None,
    }
    model._supports_cache_class = False
    try:
        yield state
    finally:
        if had_instance_attribute:
            model._supports_cache_class = previous_instance_value
        else:
            delattr(model, "_supports_cache_class")
        state["generation_model_supports_cache_class_restored"] = (
            model._supports_cache_class == original
        )


def validate_beta(beta: float) -> float:
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta must be a real number")
    beta = float(beta)
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must satisfy 0.0 <= beta <= 1.0")
    return beta


def validate_selected_heads(selected_heads: Sequence[int], head_count: int) -> List[int]:
    if not isinstance(selected_heads, (list, tuple)) or not selected_heads:
        raise ValueError("selected_heads must be a non-empty list or tuple")
    if any(type(head) is not int for head in selected_heads):
        raise TypeError("selected head indices must be integers")
    if len(set(selected_heads)) != len(selected_heads):
        raise ValueError("selected head indices contain duplicates")
    invalid = [head for head in selected_heads if not 0 <= head < head_count]
    if invalid:
        raise ValueError(
            f"selected head indices are outside 0 <= head < {head_count}: {invalid}"
        )
    return list(selected_heads)


def _max_abs(tensor: torch.Tensor) -> float:
    return float(tensor.detach().abs().max().float().cpu().item())


def suppress_antisymmetric_component(
    attention_scores: torch.Tensor,
    selected_heads: Sequence[int],
    beta: float,
    *,
    return_diagnostics: bool = False,
):
    """Scale only selected heads' antisymmetric score component by ``1-beta``.

    The input is never modified. Scores must have axes batch, head, query, key,
    and query/key lengths must be equal so transposition defines a square-matrix
    symmetric/antisymmetric decomposition.
    """

    beta = validate_beta(beta)
    if not isinstance(attention_scores, torch.Tensor):
        raise TypeError("attention_scores must be a torch.Tensor")
    if attention_scores.ndim != 4:
        raise ValueError(
            "attention_scores must have shape batch,head,query,key; "
            f"received ndim={attention_scores.ndim}"
        )
    if not attention_scores.is_floating_point():
        raise TypeError("attention_scores must use a floating-point dtype")
    query_length, key_length = attention_scores.shape[-2:]
    if query_length != key_length:
        raise NonSquareAttentionScoreError(
            "antisymmetric score intervention requires query_length == key_length; "
            f"received query_length={query_length}, key_length={key_length}"
        )
    selected_heads = validate_selected_heads(
        selected_heads, attention_scores.shape[1]
    )

    transpose = attention_scores.transpose(-1, -2)
    all_antisymmetric = (attention_scores - transpose) / 2
    selected_pre = attention_scores[:, selected_heads, :, :]
    selected_pre_asym = all_antisymmetric[:, selected_heads, :, :]
    selected_pre_sym = (
        selected_pre + selected_pre.transpose(-1, -2)
    ) / 2

    head_mask = torch.zeros(
        attention_scores.shape[1],
        dtype=torch.bool,
        device=attention_scores.device,
    )
    head_mask[selected_heads] = True
    modified = torch.where(
        head_mask.view(1, -1, 1, 1),
        attention_scores - beta * all_antisymmetric,
        attention_scores,
    )

    if not return_diagnostics:
        return modified

    selected_post = modified[:, selected_heads, :, :]
    selected_post_transpose = selected_post.transpose(-1, -2)
    selected_post_sym = (selected_post + selected_post_transpose) / 2
    selected_post_asym = (selected_post - selected_post_transpose) / 2
    pre_norm = float(
        torch.linalg.vector_norm(selected_pre_asym.detach().float()).cpu().item()
    )
    post_norm = float(
        torch.linalg.vector_norm(selected_post_asym.detach().float()).cpu().item()
    )
    observed_scale = post_norm / pre_norm if pre_norm > 0 else None
    expected = selected_pre - beta * selected_pre_asym

    if len(selected_heads) == attention_scores.shape[1]:
        unselected_error = 0.0
    else:
        unselected_mask = ~head_mask
        unselected_error = _max_abs(
            modified[:, unselected_mask, :, :]
            - attention_scores[:, unselected_mask, :, :]
        )

    diagnostics = {
        "selected_head_count": len(selected_heads),
        "selected_head_indices": list(selected_heads),
        "query_length": query_length,
        "key_length": key_length,
        "beta": beta,
        "pre_asym_fro_norm": pre_norm,
        "post_asym_fro_norm": post_norm,
        "observed_asym_scale": observed_scale,
        "expected_asym_scale": 1.0 - beta,
        "sym_component_max_abs_error": _max_abs(
            selected_post_sym - selected_pre_sym
        ),
        "intervention_identity_max_abs_error": _max_abs(
            selected_post - expected
        ),
        "unselected_head_max_abs_error": unselected_error,
    }
    return modified, diagnostics


class ScoreInterventionDiagnostics:
    """Collect per-layer calls without retaining score tensors."""

    def __init__(self, beta: float):
        self.beta = validate_beta(beta)
        self.calls: List[Dict[str, object]] = []

    def clear(self) -> None:
        self.calls.clear()

    def record(self, layer: int, diagnostics: Dict[str, object]) -> None:
        self.calls.append({"layer": layer, **diagnostics})

    def summary(self) -> Dict[str, object]:
        observed = [
            float(call["observed_asym_scale"])
            for call in self.calls
            if call["observed_asym_scale"] is not None
        ]

        def maximum(key: str) -> float:
            return max((float(call[key]) for call in self.calls), default=0.0)

        query_lengths = sorted({int(call["query_length"]) for call in self.calls})
        key_lengths = sorted({int(call["key_length"]) for call in self.calls})
        selected_layer_heads = {
            (int(call["layer"]), int(head))
            for call in self.calls
            for head in call["selected_head_indices"]
        }
        pre_norm = math.sqrt(
            sum(float(call["pre_asym_fro_norm"]) ** 2 for call in self.calls)
        )
        post_norm = math.sqrt(
            sum(float(call["post_asym_fro_norm"]) ** 2 for call in self.calls)
        )
        return {
            "call_count": len(self.calls),
            "selected_head_count": len(selected_layer_heads),
            "beta": self.beta,
            "pre_asym_fro_norm": pre_norm,
            "post_asym_fro_norm": post_norm,
            "aggregate_observed_asym_scale": (
                post_norm / pre_norm if pre_norm > 0 else None
            ),
            "mean_observed_asym_scale": (
                sum(observed) / len(observed) if observed else None
            ),
            "expected_asym_scale": 1.0 - self.beta,
            "query_lengths": query_lengths,
            "key_lengths": key_lengths,
            "all_score_matrices_square": (
                all(
                    int(call["query_length"]) == int(call["key_length"])
                    for call in self.calls
                )
                if self.calls
                else None
            ),
            "max_intervention_identity_error": maximum(
                "intervention_identity_max_abs_error"
            ),
            "max_symmetric_component_error": maximum(
                "sym_component_max_abs_error"
            ),
            "max_unselected_head_error": maximum(
                "unselected_head_max_abs_error"
            ),
            "max_observed_scale_abs_error": max(
                (
                    abs(value - (1.0 - self.beta))
                    for value in observed
                ),
                default=None,
            ),
        }


def validate_layer_head_pairs(
    layer_head_pairs: Sequence[Tuple[int, int]],
    *,
    layer_count: int,
    head_count: int,
) -> Dict[int, List[int]]:
    if not isinstance(layer_head_pairs, (list, tuple)) or not layer_head_pairs:
        raise ValueError("layer_head_pairs must be non-empty")
    normalized = []
    for pair in layer_head_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each intervention location must be (layer, head)")
        layer, head = pair
        if type(layer) is not int or type(head) is not int:
            raise TypeError("layer/head indices must be integers")
        if not 0 <= layer < layer_count:
            raise ValueError(f"layer index out of range: {layer}")
        if not 0 <= head < head_count:
            raise ValueError(f"head index out of range: {head}")
        normalized.append((layer, head))
    if len(set(normalized)) != len(normalized):
        raise ValueError("layer_head_pairs contain duplicates")
    grouped: Dict[int, List[int]] = {}
    for layer, head in normalized:
        grouped.setdefault(layer, []).append(head)
    return grouped


def _patched_llama_attention_forward(
    selected_heads: Sequence[int],
    beta: float,
    diagnostics: ScoreInterventionDiagnostics,
):
    """Return the audited Transformers 4.42.3 eager forward plus one score edit."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if use_cache:
            raise UnsupportedAttentionCacheError(
                "asym score intervention requires use_cache=False"
            )
        if past_key_value is not None:
            raise UnsupportedAttentionCacheError(
                "asym score intervention does not support past_key_value"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp,
                dim=0,
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)
            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)
            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = modeling_llama.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states = modeling_llama.repeat_kv(
            key_states, self.num_key_value_groups
        )
        value_states = modeling_llama.repeat_kv(
            value_states, self.num_key_value_groups
        )
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # The only semantic change from Transformers 4.42.3 LlamaAttention:
        # edit selected post-RoPE/repeat_kv raw scores before the causal mask.
        attn_weights, call_diagnostics = suppress_antisymmetric_component(
            attn_weights,
            selected_heads,
            beta,
            return_diagnostics=True,
        )
        diagnostics.record(self.layer_idx, call_diagnostics)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
            bsz,
            self.num_heads,
            q_len,
            self.head_dim,
        ):
            raise ValueError(
                "`attn_output` should be of size "
                f"{(bsz, self.num_heads, q_len, self.head_dim)}, but is "
                f"{attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    return forward


@contextmanager
def patch_llama_attention_scores(
    model,
    layer_head_pairs: Sequence[Tuple[int, int]],
    beta: float,
):
    """Temporarily patch selected eager Llama attention layers and restore them."""

    beta = validate_beta(beta)
    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "audited LlamaAttention forward supports Transformers "
            f"{SUPPORTED_TRANSFORMERS_VERSION}, found {transformers.__version__}"
        )
    layers = model.model.layers
    if not layers:
        raise ValueError("model has no decoder layers")
    first_attention = layers[0].self_attn
    grouped = validate_layer_head_pairs(
        layer_head_pairs,
        layer_count=len(layers),
        head_count=first_attention.num_heads,
    )
    diagnostics = ScoreInterventionDiagnostics(beta)
    patched = []
    try:
        for layer_index, selected_heads in grouped.items():
            module = layers[layer_index].self_attn
            if type(module) is not modeling_llama.LlamaAttention:
                raise TypeError(
                    "score intervention requires eager LlamaAttention; "
                    f"layer={layer_index}, found={type(module).__name__}"
                )
            if module.layer_idx != layer_index:
                raise ValueError(
                    f"attention layer_idx mismatch at layer {layer_index}"
                )
            had_instance_forward = "forward" in module.__dict__
            previous_instance_forward = module.__dict__.get("forward")
            replacement = _patched_llama_attention_forward(
                selected_heads, beta, diagnostics
            )
            module.forward = types.MethodType(replacement, module)
            patched.append(
                (module, had_instance_forward, previous_instance_forward)
            )
        yield diagnostics
    finally:
        for module, had_instance_forward, previous_instance_forward in reversed(
            patched
        ):
            if had_instance_forward:
                module.forward = previous_instance_forward
            else:
                module.__dict__.pop("forward", None)
