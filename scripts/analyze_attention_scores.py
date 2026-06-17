import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

try:
    from transformers.models.llama.modeling_llama import repeat_kv
except ImportError:
    repeat_kv = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import apply_template


CSV_COLUMNS = [
    "prompt_id",
    "source_index",
    "layer",
    "head",
    "seq_len",
    "fro_norm",
    "sym_norm",
    "skew_norm",
    "fro_norm_per_token",
    "sym_norm_per_token",
    "skew_norm_per_token",
    "sym_ratio",
    "skew_ratio",
    "reconstruction_rel_error",
    "energy_rel_error",
]


def get_args():
    parser = argparse.ArgumentParser(description="Analyze symmetry of post-RoPE pre-mask attention score matrices.")
    parser.add_argument("--model", required=True, help="Model path.")
    parser.add_argument("--prompts_file", required=True, help="JSONL file with at least a prompt key.")
    parser.add_argument("--heads", required=True, help="Semicolon-separated layer,head list, e.g. '3,0;7,16'.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument("--max_prompts", type=int, default=None, help="Optional limit on the number of prompts.")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt passed to apply_template().")
    parser.add_argument("--debug", action="store_true", help="Print shape and decomposition diagnostics for the first prompt/head.")
    return parser.parse_args()


def read_prompt_rows(path):
    rows = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_no} is not a JSON object")
            if "prompt" not in row:
                raise ValueError(f"{path}: line {line_no} missing required key: prompt")
            rows.append(row)
    return rows


def parse_heads(heads, model_config):
    parsed = []
    for item in heads.split(";"):
        item = item.strip()
        if not item:
            continue

        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid head spec {item!r}; expected 'layer,head'")

        try:
            layer, head = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"Invalid head spec {item!r}; layer/head must be integers") from exc

        if not 0 <= layer < model_config["n_layers"]:
            raise ValueError(f"Invalid layer {layer}; expected 0 <= layer < {model_config['n_layers']}")
        if not 0 <= head < model_config["n_heads"]:
            raise ValueError(f"Invalid head {head}; expected 0 <= head < {model_config['n_heads']}")

        parsed.append((layer, head))

    if not parsed:
        raise ValueError("--heads did not contain any valid layer,head pairs")

    return parsed


def capture_self_attn_inputs(model, layers, inputs):
    captured = {}
    handles = []

    for layer in sorted(set(layers)):
        self_attn = model.model.layers[layer].self_attn

        def hook(module, module_inputs, layer=layer):
            captured[layer] = module_inputs[0].detach()

        handles.append(self_attn.register_forward_pre_hook(hook))

    try:
        with torch.no_grad():
            model.model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(set(layers) - set(captured))
    if missing:
        raise RuntimeError(f"Failed to capture self_attn inputs for layers: {missing}")

    return captured


def get_rotary_cos_sin(self_attn, key_states, position_ids, seq_len):
    try:
        return self_attn.rotary_emb(key_states, position_ids)
    except TypeError:
        try:
            return self_attn.rotary_emb(key_states, seq_len=seq_len)
        except TypeError as exc:
            raise RuntimeError(
                "Unable to compute post-RoPE Q/K with this transformers rotary embedding API. "
                "Refusing to fall back to pre-RoPE scores."
            ) from exc


def apply_rope(query_states, key_states, cos, sin, position_ids):
    try:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    except TypeError:
        try:
            return apply_rotary_pos_emb(query_states, key_states, cos, sin)
        except TypeError as exc:
            raise RuntimeError(
                "Unable to apply rotary position embedding. Refusing to fall back to pre-RoPE scores."
            ) from exc
    except RuntimeError as exc:
        raise RuntimeError(
            "Unable to apply rotary position embedding. Refusing to fall back to pre-RoPE scores."
        ) from exc


def compute_post_rope_qk(self_attn, hidden_states, position_ids):
    batch_size, seq_len, _ = hidden_states.shape
    num_heads = self_attn.num_heads
    num_key_value_heads = getattr(self_attn, "num_key_value_heads", num_heads)
    head_dim = self_attn.head_dim

    query_states = self_attn.q_proj(hidden_states)
    key_states = self_attn.k_proj(hidden_states)

    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    cos, sin = get_rotary_cos_sin(self_attn, key_states, position_ids, seq_len)
    query_states, key_states = apply_rope(query_states, key_states, cos, sin, position_ids)

    if key_states.shape[1] != query_states.shape[1]:
        if repeat_kv is None:
            raise RuntimeError("repeat_kv is unavailable, but key/value heads differ from query heads")
        num_key_value_groups = getattr(
            self_attn,
            "num_key_value_groups",
            query_states.shape[1] // key_states.shape[1],
        )
        key_states = repeat_kv(key_states, num_key_value_groups)

    return query_states, key_states


def compute_score_matrix(query_states, key_states, head):
    head_dim = query_states.shape[-1]
    query = query_states[0, head].float()
    key = key_states[0, head].float()
    return torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)


def compute_metrics(score_matrix, eps=1e-12):
    seq_len = score_matrix.shape[-1]
    sym = (score_matrix + score_matrix.transpose(-1, -2)) / 2
    skew = (score_matrix - score_matrix.transpose(-1, -2)) / 2

    fro_norm = torch.linalg.matrix_norm(score_matrix, ord="fro")
    sym_norm = torch.linalg.matrix_norm(sym, ord="fro")
    skew_norm = torch.linalg.matrix_norm(skew, ord="fro")
    reconstruction_rel_error = torch.linalg.matrix_norm(score_matrix - (sym + skew), ord="fro") / (fro_norm + eps)
    energy_rel_error = torch.abs(fro_norm ** 2 - sym_norm ** 2 - skew_norm ** 2) / (fro_norm ** 2 + eps)

    return {
        "seq_len": seq_len,
        "fro_norm": float(fro_norm.item()),
        "sym_norm": float(sym_norm.item()),
        "skew_norm": float(skew_norm.item()),
        "fro_norm_per_token": float((fro_norm / seq_len).item()),
        "sym_norm_per_token": float((sym_norm / seq_len).item()),
        "skew_norm_per_token": float((skew_norm / seq_len).item()),
        "sym_ratio": float((sym_norm / (fro_norm + eps)).item()),
        "skew_ratio": float((skew_norm / (fro_norm + eps)).item()),
        "reconstruction_rel_error": float(reconstruction_rel_error.item()),
        "energy_rel_error": float(energy_rel_error.item()),
    }


def print_debug(metrics, score_matrix, selected_heads):
    print(f"tokenized seq_len: {metrics['seq_len']}")
    print(f"selected heads: {selected_heads}")
    print(f"S shape: {tuple(score_matrix.shape)}")
    print(f"dtype: {score_matrix.dtype}")
    print(f"device: {score_matrix.device}")
    print(f"reconstruction_rel_error: {metrics['reconstruction_rel_error']}")
    print(f"energy_rel_error: {metrics['energy_rel_error']}")
    print(f"sym_ratio: {metrics['sym_ratio']}")
    print(f"skew_ratio: {metrics['skew_ratio']}")
    if metrics["reconstruction_rel_error"] > 1e-5:
        print("WARNING: reconstruction_rel_error > 1e-5")
    if metrics["energy_rel_error"] > 1e-4:
        print("WARNING: energy_rel_error > 1e-4")


def main():
    args = get_args()
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max_prompts must be positive when specified")

    rows = read_prompt_rows(args.prompts_file)
    if args.max_prompts is not None:
        rows = rows[:args.max_prompts]

    model, tokenizer, model_config = load_model_and_tokenizer(args.model)
    if model_config["model_name"] != "Llama":
        raise ValueError("analyze_attention_scores.py currently supports Llama models only")
    model.eval()

    selected_heads = parse_heads(args.heads, model_config)
    selected_layers = [layer for layer, _ in selected_heads]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_rows = []
    debug_printed = False
    for prompt_id, row in enumerate(tqdm(rows)):
        templated_prompt = apply_template(
            prompts=[row["prompt"]],
            system_prompt=args.system_prompt,
            tokenizer=tokenizer,
            model_config=model_config,
        )[0]
        inputs = tokenizer(templated_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        seq_len = inputs.input_ids.shape[-1]
        position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)

        captured_inputs = capture_self_attn_inputs(model, selected_layers, inputs)
        qk_by_layer = {}
        for layer in sorted(set(selected_layers)):
            self_attn = model.model.layers[layer].self_attn
            qk_by_layer[layer] = compute_post_rope_qk(
                self_attn=self_attn,
                hidden_states=captured_inputs[layer],
                position_ids=position_ids,
            )

        for layer, head in selected_heads:
            query_states, key_states = qk_by_layer[layer]
            score_matrix = compute_score_matrix(query_states, key_states, head)
            metrics = compute_metrics(score_matrix)
            if args.debug and not debug_printed:
                print_debug(metrics, score_matrix, selected_heads)
                debug_printed = True

            output_rows.append({
                "prompt_id": prompt_id,
                "source_index": row.get("index", ""),
                "layer": layer,
                "head": head,
                **metrics,
            })

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"wrote rows: {len(output_rows)}")
    print(f"output_csv: {output_path}")


if __name__ == "__main__":
    main()
