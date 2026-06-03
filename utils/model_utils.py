import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,GenerationConfig

def load_model_and_tokenizer(model_path,device='cuda'):
    if "llama" in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype = torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        MODEL_CONFIG={
            "model_name": "Llama",
            "n_heads":model.config.num_attention_heads, #32
            "n_layers":model.config.num_hidden_layers, #32
            "resid_dim":model.config.hidden_size, #4096
            "name_or_path":model.config._name_or_path,
            "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
            "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "prepend_bos":True
        }
    elif "mistral" in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype = torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        MODEL_CONFIG={
            "model_name": "Mistral",
            "n_heads":model.config.num_attention_heads, #32
            "n_layers":model.config.num_hidden_layers, #32
            "resid_dim":model.config.hidden_size, #4096
            "name_or_path":model.config._name_or_path,
            "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
            "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "prepend_bos":True
        }
    return model,tokenizer,MODEL_CONFIG
