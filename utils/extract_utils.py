import torch
from baukit import TraceDict

def gather_attn_activations(prompt,layers,model,tokenizer,model_config):
    """
    Parameters:
    prompt: input prompt
    layers: layer names to get activatons from
    model: huggingface model
    tokenizer: huggingface tokenizer

    Returns:
    td: tracedict with stored activations (td[hook_name].input : 1×n_tokens×resid_dim)
    """

    inputs = tokenizer(prompt, return_tensors='pt',add_special_tokens=False).to(model.device) #ここでまた<s>に対応するトークンIDが付けられる

    # Access Activations
    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:                
        model(**inputs) # batch_size x n_tokens x vocab_size, only want last token prediction

    return td

def get_mean_head_activations(prompts,model,model_config,tokenizer):
    """
    Parameters:
    prompts: a list of prompts to average over
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer

    Returns:
    mean_activations: avg activation of each attention head over prompts
    """
    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations
    
    N_TRIALS = len(prompts)

    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['n_heads'], 1, model_config['resid_dim']//model_config['n_heads'])

    for n in range(N_TRIALS):
        activations_td = gather_attn_activations(
            prompt = prompts[n],
            layers = model_config['attn_hook_names'],
            model = model,
            tokenizer = tokenizer,
            model_config = model_config)
        
        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_config) for layer in model_config['attn_hook_names']]).permute(0,2,1,3)
        #stack_initial : Layer × head × n_token × head_hidden_dim(resid_dim/h)
        activation_storage[n] = stack_initial[:,:,[-1]]
    
    mean_activations = activation_storage.mean(dim=0)
    return mean_activations #n_layer × n_head × 1 × head_hidden_dim