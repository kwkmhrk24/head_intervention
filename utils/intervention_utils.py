import torch
from tqdm import tqdm
from baukit import TraceDict

def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)

def replace_activation_w_avg(layer_head_pairs,avg_activations,interpolation_rate,intervention_times,model,model_config):
    """
    An intervention function for replacing activations with a computed average value.
    Returns: 
    rep_act: A function that specifies how to replace activations with an average when given a hooked pytorch module.
    """
    edit_layer = [x[0] for x in layer_head_pairs]
    total_intervention = 0

    def rep_act(output,layer_name,inputs):
        nonlocal total_intervention
        current_layer = int(layer_name.split('.')[2])
        if (current_layer in edit_layer) and (total_intervention < intervention_times):
            if isinstance(inputs,tuple):
                inputs = inputs[0]
            
            #Determine the shape for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads'])
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

            #Perform intervention
            for (layer,head_n) in layer_head_pairs:
                if layer == current_layer:
                    inputs[-1,-1,head_n] = avg_activations[layer,head_n,0].to(
                        device=inputs.device,
                        dtype=inputs.dtype,
                    )
                    total_intervention += 1
            
            inputs = inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight
            new_output = torch.matmul(inputs, out_proj.T)
            #print("intervened")
            return (1-interpolation_rate)*output + interpolation_rate*new_output
        else:
            return output
    
    return rep_act


def compute_CIE(prompt, avg_activations, model, model_config, tokenizer):
    # set token id to corresponding to Sure
    if model_config["model_name"] == "Llama":
        token_id_of_interest = [18585]
    elif model_config["model_name"] == "Mistral":
        token_id_of_interest = [12875]
    elif model_config["model_name"] == "Qwen":
        sure_ids = tokenizer.encode("Sure", add_special_tokens=False)
        token_id_of_interest = [sure_ids[0]]
    else:
        sure_ids = tokenizer.encode("Sure", add_special_tokens=False)
        token_id_of_interest = [sure_ids[0]]
    token_id_of_interest = torch.tensor(
        token_id_of_interest,
        dtype=torch.long,
        device=model.device,
    )
    
    inputs = tokenizer(prompt, return_tensors='pt',add_special_tokens=False).to(model.device)

    indirect_effect_storage = torch.zeros(model_config['n_layers'],model_config['n_heads'],1)
    
    # probability distribution before intervention
    clean_output = model(**inputs).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
    clean_probs = torch.softmax(clean_output, dim=-1)

    # compute CIE for each attention head
    for layer in tqdm(range(0,model_config['n_layers'])):
        head_hook_layer = [model_config['attn_hook_names'][layer]]

        for head_n in range(model_config['n_heads']):
            intervention_locations = [(layer,head_n)]
            intervention_fn = replace_activation_w_avg(layer_head_pairs=intervention_locations,
                                                       avg_activations=avg_activations,
                                                       interpolation_rate=1.0,
                                                       intervention_times=1, 
                                                       model=model, model_config=model_config)
            with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:                
                    output = model(**inputs).logits[:,-1,:]
                    
            # TRACK probs of tokens of interest
            intervention_probs = torch.softmax(output[0], dim=-1)
            #print(f"layer{layer+1},head{head_n+1} intervened_prob: {intervention_probs[0][token_id_of_interest[0]]}")
            indirect_effect_storage[layer,head_n,0] = (
                clean_probs - intervention_probs
            ).index_select(1, token_id_of_interest).squeeze().detach().float().cpu()

    return indirect_effect_storage

def generate_with_intervention(prompt,intervention_locations,interpolation_rate,intervention_activation,model,model_config,tokenizer,generation_config):
    inputs = tokenizer(prompt, return_tensors='pt',add_special_tokens=False).to(model.device)
    
    head_hook_layer = [model_config['attn_hook_names'][il[0]] for il in intervention_locations]
    intervention_fn = replace_activation_w_avg(layer_head_pairs=intervention_locations,
                                                       avg_activations= intervention_activation, 
                                                       interpolation_rate= interpolation_rate,
                                                       intervention_times = len(intervention_locations),
                                                       model=model, 
                                                       model_config=model_config
                                                       )
    with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:   
        outputs = model.generate(**inputs, **generation_config)
        outputs = tokenizer.decode(outputs[0, inputs.input_ids.shape[-1]:])
    return outputs
