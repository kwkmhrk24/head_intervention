import torch
import argparse
from tqdm import tqdm

from utils.prompt_utils import get_prompts_list, apply_template
from utils.model_utils import load_model_and_tokenizer
from utils.extract_utils import get_mean_head_activations
from utils.intervention_utils import compute_CIE

def compute_AIE(jailbreak_prompts,original_prompts,model,model_config,tokenizer):
    AIE = torch.zeros(model_config['n_layers'],model_config['n_heads'])
    for i,(jailbreak_prompt,original_prompt) in enumerate(tqdm(zip(jailbreak_prompts,original_prompts))):
        activation = get_mean_head_activations(
            prompts=[original_prompt],
            model = model,
            model_config=model_config,
            tokenizer=tokenizer
            )
        with torch.no_grad():
            CIE = compute_CIE(
                prompt=jailbreak_prompt,
                avg_activations=activation,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer
                ).squeeze().detach().cpu()
        AIE += CIE
    
    AIE /= len(jailbreak_prompts)
    return AIE
        
def get_args():
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--jailbreak_prompts",type = str,help="Path to the jsonl file containing the jailbreak prompts to compute AIE. Each line must have \"prompt\" key.")
    parser.add_argument("--original_prompts",type=str,help="Path to the jsonl file containing the original prompts corresponding to jailbreak prompts")
    parser.add_argument("--system_prompt",type=str, default=None, help="System prompt. Default is None. If you set \"self-reminder\", system prompt is automatically set to the prompt we used for self-reminder.")
    parser.add_argument("--output_file",type=str,default=None,help="output file name. Result will be stored in \"./data/output_file.pt\"")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    model,tokenizer,model_config = load_model_and_tokenizer(args.model)

    jailbreak_prompts = get_prompts_list(args.jailbreak_prompts)
    jailbreak_prompts = apply_template(
        prompts=jailbreak_prompts,
        system_prompt=args.system_prompt,
        tokenizer=tokenizer,
        model_config=model_config)
    
    original_prompts = get_prompts_list(args.original_prompts)
    original_prompts = apply_template(
        prompts=original_prompts,
        system_prompt=args.system_prompt,
        tokenizer=tokenizer,
        model_config=model_config
        )
    AIE = compute_AIE(
        jailbreak_prompts=jailbreak_prompts,
        original_prompts=original_prompts,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer
        )
    if args.output_file != None:
        torch.save(AIE,f"./data/{args.output_file}.pt")
