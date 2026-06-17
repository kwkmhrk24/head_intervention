import argparse
import torch
from pathlib import Path

from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import get_prompts_list, apply_template
from utils.extract_utils import get_mean_head_activations

def get_args():
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument("--model",type=str,help="Model path")
    parser.add_argument("--prompts_file",type=str,help="path to jsonl file containing prompts for computing the average head")
    parser.add_argument("--system_prompt",type=str,default=None,help="system prompt which is used when computing AIE")
    parser.add_argument("--output_file",type=str,default=None,help="Filename for saving intervention vector. This will be stored in ./data/output_file.pt")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    model,tokenizer,model_config = load_model_and_tokenizer(args.model)
    prompts = get_prompts_list(args.prompts_file)
    prompts = apply_template(prompts=prompts,system_prompt=args.system_prompt,tokenizer=tokenizer,model_config=model_config)

    mean_activation = get_mean_head_activations(
        prompts=prompts,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer
        )
    if args.output_file is not None:
        output_path = Path("./data") / f"{args.output_file}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mean_activation, output_path)
    
