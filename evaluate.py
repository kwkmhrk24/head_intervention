import argparse
import torch
from tqdm import tqdm

from utils.model_utils import load_model_and_tokenizer
from utils.prompt_utils import get_prompts_list,apply_template
from utils.evaluate_utils import check_success
from utils.intervention_utils import generate_with_intervention

intervention_locations = { #set intervened head based on AIE score
    "Llama":  [(3,0),(7,16),(9,9),(13,4),(3,5)],
    "Mistral": [(16,14),(31,14),(15,14),(30,3),(16,31),(31,13),(11,9),(14,20),(30,9),(14,27)]
}
generation_config = { #set generation parameters
    "do_sample": False,
    "temperature" : None,
    "top_p": None,
    "max_new_tokens": 32
}

def calc_ASR(prompts,intervention_locations,interpolation_rate,intervention_activation,model,model_config,tokenizer,generation_config):
    ASN = 0
    for prompt in tqdm(prompts,leave=False):
        output = generate_with_intervention(
            prompt=prompt,
            intervention_locations=intervention_locations,
            interpolation_rate=interpolation_rate,
            intervention_activation=intervention_activation,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            generation_config=generation_config
            )
        #print(output)
        if check_success(output):
            ASN+=1
    ASR = ASN / len(prompts)
    return ASR

def get_args():
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--input_file",type = str,help="Path to the jsonl file containing the prompts to be evaluated. Each line must have \"prompt\" key.")
    parser.add_argument("--system_prompt",type=str, default=None, help="System prompt. Default is None. If you set \"self-reminder\", system prompt is automatically set to the prompt we used for self-reminder.")
    parser.add_argument("--intervention_activation",type=str,help="Path to .pt file containing head output vectors for intervention")
    parser.add_argument("--intervened_head_num",type=int,default=1,help="number of intervened head")
    parser.add_argument("--interpolation_rate",type=float,default=1.0,help="interpolation rate λ in the paper")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    model,tokenizer,model_config = load_model_and_tokenizer(args.model)
    prompts = get_prompts_list(args.input_file)
    prompts = apply_template(prompts=prompts,system_prompt=args.system_prompt,tokenizer=tokenizer,model_config=model_config)
    intervention_activation = torch.load(args.intervention_activation)

    ASR = calc_ASR(
        prompts=prompts,
        intervention_locations=intervention_locations[model_config["model_name"]][:args.intervened_head_num],
        interpolation_rate=args.interpolation_rate,
        intervention_activation=intervention_activation,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        generation_config=generation_config
    )
    print(f"ASR : {ASR}")
