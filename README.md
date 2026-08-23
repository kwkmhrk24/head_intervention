# Jailbreak Defense in LLM via Attention Head Analysis and Selective Intervention

## Requirements
- Python: 3.12.4
- Library: Please see `requirements.txt`

## Directory structure
```
head_intervention_defense/
    ├── README.md
    ├── requirements.txt
    ├── data/
    │   ├── advbench.jsonl
    │   ├── ...
    ├── utils/
    │   ├── __init__.py
    │   ├── evaluate_utils.py
    │   ├── extract_utils.py
    │   ├── intervention_utils.py
    │   ├── model_utils.py
    │   └── prompt_utils.py
    ├── compute_AIE.py
    ├── extract_itervention_vector.py
    └── evaluate.py
```


## How to execute experiments
### Collect jailbreak prompts
First, run [GCG](https://github.com/llm-attacks/llm-attacks) and [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN) to collect jailbreak prompts.

Please store the prompts in a JSONL file, where each line is a JSON object containing a key named `prompt`.

Save this file in the ./data directory.

### Compute AIE (Section 3.2 and 3.3 Step1)
The main script for this step is `compute_AIE.py`

To run the experiment:
```
$ python compute_AIE.py --model <model> --jailbreak_prompts <jailbreak prompts> --original_prompts <original prompts> --output_file <output file>
```

Key arguments:
1. `--model`: Path to the model.
1. `--jailbreak_prompts`: Path to the file containing jailbreak prompts used for computing AIE (referred to as $P_s$ in Section 3.2 of the paper).
1. `--original_prompts`: Path to the corresponding original prompts before the attack.
1. `--output_file`: Name of the output file. The result will be saved at `./data/{output_file}.pt`.

### Extract average attention head output for intervention (Section 3.3 Step2)
The main script for this step is `extract_intervention_vector.py`

To run the experiment:
```
$ python extract_intervention_vector.py --model <model> --prompts_file <prompts file> --system_prompt <system prompt>
```

Key arguments:
1. `--model`: Path to the model.
1. `--prompts_file`: Path to the file containing prompts used to extract intervention values (referred to as in Section 3.3).
1. `--system_prompt`: Sets the system prompt. Default is `None`. If set to `self-reminder`, the script will automatically use the system prompt used in the self-reminder setting.


### Evaluate ASR
The main script for this step is `evaluate.py`.

To run the experiment:
```
$ python evaluate.py --model <model> --input_file <input file> --system_prompt <system prompt> --intervention_activation <intervention activation> --intervened_head_num <intervened head number> --interpolation_rate <interpolation rate>
```

Key arguments:
1. `--model`: Path to the model.
1. `--input_file`: Path to the JSONL file containing the prompts to be evaluated.
1. `--system_prompt`: Sets the system prompt. Default is `None`. If set to `self-reminder`, the script will automatically use the system prompt used in the self-reminder setting.
1. `--intervention_activation`: Path to the `.pt` file containing the intervention values.
1. `--intervened_head_num`: Number of attention heads to intervene (referred to as $S$ in the paper).
1. `--interpolation_rate`: Interpolation rate (referred to as $\lambda$ in the paper).

If you want to change the specific heads to be intervened on, please modify the `intervention_location` variable directly inside `evaluate.py`.

## AutoDAN internal hold-out head selection

The Llama2 AutoDAN selection pipeline is documented in
[`docs/autodan_selection_holdout.md`](docs/autodan_selection_holdout.md). All 130
AutoDAN pairs had already been used for exploratory attention-score analysis
before this split was defined. Therefore, the 20/110 split is an internal
AutoDAN hold-out validation and the 110-prompt evaluation subset must not be
described as a fully unseen test set.

The distinction between the stored AutoDAN source labels (84/130) and the
current no-intervention evaluation (17/130), together with prompt-level
diagnostic output, is documented in
[`docs/autodan_baseline_alignment.md`](docs/autodan_baseline_alignment.md).

## Current research workflow

The canonical branch for ongoing work is `research/jailbreak-defense`. The
original reproduction commands above remain available, but new analysis should
start from these maintained entrypoints:

- `scripts/analyze_attention_scores.py`: capture tail-window attention scores.
- `scripts/analyze_matched_attention_head_metrics.py`: compare matched
  AutoDAN/GCG attention metrics.
- `scripts/evaluate_asym_score_intervention.py`: run the shared Llama/Mistral,
  AutoDAN/GCG direct `S_asym` evaluation.
- `scripts/prepare_gcg_wasserstein_reference.py`: build the fixed GCG split,
  Wasserstein ranking, and reference artifacts.
- `scripts/evaluate_gcg_wasserstein_reference_intervention.py`: run the fixed
  GCG reference intervention.
- `scripts/evaluate_gcg_first_token_reference_ablation.py` and
  `scripts/evaluate_gcg_first_token_holdout.py`: run the selection and holdout
  first-token-only evaluations.
- `scripts/diagnose_gcg_first_token_generalization.py` and
  `scripts/diagnose_gcg_reference_factorization.py`: run the current
  generalization and reference/head factorization diagnostics.
- `scripts/audit_gcg_reference.py`: audit artifacts and saved evaluation runs
  with the `artifacts`, `summary`, `evaluation`, or `first-token` subcommand.
- `scripts/analyze_cross_run_intervention_transitions.py`: compare prompt-level
  transitions across saved multidataset runs without loading a model.

Reusable logic and branch policy are mapped in
[`docs/repository_structure.md`](docs/repository_structure.md).
