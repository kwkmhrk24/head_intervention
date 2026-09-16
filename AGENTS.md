# AGENTS.md

## Purpose

This repository is for research on LLM jailbreak defense and mechanistic intervention.

Codex handles repository inspection, implementation, testing, validation,
explicitly requested experiments, and reproducible reporting.
Scientific direction, hypotheses, and major methodological decisions are determined
through discussion between the user and ChatGPT.
Do not independently redefine the research question or scientific conditions.

## Instruction Priority

Follow explicit instructions in the current task.
If ambiguity could affect scientific validity, dataset independence, provenance,
reproducibility, or experimental conditions, do not guess.
Stop at the safe boundary and report the ambiguity.
Never silently substitute a scientifically different condition.

## Before Starting Work

Before modifying code or running experiments:

1. Confirm Git branch, HEAD, and worktree status.
2. Inspect relevant scripts, utilities, tests, manifests, and artifacts.
3. Reuse existing infrastructure where appropriate.

For dataset- or experiment-related work, perform a read-only provenance check when needed.
Do not infer dataset mappings, indices, or historical conditions without evidence.

## Scientific Integrity

Keep these roles clearly separated:

* discovery / selection
* development / diagnostic
* holdout
* final independent test

Do not use evaluation or final-test data to select Heads/elements, construct references,
choose hyperparameters, design ranking criteria, or tune the method,
unless the task explicitly defines that dataset as development data.

Do not change scientific conditions after seeing evaluation results unless explicitly asked.
Scientific conditions include selected Heads/elements, Top-k, alpha, ranking criterion,
reference construction, intervention timing, dataset split, prompt filtering,
and success criterion.

Do not select or remove prompts after inspecting model outputs unless the rule was predefined.
Do not silently replace missing datasets or infer uncertain dataset-index mappings.
Clearly distinguish exploratory, diagnostic, holdout, and independent-test results.
Do not overstate generalization from development or diagnostic data.

## Reproducibility and Provenance

When relevant, preserve or report:

* Git commit
* dataset source / indices / hashes
* model and tokenizer identifiers / revisions / local sources
* resolved config and random seed
* dtype and device / GPU model
* run ID / timestamp
* source artifact hashes

For newly generated datasets, prefer versioned configs, manifests, schemas, and validators.
Report unknown provenance as `unknown`; do not infer it.
Never modify raw source data in place.
Do not overwrite existing derived data or experiment outputs.
Use a new run directory for new experiments unless explicitly instructed otherwise.

## Implementation

Make minimal, task-focused changes.
Reuse existing utilities and shared infrastructure instead of duplicating logic.
Do not refactor unrelated code during experiment tasks.
Preserve existing scientific behavior unless a behavior change is explicitly requested.
Use deterministic ordering and tie-breaking for ranking or selection.
Add fail-fast validation for assumptions that could invalidate an experiment.
Never silently fall back to a scientifically different implementation.

## Testing

For code changes, normally run:

1. targeted tests
2. relevant broader tests
3. Python compilation checks when applicable
4. `git diff --check`

Report validation results.
Do not bypass scientific invariant tests or alter scientific conditions merely to make tests pass.

## GPU Experiments

Run GPU-intensive experiments only when explicitly requested.
Prefer CPU-only audits and preflight validation before GPU execution.
Do not proceed if preflight validation fails.

For explicitly requested GPU experiments, first check within the normal Codex sandbox:

* `nvidia-smi` detects the expected GPU
* PyTorch reports `torch.cuda.is_available() == True` and at least one CUDA device
* target GPU / VRAM / existing GPU processes
* frozen scientific conditions

If `nvidia-smi` cannot connect to the NVIDIA driver, PyTorch reports CUDA unavailable,
or the CUDA device count is zero in the normal sandbox, do not yet conclude that GPU
execution is unavailable. Obtain user approval if required, then recheck without
`sudo` using Codex execution permissions outside the normal sandbox. Recheck
`nvidia-smi`, `torch.cuda.is_available()`, `torch.cuda.device_count()`, and the GPU
name; check VRAM and GPU processes when relevant. If the GPU is available there,
continue the explicitly requested experiment in that same execution environment.
Record the change in execution permissions as a runtime adjustment; it does
not change the scientific conditions.

Never use `sudo` to obtain GPU access. Do not reinstall or change the NVIDIA driver,
CUDA, or PyTorch to make the GPU visible. Do not silently switch to CPU, another
machine, another Python environment, or another execution method beyond the
approved sandbox-exempt run. Do not change the scientific conditions.
If GPU or CUDA remains unavailable outside the normal sandbox, stop and report it.

Use fail-fast runtime invariants when applicable.
Record generation failures, NaN/Inf, skipped inputs, unexpected intervention counts,
and other anomalies.
Do not change Top-k, alpha, dataset, ranking, reference, or other scientific conditions
in response to intermediate results unless explicitly instructed.
Record runtime adjustments even when they do not change the scientific experiment.

## Artifacts

Treat existing research artifacts as immutable unless explicitly instructed otherwise.
Do not overwrite prior experiment outputs.
Do not commit experiment artifacts, generated outputs, logs, caches, or large artifacts.
Versioned configs, manifests, schemas, and validators may be committed.
Never commit model weights, secrets, caches, or unintended large files.

## Git

Do not perform destructive Git operations without explicit instruction.
Do not merge, delete branches/tags, or reset published work unless explicitly requested.
Treat pushed history as immutable: never force-push (including force-with-lease),
amend or rebase pushed commits. Add corrections as new commits.
When a logical work unit is complete and its required tests and preflight checks pass,
normally commit the code and tests and push them to the current feature branch.
Do not mechanically push unverified or incomplete work.
Do not push routine research implementation directly to `main`.
After repository changes, report branch, HEAD, and worktree status.

## Reporting

At task completion, report as applicable:

* Git state and changed files
* exact scientific conditions
* dataset / sample counts
* run directory
* validation / test results
* primary / diagnostic metrics
* anomalies / failures
* intentionally skipped work

For jailbreak-defense evaluation, report newly defended, still-success,
newly worsened, and still-failure when available.
Separate observations from interpretation.
Do not claim causality or generalization beyond what the experiment supports.

## Research Workflow

Normal workflow:

User + ChatGPT:
research discussion -> hypothesis -> experiment design -> frozen scientific conditions

Codex:
repository audit -> implementation -> CPU validation -> requested GPU experiment
-> reproducible report

Codex may suggest follow-up experiments, but must not automatically launch
new scientific experiments or tune the method without explicit instruction.
