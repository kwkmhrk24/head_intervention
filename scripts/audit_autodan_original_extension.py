"""Inventory saved AutoDAN evaluations without loading model weights or scores."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import re
from statistics import median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_autodan_id15_controls import edit_distance
from utils.evaluate_utils import check_success_with_details
from utils.prompt_utils import apply_template

BASE = ROOT / "data/outputs/llama2/autodan"
DERIVED = ROOT / "data/derived/llama2/autodan"
RAW = ROOT / "data/raw/llama2"
FAST = BASE / "formatting_ablation/official_fastchat_34062e9_full130_20260914T151610Z"
OLD = ROOT / "data/outputs/gcg_position_element_topk"
RECHECK = ROOT / "results/harmbench_autodan_old_strong_pilot_recheck/trajectory_run0_current_baseline.jsonl"
MODEL = Path("/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *args], text=True
    ).strip()


def output_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def output_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def locate(tokenizer, formatted, body, original, add_special_tokens):
    encoded = tokenizer(formatted, add_special_tokens=add_special_tokens,
                        return_offsets_mapping=True)
    require(formatted.count(body) == 1, "ambiguous user message")
    match = list(re.finditer(re.escape(original), body))
    retention = "exact"
    if not match:
        match = list(re.finditer(re.escape(original), body, re.IGNORECASE))
        retention = "case_only"
    if len(match) != 1:
        return encoded.input_ids, None, None, "missing_or_ambiguous", None
    start, end = (formatted.index(body) + pos for pos in match[0].span())
    positions = [i for i, (a, b) in enumerate(encoded.offset_mapping)
                 if a < end and start < b and a < b]
    require(positions and positions == list(range(positions[0], positions[-1] + 1)),
            "noncontiguous Original Token span")
    boundary = any(encoded.offset_mapping[i][0] < start or encoded.offset_mapping[i][1] > end
                   for i in positions)
    return encoded.input_ids, [positions[0], positions[-1] + 1], [start, end], retention, boundary


def candidate(tokenizer, path, line, index, original, body, formatted, response, label,
              evaluation_format, evaluation_version, revision, add_special_tokens, saved_ids=None,
              run=None, step=None, seed=None, details=None):
    ids, span, char_span, retention, boundary = locate(
        tokenizer, formatted, body, original, add_special_tokens)
    if saved_ids is not None:
        require(ids == saved_ids, f"saved Token mismatch: {path}:{line}")
    require(type(label) is bool and isinstance(response, str), f"missing stored evaluation: {path}:{line}")
    replayed, reason = check_success_with_details(response)
    require(replayed == label, f"stored label disagreement: {path}:{line}")
    if details is not None:
        require(details == reason["matched_refusal_substrings"], f"refusal mismatch: {path}:{line}")
    return {
        "original_id": index, "original_text": original, "candidate_id": sha(body),
        "candidate_source": str(path.relative_to(ROOT)), "source_line": line,
        "prompt_text": body, "input_token_ids": ids, "input_length": len(ids),
        "original_token_span": span, "original_token_ids": ids[slice(*span)] if span else None,
        "original_character_span_in_formatted_input": char_span,
        "original_token_crosses_character_boundary": boundary, "original_retention": retention,
        "evaluation_format": evaluation_format, "model_revision": revision,
        "response": response, "response_sha256": sha(response), "stored_label": label,
        "label_reason": reason["decision_reason"], "matched_refusal_substrings": reason["matched_refusal_substrings"],
        "empty_response": reason["empty_response"], "evaluation_version": evaluation_version,
        "run_id": run, "step_index": step, "seed": seed,
        "input_token_ids_provenance": "stored_and_retokenized" if saved_ids is not None else "retokenized",
    }


def inventory(tokenizer, input_files):
    original_path = DERIVED / "original_prompts.jsonl"
    jailbreak_path = DERIVED / "jailbreak_prompts.jsonl"
    raw_path = RAW / "llama2_AutoDAN.json"
    raw_jsonl_path = RAW / "AutoDAN_llama2.jsonl"
    fast_path = FAST / "paired_results.jsonl"
    hf_path = OLD / "autodan130_transfer/full130_20260914T085700Z/autodan130_baseline.jsonl"
    fast_manifest_path = FAST / "run_manifest.json"
    hf_manifest_path = hf_path.with_name("run_manifest.json")
    input_files.update((original_path, jailbreak_path, raw_path, raw_jsonl_path, fast_path, hf_path,
                        fast_manifest_path, hf_manifest_path, ROOT / "utils/evaluate_utils.py"))
    originals = [r["prompt"] for r in rows(original_path)]
    require(len(originals) == len(set(originals)) == 130, "duplicate or missing Original")
    fast_manifest = read(fast_manifest_path)
    hf_manifest = read(hf_manifest_path)
    fast_rows = rows(fast_path)
    archive = {}
    result = []
    for i, (raw, raw_jsonl, derived, f, h) in enumerate(zip(
        read(raw_path), rows(raw_jsonl_path), rows(jailbreak_path), fast_rows, rows(hf_path), strict=True
    )):
        body = raw["user prompt"]
        require(i == raw["index"] == raw_jsonl["index"] == f["dataset_index"] == h["dataset_index"] and
                body == raw_jsonl["prompt"] == derived["prompt"] == f["prompt"] == h["prompt"],
                f"legacy source mapping {i}")
        require(not f["generation_failed"] and not h["generation_failed"] and
                (f["current_response"], f["current_success"]) == (h["response"], h["success"]),
                f"legacy response mapping {i}")
        require(len(f["official_input_ids"]) == f["official_prompt_length"] and
                len(h["input_ids"]) == h["prompt_length"], f"legacy input length {i}")
        fast_candidate = candidate(
            tokenizer, fast_path, i + 1, i, originals[i], body, f["official_formatted_prompt"],
            f["official_format_response"], f["official_format_success"], "FastChat",
            "legacy_fastchat_full130_20260914", fast_manifest["model_repository_head"], True,
            f["official_input_ids"], details=f["matched_refusal_substrings"])
        require(bool(fast_candidate["original_token_span"]) != f["contains_unreplaced_placeholder"],
                f"unexpected placeholder/retention at Original {i}")
        result.append(fast_candidate)
        result.append(candidate(
            tokenizer, hf_path, i + 1, i, originals[i], body, h["formatted_prompt"], h["response"],
            h["success"], "HF", "legacy_hf_full130_20260914", None, False, h["input_ids"],
            details=h["matched_refusal_substrings"]))
        archive[i] = {"original_id": i, "candidate_id": sha(body), "raw_stored_label": raw["success"],
                      "hf_label": h["success"], "fastchat_label": f["official_format_success"]}
    conditions = {
        "legacy_fastchat_full130_20260914": {
            "manifest": str(fast_manifest_path.relative_to(ROOT)),
            "settings": fast_manifest["generation_config"],
            "input_format": fast_manifest["condition_b_format"],
            "model_revision": fast_manifest["model_repository_head"],
        },
        "legacy_hf_full130_20260914": {
            "manifest": str(hf_manifest_path.relative_to(ROOT)),
            "settings": hf_manifest["generation_settings"],
            "model_revision": None,
        },
    }
    template_hash = sha(tokenizer.chat_template)
    sources = sorted(set(BASE.glob("harmbench_*/prompt_results.jsonl")) |
                     set((ROOT / "results").glob("harmbench_autodan_*/*.jsonl")))
    for path in sources:
        source_rows = rows(path)
        summary_path = path.with_name("summary.json" if path.name == "prompt_results.jsonl"
                                      else path.stem + "_summary.json")
        summary = read(summary_path)
        input_files.update((path, summary_path))
        version = path.parent.name
        conditions[version] = {
            "manifest": str(summary_path.relative_to(ROOT)), "settings": summary["decoding_config"],
            "model_revision": None, "template_sha256": template_hash,
            "evaluation_timestamp": source_rows[0]["run_timestamp_utc"],
            "transformers_version": summary["transformers_version"],
            "generation_commit": summary["source_harmbench_generation_commit"],
        }
        seen = set()
        for line, row in enumerate(source_rows, 1):
            i, body = row["raw_index"], row["attack_prompt"]
            require(row["goal"] == originals[i] and row["behavior_id"] == f"advbench_{i + 1}",
                    f"Original mapping {path}:{line}")
            require(sha(body) == row["attack_prompt_sha256"] and sha(row["response"]) == row["response_sha256"],
                    f"source hashes {path}:{line}")
            require((i, sha(body)) not in seen and not row["generation_failed"] and
                    row["chat_template_sha256"] == template_hash and row["model_path"] == str(MODEL) and
                    row["decoding_config"] == summary["decoding_config"] and
                    row["transformers_version"] == summary["transformers_version"],
                    f"source evaluation metadata {path}:{line}")
            seen.add((i, sha(body)))
            formatted = apply_template([body], None, tokenizer, {"model_name": "Llama"})[0]
            record = candidate(
                tokenizer, path, line, i, originals[i], body, formatted, row["response"],
                row["keyword_success"], "HF", version, None, False, run=row.get("harmbench_run_index"),
                step=row.get("harmbench_step_index"), seed=row.get("run_seed"),
                details=row["matched_refusal_substrings"])
            require(record["original_retention"] == "exact", f"HarmBench Original span {path}:{line}")
            record["generation_source_path"] = row.get("source_logs_path") or row.get("source_test_cases_path")
            record["generation_source_sha256"] = row.get("source_logs_sha256") or row.get("source_test_cases_sha256")
            generation_path = Path(record["generation_source_path"])
            if generation_path.is_file():
                input_files.add(generation_path)
            result.append(record)
    return result, list(archive.values()), conditions


def audit_reuses(candidates, input_files):
    fast = {r["original_id"]: r for r in candidates if r["evaluation_format"] == "FastChat"}
    paths = [
        OLD / "autodan130_official_format_transfer/full130_20260914T152933Z/baseline_reproduction.jsonl",
        OLD / "autodan_selection_reference_ablation/evaluation110_official_fastchat_34062e9_20260914T163404Z/evaluation110_baseline.jsonl",
        BASE / "holdout_seed42/rank_shift_cancellation_v1_cross_attack_20260916T0340Z/autodan_evaluation110_baseline.jsonl",
        BASE / "baseline_alignment/no_intervention_prompt_results.jsonl",
    ]
    counts = {}
    for path in paths:
        input_files.add(path)
        count = Counter()
        for row in rows(path):
            reference = fast[row.get("dataset_index", row.get("source_index"))]
            require(row["prompt"] == reference["prompt_text"], f"alternate source includes a new candidate: {path}")
            count["same_candidate_text"] += 1
            count["same_fastchat_input_ids"] += row.get("input_ids") == reference["input_token_ids"]
            count["same_fastchat_label"] += row["success"] == reference["stored_label"]
        counts[str(path.relative_to(ROOT))] = count
    return counts


def audit_generation(candidates, input_files):
    """Locate saved generation-only cases without deriving a label from search logs."""
    evaluated = {(r["original_id"], r["candidate_id"]) for r in candidates
                 if "generation_source_path" in r}
    inventory = defaultdict(set)
    sources = defaultdict(set)
    dirs = ROOT.parent / "HarmBench/test_cases"
    for path in sorted(dirs.glob("autodan_*/test_cases_individual_behaviors/*/*.json")):
        if path.name not in ("logs.json", "test_cases.json"):
            continue
        input_files.add(path)
        for behavior, entries in read(path).items():
            if path.name == "logs.json":
                entries = [step for run in entries for step in run]
            for entry in entries:
                body = entry["test_case"] if isinstance(entry, dict) else entry
                digest = sha(body)
                inventory[behavior].add(digest)
                sources[behavior, digest].add(str(path.relative_to(ROOT.parent)))
    result = []
    no_evaluation = []
    for behavior, digests in sorted(inventory.items()):
        index = int(behavior.removeprefix("advbench_")) - 1 if behavior.startswith("advbench_") else None
        unmatched = [digest for digest in sorted(digests) if (index, digest) not in evaluated]
        result.append({"behavior_id": behavior, "legacy_original_id": index,
                       "distinct_saved_candidates": len(digests),
                       "no_matching_saved_hf_evaluation": len(unmatched)})
        no_evaluation.extend({"behavior_id": behavior, "legacy_original_id": index,
                              "candidate_id": digest, "generation_sources": sorted(sources[behavior, digest])}
                             for digest in unmatched)
    return result, no_evaluation


def pair_and_summarize(candidates):
    cohorts = defaultdict(list)
    for row in candidates:
        cohorts[row["evaluation_version"], row["original_id"]].append(row)
    pairs, state = [], []
    for (version, original_id), cohort in sorted(cohorts.items()):
        valid = [r for r in cohort if r["original_token_span"] and not r["empty_response"]]
        yes = [r for r in valid if r["stored_label"]]
        no = [r for r in valid if not r["stored_label"]]
        state.append({
            "evaluation_format": cohort[0]["evaluation_format"], "evaluation_version": version,
            "original_id": original_id, "candidates": len(cohort), "success": len(yes), "failure": len(no),
            "excluded_missing_original_or_empty_response": len(cohort) - len(valid),
            "same_evaluation_opposite_label_pairs": len(yes) * len(no),
        })
        for s, f in product(yes, no):
            require(s["candidate_id"] != f["candidate_id"], "same candidate with conflicting labels")
            length = s["input_length"] - f["input_length"]
            pairs.append({
                "original_id": original_id, "evaluation_format": s["evaluation_format"],
                "evaluation_version": version, "success_candidate_id": s["candidate_id"],
                "failure_candidate_id": f["candidate_id"], "success_source_line": s["source_line"],
                "failure_source_line": f["source_line"], "success_run": s["run_id"], "failure_run": f["run_id"],
                "success_step": s["step_index"], "failure_step": f["step_index"],
                "success_input_length": s["input_length"], "failure_input_length": f["input_length"],
                "length_delta": length, "absolute_length_delta": abs(length),
                "relative_length_delta": abs(length) / max(s["input_length"], f["input_length"]),
                "original_start_delta": s["original_token_span"][0] - f["original_token_span"][0],
                "original_end_delta": s["original_token_span"][1] - f["original_token_span"][1],
                "same_original_token_ids": s["original_token_ids"] == f["original_token_ids"],
                "token_edit_distance": edit_distance(s["input_token_ids"], f["input_token_ids"]),
                "success_label_reason": s["label_reason"], "failure_label_reason": f["label_reason"],
            })
    return pairs, state


def stats(values):
    return {"min": min(values), "median": median(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output directory already exists")
    from transformers import AutoTokenizer
    import transformers

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer_files = {MODEL / name for name in ("tokenizer.json", "tokenizer.model",
                                                "tokenizer_config.json", "special_tokens_map.json")}
    input_files = {Path(__file__).resolve(), *tokenizer_files, ROOT / "utils/prompt_utils.py",
                   ROOT / "scripts/analyze_autodan_id15_controls.py", ROOT / "AGENTS.md"}
    candidates, raw_labels, conditions = inventory(tokenizer, input_files)
    duplicates = audit_reuses(candidates, input_files)
    generation, unmatched = audit_generation(candidates, input_files)
    prior = BASE / "id15_controls_20261123/plan.json"
    revision_evidence = BASE / "pair_score_id15_20261123/plan.json"
    input_files.update((prior, revision_evidence))
    revision = git(MODEL, "rev-parse", "HEAD")
    plan = read(revision_evidence)
    require(revision == plan["revision"] and sha(tokenizer.chat_template) == plan["template_sha256"],
            "local tokenizer revision/template differs from the ID 15 reconstruction reference")
    prior_candidates = read(prior)["candidates"]
    anchor = {r["candidate_id"]: r for r in candidates
              if r["candidate_source"] == str(RECHECK.relative_to(ROOT)) and r["original_id"] == 15}
    require(set(anchor) == {r["candidate_id"] for r in prior_candidates.values()} and
            all(anchor[r["candidate_id"]]["input_token_ids"] == r["input_ids"] and
                anchor[r["candidate_id"]]["original_token_span"] == r["original_span"] and
                anchor[r["candidate_id"]]["stored_label"] == r["success"] and
                anchor[r["candidate_id"]]["response_sha256"] == r["response_sha256"]
                for r in prior_candidates.values()), "ID 15 prior reconstruction mismatch")
    pairs, state = pair_and_summarize(candidates)
    signatures = defaultdict(set)
    revisions = defaultdict(set)
    for r in candidates:
        key = (r["original_id"], r["candidate_id"])
        signatures[key].add((r["evaluation_format"], r["stored_label"]))
        if r["original_token_span"] and not r["empty_response"]:
            revisions[r["original_id"]].add((r["evaluation_format"], r["stored_label"], r["candidate_id"]))
    observed = {}
    for name, subset in (
        ("FastChat", [r for r in candidates if r["evaluation_format"] == "FastChat"]),
        ("HF_legacy", [r for r in candidates if r["evaluation_version"] == "legacy_hf_full130_20260914"]),
        ("HF_HarmBench", [r for r in candidates if r["evaluation_version"] not in
                         ("legacy_fastchat_full130_20260914", "legacy_hf_full130_20260914")]),
        ("HF_all", [r for r in candidates if r["evaluation_format"] == "HF"]),
    ):
        valid = [r for r in subset if r["original_token_span"] and not r["empty_response"]]
        labels = defaultdict(set)
        for r in valid:
            labels[r["original_id"]].add(r["stored_label"])
        sources = {r["evaluation_version"] for r in subset}
        subset_pairs = [p for p in pairs if p["evaluation_version"] in sources]
        observed[name] = {
            "observed_originals": len({r["original_id"] for r in subset}),
            "eligible_originals": len(labels),
            "unique_candidates": len({(r["original_id"], r["candidate_id"]) for r in subset}),
            "evaluation_records": len(subset),
            "success_only_originals": sum(v == {True} for v in labels.values()),
            "failure_only_originals": sum(v == {False} for v in labels.values()),
            "paired_original_ids": sorted({p["original_id"] for p in subset_pairs}),
            "same_evaluation_pairs_including_reevaluations": len(subset_pairs),
            "distinct_candidate_pairs": len({(p["success_candidate_id"], p["failure_candidate_id"])
                                            for p in subset_pairs}),
        }
    anchor_pairs = [p for p in pairs if p["evaluation_version"] == RECHECK.parent.name]
    exact_ids = [p for p in anchor_pairs if p["same_original_token_ids"]]
    strong_candidates = {r["candidate_id"]: r for r in candidates
                         if r["evaluation_version"] == "harmbench_strong_pilot_selection5_trajectory"}
    recheck_candidates = {r["candidate_id"]: r for r in candidates
                          if r["evaluation_version"] == RECHECK.parent.name}
    require(strong_candidates.keys() == recheck_candidates.keys() and all(
        (r["response_sha256"], r["stored_label"], r["input_token_ids"]) ==
        (recheck_candidates[key]["response_sha256"], recheck_candidates[key]["stored_label"],
         recheck_candidates[key]["input_token_ids"]) for key, r in strong_candidates.items()),
        "Strong pilot and recheck differ")
    different_candidate_cross_format = sorted(i for i, choices in revisions.items()
        if any(a[2] != b[2] and a[0] != b[0] and a[1] != b[1] for a, b in product(choices, repeat=2)))
    summary = {
        "formats": observed,
        "raw_legacy_label_counts": dict(Counter(r["raw_stored_label"] for r in raw_labels)),
        "excluded_legacy_original_ids": sorted({r["original_id"] for r in candidates if not r["original_token_span"]}),
        "same_candidate_cross_format_opposite_labels": sorted(i for (i, _), v in signatures.items()
            if i != 10 and ({("HF", True), ("FastChat", False)} <= v or
                            {("HF", False), ("FastChat", True)} <= v)),
        "different_candidate_cross_format_opposite_labels": different_candidate_cross_format,
        "generation_only_inventory": generation,
        "id15_anchor_pairs": len(anchor_pairs),
        "id15_anchor_pairs_with_identical_original_token_ids": len(exact_ids),
        "id15_anchor_pair_distributions": {
            key: stats([p[key] for p in anchor_pairs])
            for key in ("absolute_length_delta", "relative_length_delta", "original_start_delta",
                        "original_end_delta", "token_edit_distance")
        },
        "id15_anchor_equal_original_token_pair_distributions": {
            key: stats([p[key] for p in exact_ids])
            for key in ("absolute_length_delta", "relative_length_delta", "original_start_delta",
                        "original_end_delta", "token_edit_distance")
        },
    }
    harmbench = ROOT.parent / "HarmBench"
    repo_status = {str(repo): {"branch": git(repo, "branch", "--show-current"),
                               "head": git(repo, "rev-parse", "HEAD"),
                               "status": git(repo, "status", "--short").splitlines()}
                   for repo in (ROOT, harmbench)}
    validation = {
        "saved_fastchat_and_legacy_hf_ids_retokenized": 260,
        "id15_full_ids_original_span_labels_responses_match_prior": len(anchor),
        "labels_replayed_against_saved_response": len(candidates),
        "empty_responses": sum(r["empty_response"] for r in candidates),
        "strong_pilot_reevaluation_identical_records": len(strong_candidates),
        "baseline_reuses": duplicates,
        "historical_hf_revision_recorded": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("summary", summary), ("validation", validation)):
        output_json(args.output / f"{name}.json", data)
    with (args.output / "candidates.jsonl").open("w") as stream:
        for row in candidates:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    output_csv(args.output / "pairs.csv", [
        p for p in pairs if p["evaluation_version"] != "harmbench_strong_pilot_selection5_trajectory"
    ])
    output_csv(args.output / "pairs_by_evaluation.csv", pairs)
    output_csv(args.output / "originals_by_evaluation.csv", state)
    output_csv(args.output / "legacy_label_audit.csv", raw_labels)
    output_json(args.output / "generation_without_matching_evaluation.json", unmatched)
    output_json(args.output / "manifest.json", {
        "schema_version": 1,
        "span_convention": "zero-based half-open; tokens overlapping the Original characters are included",
        "candidate_id": "SHA-256 of exact prompt_text; candidate evaluation key also includes original_id and evaluation_version",
        "label_policy": "saved no-intervention keyword label; no cross-evaluation pairing",
        "evaluation_versions": conditions,
        "local_tokenizer_model_repository_head": revision,
        "transformers_version": transformers.__version__,
        "repositories": repo_status,
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(input_files)},
    })
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
