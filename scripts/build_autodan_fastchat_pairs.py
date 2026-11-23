"""Build a dataset from saved FastChat evaluations and fixed AutoDAN references."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from statistics import median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_autodan_id15_controls import edit_distance
from scripts.audit_autodan_original_extension import BASE, DERIVED, FAST, MODEL, locate, require
from scripts.evaluate_gcg_position_element_topk_intervention import (
    AUTODAN_OFFICIAL_PREFIX,
    AUTODAN_OFFICIAL_SUFFIX,
)
from utils.artifact_io import (
    ensure_output_directory_absent, read_json, read_jsonl, sha256_file, sha256_text,
    utc_now, write_json, write_jsonl,
)
from utils.evaluate_utils import REFUSAL_SUBSTRINGS, check_success_with_details

AUDIT = BASE / "original_extension_audit_20261123"
FORMAT = "FastChat"
VERSION = "legacy_fastchat_full130_20260914"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json")
CONDITION_FIELDS = (
    "autodan_repository_commit", "fastchat_version_used_for_cpu_validation",
    "fastchat_conversation_source_sha256", "condition_b_format", "generation_config",
    "decode", "model_dtype", "attention_implementation", "model_eval",
    "tokenizer_class", "tokenizer_is_fast", "transformers_version", "success_checker",
    "batch_size", "seed", "intervention_enabled",
)
PAIR_FIELDS = (
    "original_id", "success_candidate_id", "failure_candidate_id", "same_original_token_ids",
    "success_input_length", "failure_input_length", "length_delta", "absolute_length_delta",
    "relative_length_delta", "original_start_delta", "original_end_delta", "token_edit_distance",
    "success_boundary_crossing", "failure_boundary_crossing", "success_label_reason", "failure_label_reason",
    "selected",
)


def formatted_prompt(body):
    return AUTODAN_OFFICIAL_PREFIX + body + AUTODAN_OFFICIAL_SUFFIX


def load_frozen(tokenizer, audit_dir, fast_dir, conditions, revision):
    source_path = fast_dir / "paired_results.jsonl"
    sources = read_jsonl(source_path)
    originals = read_jsonl(DERIVED / "original_prompts.jsonl")
    require(len(sources) == 130 and [r["dataset_index"] for r in sources] == list(range(130)),
            "legacy FastChat index mismatch")
    require(len(originals) == len(sources), "legacy Original index mismatch")
    successful = [r for r in sources if r["official_format_success"]]
    require(len(successful) == 49 and {r["dataset_index"] for r in successful
                                     if r["contains_unreplaced_placeholder"]} == {10},
            "unexpected FastChat success/placeholder records")
    from_audit = [r for r in read_jsonl(audit_dir / "candidates.jsonl")
                  if r["evaluation_format"] == FORMAT and r["stored_label"] and r["original_id"] != 10]
    require(len(from_audit) == 48 and {r["original_id"] for r in from_audit} ==
            {r["dataset_index"] for r in successful} - {10}, "audit reference IDs differ")
    anchors = {}
    for a in from_audit:
        i = a["original_id"]
        r = sources[i]
        body = a["prompt_text"]
        rendered = formatted_prompt(body)
        ids, span, chars, retention, boundary = locate(tokenizer, rendered, body, a["original_text"], True)
        verdict, detail = check_success_with_details(a["response"])
        require(a["original_text"] == originals[i]["prompt"] and a["evaluation_version"] == VERSION and
                body == r["prompt"] and rendered == r["official_formatted_prompt"] and
                ids == a["input_token_ids"] == r["official_input_ids"] and
                len(ids) == a["input_length"] == r["official_prompt_length"] and
                span and span == a["original_token_span"] and ids[slice(*span)] == a["original_token_ids"] and
                chars == a["original_character_span_in_formatted_input"] and
                boundary == a["original_token_crosses_character_boundary"] and
                retention == a["original_retention"] and a["model_revision"] == revision and
                a["candidate_id"] == sha256_text(body) and a["response"] == r["official_format_response"] and
                a["response_sha256"] == sha256_text(a["response"]) and verdict and
                not detail["empty_response"] and not r["generation_failed"] and
                detail["matched_refusal_substrings"] == r["matched_refusal_substrings"],
                f"legacy reference mismatch at Original {i}")
        anchors[i] = dict(a, source="legacy_autodan_fastchat", generation_method="legacy_autodan",
                          frozen_original_surface=rendered[slice(*chars)], status="reference",
                          evaluation_conditions=conditions,
                          provenance={"artifact": str(source_path), "data_row": i + 1,
                                      "evaluation_artifact": str(fast_dir / "run_manifest.json"),
                                      "prior_audit": str(audit_dir / "candidates.jsonl")})
    return dict(sorted(anchors.items()))


def inspect_addition(raw, anchors, tokenizer, conditions, revision, file, row_number):
    for key in ("candidate_id", "prompt_text", "source", "generation_method"):
        require(isinstance(raw.get(key), str) and raw[key].strip(), f"missing or invalid {key}")
    index = raw.get("original_id")
    require(type(index) is int and index in anchors, "original_id outside the fixed reference set")
    provenance = raw.get("provenance")
    require(isinstance(provenance, dict) and isinstance(provenance.get("artifact"), str) and
            provenance["artifact"].strip(), "provenance.artifact is required")
    anchor = anchors[index]
    body = raw["prompt_text"]
    surface = anchor["frozen_original_surface"]
    ids, span, chars, retention, boundary = locate(tokenizer, formatted_prompt(body), body, surface, True)
    issues = []
    if retention != "exact" or len(re.findall(re.escape(surface), body, re.IGNORECASE)) != 1:
        issues.append("original_surface_changed_or_ambiguous")
    supplied_tokens = raw.get("input_token_ids")
    supplied_span = raw.get("original_token_span")
    require(supplied_tokens is None or (isinstance(supplied_tokens, list) and
            all(type(t) is int and t >= 0 for t in supplied_tokens)), "invalid input_token_ids")
    require(supplied_span is None or (isinstance(supplied_span, list) and len(supplied_span) == 2 and
            all(type(t) is int for t in supplied_span) and 0 <= supplied_span[0] < supplied_span[1]),
            "invalid original_token_span")
    require(raw.get("input_length") is None or type(raw["input_length"]) is int, "invalid input_length")
    computed = {"input_token_ids": ids, "input_length": len(ids), "original_token_span": span}
    for key, expected in computed.items():
        if raw.get(key) is not None and raw[key] != expected:
            issues.append(f"{key}_mismatch")
    response, label = raw.get("response"), raw.get("stored_label")
    require(response is None or isinstance(response, str), "response must be text or null")
    require(label is None or type(label) is bool, "stored_label must be boolean or null")
    reason = None
    refusals = None
    status = "unevaluated"
    if response is not None and label is not None:
        status = "evaluated"
        expected = {"evaluation_format": FORMAT, "evaluation_version": VERSION,
                    "model_revision": revision, "evaluation_conditions": conditions}
        issues.extend(f"{key}_mismatch" for key, value in expected.items() if raw.get(key) != value)
        evidence = provenance.get("evaluation_artifact")
        if not isinstance(evidence, str) or not evidence.strip():
            issues.append("missing_evaluation_artifact")
        if raw.get("input_token_ids") is None:
            issues.append("missing_evaluated_input_token_ids")
        replay, detail = check_success_with_details(response)
        reason, refusals = detail["decision_reason"], detail["matched_refusal_substrings"]
        if detail["empty_response"]:
            issues.append("empty_response")
        if replay != label:
            issues.append("stored_label_disagrees_with_rule")
    normalized = {key: raw.get(key) for key in (
        "evaluation_format", "evaluation_version", "model_revision", "evaluation_conditions",
        "response", "stored_label",
    )}
    return dict(raw | normalized | computed, status="ineligible" if issues else status, issues=issues,
                submission={"file": str(file), "data_row": row_number},
                submitted_token_fields={key: raw[key] for key in computed if raw.get(key) is not None},
                input_token_ids_provenance=("stored_and_retokenized" if supplied_tokens == ids
                                            else "retokenized_reference_format"),
                original_text=anchor["original_text"], frozen_original_surface=surface,
                prompt_sha256=sha256_text(body), original_token_ids=ids[slice(*span)] if span else None,
                original_character_span_in_formatted_input=chars,
                original_token_crosses_character_boundary=boundary, original_retention=retention,
                label_reason=reason, matched_refusal_substrings=refusals)


def import_records(paths, anchors, tokenizer, conditions, revision):
    records, rejected = [], []
    for path in paths:
        for number, raw in enumerate(read_jsonl(path), 1):
            try:
                records.append(inspect_addition(raw, anchors, tokenizer, conditions, revision, path, number))
            except ValueError as exc:
                rejected.append({"file": str(path), "data_row": number, "issue": str(exc), "record": raw})
    return records, rejected


def select_pairs(anchors, records):
    pairs, states = [], []
    by_original = defaultdict(list)
    for row in records:
        by_original[row["original_id"]].append(row)
    for index, anchor in anchors.items():
        rows = by_original[index]
        distinct_inputs_by_id = defaultdict(set)
        input_groups = defaultdict(list)
        for row in rows:
            distinct_inputs_by_id[row["candidate_id"]].add(tuple(row["input_token_ids"]))
            if row["status"] == "evaluated":
                input_groups[tuple(row["input_token_ids"])].append(row)
        anchor_input = tuple(anchor["input_token_ids"])
        canonical_failures = []
        for tokens, group in input_groups.items():
            labels = {r["stored_label"] for r in group}
            if tokens == anchor_input:
                labels.add(True)
            group.sort(key=lambda r: (r["candidate_id"], r["submission"]["file"], r["submission"]["data_row"]))
            representative = None
            for row in group:
                if len(labels) > 1:
                    issue = "conflicting_labels_for_same_input"
                elif len(distinct_inputs_by_id[row["candidate_id"]]) > 1 or (
                    row["candidate_id"] == anchor["candidate_id"] and tokens != anchor_input
                ):
                    issue = "candidate_id_used_for_distinct_inputs"
                elif tokens == anchor_input or representative is not None:
                    issue = "duplicate_evaluated_input"
                else:
                    issue = None
                    representative = row
                if issue:
                    row["issues"].append(issue)
                    row["status"] = "ineligible"
            if representative is not None and not representative["stored_label"]:
                canonical_failures.append(representative)
        matches = []
        for failure in canonical_failures:
            a = anchor["original_token_span"]
            b = failure["original_token_span"]
            a_len, b_len = anchor["input_length"], failure["input_length"]
            matches.append(dict(zip(PAIR_FIELDS, (
                index, anchor["candidate_id"], failure["candidate_id"],
                anchor["original_token_ids"] == failure["original_token_ids"],
                a_len, b_len, a_len - b_len, abs(a_len - b_len), abs(a_len - b_len) / max(a_len, b_len),
                a[0] - b[0], a[1] - b[1],
                edit_distance(anchor["input_token_ids"], failure["input_token_ids"]),
                anchor["original_token_crosses_character_boundary"],
                failure["original_token_crosses_character_boundary"], anchor["label_reason"],
                failure["label_reason"], False,
            ), strict=True)))
        matches.sort(key=lambda p: (
            not p["same_original_token_ids"], p["absolute_length_delta"],
            abs(p["original_start_delta"]) + abs(p["original_end_delta"]),
            p["token_edit_distance"], p["failure_candidate_id"],
        ))
        if matches:
            matches[0]["selected"] = True
        pairs.extend(matches)
        states.append({"original_id": index, "additional_records": len(rows),
                       "eligible_failure_candidates": len(matches),
                       "unevaluated_candidates": sum(r["status"] == "unevaluated" for r in rows),
                       "ineligible_candidates": sum(r["status"] == "ineligible" for r in rows),
                       "status": "paired" if matches else "missing_eligible_failure"})
    return pairs, states


def write_csv(path, fieldnames, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additional", action="append", type=Path, default=[], metavar="JSONL")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ensure_output_directory_absent(args.output)
    from transformers import AutoTokenizer, __version__ as transformers_version

    source = read_json(FAST / "run_manifest.json")
    audit = read_json(AUDIT / "manifest.json")
    revision = source["model_repository_head"]
    require(revision == audit["local_tokenizer_model_repository_head"] and
            transformers_version == source["transformers_version"], "legacy revision/runtime mismatch")
    frozen_input_files = [FAST / "paired_results.jsonl", FAST / "run_manifest.json", DERIVED / "original_prompts.jsonl",
                          *(MODEL / name for name in TOKENIZER_FILES)]
    for path in frozen_input_files:
        require(sha256_file(path) == audit["inputs"].get(str(path)), f"prior input hash mismatch: {path}")
    require(len(REFUSAL_SUBSTRINGS) == 43, "unexpected refusal rule")
    conditions = {key: source[key] for key in CONDITION_FIELDS}
    conditions.update(template="llama-2", model_name="Llama-2-7b-chat-hf",
                      refusal_substrings_sha256=sha256_text(json.dumps(list(REFUSAL_SUBSTRINGS))),
                      tokenizer_config_sha256=sha256_file(MODEL / "tokenizer_config.json"),
                      tokenizer_json_sha256=sha256_file(MODEL / "tokenizer.json"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    anchors = load_frozen(tokenizer, AUDIT, FAST, conditions, revision)
    additions, rejected = import_records(args.additional, anchors, tokenizer, conditions, revision)
    pairs, states = select_pairs(anchors, additions)
    selected = [p for p in pairs if p["selected"]]
    assert len(selected) == len({p["original_id"] for p in selected})
    stats = {}
    for field in ("absolute_length_delta", "relative_length_delta", "original_start_delta",
                  "original_end_delta", "token_edit_distance"):
        values = [p[field] for p in selected]
        stats[field] = dict(min=min(values), median=median(values), max=max(values)) if values else None
    n = len(selected)
    missing = {str(target): max(0, target - n) for target in (10, 20, 30)}
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "candidates.jsonl", [*anchors.values(), *additions])
    write_jsonl(args.output / "rejected_records.jsonl", rejected)
    write_csv(args.output / "pairs.csv", PAIR_FIELDS, pairs)
    write_jsonl(args.output / "pairs.jsonl", pairs)
    write_csv(args.output / "original_status.csv", list(states[0]), states)
    write_json(args.output / "pair_statistics.json", {"selected_originals": n, "statistics": stats})
    write_json(args.output / "validation.json", {
        "frozen_candidates_verified": len(anchors), "excluded_original_ids": [10],
        "additional_status_counts": dict(Counter(r["status"] for r in additions)),
        "additional_issue_counts": dict(Counter(issue for r in additions for issue in r["issues"])),
        "rejected_records": len(rejected), "eligible_pairs": len(pairs), "selected_originals": n,
        "selected_identical_original_token_ids": sum(p["same_original_token_ids"] for p in selected),
        "remaining_originals_per_target": missing,
    })
    inputs = [Path(__file__), AUDIT / "manifest.json", AUDIT / "candidates.jsonl",
              ROOT / "utils/evaluate_utils.py", ROOT / "scripts/analyze_autodan_id15_controls.py",
              ROOT / "scripts/audit_autodan_original_extension.py",
              ROOT / "scripts/evaluate_gcg_position_element_topk_intervention.py",
              *frozen_input_files, *args.additional]
    write_json(args.output / "manifest.json", {
        "schema_version": 1, "created_at": utc_now(),
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "evaluation_format": FORMAT, "evaluation_version": VERSION, "model_revision": revision,
        "evaluation_conditions": conditions,
        "span": "0-based half-open; includes tokens with overlapping character offsets",
        "difference_direction": "frozen success minus failure",
        "selection_order": ["identical Original Token IDs", "absolute length delta",
                            "sum of absolute Original position deltas", "Token edit distance",
                            "candidate_id"],
        "inputs": {str(p.resolve()): sha256_file(p) for p in inputs},
    })
    (args.output / "report.md").write_text(
        "# AutoDAN FastChat比較データセット\n\n"
        f"固定成功候補 {len(anchors)} Originalの保存入力・応答・ラベルを照合した。"
        f"追加候補は取り込み {len(additions)} 件、形式不正 {len(rejected)} 件。"
        f"適格ペアは {len(pairs)} 組、代表ペアは {n} Original。\n\n"
        f"最低10 Originalまで {missing['10']}、20 Originalまで {missing['20']}、"
        f"30 Originalまで {missing['30']} の同条件の失敗記録が必要。"
        "ラベルは保存応答と拒否文字列規則で照合した。\n\n"
        "代表はOriginalのToken ID一致、絶対長さ差、位置差、Token編集距離、候補IDの順で選定。"
        "位置スパンは0始まりの半開区間。Originalに一部でも重なる境界Tokenを含む。"
        "長さと位置の差は固定成功側から失敗側を引いた値。"
        + (" 代表ペアがないため分布統計は欠損。" if not selected else "") +
        "\n\n[候補](candidates.jsonl) · [ペア](pairs.csv) · [Original別状況](original_status.csv) · "
        "[代表ペア統計](pair_statistics.json) · [検証](validation.json) · [来歴](manifest.json)\n",
        encoding="utf-8",
    )
    print(f"verified references={len(anchors)} eligible_pairs={len(pairs)} selected_originals={n}")


if __name__ == "__main__":
    main()
