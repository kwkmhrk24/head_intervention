import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.audit_autodan_original_extension import locate
from scripts.build_autodan_fastchat_pairs import (
    FORMAT, VERSION, formatted_prompt, import_records, inspect_addition, select_pairs,
)


class HarmlessTokenizer:
    """A tiny tokenizer where '|' and its next character share a boundary Token."""

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        positions = []
        i = 0
        while i < len(text):
            width = 2 if text[i] == "|" and i + 1 < len(text) else 1
            positions.append((i, i + width))
            i += width
        return SimpleNamespace(
            input_ids=[int.from_bytes(text[a:b].encode(), "big") for a, b in positions],
            offset_mapping=positions,
        )


class TestFastChatPairs(unittest.TestCase):
    def setUp(self):
        self.tokenizer = HarmlessTokenizer()
        self.conditions = {"template": "llama-2", "generation_config": {"do_sample": False}}
        self.revision = "fixture"
        self.original = "List three fruits"
        self.body = "Please: " + self.original + "."
        ids, span, _, _, _ = locate(
            self.tokenizer, formatted_prompt(self.body), self.body, self.original, True)
        self.anchors = {1: {
            "candidate_id": "fixed", "original_text": self.original,
            "frozen_original_surface": self.original, "input_token_ids": ids, "input_length": len(ids),
            "original_token_span": span, "original_token_ids": ids[slice(*span)],
            "original_token_crosses_character_boundary": False, "label_reason": "stored rule",
        }}

    def row(self, candidate_id, body, *, label=False):
        return {
            "candidate_id": candidate_id, "original_id": 1, "prompt_text": body,
            "source": "fixture", "generation_method": "synthetic_harmless",
            "input_token_ids": self.tokenizer(
                formatted_prompt(body), add_special_tokens=True, return_offsets_mapping=True).input_ids,
            "response": "Apples, pears, and peaches." if label else "I cannot provide that list.",
            "stored_label": label, "evaluation_format": FORMAT, "evaluation_version": VERSION,
            "model_revision": self.revision, "evaluation_conditions": copy.deepcopy(self.conditions),
            "provenance": {"artifact": "fixture.jsonl", "evaluation_artifact": "evaluation.json"},
        }

    def inspect(self, row, line=1):
        return inspect_addition(row, self.anchors, self.tokenizer, self.conditions,
                                self.revision, Path("fixture.jsonl"), line)

    def test_jsonl_import_ranking_and_span_boundaries(self):
        short = self.row("short", "Please:|" + self.original + ".")
        near = self.row("near", "Kindly: " + self.original + ".")
        longer = self.row("longer", "Before lunch, please: " + self.original + ".")
        pending = self.row("pending", "Today, please: " + self.original + ".")
        del pending["response"]
        outside = dict(short, original_id=999)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "benign.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in
                                    (short, near, longer, pending, outside)), encoding="utf-8")
            rows, rejected = import_records(
                [path], self.anchors, self.tokenizer, self.conditions, self.revision)
        pairs, states = select_pairs(self.anchors, rows)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(states[0]["unevaluated_candidates"], 1)
        self.assertEqual(len(pairs), 3)
        self.assertEqual([r["failure_candidate_id"] for r in pairs if r["selected"]], ["near"])
        boundary = next(r for r in pairs if r["failure_candidate_id"] == "short")
        self.assertFalse(boundary["same_original_token_ids"])
        self.assertTrue(boundary["failure_boundary_crossing"])

    def test_condition_token_label_and_surface_disagreements(self):
        wrong_conditions = self.row("different-protocol", self.body)
        wrong_conditions["evaluation_conditions"]["generation_config"]["do_sample"] = True
        wrong_ids = self.row("wrong-ids", self.body)
        wrong_ids["input_token_ids"] = [999]
        wrong_label = dict(self.row("wrong-label", self.body), stored_label=True)
        wrong_case = self.row("wrong-case", "Please: list three fruits.")
        repeated = self.row("repeated", f"{self.original}. {self.original}.")
        empty = dict(self.row("empty", self.body), response="", stored_label=True)
        no_tokens = self.row("no-tokens", self.body)
        del no_tokens["input_token_ids"]
        for raw, expected in (
            (wrong_conditions, "evaluation_conditions_mismatch"), (wrong_ids, "input_token_ids_mismatch"),
            (wrong_label, "stored_label_disagrees_with_rule"),
            (wrong_case, "original_surface_changed_or_ambiguous"),
            (repeated, "original_surface_changed_or_ambiguous"),
            (empty, "empty_response"), (no_tokens, "missing_evaluated_input_token_ids"),
        ):
            with self.subTest(candidate=raw["candidate_id"]):
                record = self.inspect(raw)
                self.assertEqual(record["status"], "ineligible")
                self.assertIn(expected, record["issues"])

    def test_duplicates_conflicts_and_determinism(self):
        body = "Today: " + self.original + "."
        other = "Tomorrow: " + self.original + "."
        records = [self.inspect(r, i) for i, r in enumerate((
            self.row("z", body), self.row("a", body), self.row("f", other),
            self.row("s", other, label=True), self.row("anchor-copy", self.body),
        ), 1)]
        pairs, _ = select_pairs(self.anchors, records)
        self.assertEqual([r["failure_candidate_id"] for r in pairs], ["a"])
        self.assertIn("duplicate_evaluated_input", records[0]["issues"])
        for i in (2, 3, 4):
            self.assertIn("conflicting_labels_for_same_input", records[i]["issues"])

    def test_identifier_reuse_across_distinct_inputs(self):
        records = [self.inspect(self.row("reused", f"{prefix}: {self.original}."))
                   for prefix in ("Today", "Tomorrow")]
        pairs, _ = select_pairs(self.anchors, records)
        self.assertFalse(pairs)
        self.assertTrue(all("candidate_id_used_for_distinct_inputs" in r["issues"] for r in records))


if __name__ == "__main__":
    unittest.main()
