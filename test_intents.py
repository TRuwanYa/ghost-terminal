"""Policy/data tests use synthetic scores; they do not measure NLP accuracy."""
import csv
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import Mock

import numpy as np

from intent_model import (INTENTS, UNKNOWN, read_examples, text_hash, ensure_disjoint,
    decisions, select_thresholds, evaluation_report, IntentRecognizer)


class IntentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "examples.csv"

    # Helper to write an example for each intent in the list of 7 plus any  extra rows that test provides
    def write_rows(self, extra=()):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text", "intent"])
            writer.writerows([(f"example {label}", label) for label in INTENTS])
            writer.writerows(extra)

    def test_duplicates_removed_only_from_training(self):
        self.write_rows([("EXAMPLE greeting!", "greeting")])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(len(read_examples(self.path, training=True)), 7)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_examples(self.path)

    def test_conflicting_labels_rejected(self):
        self.write_rows([("example greeting", "goodbye")])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            read_examples(self.path, training=True)

    def test_out_of_scope_cannot_be_training_label(self):
        self.write_rows([("weather please", UNKNOWN)])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            read_examples(self.path, training=True)

    def test_cross_split_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            ensure_disjoint([("HELLO!", "greeting")], [text_hash("hello")])

    def test_low_score_and_small_margin_both_reject(self):
        labels, _, _ = decisions([[.8, .2], [.55, .45], [.51, .49]],
                                 ["greeting", "goodbye"], .52, .15)
        self.assertEqual(labels.tolist(), ["greeting", UNKNOWN, UNKNOWN])

    def test_threshold_selection_uses_validation(self):
        p = np.array([[.90, .10], [.40, .60], [.51, .49], [.52, .48]])
        truth = ["greeting", "goodbye", UNKNOWN, UNKNOWN]
        policy = select_thresholds(p, ["greeting", "goodbye"], truth)
        predicted, _, _ = decisions(p, ["greeting", "goodbye"],
                                    policy["threshold"], policy["margin"])
        self.assertEqual(predicted.tolist(), truth)

    def test_metrics_distinguish_rejection_from_correct_classification(self):
        examples = [("a", "greeting"), ("b", "goodbye"), ("c", UNKNOWN)]
        report = evaluation_report(examples, [[.9, .1], [.55, .45], [.52, .48]],
                                   ["greeting", "goodbye"], {"threshold": .6, "margin": 0})
        self.assertEqual(report["supported_correct_rate"], .5)
        self.assertEqual(report["supported_rejection_rate"], .5)
        self.assertEqual(report["out_of_scope_recall"], 1)

    def test_blank_or_long_input_never_calls_classifier(self):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        recognizer.probabilities = Mock(side_effect=AssertionError("Should not classify"))
        recognizer.encoder = Mock(max_seq_length=3)
        recognizer.encoder.tokenizer.encode.return_value = [1, 2, 3, 4]
        self.assertEqual(recognizer.predict("   ")["reason"], "empty_input")
        self.assertEqual(recognizer.predict("a long message")["reason"], "message_too_long")


if __name__ == "__main__":
    unittest.main()
