"""Evaluate the fixed model on held-out test data. No fitting or tuning."""
import argparse
from pathlib import Path

from intent_model import (ROOT, MODEL_DIR, IntentRecognizer, read_examples,
    ensure_disjoint, evaluation_report, save_json, print_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, default=ROOT / "data" / "intents_test.csv")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    output = args.model_dir / "test_report.json"
    if output.exists():
        raise ValueError("test_report.json already exists. Read that report; repeated tuning against this test makes it development data.")
    examples = read_examples(args.test_file)
    recognizer = IntentRecognizer(args.model_dir)
    metadata = recognizer.metadata
    ensure_disjoint(examples, metadata["train_hashes"] + metadata["validation_hashes"])
    report = evaluation_report(examples, recognizer.probabilities([text for text, _ in examples]),
                               recognizer.classifier.classes_, metadata["policy"])
    save_json(output, report)
    print_summary(report)
    print("Saved:", output.resolve())
    print("This small drafted dataset is an initial check, not a real-user performance guarantee.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit(f"Evaluation stopped: {error}")
