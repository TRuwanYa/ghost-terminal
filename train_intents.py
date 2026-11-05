"""Fit the classifier on train; choose rejection thresholds on validation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform

import joblib
from sklearn.linear_model import LogisticRegression

from intent_model import (ROOT, MODEL_DIR, ENCODER_DIR, ENCODER_NAME, read_examples,
    text_hash, ensure_disjoint, encode, select_thresholds, evaluation_report,
    save_json, print_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--check-data", action="store_true", help="Check train and validation without downloading or training")
    args = parser.parse_args()
    train = read_examples(args.data_dir / "intents_train.csv", training=True)
    validation = read_examples(args.data_dir / "intents_validation.csv")
    train_hashes = [text_hash(text) for text, _ in train]
    ensure_disjoint(validation, train_hashes)
    print("Training examples used:", len(train), dict(Counter(label for _, label in train)))
    print("Validation examples:", len(validation))
    print("The test CSV is not read by this script.")
    if args.check_data:
        return
    if args.model_dir.exists():
        raise ValueError("Model directory already exists. Choose a new --model-dir for another experiment.")

    from sentence_transformers import SentenceTransformer
    if ENCODER_DIR.exists():
        encoder = SentenceTransformer(str(ENCODER_DIR), device="cpu", local_files_only=True,
                                      trust_remote_code=False)
    else:
        print("Downloading the public sentence encoder once; inference runs on this computer.")
        encoder = SentenceTransformer(ENCODER_NAME, device="cpu", trust_remote_code=False)
        encoder.save_pretrained(str(ENCODER_DIR))

    # The pretrained encoder stays frozen. Only logistic regression is fitted.
    classifier = LogisticRegression(C=10.0, max_iter=2000, random_state=42)
    classifier.fit(encode(encoder, [text for text, _ in train]), [label for _, label in train])
    probabilities = classifier.predict_proba(encode(encoder, [text for text, _ in validation]))
    policy = select_thresholds(probabilities, classifier.classes_, [label for _, label in validation])
    report = evaluation_report(validation, probabilities, classifier.classes_, policy)
    args.model_dir.mkdir(parents=True)
    joblib.dump(classifier, args.model_dir / "classifier.joblib")
    save_json(args.model_dir / "metadata.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "encoder": ENCODER_NAME,
        "policy": policy, "classes": classifier.classes_.tolist(),
        "training_rows_used": len(train), "validation_rows": len(validation),
        "train_hashes": train_hashes,
        "validation_hashes": [text_hash(text) for text, _ in validation],
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in
            ["sentence-transformers", "transformers", "torch", "scikit-learn", "numpy", "joblib"]},
    })
    save_json(args.model_dir / "validation_report.json", report)
    print("\nChosen rejection policy:", policy)
    print_summary(report)
    print("Saved:", args.model_dir.resolve())
    print("Review validation_report.json before the final test evaluation.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit(f"Training stopped: {error}")
