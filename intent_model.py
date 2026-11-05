"""Shared data checks, rejection policy and local intent prediction."""
from pathlib import Path
import csv
import hashlib
import json
import re
import warnings

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "intent_v1"
ENCODER_DIR = ROOT / "models" / "sentence_encoder"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INTENTS = (
    "greeting", "goodbye", "ask_user_name", "ask_ghost_name",
    "change_user_name", "request_help", "ask_identity",
)
UNKNOWN = "out_of_scope"
LABELS = (*INTENTS, UNKNOWN)


def text_key(text):
    """For duplicate checks only; never use this to preprocess model input."""
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


def text_hash(text):
    return hashlib.sha256(text_key(text).encode("utf-8")).hexdigest()


def read_examples(path, *, training=False):
    """Reject malformed data and conflicts; deduplicate training only."""
    allowed = set(INTENTS if training else LABELS)
    examples, seen = [], {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "intent"]:
            raise ValueError(f"{path}: expected exactly the columns text,intent")
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            text, label = row["text"].strip(), row["intent"].strip()
            key = text_key(text)
            if not key or label not in allowed:
                raise ValueError(f"{path}:{line}: empty text or unsupported label {label!r}")
            if key in seen:
                if seen[key] != label:
                    raise ValueError(f"{path}:{line}: same text has conflicting labels")
                if training:
                    warnings.warn(f"{path}:{line}: skipping duplicate {text!r}")
                    continue
                raise ValueError(f"{path}:{line}: duplicate evaluation example")
            seen[key] = label
            examples.append((text, label))
    if not examples:
        raise ValueError(f"{path}: no examples")
    missing = allowed - {label for _, label in examples}
    if missing:
        raise ValueError(f"{path}: missing labels: {sorted(missing)}")
    return examples


def ensure_disjoint(examples, used_hashes):
    overlap = [text for text, _ in examples if text_hash(text) in set(used_hashes)]
    if overlap:
        raise ValueError(f"Dataset overlaps with earlier data: {overlap[:3]}")


def decisions(probabilities, classes, threshold, margin):
    """Abstain if the top score is low or the top two scores are too close."""
    p = np.asarray(probabilities)
    order = np.argsort(p, axis=1)
    rows = np.arange(len(p))
    top = order[:, -1]
    scores = p[rows, top]
    gaps = scores - p[rows, order[:, -2]]
    labels = np.asarray(classes, dtype=object)[top].copy()
    labels[(scores < threshold) | (gaps < margin)] = UNKNOWN
    return labels, scores, gaps


def select_thresholds(probabilities, classes, expected):
    """Tune on validation only, balancing supported accuracy and OOS recall."""
    expected = np.asarray(expected)
    known = expected != UNKNOWN
    if not known.any() or known.all():
        raise ValueError("Validation needs supported and out_of_scope examples")
    best = None
    # Fixed, modest search; these scores are not calibrated probabilities.
    for threshold in np.arange(0.20, 0.951, 0.05):
        for margin in (0.0, 0.10, 0.20):
            predicted, _, _ = decisions(probabilities, classes, threshold, margin)
            score = (float(np.mean(predicted[known] == expected[known]))
                     + float(np.mean(predicted[~known] == UNKNOWN))) / 2
            candidate = (score, float(threshold), margin)
            if best is None or candidate > best:
                best = candidate
    score, threshold, margin = best
    return {"threshold": round(threshold, 2), "margin": margin,
            "validation_objective": score}


def evaluation_report(examples, probabilities, classes, policy):
    expected = np.array([label for _, label in examples])
    predicted, scores, gaps = decisions(
        probabilities, classes, policy["threshold"], policy["margin"])
    known = expected != UNKNOWN
    accepted = predicted != UNKNOWN
    return {
        "total": len(examples),
        "accuracy": float(np.mean(expected == predicted)),
        "supported_correct_rate": float(np.mean(predicted[known] == expected[known])),
        "out_of_scope_recall": float(np.mean(predicted[~known] == UNKNOWN)),
        "supported_rejection_rate": float(np.mean(~accepted[known])),
        "accepted_accuracy": float(np.mean(expected[accepted] == predicted[accepted]))
        if accepted.any() else None,
        "classification_report": classification_report(
            expected, predicted, labels=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix_label_order": list(LABELS),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=list(LABELS)).tolist(),
        "predictions": [
            {"text": text, "expected": label, "predicted": str(prediction),
             "top_score": float(score), "margin": float(gap), "correct": label == prediction}
            for (text, label), prediction, score, gap in zip(examples, predicted, scores, gaps)
        ],
    }


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_summary(report):
    for key in ("accuracy", "supported_correct_rate", "out_of_scope_recall",
                "supported_rejection_rate", "accepted_accuracy"):
        value = report[key]
        print(f"{key}: {value:.1%}" if value is not None else f"{key}: n/a")
    print(f"macro_f1: {report['classification_report']['macro avg']['f1-score']:.3f}")


def load_encoder(path=ENCODER_DIR):
    from sentence_transformers import SentenceTransformer
    if not Path(path).is_dir():
        raise FileNotFoundError("Local sentence encoder missing. Run train_intents.py first.")
    return SentenceTransformer(str(path), device="cpu", local_files_only=True,
                               trust_remote_code=False)


def encode(encoder, texts):
    texts = list(texts)
    for text in texts:
        if len(encoder.tokenizer.encode(text, add_special_tokens=True)) > encoder.max_seq_length:
            raise ValueError("A dataset example exceeds the encoder's token limit. Shorten or split it before using it.")
    return encoder.encode(texts, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)


class IntentRecognizer:
    def __init__(self, model_dir=MODEL_DIR, encoder_dir=ENCODER_DIR):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        # Only load the artifact produced by your own training script.
        self.classifier = joblib.load(model_dir / "classifier.joblib")
        self.encoder = load_encoder(encoder_dir)

    def probabilities(self, texts):
        return self.classifier.predict_proba(encode(self.encoder, texts))

    def predict(self, text):
        text = text.strip()
        if not text:
            return {"intent": UNKNOWN, "reason": "empty_input", "top_score": None}
        # Avoid silently interpreting only the beginning of a long message.
        token_count = len(self.encoder.tokenizer.encode(text, add_special_tokens=True))
        if token_count > self.encoder.max_seq_length:
            return {"intent": UNKNOWN, "reason": "message_too_long", "top_score": None}
        p = self.probabilities([text])
        policy = self.metadata["policy"]
        labels, scores, gaps = decisions(p, self.classifier.classes_,
                                        policy["threshold"], policy["margin"])
        return {"intent": str(labels[0]),
                "candidate": str(self.classifier.classes_[p[0].argmax()]),
                "top_score": round(float(scores[0]), 4),
                "margin": round(float(gaps[0]), 4),
                "reason": "uncertain" if labels[0] == UNKNOWN else "accepted"}
