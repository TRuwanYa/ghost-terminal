"""Try intent prediction without changing Ghost's names or memory."""
import argparse
import json
from pathlib import Path
from intent_model import MODEL_DIR, IntentRecognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Omit to enter an interactive prediction loop")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    recognizer = IntentRecognizer(args.model_dir)
    if args.text is not None:
        print(json.dumps(recognizer.predict(args.text), indent=2))
        return
    print("Intent recognizer only. Type /exit to leave. No chat replies are generated.")
    while True:
        try:
            text = input("Text: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.strip().casefold() == "/exit":
            return
        print(json.dumps(recognizer.predict(text), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit(f"Prediction stopped: {error}")
