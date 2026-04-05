# src/predict.py
"""
Command-line inference using the saved best pipeline.

Usage (from project root):
    python -m src.predict "Free entry! Win a £1000 prize now"
    python -m src.predict "Hey, are we meeting tomorrow?" --json
    python src/predict.py "Some message here"

The script loads models/best_pipeline.pkl and prints the predicted label.

Score reporting:
- If the winning classifier supports predict_proba() (LogisticRegression,
  MultinomialNB), the model-estimated spam probability is reported.
- If it only exposes decision_function() (LinearSVC), the raw decision score
  is reported and labelled as such.  No sigmoid transformation is applied.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BEST_PIPELINE_PATH, METRICS_PATH


def load_pipeline(path: Path = BEST_PIPELINE_PATH):
    """Load the saved sklearn pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"No saved pipeline found at {path}.\n"
            "Please run `python -m src.train` first."
        )
    return joblib.load(path)


def _get_score(pipeline, text: str) -> Dict[str, Any]:
    """
    Return {'score': float|None, 'score_type': str}.

    - predict_proba       → model-estimated P(spam)  score_type = 'probability'
    - decision_function   → raw SVM margin           score_type = 'decision_score'
    - neither             → None                     score_type = 'not_available'
    """
    clf = pipeline.named_steps.get("clf")

    if hasattr(clf, "predict_proba"):
        proba     = pipeline.predict_proba([text])[0]
        spam_prob = float(proba[1])
        return {"score": round(spam_prob, 6), "score_type": "probability"}

    if hasattr(clf, "decision_function"):
        raw = pipeline.decision_function([text])[0]
        return {"score": round(float(raw), 6), "score_type": "decision_score"}

    return {"score": None, "score_type": "not_available"}


def predict(text: str, pipeline=None) -> Dict[str, Any]:
    """
    Classify a single raw SMS string.

    Parameters
    ----------
    text     : raw SMS text (preprocessing is done inside the pipeline)
    pipeline : loaded sklearn Pipeline; loads from disk if None

    Returns
    -------
    dict with keys:
        text, predicted_label, predicted_label_index, score, score_type, model_name
    """
    text = text.strip()
    if not text:
        raise ValueError("Input text is empty.")

    if pipeline is None:
        pipeline = load_pipeline()

    pred_idx   = int(pipeline.predict([text])[0])
    pred_label = "spam" if pred_idx == 1 else "ham"
    score_info = _get_score(pipeline, text)

    model_name = "unknown"
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, encoding="utf-8") as fh:
                meta = json.load(fh)
            model_name = meta.get("selected_pipeline_name", "unknown")
        except Exception:
            pass

    return {
        "text":                  text,
        "predicted_label":       pred_label,
        "predicted_label_index": pred_idx,
        "score":                 score_info["score"],
        "score_type":            score_info["score_type"],
        "model_name":            model_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict spam / ham for an SMS message.")
    parser.add_argument("text", nargs="+", help="SMS message text (wrap in quotes)")
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Output result as a JSON object instead of formatted text",
    )
    args = parser.parse_args()

    message = " ".join(args.text)
    result  = predict(message)

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\nMessage : {result['text']}")
        print(f"Label   : {result['predicted_label'].upper()}")
        print(f"Score   : {result['score']}  ({result['score_type']})")
        print(f"Model   : {result['model_name']}\n")


if __name__ == "__main__":
    main()
