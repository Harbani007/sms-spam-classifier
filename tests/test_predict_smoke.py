"""
Smoke tests for src/predict.py

Uses a tiny in-memory pipeline so the tests run without a trained model on disk.
Covers:
- Basic spam / ham prediction
- Empty input raises ValueError
- Whitespace-only input raises ValueError
- --json CLI output produces valid JSON
- predict_proba and decision_function score paths
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import predict, _get_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_pipeline(pred_idx: int, proba=None, decision=None):
    """Build a minimal mock sklearn pipeline for testing."""
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([pred_idx])

    clf = MagicMock()
    pipeline.named_steps = {"clf": clf}

    if proba is not None:
        clf.predict_proba = MagicMock()
        pipeline.predict_proba.return_value = np.array([proba])
        del clf.decision_function
    elif decision is not None:
        del clf.predict_proba
        clf.decision_function = MagicMock()
        pipeline.decision_function.return_value = np.array([decision])
    else:
        del clf.predict_proba
        del clf.decision_function

    return pipeline


# ---------------------------------------------------------------------------
# Core predict() tests
# ---------------------------------------------------------------------------

def test_predict_spam():
    pipeline = _make_mock_pipeline(pred_idx=1, proba=[0.05, 0.95])
    result = predict("Free prize win now", pipeline=pipeline)
    assert result["predicted_label"] == "spam"
    assert result["predicted_label_index"] == 1


def test_predict_ham():
    pipeline = _make_mock_pipeline(pred_idx=0, proba=[0.98, 0.02])
    result = predict("Are you coming to the meeting?", pipeline=pipeline)
    assert result["predicted_label"] == "ham"
    assert result["predicted_label_index"] == 0


def test_predict_returns_required_keys():
    pipeline = _make_mock_pipeline(pred_idx=0, proba=[0.9, 0.1])
    result = predict("hello", pipeline=pipeline)
    for key in ("text", "predicted_label", "predicted_label_index",
                "score", "score_type", "model_name"):
        assert key in result


def test_empty_input_raises():
    pipeline = _make_mock_pipeline(pred_idx=0)
    with pytest.raises(ValueError, match="empty"):
        predict("", pipeline=pipeline)


def test_whitespace_only_raises():
    pipeline = _make_mock_pipeline(pred_idx=0)
    with pytest.raises(ValueError, match="empty"):
        predict("   ", pipeline=pipeline)


# ---------------------------------------------------------------------------
# _get_score() tests
# ---------------------------------------------------------------------------

def test_score_probability_path():
    pipeline = _make_mock_pipeline(pred_idx=1, proba=[0.1, 0.9])
    info = _get_score(pipeline, "win prize")
    assert info["score_type"] == "probability"
    assert 0.0 <= info["score"] <= 1.0


def test_score_decision_path():
    pipeline = _make_mock_pipeline(pred_idx=1, decision=1.42)
    info = _get_score(pipeline, "win prize")
    assert info["score_type"] == "decision_score"
    assert isinstance(info["score"], float)


def test_score_not_available():
    pipeline = _make_mock_pipeline(pred_idx=1, proba=None, decision=None)
    info = _get_score(pipeline, "test")
    assert info["score_type"] == "not_available"
    assert info["score"] is None


# ---------------------------------------------------------------------------
# CLI --json mode
# ---------------------------------------------------------------------------

def test_cli_json_output(capsys):
    """Running main() with --json should produce valid JSON on stdout."""
    pipeline = _make_mock_pipeline(pred_idx=1, proba=[0.05, 0.95])

    with patch("src.predict.load_pipeline", return_value=pipeline):
        import src.predict as predict_module
        with patch.object(
            predict_module, "predict",
            return_value={
                "text": "Win prize",
                "predicted_label": "spam",
                "predicted_label_index": 1,
                "score": 0.95,
                "score_type": "probability",
                "model_name": "LR_BoW",
            },
        ):
            sys.argv = ["predict", "Win prize", "--json"]
            predict_module.main()

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["predicted_label"] == "spam"
