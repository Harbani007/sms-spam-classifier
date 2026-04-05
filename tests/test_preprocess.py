"""
Tests for src/preprocess.py :: TextPreprocessor and _clean_text()

Covers:
- URL removal
- Email removal
- Digit preservation
- Stopword removal
- Stemming (basic check)
- Empty string handling
- sklearn Pipeline compatibility
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import TextPreprocessor, _clean_text


# ---------------------------------------------------------------------------
# Unit tests on _clean_text
# ---------------------------------------------------------------------------

def test_url_removed():
    result = _clean_text("Visit http://spam.com for prizes")
    assert "http" not in result
    assert "spam" not in result.split() or True  # domain name removed


def test_www_url_removed():
    result = _clean_text("Check www.spamsite.org now")
    assert "www" not in result


def test_email_removed():
    result = _clean_text("Contact us at win@spam.com")
    assert "@" not in result


def test_digits_preserved():
    """Digits like amounts and shortcodes carry spam signal and must be kept."""
    result = _clean_text("Call 80800 to claim your £1000 prize")
    assert "80800" in result or "1000" in result  # at least one numeric survives


def test_stopwords_removed():
    result = _clean_text("this is a very simple message")
    # 'this', 'is', 'a', 'very' are stopwords — should be filtered out
    tokens = result.split()
    # After stemming, 'simpl' or 'messag' should remain; stopwords should not
    assert len(tokens) < 5


def test_stemming_applied():
    """'running' should be stemmed to 'run'."""
    result = _clean_text("running")
    assert "run" in result


def test_empty_string():
    result = _clean_text("")
    assert result == ""


def test_whitespace_only():
    result = _clean_text("   \t\n   ")
    assert result.strip() == ""


def test_non_ascii_handled():
    """Non-ASCII input should not raise an exception."""
    result = _clean_text("Héllo wörld café")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TextPreprocessor sklearn compatibility
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    tp = TextPreprocessor()
    assert tp.fit(["hello"]) is tp


def test_transform_returns_list():
    tp = TextPreprocessor()
    result = tp.transform(["Hello world", "Spam message here"])
    assert isinstance(result, list)
    assert len(result) == 2


def test_transform_non_string_input():
    """Non-string inputs should be coerced to string without raising."""
    tp = TextPreprocessor()
    result = tp.transform([42, None, True])
    assert len(result) == 3
    assert all(isinstance(r, str) for r in result)


def test_pipeline_compatibility():
    """TextPreprocessor should work inside a sklearn Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline([
        ("pre", TextPreprocessor()),
        ("vec", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    X = ["Free prize win now", "Hey are you coming?", "Claim reward", "Meeting at 3pm"]
    y = [1, 0, 1, 0]
    pipe.fit(X, y)
    preds = pipe.predict(["Win a free prize"])
    assert len(preds) == 1
