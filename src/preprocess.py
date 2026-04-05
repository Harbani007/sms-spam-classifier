# src/preprocess.py
"""
Custom sklearn-compatible text preprocessor for SMS spam detection.

Design decisions:
- Subclasses BaseEstimator + TransformerMixin so it slots into any Pipeline.
- fit() is a no-op (stateless transform); included for API compatibility.
- Stopwords come from sklearn's built-in ENGLISH_STOP_WORDS — no nltk corpus
  download needed at runtime.
- Stemming uses NLTK's PorterStemmer, which is purely algorithmic and requires
  no corpus download.
- Digits are preserved: amounts like "1000" and shortcodes like "80800" are
  genuine spam signals in SMS data.
- All regex operations are compiled once at import time for speed.
"""

import re
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer

# ---------------------------------------------------------------------------
# Compile patterns once — cheaper than re-compiling per document
# ---------------------------------------------------------------------------
_URL_PATTERN        = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_PATTERN      = re.compile(r"\S+@\S+\.\S+", re.IGNORECASE)
_NON_ALNUM_PATTERN  = re.compile(r"[^a-zA-Z0-9\s]")   # keeps letters, digits, spaces
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Build a plain set for O(1) lookup
_STOP_WORDS: set = set(ENGLISH_STOP_WORDS)

# Single shared stemmer instance
_stemmer = PorterStemmer()


def _clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to a single string.

    Steps (in order):
    1. Lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove non-alphanumeric characters (keeps spaces and digits)
    5. Collapse repeated whitespace / strip
    6. Tokenise on whitespace
    7. Remove stopwords and single-character tokens
    8. Apply Porter stemming
    9. Re-join tokens into a string
    """
    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _NON_ALNUM_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    tokens = [
        _stemmer.stem(word)
        for word in text.split()
        if word not in _STOP_WORDS and len(word) > 1
    ]

    return " ".join(tokens)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that cleans raw SMS text.

    Usage inside a Pipeline:
        Pipeline([
            ("pre", TextPreprocessor()),
            ("vec", TfidfVectorizer()),
            ("clf", LogisticRegression()),
        ])

    After fitting the pipeline, pipeline.predict(["raw text"]) triggers the
    full cleaning chain automatically — no manual preprocessing step needed.
    """

    def fit(self, X, y=None) -> "TextPreprocessor":
        """No-op; returns self for sklearn compatibility."""
        return self

    def transform(self, X) -> List[str]:
        """
        Parameters
        ----------
        X : iterable of str
            Raw SMS message strings.

        Returns
        -------
        list of str
            Cleaned, stemmed, stopword-filtered strings.
        """
        return [_clean_text(str(doc)) for doc in X]

    def get_feature_names_out(self, input_features=None):
        """Not applicable for a text transformer; returns empty array."""
        return []
