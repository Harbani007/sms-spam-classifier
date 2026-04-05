"""
Tests for src/train.py :: load_dataset()

Covers:
- Standard CSV with v1/v2 header (UCI default)
- Standard CSV with label/text header
- Headerless two-column TSV (raw UCI format)
- Validation: unknown label values raise ValueError
- Validation: missing file raises FileNotFoundError
- Deduplication and null-dropping
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import load_dataset, _validate_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_path: Path, content: str, filename: str = "spam.csv") -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Format variants
# ---------------------------------------------------------------------------

def test_load_v1v2_header(tmp_path):
    """UCI default CSV format: v1=label, v2=text."""
    content = "v1,v2\nham,Hello there\nspam,Win a free prize now\n"
    path = _write_csv(tmp_path, content)
    df = load_dataset(path)
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 2
    assert df.loc[df["text"] == "Hello there", "label"].iloc[0] == 0
    assert df.loc[df["text"] == "Win a free prize now", "label"].iloc[0] == 1


def test_load_label_text_header(tmp_path):
    """label/text header variant."""
    content = "label,text\nham,How are you?\nspam,Claim your reward\n"
    path = _write_csv(tmp_path, content)
    df = load_dataset(path)
    assert len(df) == 2
    assert df["label"].tolist() == [0, 1]


def test_load_headerless_tsv(tmp_path):
    """Headerless two-column TSV — the most common raw UCI format."""
    content = "ham\tGo until jurong point\nspam\tFREE entry in 2 a wkly comp\n"
    path = _write_csv(tmp_path, content, "spam.csv")
    df = load_dataset(path)
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 2
    assert 0 in df["label"].values
    assert 1 in df["label"].values


def test_load_binary_labels(tmp_path):
    """Numeric 0/1 labels should be accepted."""
    content = "label,text\n0,This is ham\n1,This is spam\n"
    path = _write_csv(tmp_path, content)
    df = load_dataset(path)
    assert df["label"].tolist() == [0, 1]


def test_deduplication(tmp_path):
    """Duplicate message texts should be removed."""
    content = "label,text\nham,Hello\nham,Hello\nspam,Win prize\n"
    path = _write_csv(tmp_path, content)
    df = load_dataset(path)
    assert len(df) == 2


def test_null_rows_dropped(tmp_path):
    """Rows with null text or label should be dropped."""
    content = "label,text\nham,Valid message\n,Missing label\nspam,\n"
    path = _write_csv(tmp_path, content)
    df = load_dataset(path)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nonexistent.csv")


def test_invalid_label_raises(tmp_path):
    """Unknown label values should raise ValueError via _validate_labels."""
    s = pd.Series(["ham", "spam", "unknown"])
    with pytest.raises(ValueError, match="Unexpected label values"):
        _validate_labels(s)


def test_valid_labels_pass():
    """Known label variants should not raise."""
    for val in ["spam", "ham", "0", "1"]:
        _validate_labels(pd.Series([val]))   # should not raise
