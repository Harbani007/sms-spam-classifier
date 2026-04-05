# src/train.py
"""
Training script for SMS spam detection.

What this script does:
1. Loads and normalises the dataset from data/spam.csv
2. Performs a stratified 80/20 train/test split
3. Builds 6 sklearn pipelines (3 classifiers × 2 vectorisers)
4. Runs stratified 5-fold CV on the training split only
5. Selects the pipeline with the highest mean CV F1 score
6. Fits the best pipeline on the full training split
7. Evaluates on the holdout test split
8. Saves the best pipeline and all result artefacts

Run:
    python -m src.train          (from project root)
    python src/train.py          (from project root)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BEST_PIPELINE_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_IMG,
    COMPARISON_PLOT_IMG,
    CV_FOLDS,
    CV_RESULTS_CSV,
    DATASET_PATH,
    ERROR_ANALYSIS_PATH,
    LABEL_NAMES,
    LR_CLASS_WEIGHT,
    LR_MAX_ITER,
    LR_SOLVER,
    MAX_ERROR_EXAMPLES,
    MAX_FEATURES,
    METRICS_PATH,
    MODEL_COMPARISON_CSV,
    MODELS_DIR,
    MNB_ALPHA,
    N_JOBS,
    NGRAM_RANGE,
    RANDOM_STATE,
    RESULTS_DIR,
    SVC_CLASS_WEIGHT,
    SVC_MAX_ITER,
    TEST_SIZE,
    TOP_N_FEATURES,
    TOP_SPAM_FEATURES_CSV,
    TOP_HAM_FEATURES_CSV,
)
from src.preprocess import TextPreprocessor
from src.evaluate import save_confusion_matrix, save_comparison_plot


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_LABEL_VARIANTS = {
    "spam": 1, "1": 1,
    "ham":  0, "0": 0,
}

# Known (text_col, label_col) pairs across common dataset formats
_COL_CANDIDATES = [
    ("v2", "v1"),            # UCI default with header
    ("text", "label"),
    ("message", "label"),
    ("message", "category"),
    ("sms", "label"),
]


def _identify_columns(df: pd.DataFrame):
    """Return (text_col, label_col) from a DataFrame, or raise ValueError."""
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    for tc, lc in _COL_CANDIDATES:
        if tc in cols and lc in cols:
            return tc, lc

    # Fallback: two-column file — assume first=label, second=text
    if len(cols) >= 2:
        return cols[1], cols[0]

    raise ValueError(
        f"Cannot identify label/text columns. Got columns: {list(df.columns)}"
    )


def _validate_labels(series: pd.Series) -> None:
    """Raise ValueError if any label value is not in the expected set."""
    normalised = series.astype(str).str.strip().str.lower()
    unexpected = set(normalised.unique()) - set(_LABEL_VARIANTS.keys())
    if unexpected:
        raise ValueError(
            f"Unexpected label values: {unexpected}. "
            f"Expected one of: {set(_LABEL_VARIANTS.keys())}"
        )


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load the SMS spam dataset and return a DataFrame with exactly two columns:
    'text' (str) and 'label' (int: 0=ham, 1=spam).

    Handles the two most common formats:
      - File with a header row (UCI/Kaggle default: v1/v2, label/text, etc.)
      - Headerless two-column TSV (raw UCI format)

    Raises FileNotFoundError or ValueError for unrecognisable inputs.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Please download the SMS Spam Collection and place it at "
            f"{path}.\nSee data/README.md for instructions."
        )

    df = None

    # Known first-column names that unambiguously mark a headered file.
    # We check column NAMES (not values) to avoid swallowing the first
    # data row of a headerless file.
    _HEADER_LABEL_COL_NAMES = {"v1", "label", "category"}

    # ------------------------------------------------------------------ #
    # Pass 1: try each separator with a header row                        #
    # Accept ONLY when the first column name is a known header keyword.   #
    # ------------------------------------------------------------------ #
    for sep in (",", "\t"):
        try:
            candidate = pd.read_csv(
                path, sep=sep, encoding="ISO-8859-1", on_bad_lines="skip"
            )
            if candidate.shape[1] < 2:
                continue
            first_col_name = candidate.columns[0].strip().lower()
            if first_col_name in _HEADER_LABEL_COL_NAMES:
                df = candidate
                break
        except Exception:
            continue

    # ------------------------------------------------------------------ #
    # Pass 2: headerless TSV (first row is real data, not a header)       #
    # ------------------------------------------------------------------ #
    if df is None:
        for sep in ("\t", ","):
            try:
                candidate = pd.read_csv(
                    path,
                    sep=sep,
                    header=None,
                    encoding="ISO-8859-1",
                    on_bad_lines="skip",
                )
                if candidate.shape[1] >= 2:
                    # Explicitly rename: first column = label, second = text.
                    # Extra columns (unnamed extras from bad parses) are dropped.
                    candidate = candidate.iloc[:, :2].copy()
                    candidate.columns = ["label", "text"]
                    df = candidate
                    break
            except Exception:
                continue

    if df is None:
        raise ValueError(f"Could not parse dataset at {path}.")

    # Identify and rename columns
    text_col, label_col = _identify_columns(df)
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    # Drop nulls and duplicates
    df.dropna(subset=["text", "label"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["text"] = df["text"].astype(str)

    # Validate and map labels
    _validate_labels(df["label"])
    df["label"] = (
        df["label"].astype(str).str.strip().str.lower().map(_LABEL_VARIANTS)
    )

    n_spam = int(df["label"].sum())
    n_ham  = len(df) - n_spam
    print(f"[data] Loaded {len(df)} messages  (spam={n_spam}, ham={n_ham})")
    return df


# ---------------------------------------------------------------------------
# Pipeline definitions
# ---------------------------------------------------------------------------

def _make_pipeline(vec, clf) -> Pipeline:
    """Create a text classification pipeline with shared preprocessor."""
    return Pipeline([
        ("pre", TextPreprocessor()),
        ("vec", vec),
        ("clf", clf),
    ])


def build_pipelines() -> Dict[str, Pipeline]:
    """
    Return a dict of name → sklearn Pipeline for all 6 combinations.
    Each pipeline is fully end-to-end: raw text → prediction.
    """
    # Two vectorisers × three classifiers = six pipelines.
    # Each component is instantiated fresh inside _make_pipeline so that
    # sklearn's clone() works correctly during cross-validation.
    return {
        "LR_BoW": _make_pipeline(
            CountVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES, lowercase=False),
            LogisticRegression(solver=LR_SOLVER, max_iter=LR_MAX_ITER,
                               class_weight=LR_CLASS_WEIGHT, random_state=RANDOM_STATE),
        ),
        "LR_TF-IDF": _make_pipeline(
            TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES,
                            lowercase=False, sublinear_tf=True),
            LogisticRegression(solver=LR_SOLVER, max_iter=LR_MAX_ITER,
                               class_weight=LR_CLASS_WEIGHT, random_state=RANDOM_STATE),
        ),
        "LinearSVC_BoW": _make_pipeline(
            CountVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES, lowercase=False),
            LinearSVC(max_iter=SVC_MAX_ITER, class_weight=SVC_CLASS_WEIGHT,
                      random_state=RANDOM_STATE),
        ),
        "LinearSVC_TF-IDF": _make_pipeline(
            TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES,
                            lowercase=False, sublinear_tf=True),
            LinearSVC(max_iter=SVC_MAX_ITER, class_weight=SVC_CLASS_WEIGHT,
                      random_state=RANDOM_STATE),
        ),
        "MNB_BoW": _make_pipeline(
            CountVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES, lowercase=False),
            MultinomialNB(alpha=MNB_ALPHA),
        ),
        "MNB_TF-IDF": _make_pipeline(
            TfidfVectorizer(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES,
                            lowercase=False, sublinear_tf=True),
            MultinomialNB(alpha=MNB_ALPHA),
        ),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(
    pipelines: Dict[str, Pipeline],
    X_train: pd.Series,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Run stratified k-fold CV on every pipeline (training split only).

    Returns
    -------
    summary_df  : one row per pipeline (mean/std across folds)
    fold_df     : one row per (pipeline × fold) with raw fold scores
    best_name   : name of the best pipeline by mean CV F1
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["f1", "accuracy", "precision", "recall"]

    summary_records: List[dict] = []
    fold_records:    List[dict] = []

    for name, pipeline in pipelines.items():
        print(f"[cv]  Running {CV_FOLDS}-fold CV for {name} …", end=" ", flush=True)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=N_JOBS,
            return_train_score=False,
        )

        # Summary row
        record = {"pipeline_name": name}
        for metric in scoring:
            key = f"test_{metric}"
            record[f"cv_{metric}_mean"] = float(np.mean(scores[key]))
            record[f"cv_{metric}_std"]  = float(np.std(scores[key]))
        summary_records.append(record)
        print(f"F1={record['cv_f1_mean']:.4f} ± {record['cv_f1_std']:.4f}")

        # Per-fold rows
        for fold_idx in range(CV_FOLDS):
            fold_records.append({
                "pipeline_name": name,
                "fold":          fold_idx + 1,
                **{metric: float(scores[f"test_{metric}"][fold_idx])
                   for metric in scoring},
            })

    summary_df = (
        pd.DataFrame(summary_records)
        .sort_values("cv_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    fold_df = pd.DataFrame(fold_records)

    best_name = summary_df.iloc[0]["pipeline_name"]
    print(
        f"\n[cv]  Best pipeline by F1: {best_name}  "
        f"(mean CV F1 = {summary_df.iloc[0]['cv_f1_mean']:.4f})"
    )
    return summary_df, fold_df, best_name


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

def evaluate_best(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> Tuple[dict, np.ndarray]:
    """Compute and return classification metrics on the holdout test split."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "test_accuracy":  float(accuracy_score(y_test, y_pred)),
        "test_f1":        float(f1_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall":    float(recall_score(y_test, y_pred)),
    }

    report_str  = classification_report(y_test, y_pred, target_names=["ham", "spam"])
    report_dict = classification_report(
        y_test, y_pred, target_names=["ham", "spam"], output_dict=True
    )

    print("\n[holdout] Test-set metrics:")
    for k, v in metrics.items():
        print(f"          {k:<20s} = {v:.4f}")
    print()
    print(report_str)

    return metrics, y_pred, report_dict


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def build_error_analysis(
    texts: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    pipeline: Pipeline,
    n: int = MAX_ERROR_EXAMPLES,
) -> dict:
    """
    Return false-positive and false-negative examples, each enriched with
    a prediction score and sorted by highest model confidence in the wrong
    answer (most instructive mistakes first).
    """
    texts_arr  = texts.reset_index(drop=True)
    y_true_arr = y_true.reset_index(drop=True)
    y_pred_arr = pd.Series(y_pred)

    clf = pipeline.named_steps.get("clf")
    has_proba = hasattr(clf, "predict_proba")
    has_decision = hasattr(clf, "decision_function")

    def _score(idx_list):
        if not idx_list:
            return []
        batch = texts_arr[idx_list].tolist()
        if has_proba:
            probas = pipeline.predict_proba(batch)
            # confidence = probability assigned to the predicted (wrong) class
            scores     = [float(p[1]) for p in probas]
            score_type = "probability"
        elif has_decision:
            raw        = pipeline.decision_function(batch)
            scores     = [float(v) for v in raw]
            score_type = "decision_score"
        else:
            scores     = [None] * len(idx_list)
            score_type = "not_available"
        return scores, score_type

    fp_idx = y_pred_arr[(y_pred_arr == 1) & (y_true_arr == 0)].index.tolist()
    fn_idx = y_pred_arr[(y_pred_arr == 0) & (y_true_arr == 1)].index.tolist()

    def _build_list(idx_list):
        if not idx_list:
            return []
        scores_and_type = _score(idx_list)
        scores, score_type = scores_and_type
        items = [
            {
                "text":            str(texts_arr[i]),
                "true_label":      LABEL_NAMES[int(y_true_arr[i])],
                "predicted_label": LABEL_NAMES[int(y_pred_arr[i])],
                "score":           scores[k],
                "score_type":      score_type,
            }
            for k, i in enumerate(idx_list)
        ]
        # Sort by highest confidence in the wrong prediction
        items.sort(key=lambda x: x["score"] if x["score"] is not None else 0, reverse=True)
        return items[:n]

    return {
        "false_positives": _build_list(fp_idx),
        "false_negatives": _build_list(fn_idx),
    }


# ---------------------------------------------------------------------------
# Feature interpretability
# ---------------------------------------------------------------------------

def extract_feature_importance(
    pipeline: Pipeline,
    n: int = TOP_N_FEATURES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract top spam-indicating and ham-indicating tokens from a linear model.

    Works for LogisticRegression, LinearSVC (via coef_) and MultinomialNB
    (via log-probability ratio).  Returns (None, None) for unsupported models.

    Returns
    -------
    spam_df, ham_df : DataFrames with columns ['feature', 'score']
    """
    vec = pipeline.named_steps.get("vec")
    clf = pipeline.named_steps.get("clf")

    if vec is None or clf is None:
        return None, None

    feature_names = np.array(vec.get_feature_names_out())

    if hasattr(clf, "coef_"):
        # LogisticRegression or LinearSVC — binary coef_ is shape (1, n_features)
        coefs = clf.coef_.ravel()
        spam_idx = np.argsort(coefs)[::-1][:n]
        ham_idx  = np.argsort(coefs)[:n]

        spam_df = pd.DataFrame({
            "feature": feature_names[spam_idx],
            "score":   coefs[spam_idx].round(6),
            "note":    "coefficient (higher = stronger spam signal)",
        })
        ham_df = pd.DataFrame({
            "feature": feature_names[ham_idx],
            "score":   coefs[ham_idx].round(6),
            "note":    "coefficient (more negative = stronger ham signal)",
        })
        return spam_df, ham_df

    if hasattr(clf, "feature_log_prob_"):
        # MultinomialNB — rows are [ham, spam]
        log_ratio = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        spam_idx  = np.argsort(log_ratio)[::-1][:n]
        ham_idx   = np.argsort(log_ratio)[:n]

        spam_df = pd.DataFrame({
            "feature": feature_names[spam_idx],
            "score":   log_ratio[spam_idx].round(6),
            "note":    "log P(feat|spam)/P(feat|ham)",
        })
        ham_df = pd.DataFrame({
            "feature": feature_names[ham_idx],
            "score":   log_ratio[ham_idx].round(6),
            "note":    "log P(feat|spam)/P(feat|ham) — negative = ham-indicative",
        })
        return spam_df, ham_df

    return None, None


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------

def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_dataset()

    # 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    print(f"[split] train={len(X_train)}  test={len(X_test)}")

    # 3. Build all pipelines
    pipelines = build_pipelines()

    # 4. CV on training split only
    summary_df, fold_df, best_name = run_cv(pipelines, X_train, y_train)

    parts = best_name.split("_", 1)
    best_model_name = parts[0]
    best_vectorizer = parts[1] if len(parts) > 1 else "BoW"

    # 5. Fit best pipeline on full training split
    print(f"[train] Fitting best pipeline ({best_name}) on full training split …")
    best_pipeline = pipelines[best_name]
    best_pipeline.fit(X_train, y_train)

    # 6. Holdout evaluation
    holdout_metrics, y_pred, report_dict = evaluate_best(best_pipeline, X_test, y_test)

    # 7. Save best pipeline
    joblib.dump(best_pipeline, BEST_PIPELINE_PATH)
    print(f"[save] Pipeline saved to {BEST_PIPELINE_PATH}")

    # 8. Confusion matrix image
    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, best_name, CONFUSION_MATRIX_IMG)

    # 9. Comparison bar chart
    save_comparison_plot(summary_df, COMPARISON_PLOT_IMG)

    # 10. model_comparison.csv — one row per pipeline (summary)
    summary_df["classifier"] = summary_df["pipeline_name"].apply(
        lambda n: n.split("_", 1)[0]
    )
    summary_df["vectorizer"] = summary_df["pipeline_name"].apply(
        lambda n: n.split("_", 1)[1] if "_" in n else "BoW"
    )
    summary_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    print(f"[save] model_comparison.csv → {MODEL_COMPARISON_CSV}")

    # 11. cv_results.csv — one row per (pipeline × fold)
    fold_df.to_csv(CV_RESULTS_CSV, index=False)
    print(f"[save] cv_results.csv → {CV_RESULTS_CSV}")

    # 12. classification_report.json
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2)
    print(f"[save] classification_report.json → {CLASSIFICATION_REPORT_PATH}")

    # 13. error_analysis.json
    error_data = build_error_analysis(X_test, y_test, y_pred, best_pipeline)
    with open(ERROR_ANALYSIS_PATH, "w", encoding="utf-8") as fh:
        json.dump(error_data, fh, indent=2, ensure_ascii=False)
    print(f"[save] error_analysis.json → {ERROR_ANALYSIS_PATH}")

    # 14. Feature interpretability
    spam_features, ham_features = extract_feature_importance(best_pipeline)
    if spam_features is not None:
        spam_features.to_csv(TOP_SPAM_FEATURES_CSV, index=False)
        ham_features.to_csv(TOP_HAM_FEATURES_CSV, index=False)
        print(f"[save] top_spam_features.csv → {TOP_SPAM_FEATURES_CSV}")
        print(f"[save] top_ham_features.csv  → {TOP_HAM_FEATURES_CSV}")
    else:
        print("[info] Feature importance not available for this classifier type.")

    # 15. metrics.json — summary of the best pipeline's performance
    best_row = summary_df[summary_df["pipeline_name"] == best_name].iloc[0]
    metrics_doc = {
        "best_model_name":        best_model_name,
        "best_vectorizer":        best_vectorizer,
        "selected_pipeline_name": best_name,
        "cv_mean_f1":             round(float(best_row["cv_f1_mean"]), 6),
        "cv_std_f1":              round(float(best_row["cv_f1_std"]),  6),
        "test_accuracy":          round(holdout_metrics["test_accuracy"],  6),
        "test_precision":         round(holdout_metrics["test_precision"], 6),
        "test_recall":            round(holdout_metrics["test_recall"],    6),
        "test_f1":                round(holdout_metrics["test_f1"],        6),
        "train_size":             int(len(X_train)),
        "test_size":              int(len(X_test)),
        "label_mapping":          {"0": "ham", "1": "spam"},
        "random_state":           RANDOM_STATE,
        "cv_folds":               CV_FOLDS,
        "test_split":             TEST_SIZE,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics_doc, fh, indent=2)
    print(f"[save] metrics.json → {METRICS_PATH}")

    print("\n✓ Training complete.\n")
    print(f"  Best pipeline : {best_name}")
    print(f"  CV F1         : {metrics_doc['cv_mean_f1']:.4f} ± {metrics_doc['cv_std_f1']:.4f}")
    print(f"  Test F1       : {metrics_doc['test_f1']:.4f}")
    print(f"  Test Accuracy : {metrics_doc['test_accuracy']:.4f}")


if __name__ == "__main__":
    train()
