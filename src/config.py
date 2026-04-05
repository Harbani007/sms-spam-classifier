# src/config.py
"""
Central configuration for the SMS Spam Detection project.
All paths, hyperparameters, and constants live here so that
other modules stay clean and consistent.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (all relative to the project root, one level above /src)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR      = PROJECT_ROOT / "data"
MODELS_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR   = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Default dataset file — adjust if your file has a different name
DATASET_PATH = DATA_DIR / "spam.csv"

# Saved pipeline artifact
BEST_PIPELINE_PATH = MODELS_DIR / "best_pipeline.pkl"

# Result artifacts
METRICS_PATH               = RESULTS_DIR / "metrics.json"
MODEL_COMPARISON_CSV       = RESULTS_DIR / "model_comparison.csv"
CV_RESULTS_CSV             = RESULTS_DIR / "cv_results.csv"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.json"
ERROR_ANALYSIS_PATH        = RESULTS_DIR / "error_analysis.json"
CONFUSION_MATRIX_IMG       = RESULTS_DIR / "confusion_matrix_best.png"
COMPARISON_PLOT_IMG        = RESULTS_DIR / "comparison_plot.png"
TOP_SPAM_FEATURES_CSV      = RESULTS_DIR / "top_spam_features.csv"
TOP_HAM_FEATURES_CSV       = RESULTS_DIR / "top_ham_features.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Train / Test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20            # 80 / 20 stratified split

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS = 5                # stratified k-fold on training split only

# ---------------------------------------------------------------------------
# Parallelism
# n_jobs=1 is safer and more portable across environments than -1.
# Set to -1 here to use all CPU cores if your environment supports it.
# ---------------------------------------------------------------------------
N_JOBS = 1

# ---------------------------------------------------------------------------
# Vectorizer defaults (shared across BoW and TF-IDF)
# ---------------------------------------------------------------------------
NGRAM_RANGE  = (1, 2)       # unigrams + bigrams
MAX_FEATURES = 5000         # vocabulary size cap

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
LR_MAX_ITER     = 1000      # enough iterations for lbfgs to converge
LR_SOLVER       = "lbfgs"
LR_CLASS_WEIGHT = "balanced"   # important: SMS datasets are imbalanced (~87% ham)

SVC_MAX_ITER     = 2000
SVC_CLASS_WEIGHT = "balanced"  # same rationale

MNB_ALPHA = 1.0             # Laplace smoothing

# ---------------------------------------------------------------------------
# Error-analysis cap
# ---------------------------------------------------------------------------
MAX_ERROR_EXAMPLES = 10     # how many FP / FN to save in error_analysis.json

# Top-N features to export for interpretability
TOP_N_FEATURES = 25

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL_SPAM  = 1
LABEL_HAM   = 0
LABEL_NAMES = {LABEL_HAM: "ham", LABEL_SPAM: "spam"}
