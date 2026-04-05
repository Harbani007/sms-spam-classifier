# src/evaluate.py
"""
Visualisation helpers for training results.

Called by train.py, but also importable standalone for re-generating
plots after training without re-running the full train loop.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    COMPARISON_PLOT_IMG,
    CONFUSION_MATRIX_IMG,
    CV_FOLDS,
    CV_RESULTS_CSV,
    MODEL_COMPARISON_CSV,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def save_confusion_matrix(cm: np.ndarray, pipeline_name: str, out_path: Path) -> None:
    """
    Save a clean confusion-matrix figure for the best pipeline.

    Parameters
    ----------
    cm            : 2×2 ndarray from sklearn.metrics.confusion_matrix
    pipeline_name : label for the figure title
    out_path      : where to write the .png
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    class_labels = ["ham (0)", "spam (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=11)
    ax.set_yticklabels(class_labels, fontsize=11)

    for (r, c), val in np.ndenumerate(cm):
        color = "white" if val > cm.max() / 2 else "black"
        ax.text(c, r, str(val), ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)

    ax.set_xlabel("Predicted label", fontsize=12, labelpad=10)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {pipeline_name}", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Confusion matrix saved to {out_path}")


# ---------------------------------------------------------------------------
# Comparison bar chart
# ---------------------------------------------------------------------------

def save_comparison_plot(summary_df: pd.DataFrame, out_path: Path) -> None:
    """
    Bar chart comparing mean CV F1 across all pipelines.
    Bars are sorted descending by F1; error bars show ± 1 std.

    Expects summary_df to have one row per pipeline (model_comparison.csv format).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = summary_df.sort_values("cv_f1_mean", ascending=False).copy()
    names  = df["pipeline_name"].tolist()
    means  = df["cv_f1_mean"].tolist()
    stds   = df["cv_f1_std"].tolist()
    n_pipelines = len(names)

    x = np.arange(n_pipelines)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        x, means,
        yerr=stds,
        capsize=5,
        color=plt.cm.tab10.colors[:n_pipelines],
        edgecolor="grey",
        linewidth=0.6,
        error_kw={"elinewidth": 1.5, "ecolor": "#444444"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(max(0.0, min(means) - 0.05), 1.0)
    ax.set_ylabel("Mean CV F1 (spam class)", fontsize=11)
    ax.set_title(
        f"Pipeline Comparison — {n_pipelines} Pipelines, "
        f"Stratified {CV_FOLDS}-Fold CV",
        fontsize=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Comparison plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Optional standalone: regenerate plots from saved CSV artefacts
# ---------------------------------------------------------------------------

def regenerate_plots() -> None:
    """
    Re-generate both plots from the CSV artefacts already on disk.
    Useful for tweaking visuals without re-running training.
    """
    if not MODEL_COMPARISON_CSV.exists():
        print(f"[error] {MODEL_COMPARISON_CSV} not found. Run train.py first.")
        return

    summary_df = pd.read_csv(MODEL_COMPARISON_CSV)
    save_comparison_plot(summary_df, COMPARISON_PLOT_IMG)
    print("[info] Confusion matrix requires the original test split — skipping. "
          "Re-run train.py to regenerate it.")


if __name__ == "__main__":
    regenerate_plots()
