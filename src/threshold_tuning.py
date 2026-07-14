"""Threshold tuning: instead of the default 0.5 cutoff, scan all thresholds to find
the one that maximizes F1, and visualize the precision/recall trade-off."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

from data_preprocessing import clean_data, load_raw_data
from feature_engineering import split_features_target
from train_models import MODEL_CONFIGS, build_pipeline

FIGURES_DIR = "reports/figures"


def find_best_threshold(pipeline, X_test, y_test):
    # predict_proba returns probability for each class; column [:, 1] = probability of default (class 1)
    probs = pipeline.predict_proba(X_test)[:, 1]

    # precision_recall_curve gives precision & recall at every possible threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    # Compute F1 at each threshold. (precisions/recalls have one extra entry vs thresholds,
    # so we align by slicing off the last point.)
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]

    return best_threshold, probs, precisions, recalls, thresholds


def metrics_at_threshold(y_test, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
    }


def plot_threshold_curve(precisions, recalls, thresholds, best_threshold):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions[:-1], label="Precision")
    ax.plot(thresholds, recalls[:-1], label="Recall")
    ax.axvline(best_threshold, color="red", linestyle="--", label=f"Best F1 threshold = {best_threshold:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", label="Default threshold = 0.50")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs Decision Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/threshold_tuning.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    clean_df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_features_target(clean_df)

    pipeline = build_pipeline(MODEL_CONFIGS["random_forest"])
    pipeline.fit(X_train, y_train)

    best_threshold, probs, precisions, recalls, thresholds = find_best_threshold(
        pipeline, X_test, y_test
    )

    print("At DEFAULT threshold (0.50):")
    print(f"  {metrics_at_threshold(y_test, probs, 0.5)}")
    print("\nAt BEST-F1 threshold:")
    print(f"  {metrics_at_threshold(y_test, probs, best_threshold)}")

    plot_threshold_curve(precisions, recalls, thresholds, best_threshold)
    print(f"\nSaved threshold curve to {FIGURES_DIR}/threshold_tuning.png")
