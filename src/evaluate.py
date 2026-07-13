"""Score trained models on the held-out test set."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

FIGURES_DIR = "reports/figures"


def evaluate_model(pipeline, X_test, y_test, name: str) -> dict:
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)

    return metrics


def evaluate_all(fitted_models: dict, X_test, y_test) -> pd.DataFrame:
    rows = [evaluate_model(model, X_test, y_test, name) for name, model in fitted_models.items()]
    return pd.DataFrame(rows).set_index("model").round(4)
