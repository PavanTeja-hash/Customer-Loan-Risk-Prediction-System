"""Feature importance: rank which features the tuned XGBoost model relies on most,
and visualize them. Makes the 'black box' model explainable."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from data_preprocessing import clean_data, load_raw_data
from feature_engineering import split_features_target
from train_models import build_pipeline, build_xgboost

FIGURES_DIR = "reports/figures"


def get_feature_importances(pipeline) -> pd.Series:
    # The pipeline has two named steps: "preprocessor" and "classifier".
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # After one-hot encoding, feature names change (loan_grade -> loan_grade_A, _B, ...).
    # get_feature_names_out() gives the real, post-encoding column names.
    feature_names = preprocessor.get_feature_names_out()

    # XGBoost exposes how much it relied on each feature via .feature_importances_
    importances = classifier.feature_importances_

    # Pair names with scores, sort highest-first.
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def plot_feature_importances(importance_series: pd.Series, top_n: int = 15) -> None:
    top = importance_series.head(top_n).iloc[::-1]  # reverse so biggest is at the top of the barh
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values)
    ax.set_xlabel("Importance score")
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost)")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    clean_df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_features_target(clean_df)

    model = build_xgboost(y_train)   # tuned XGBoost from the single source of truth
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    importances = get_feature_importances(pipeline)
    print("Top 10 most important features:\n")
    print(importances.head(10).round(4))

    plot_feature_importances(importances, top_n=15)
    print(f"\nSaved feature importance chart to {FIGURES_DIR}/feature_importance.png")
