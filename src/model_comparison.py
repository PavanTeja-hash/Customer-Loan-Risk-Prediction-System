"""Compare all three models (Logistic Regression, Random Forest, XGBoost) fairly
on the same train/test split, using imbalance handling for each."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_preprocessing import clean_data, load_raw_data
from feature_engineering import split_features_target
from train_models import MODEL_CONFIGS, build_pipeline, build_xgboost
from evaluate import evaluate_model


def build_models(y_train) -> dict:
    # Random Forest and XGBoost come from train_models.py (single source of truth for
    # their tuned settings). LR here uses class_weight="balanced" for a fair comparison.
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": MODEL_CONFIGS["random_forest"],
        "xgboost": build_xgboost(y_train),
    }


def compare_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    models = build_models(y_train)
    rows = []
    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        rows.append(evaluate_model(pipeline, X_test, y_test, name))
    return pd.DataFrame(rows).set_index("model").round(4)


if __name__ == "__main__":
    clean_df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_features_target(clean_df)

    results = compare_models(X_train, X_test, y_train, y_test)
    print(results)
    print(f"\nBest model by F1: {results['f1_score'].idxmax()}")
