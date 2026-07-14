"""End-to-end pipeline: load -> clean -> EDA -> feature engineer -> train -> evaluate."""

import sys

import joblib

sys.path.insert(0, "src")

from data_preprocessing import load_raw_data, clean_data, save_processed_data
from eda import run_eda
from feature_engineering import split_features_target
from train_models import train_all_models, build_pipeline, build_xgboost
from evaluate import evaluate_all


def main():
    print("1. Loading raw data...")
    raw_df = load_raw_data()
    print(f"   {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    print("2. Cleaning data...")
    clean_df = clean_data(raw_df)
    save_processed_data(clean_df)
    print(f"   {clean_df.shape[0]} rows remain after cleaning")

    print("3. Running EDA (saving plots to reports/figures/)...")
    run_eda(clean_df)

    print("4. Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_features_target(clean_df)
    print(f"   Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    print("5. Training models (Logistic Regression, Random Forest, XGBoost)...")
    fitted_models = train_all_models(X_train, y_train)
    # train_all_models covers LR and RF (from MODEL_CONFIGS); add the tuned XGBoost,
    # which needs y_train for its imbalance setting, so it's built via a factory.
    xgb_pipeline = build_pipeline(build_xgboost(y_train))
    xgb_pipeline.fit(X_train, y_train)
    fitted_models["xgboost"] = xgb_pipeline

    print("6. Evaluating on test set...")
    results = evaluate_all(fitted_models, X_test, y_test)
    print(results)

    best_name = results["f1_score"].idxmax()
    joblib.dump(fitted_models[best_name], "models/best_model.joblib")
    print(f"\nBest model by F1-score: {best_name} -> saved to models/best_model.joblib")


if __name__ == "__main__":
    main()
