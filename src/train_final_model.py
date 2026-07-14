"""Train the final tuned XGBoost pipeline on all training data and save it to disk,
so the Streamlit app can load it instantly instead of retraining every time."""

import joblib

from data_preprocessing import clean_data, load_raw_data
from feature_engineering import split_features_target
from train_models import build_pipeline, build_xgboost

MODEL_PATH = "models/best_model.joblib"


def train_and_save() -> None:
    clean_df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_features_target(clean_df)

    # Build and train the tuned XGBoost pipeline (preprocessing + model bundled).
    pipeline = build_pipeline(build_xgboost(y_train))
    pipeline.fit(X_train, y_train)

    # Save the WHOLE pipeline (not just the model): this way the saved object also
    # remembers how to scale/encode raw inputs, so the app can feed it raw customer data.
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
