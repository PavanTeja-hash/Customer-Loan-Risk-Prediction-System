"""Train classification models to predict loan default risk."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from feature_engineering import build_preprocessor

MODEL_CONFIGS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
}


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )


def train_all_models(X_train, y_train) -> dict:
    fitted = {}
    for name, model in MODEL_CONFIGS.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    return fitted
