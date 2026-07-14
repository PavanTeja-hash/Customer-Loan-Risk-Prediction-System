"""Train classification models to predict loan default risk."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from feature_engineering import build_preprocessor

# Tuned Random Forest settings, chosen via GridSearchCV (see hyperparameter_tuning.py):
# n_estimators=100, max_depth=20, min_samples_leaf=1 was the best defensible config.
# class_weight="balanced" was added after the imbalance experiment to improve recall.
MODEL_CONFIGS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    ),
}

# Tuned XGBoost settings, chosen via GridSearchCV (hyperparameter_tuning.py xgb).
# XGBoost needs scale_pos_weight computed from y_train (imbalance handling), which
# isn't known until runtime, so it's built via a factory function instead of a
# frozen config. This is the single source of truth for the tuned XGBoost.
def build_xgboost(y_train) -> XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    return XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.2,
        subsample=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )


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
