"""Class imbalance experiment: compare Random Forest WITH vs WITHOUT class_weight='balanced'
to see the effect on recall (catching more defaulters) and the precision trade-off."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import clean_data, load_raw_data
from feature_engineering import split_features_target
from train_models import build_pipeline
from evaluate import evaluate_model

# Two versions of the SAME model, using our tuned settings: the only difference is
# class_weight, so the experiment isolates exactly that one change.
MODELS = {
    "rf_default": RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_leaf=1, random_state=42
    ),
    "rf_balanced": RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_leaf=1,
        class_weight="balanced", random_state=42,
    ),
}


def run_imbalance_experiment(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    for name, model in MODELS.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        rows.append(evaluate_model(pipeline, X_test, y_test, name))
    return pd.DataFrame(rows).set_index("model").round(4)


if __name__ == "__main__":
    clean_df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_features_target(clean_df)

    results = run_imbalance_experiment(X_train, X_test, y_train, y_test)
    print(results)
