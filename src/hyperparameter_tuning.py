"""Hyperparameter tuning: systematically search for the best model settings
using GridSearchCV (which uses 5-fold cross-validation to grade each combination).
Covers both Random Forest and XGBoost."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier # type: ignore

from data_preprocessing import clean_data, load_raw_data
from feature_engineering import split_features_target
from train_models import build_pipeline

# The "grid" of settings to try. GridSearchCV will test EVERY combination of these.
# Note the "classifier__" prefix: it tells the Pipeline which step these settings
# belong to (our model step is named "classifier" in build_pipeline).
PARAM_GRID = {
    "classifier__n_estimators": [100, 300],        # how many trees in the forest
    "classifier__max_depth": [5, 10, 20],          # restricted depths only (removed None to force generalization)
    "classifier__min_samples_leaf": [1, 5, 10],    # min samples per leaf; bigger = simpler, less noisy leaves
}

# XGBoost has more knobs because boosting is a sequential, more delicate process.
# Kept tight so the total grid (3x3x3x2 = 54 combos x 5 folds = 270 fits) stays runnable.
XGB_PARAM_GRID = {
    "classifier__n_estimators": [200, 400],        # number of sequential correction-rounds (trees)
    "classifier__max_depth": [3, 5, 7],            # shallow trees: boosting prefers many weak learners
    "classifier__learning_rate": [0.05, 0.1, 0.2], # how big a correction each tree makes (smaller = steadier)
    "classifier__subsample": [0.8, 1.0],           # fraction of rows each tree trains on (<1 fights overfitting)
}


def tune_random_forest(X_train, y_train, cv: int = 5) -> GridSearchCV:
    pipeline = build_pipeline(RandomForestClassifier(random_state=42))

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="f1",   # pick the combination with the best F1 (same metric we care about)
        cv=cv,          # 5-fold cross-validation for each combination
        n_jobs=-1,      # use all CPU cores to speed it up
        verbose=1,      # print progress while it runs
    )

    search.fit(X_train, y_train)
    return search


def tune_xgboost(X_train, y_train, cv: int = 5) -> GridSearchCV:
    # scale_pos_weight handles imbalance (kept fixed, not tuned): #negatives / #positives.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    pipeline = build_pipeline(base_model)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=XGB_PARAM_GRID,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    return search


if __name__ == "__main__":
    import sys

    clean_df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_features_target(clean_df)

    # Choose which model to tune from the command line: "rf" (default) or "xgb".
    which = sys.argv[1] if len(sys.argv) > 1 else "rf"

    if which == "xgb":
        search = tune_xgboost(X_train, y_train, cv=5)
        print("\n=== XGBoost tuning results ===")
    else:
        search = tune_random_forest(X_train, y_train, cv=5)
        print("\n=== Random Forest tuning results ===")

    print(f"Best F1 (cross-validated): {search.best_score_:.4f}")
    print(f"Best settings found: {search.best_params_}")
