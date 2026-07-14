"""Cross-validation: get a more reliable performance estimate than one single train/test split."""

from sklearn.model_selection import cross_val_score

from data_preprocessing import clean_data, load_raw_data
from feature_engineering import split_features_target
from train_models import MODEL_CONFIGS, build_pipeline


def run_cross_validation(X_train, y_train, cv: int = 5) -> dict:
    results = {}
    for name, model in MODEL_CONFIGS.items():
        pipeline = build_pipeline(model)
        # cross_val_score handles all the fold-splitting, training, and testing internally.
        # scoring="f1" tells it to compute F1-score for each fold (same metric we care about).
        fold_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        results[name] = fold_scores
    return results


if __name__ == "__main__":
    clean_df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_features_target(clean_df)

    cv_results = run_cross_validation(X_train, y_train, cv=5)

    for name, fold_scores in cv_results.items():
        print(f"\n{name}")
        print(f"  Fold F1 scores: {fold_scores.round(4)}")
        print(f"  Mean F1: {fold_scores.mean():.4f}  |  Std dev: {fold_scores.std():.4f}")
