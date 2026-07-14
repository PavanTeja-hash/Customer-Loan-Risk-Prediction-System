# Customer Loan Risk Prediction System

### 🔗 [**► Try the live app**](https://customer-loan-risk-prediction-system-97fp9tmbut7vuugqvnyqqh.streamlit.app/) &nbsp;|&nbsp; enter an applicant's details and get an instant risk prediction

A machine learning system that predicts customer loan-default risk from financial and demographic data, to support lending decisions. It goes beyond a basic classifier: cross-validated evaluation, hyperparameter tuning, class-imbalance handling, model explainability, business-cost analysis, and a live interactive web app.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib / Seaborn
- Streamlit (interactive demo)

## Dataset
[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) — 32,581 records of customer financial and demographic data (age, income, home ownership, employment length, loan intent/grade/amount/interest rate, credit history) with a binary `loan_status` target (0 = repaid, 1 = default). Moderately imbalanced (~78% repaid / 22% default).

## Project Structure
```
Customer-Loan-Risk-Prediction-System/
├── app.py                       # Streamlit web app (live prediction demo)
├── main.py                      # runs the core pipeline end-to-end
├── data/
│   ├── raw/                     # original dataset
│   └── processed/               # cleaned data
├── src/
│   ├── data_preprocessing.py    # cleaning: duplicates, outliers, imputation
│   ├── eda.py                   # exploratory charts
│   ├── feature_engineering.py   # scaling + one-hot encoding pipeline, train/test split
│   ├── train_models.py          # model configs (LR, tuned RF, tuned XGBoost)
│   ├── evaluate.py              # accuracy/precision/recall/F1 + confusion matrices
│   ├── cross_validation.py      # 5-fold cross-validation
│   ├── hyperparameter_tuning.py # GridSearchCV for RF and XGBoost
│   ├── imbalance_experiment.py  # class_weight="balanced" experiment
│   ├── threshold_tuning.py      # decision-threshold analysis
│   ├── model_comparison.py      # fair 3-model comparison
│   ├── feature_importance.py    # explainability (XGBoost importances)
│   ├── cost_analysis.py         # business-cost (dollar) analysis
│   └── train_final_model.py     # trains + saves the final model for the app
├── models/best_model.joblib     # saved tuned XGBoost pipeline (~1.8MB)
├── reports/figures/             # all saved plots
├── requirements.txt
├── DEPLOYMENT.md                # how to deploy the app
└── README.md
```

## Approach

### Core pipeline
1. **Data cleaning** — remove duplicates and impossible values (age > 100, employment length > 60); impute missing `loan_int_rate` per loan grade and `person_emp_length` by median (`src/data_preprocessing.py`).
2. **EDA** — class balance, distributions, correlation heatmap, default rates by category (`src/eda.py`).
3. **Feature engineering** — `StandardScaler` for numerics, `OneHotEncoder` for categoricals, bundled in a `ColumnTransformer`; stratified 80/20 train/test split (`src/feature_engineering.py`).
4. **Modeling & evaluation** — Logistic Regression, Random Forest, XGBoost, scored on accuracy/precision/recall/F1 (`src/train_models.py`, `src/evaluate.py`).

### Advanced work
5. **Cross-validation** — 5-fold CV confirms results are stable, not a lucky split (`src/cross_validation.py`).
6. **Hyperparameter tuning** — `GridSearchCV` over RF and XGBoost settings, scored by F1 (`src/hyperparameter_tuning.py`).
7. **Class-imbalance handling** — `class_weight="balanced"` (RF/LR) and `scale_pos_weight` (XGBoost) to improve recall on defaulters (`src/imbalance_experiment.py`).
8. **Threshold tuning** — scans decision thresholds to trade off precision vs recall instead of the default 0.5 (`src/threshold_tuning.py`).
9. **Explainability** — XGBoost feature importances; top drivers are home ownership, loan grade, and loan-to-income ratio (`src/feature_importance.py`).
10. **Business-cost analysis** — translates false positives/negatives into dollar costs and finds the cost-minimizing threshold (`src/cost_analysis.py`).
11. **Interactive demo** — Streamlit app for live, per-applicant predictions with an adjustable decision threshold (`app.py`).

## Results (test set)
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.868 | 0.770 | 0.554 | 0.645 |
| Random Forest (tuned) | 0.929 | 0.902 | 0.753 | 0.821 |
| **XGBoost (tuned)** | **0.935** | 0.891 | **0.794** | **0.840** |

*(Reproducible via `python main.py`. A balanced Logistic Regression — see `src/model_comparison.py` — trades precision for higher recall: 0.556 / 0.786.)*

**Tuned XGBoost** is the best overall model — highest F1 and recall — and is saved to `models/best_model.joblib` for the app. Model choice depends on business priority: XGBoost catches the most defaulters (recall), while Random Forest has slightly higher precision.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run
Core pipeline:
```bash
python main.py
```
Any individual analysis (examples):
```bash
python src/cross_validation.py
python src/model_comparison.py
python src/feature_importance.py
python src/hyperparameter_tuning.py xgb
```
Interactive web app:
```bash
streamlit run app.py
```

## Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for deploying the Streamlit app to a public URL.

## Notes & honest limitations
- XGBoost probabilities are **risk scores, not calibrated** — rankings are reliable, but exact percentages are overconfident; `CalibratedClassifierCV` would fix this if calibrated probabilities were needed.
- Threshold tuning currently selects on the test set (mild leakage); a validation set or CV-based selection would be more rigorous.
- Business-cost figures use illustrative loss/profit assumptions, not real bank data.
