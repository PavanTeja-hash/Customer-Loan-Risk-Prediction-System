# Customer Loan Risk Prediction System

A machine learning-based system that predicts customer loan risk using financial and demographic data, to support risk assessment and lending decisions.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn

## Project Structure
```
Customer-Loan-Risk-Prediction-System/
├── data/
│   ├── raw/            # original, untouched dataset
│   └── processed/      # cleaned/feature-engineered data
├── notebooks/           # EDA and experimentation notebooks
├── src/                 # reusable pipeline code
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── evaluate.py
├── reports/
│   └── figures/         # saved plots/visualizations
├── requirements.txt
└── README.md
```

## Workflow
1. **Data Cleaning** — handle missing values, duplicates, and inconsistent types (`src/data_preprocessing.py`)
2. **EDA & Visualization** — explore distributions, correlations, class balance (`src/eda.py`, `notebooks/`)
3. **Feature Engineering** — encode categoricals, scale numerics, derive new features (`src/feature_engineering.py`)
4. **Model Training** — train classification models (Logistic Regression, Random Forest, etc.) (`src/train_models.py`)
5. **Evaluation** — accuracy, precision, recall, F1-score (`src/evaluate.py`)

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset
[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) — 32,581 records of customer financial and demographic data (age, income, home ownership, employment length, loan intent/grade/amount/interest rate, credit history) with a binary `loan_status` target (0 = repaid, 1 = default).

## Results
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.868 | 0.770 | 0.554 | 0.645 |
| Random Forest | 0.937 | 0.975 | 0.728 | 0.834 |

Random Forest was selected as the best model (highest F1-score) and saved to `models/best_model.joblib`.

## How to Run
```bash
venv\Scripts\activate
python main.py
```
This runs the full pipeline: load → clean → EDA (plots saved to `reports/figures/`) → feature engineering → train → evaluate.

Note: the trained model file (`models/best_model.joblib`) is not stored in this repo (it's ~112MB, over GitHub's size limit) — running `main.py` regenerates it locally.

## Status
✅ Pipeline complete — data cleaned, EDA generated, two models trained and evaluated.
