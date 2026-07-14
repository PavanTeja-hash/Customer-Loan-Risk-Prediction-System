"""Business cost analysis: translate model errors into dollars, and find the
threshold that MINIMIZES total business cost (not the one that maximizes F1).

NOTE: the cost assumptions below are illustrative, not real bank figures. In a real
project these would come from the finance/risk team. The METHOD is the valuable part.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data_preprocessing import clean_data, load_raw_data
from feature_engineering import split_features_target
from train_models import build_pipeline, build_xgboost

FIGURES_DIR = "reports/figures"

# --- Cost assumptions (change these to match real business numbers) ---
LGD = 0.50            # Loss Given Default: fraction of loan lost when an approved loan defaults
PROFIT_MARGIN = 0.15  # profit the bank earns on a good loan (lost when we wrongly reject one)


def total_business_cost(y_true, preds, loan_amounts) -> float:
    # False Negative: predicted repay (0) but actually defaulted (1) -> approved a bad loan.
    fn_mask = (y_true == 1) & (preds == 0)
    # False Positive: predicted default (1) but actually would repay (0) -> rejected a good loan.
    fp_mask = (y_true == 0) & (preds == 1)

    cost_fn = (loan_amounts[fn_mask] * LGD).sum()          # principal lost
    cost_fp = (loan_amounts[fp_mask] * PROFIT_MARGIN).sum()  # profit lost
    return cost_fn + cost_fp


def scan_costs(y_true, probs, loan_amounts, thresholds):
    costs = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        costs.append(total_business_cost(y_true, preds, loan_amounts))
    return np.array(costs)


def plot_cost_curve(thresholds, costs, best_threshold):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, costs, label="Total business cost ($)")
    ax.axvline(best_threshold, color="green", linestyle="--",
               label=f"Min-cost threshold = {best_threshold:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", label="Default threshold = 0.50")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total cost ($)")
    ax.set_title("Total Business Cost vs Decision Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/cost_analysis.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    clean_df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_features_target(clean_df)

    pipeline = build_pipeline(build_xgboost(y_train))
    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:, 1]
    loan_amounts = X_test["loan_amnt"].to_numpy()
    y_true = y_test.to_numpy()

    thresholds = np.linspace(0.05, 0.95, 91)
    costs = scan_costs(y_true, probs, loan_amounts, thresholds)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]

    default_cost = total_business_cost(y_true, (probs >= 0.5).astype(int), loan_amounts)
    best_cost = costs[best_idx]

    print(f"Cost assumptions: LGD={LGD}, PROFIT_MARGIN={PROFIT_MARGIN}\n")
    print(f"At default threshold 0.50: ${default_cost:,.0f} total cost")
    print(f"At min-cost threshold {best_threshold:.2f}: ${best_cost:,.0f} total cost")
    print(f"Savings from choosing the business-optimal threshold: ${default_cost - best_cost:,.0f}")

    plot_cost_curve(thresholds, costs, best_threshold)
    print(f"\nSaved cost curve to {FIGURES_DIR}/cost_analysis.png")
