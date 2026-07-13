"""Exploratory data analysis: distributions, class balance, correlations.

Saves each plot as a PNG in reports/figures/ instead of showing it on screen,
so this can run headlessly from the command line.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGURES_DIR = "reports/figures"


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/{name}.png", dpi=150)
    plt.close(fig)


def plot_target_balance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="loan_status", data=df, ax=ax)
    ax.set_title("Loan Status Class Balance (0 = repaid, 1 = default)")
    _save(fig, "target_balance")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = ["person_age", "person_income", "loan_amnt", "loan_int_rate"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
    _save(fig, "numeric_distributions")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap (numeric features)")
    _save(fig, "correlation_heatmap")


def plot_default_rate_by_category(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df.groupby("loan_grade")["loan_status"].mean().sort_index().plot(
        kind="bar", ax=axes[0], title="Default Rate by Loan Grade"
    )
    df.groupby("person_home_ownership")["loan_status"].mean().plot(
        kind="bar", ax=axes[1], title="Default Rate by Home Ownership"
    )
    for ax in axes:
        ax.set_ylabel("Default Rate")
    _save(fig, "default_rate_by_category")


def run_eda(df: pd.DataFrame) -> None:
    plot_target_balance(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_default_rate_by_category(df)


if __name__ == "__main__":
    from data_preprocessing import load_raw_data, clean_data

    clean_df = clean_data(load_raw_data())
    run_eda(clean_df)
    print(f"Saved EDA plots to {FIGURES_DIR}/")
