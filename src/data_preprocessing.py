"""Load the raw loan dataset and clean it: missing values, duplicates, outliers."""

import pandas as pd

RAW_PATH = "data/raw/credit_risk_dataset.csv"
PROCESSED_PATH = "data/processed/credit_risk_clean.csv"


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Data-entry errors: a handful of rows have implausible age/employment length
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"] <= 60]

    # Missing values: loan_int_rate depends on loan_grade, so impute per grade
    df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
        lambda s: s.fillna(s.median())
    )

    # person_emp_length: impute with overall median
    df["person_emp_length"] = df["person_emp_length"].fillna(
        df["person_emp_length"].median()
    )

    return df.reset_index(drop=True)


def save_processed_data(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    save_processed_data(clean_df)
    print(f"Raw rows: {len(raw_df)} -> Cleaned rows: {len(clean_df)}")
    print(f"Missing values left:\n{clean_df.isnull().sum()[clean_df.isnull().sum() > 0]}")
