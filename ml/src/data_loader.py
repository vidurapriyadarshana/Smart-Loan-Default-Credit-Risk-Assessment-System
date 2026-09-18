"""
Data Loader and Inspection Module
Author: Member 1 (Data Analyst)
Project: Smart Loan Default & Credit Risk Assessment System
Description: Responsible for dataset ingestion, schema validation, and quality auditing.
"""

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "credit_risk_dataset.csv")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_credit_data.csv")


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw credit risk dataset from disk and prints initial dimensions."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"[Phase 1] Successfully loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def audit_data_quality(df: pd.DataFrame) -> dict:
    """Performs comprehensive quality audit on missingness, duplicates, and target imbalance."""
    audit_results = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isnull().sum().to_dict(),
        "target_distribution": df['loan_status'].value_counts().to_dict(),
        "target_proportions": df['loan_status'].value_counts(normalize=True).round(4).to_dict(),
        "age_min": int(df['person_age'].min()),
        "age_max": int(df['person_age'].max()),
        "max_emp_length": float(df['person_emp_length'].max())
    }
    return audit_results


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset by:
    1. Removing duplicate rows.
    2. Filtering biological/physical age anomalies (18 <= age <= 85).
    3. Filtering employment duration exceeding legal working age.
    4. Grouped median imputation for missing interest rates and employment lengths.
    """
    initial_count = len(df)
    
    # 1. Deduplicate
    df = df.drop_duplicates().copy()
    dedup_count = len(df)
    print(f"[Cleaning] Removed {initial_count - dedup_count} duplicate rows.")

    # 2. Filter age & employment anomalies
    df = df[(df['person_age'] >= 18) & (df['person_age'] <= 85)].copy()
    df = df[df['person_emp_length'] <= (df['person_age'] - 16)].copy()
    filtered_count = len(df)
    print(f"[Cleaning] Filtered {dedup_count - filtered_count} anomalous rows.")

    # 3. Impute missing loan_int_rate by loan_grade median
    grade_medians = df.groupby('loan_grade')['loan_int_rate'].transform('median')
    df['loan_int_rate'] = df['loan_int_rate'].fillna(grade_medians)

    # 4. Impute missing person_emp_length by age bucket median
    age_bins = pd.cut(df['person_age'], bins=[17, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
    emp_medians = df.groupby(age_bins, observed=False)['person_emp_length'].transform('median')
    df['person_emp_length'] = df['person_emp_length'].fillna(emp_medians)

    remaining_nulls = df.isnull().sum().sum()
    print(f"[Cleaning] Final dataset shape: {df.shape}. Remaining nulls: {remaining_nulls}.")
    return df


def save_processed_data(df: pd.DataFrame, target_path: str = PROCESSED_DATA_PATH):
    """Saves verified cleaned dataset to processed directory."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    df.to_csv(target_path, index=False)
    print(f"[Export] Cleaned dataset saved to: {target_path}")


if __name__ == "__main__":
    raw_df = load_raw_data()
    audit = audit_data_quality(raw_df)
    print("\n--- Data Quality Audit Summary ---")
    for k, v in audit.items():
        print(f"  {k}: {v}")
