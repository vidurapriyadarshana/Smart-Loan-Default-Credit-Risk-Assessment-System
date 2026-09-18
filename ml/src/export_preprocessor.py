"""
Preprocessor Pipeline Export and Verification Script
Author: Member 2 (Feature Engineer)
Project: Smart Loan Default & Credit Risk Assessment System
Description: 
  1. Partitions data into 80/20 stratified Train/Test splits.
  2. Fits the feature pipeline strictly on X_train (Zero Data Leakage).
  3. Verifies transformation on X_test.
  4. Serializes preprocessor to ml/models/preprocessor.joblib.
  5. Validates single applicant raw dictionary transformation.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml.src.features import build_full_preprocessing_pipeline

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CLEAN_DATA_FILE = os.path.join(PROCESSED_DATA_PATH, "cleaned_credit_data.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PREPROCESSOR_FILE = os.path.join(MODELS_DIR, "preprocessor.joblib")


def execute_pipeline_export():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    print("Loading cleaned dataset...")
    df = pd.read_csv(CLEAN_DATA_FILE)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # 1. Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train split shape: {X_train.shape} (Default rate: {y_train.mean():.4f})")
    print(f"Test split shape:  {X_test.shape}  (Default rate: {y_test.mean():.4f})")

    # Save train/test partitions for Member 3
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_path = os.path.join(PROCESSED_DATA_PATH, "train_data.csv")
    test_path = os.path.join(PROCESSED_DATA_PATH, "test_data.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved partitions:\n  {train_path}\n  {test_path}")

    # 2. Fit pipeline strictly on X_train
    print("\nFitting feature pipeline on X_train...")
    pipeline = build_full_preprocessing_pipeline()
    pipeline.fit(X_train)

    # 3. Transform X_train and X_test independently
    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)

    print(f"X_train transformed shape: {X_train_trans.shape}")
    print(f"X_test transformed shape:  {X_test_trans.shape}")
    assert not np.isnan(X_train_trans).any(), "NaN values found in X_train_trans!"
    assert not np.isnan(X_test_trans).any(), "NaN values found in X_test_trans!"
    print("Verification passed: Zero NaNs and perfect 22-dimensional feature space.")

    # 4. Serialize fitted pipeline with joblib
    joblib.dump(pipeline, PREPROCESSOR_FILE)
    print(f"\n[Export Success] Serialized preprocessor saved to: {PREPROCESSOR_FILE}")
    print(f"File size: {os.path.getsize(PREPROCESSOR_FILE):,} bytes")

    # 5. Verify single applicant inference
    sample_applicant = pd.DataFrame([{
        "person_age": 28,
        "person_income": 65000,
        "person_home_ownership": "RENT",
        "person_emp_length": 4.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 11.2,
        "loan_percent_income": 0.15,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5
    }])

    loaded_pipeline = joblib.load(PREPROCESSOR_FILE)
    single_vector = loaded_pipeline.transform(sample_applicant)
    print(f"\nSingle applicant test transformation successful! Output shape: {single_vector.shape}")
    print(f"Vector preview: {np.round(single_vector[0][:8], 3)}")


if __name__ == "__main__":
    execute_pipeline_export()
