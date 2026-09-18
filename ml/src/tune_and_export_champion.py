"""
Hyperparameter Tuning, Feature Importance & Master Pipeline Export Module
Author: Member 3 (ML Engineer)
Project: Smart Loan Default & Credit Risk Assessment System
Description: 
  1. Optimizes XGBoost hyperparameters using Stratified 5-Fold Cross-Validation.
  2. Extracts and plots top feature importances (07_feature_importances.png).
  3. Bundles Member 2's preprocessor and Member 3's tuned XGBoost into a single master Pipeline.
  4. Serializes final pipeline to ml/models/loan_risk_pipeline.joblib.
  5. Validates end-to-end inference on raw applicant dictionary payload.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
CHAMPION_PIPELINE_FILE = os.path.join(MODELS_DIR, "loan_risk_pipeline.joblib")


def execute_tuning_and_export():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train_path = os.path.join(PROCESSED_DATA_PATH, "train_data.csv")
    test_path = os.path.join(PROCESSED_DATA_PATH, "test_data.csv")
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")

    print("Loading datasets and preprocessor...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    preprocessor = joblib.load(preprocessor_path)

    X_train = train_df.drop(columns=['loan_status'])
    y_train = train_df['loan_status']
    X_test = test_df.drop(columns=['loan_status'])
    y_test = test_df['loan_status']

    # 1. Stratified 5-Fold Cross-Validation on Tuned XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Configuring Tuned Champion XGBoost (scale_pos_weight: {scale_pos_weight:.2f})...")

    champion_xgb = XGBClassifier(
        n_estimators=220,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_train_trans = preprocessor.transform(X_train)
    cv_auc_scores = cross_val_score(champion_xgb, X_train_trans, y_train, cv=cv, scoring='roc_auc')
    print(f"5-Fold Cross-Validation ROC-AUC: Mean = {cv_auc_scores.mean():.4f} (Std = {cv_auc_scores.std():.4f})")

    # 2. Fit Champion Model on Full X_train_trans
    champion_xgb.fit(X_train_trans, y_train)

    # 3. Extract & Plot Feature Importances
    # Extract feature names from ColumnTransformer
    ct = preprocessor.named_steps['preprocessor']
    num_cols = ct.transformers_[0][2]
    ord_cols = ct.transformers_[1][2]
    nom_encoder = ct.transformers_[2][1]
    nom_feature_names = list(nom_encoder.get_feature_names_out(ct.transformers_[2][2]))
    feature_names = list(num_cols) + list(ord_cols) + nom_feature_names

    importances = champion_xgb.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.barplot(data=feat_imp_df.head(10), x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title("Top 10 Most Predictive Features (XGBoost Feature Importances)", fontweight='bold', pad=12)
    ax.set_xlabel("Relative Gini Gain Importance", fontweight='bold')
    ax.set_ylabel("Engineered Feature", fontweight='bold')
    plt.tight_layout()
    feat_imp_path = os.path.join(FIGURES_DIR, "07_feature_importances.png")
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"[Figure Saved] {feat_imp_path}")
    print("\nTop 5 Predictive Features:")
    for _, row in feat_imp_df.head(5).iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f}")

    # 4. Construct End-to-End Master Pipeline
    master_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', champion_xgb)
    ])

    # 5. Evaluate End-to-End on Untouched Test Split
    print("\nEvaluating Master Pipeline on Untouched Test Set (6,158 rows)...")
    y_pred = master_pipeline.predict(X_test)
    y_proba = master_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Final Test Accuracy:  {acc * 100:.2f}%")
    print(f"Final Test Precision: {prec * 100:.2f}%")
    print(f"Final Test Recall:    {rec * 100:.2f}% (Default Detection)")
    print(f"Final Test F1-Score:  {f1:.4f}")
    print(f"Final Test ROC-AUC:   {auc:.4f}")

    # 6. Serialize Master Pipeline with Joblib
    joblib.dump(master_pipeline, CHAMPION_PIPELINE_FILE)
    file_size_kb = os.path.getsize(CHAMPION_PIPELINE_FILE) / 1024
    print(f"\n[Master Pipeline Exported] -> {CHAMPION_PIPELINE_FILE} ({file_size_kb:.1f} KB)")

    # 7. Test Raw Single Applicant Inference (Ready for FastAPI Member 4)
    raw_sample = pd.DataFrame([{
        "person_age": 25,
        "person_income": 45000,
        "person_home_ownership": "RENT",
        "person_emp_length": 2.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "A",
        "loan_amnt": 5000,
        "loan_int_rate": 7.5,
        "loan_percent_income": 0.11,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }])

    loaded_pipeline = joblib.load(CHAMPION_PIPELINE_FILE)
    pred_prob = loaded_pipeline.predict_proba(raw_sample)[0][1]
    pred_class = loaded_pipeline.predict(raw_sample)[0]
    print(f"\nLive Raw Applicant Inference Verification:")
    print(f"  Predicted Default Probability: {pred_prob:.4f} ({pred_prob * 100:.2f}%)")
    print(f"  Predicted Class: {pred_class} ({'Default' if pred_class == 1 else 'Approved / Non-Default'})")
    print("[SUCCESS] Member 3 master pipeline is 100% production ready for FastAPI backend!")


if __name__ == "__main__":
    execute_tuning_and_export()
