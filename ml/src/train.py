"""
Model Training, Benchmarking & Evaluation Module
Author: Member 3 (ML Engineer)
Project: Smart Loan Default & Credit Risk Assessment System
Description: 
  1. Trains baseline (Logistic Regression), ensemble (Random Forest), and champion (XGBoost).
  2. Addresses class imbalance via cost-sensitive learning (scale_pos_weight = 3.6).
  3. Benchmarks candidate models on held-out test data across Accuracy, Precision, Recall, F1, and ROC-AUC.
  4. Generates ROC curves and confusion matrix comparison figures.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs")


def load_data_and_preprocessor():
    train_path = os.path.join(PROCESSED_DATA_PATH, "train_data.csv")
    test_path = os.path.join(PROCESSED_DATA_PATH, "test_data.csv")
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    preprocessor = joblib.load(preprocessor_path)

    X_train = train_df.drop(columns=['loan_status'])
    y_train = train_df['loan_status']
    X_test = test_df.drop(columns=['loan_status'])
    y_test = test_df['loan_status']

    print(f"Loaded train ({len(X_train)} rows) and test ({len(X_test)} rows).")

    # Transform using Member 2's fitted preprocessor
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    return X_train_trans, X_test_trans, y_train, y_test, preprocessor


def train_and_benchmark():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, preprocessor = load_data_and_preprocessor()

    # Calculate class imbalance weighting factor
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Class imbalance: {neg_count} non-defaults vs {pos_count} defaults. scale_pos_weight = {scale_pos_weight:.2f}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42
        )
    }

    benchmark_results = []
    roc_data = {}
    cm_data = {}

    print("\n--- Training & Evaluating Candidate Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        benchmark_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall (Default)": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)
        cm_data[name] = cm

        print(f"  [{name}] Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    results_df = pd.DataFrame(benchmark_results)

    # 1. Plot ROC Curves
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Logistic Regression": "#64748b", "Random Forest": "#3b82f6", "XGBoost": "#10b981"}
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=colors[name], linewidth=2.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label="Random Guess (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight='bold')
    ax.set_title("Candidate Models Receiver Operating Characteristic (ROC) Benchmark", fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_plot_path = os.path.join(FIGURES_DIR, "05_model_roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"\n[Figure Saved] {roc_plot_path}")

    # 2. Plot Confusion Matrices Side by Side
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, cm) in zip(axes, cm_data.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    annot_kws={'size': 13, 'weight': 'bold'})
        ax.set_title(f"{name}\nConfusion Matrix", fontweight='bold')
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_xticklabels(["Non-Default", "Default"])
        ax.set_yticklabels(["Non-Default", "Default"])
    plt.tight_layout()
    cm_plot_path = os.path.join(FIGURES_DIR, "06_confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[Figure Saved] {cm_plot_path}")

    return results_df, models['XGBoost'], preprocessor


if __name__ == "__main__":
    results, champion, preprocessor = train_and_benchmark()
    print("\n=== Final Benchmark Results Summary ===")
    print(results.to_string(index=False))
