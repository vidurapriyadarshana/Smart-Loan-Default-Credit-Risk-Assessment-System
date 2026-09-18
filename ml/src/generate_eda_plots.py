"""
EDA Visualization and Statistical Plotting Module
Author: Member 1 (Data Analyst)
Project: Smart Loan Default & Credit Risk Assessment System
Description: Generates high-resolution statistical charts and figures for EDA and documentation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
PROCESSED_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_credit_data.csv")


def generate_all_plots():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(PROCESSED_DATA)
    print(f"Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 10

    # 1. Target Class Imbalance Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    counts = df['loan_status'].value_counts()
    props = df['loan_status'].value_counts(normalize=True) * 100
    bars = ax.bar(['Non-Default (0)', 'Default (1)'], counts, color=['#10b981', '#ef4444'], width=0.45, edgecolor='black', linewidth=0.8)
    for bar, prop in zip(bars, props):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 350, f"{prop:.1f}%\n({int(yval):,})", ha='center', fontweight='bold')
    ax.set_ylabel('Applicant Count', fontweight='bold')
    ax.set_title('Target Class Distribution (loan_status)', fontweight='bold', pad=12)
    plt.tight_layout()
    target_fig = os.path.join(FIGURES_DIR, "01_target_distribution.png")
    plt.savefig(target_fig, dpi=300)
    plt.close()
    print(f"[Plot 1] Saved: {target_fig}")

    # 2. Numerical Distributions & Skewness
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    inc_skew = df['person_income'].skew()
    loan_skew = df['loan_amnt'].skew()
    age_skew = df['person_age'].skew()
    ratio_skew = df['loan_percent_income'].skew()

    sns.histplot(df['person_income'], bins=50, kde=True, ax=axes[0, 0], color='#3b82f6')
    axes[0, 0].set_title(f"Income Distribution (Skew: {inc_skew:.2f})", fontweight='bold')
    axes[0, 0].set_xlim(0, 300000)

    sns.histplot(df['loan_amnt'], bins=35, kde=True, ax=axes[0, 1], color='#8b5cf6')
    axes[0, 1].set_title(f"Loan Amount Distribution (Skew: {loan_skew:.2f})", fontweight='bold')

    sns.histplot(df['person_age'], bins=30, kde=True, ax=axes[1, 0], color='#10b981')
    axes[1, 0].set_title(f"Age Distribution (Skew: {age_skew:.2f})", fontweight='bold')

    sns.histplot(df['loan_percent_income'], bins=40, kde=True, ax=axes[1, 1], color='#f59e0b')
    axes[1, 1].set_title(f"Loan % Income (Skew: {ratio_skew:.2f})", fontweight='bold')

    plt.tight_layout()
    dist_fig = os.path.join(FIGURES_DIR, "02_numerical_distributions.png")
    plt.savefig(dist_fig, dpi=300)
    plt.close()
    print(f"[Plot 2] Saved: {dist_fig}")

    # 3. Default Rate by Loan Grade (Monotonic risk curve)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    grade_default = df.groupby('loan_grade')['loan_status'].mean().reindex(grade_order) * 100
    colors = ['#10b981', '#34d399', '#facc15', '#f59e0b', '#fb923c', '#f87171', '#dc2626']
    bars = ax.bar(grade_default.index, grade_default.values, color=colors, edgecolor='black', linewidth=0.8, width=0.55)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%", ha='center', fontweight='bold')
    ax.set_ylabel('Default Rate (%)', fontweight='bold')
    ax.set_xlabel('Credit Risk Grade', fontweight='bold')
    ax.set_title('Default Rate by Credit Risk Grade (Monotonic Risk Curve)', fontweight='bold', pad=12)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    grade_fig = os.path.join(FIGURES_DIR, "03_default_by_loan_grade.png")
    plt.savefig(grade_fig, dpi=300)
    plt.close()
    print(f"[Plot 3] Saved: {grade_fig}")

    # 4. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax, linewidths=0.5)
    ax.set_title('Numerical Features Pearson Correlation Heatmap', fontweight='bold', pad=12)
    plt.tight_layout()
    corr_fig = os.path.join(FIGURES_DIR, "04_correlation_heatmap.png")
    plt.savefig(corr_fig, dpi=300)
    plt.close()
    print(f"[Plot 4] Saved: {corr_fig}")


if __name__ == "__main__":
    generate_all_plots()
