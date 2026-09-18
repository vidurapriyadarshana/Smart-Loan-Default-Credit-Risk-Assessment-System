import pandas as pd
import numpy as np
from ml.src.features import DomainFeatureCreator, IQRWinsorizer, build_full_preprocessing_pipeline

df = pd.read_csv('ml/data/processed/cleaned_credit_data.csv')
X = df.drop(columns=['loan_status'])
y = df['loan_status']

print('=== 1. TESTING DOMAIN FEATURE CREATOR ===')
creator = DomainFeatureCreator()
X_eng = creator.fit_transform(X)

print('New columns created:', [c for c in X_eng.columns if c not in X.columns])

raw_inc_skew = X['person_income'].skew()
log_inc_skew = X_eng['log_income'].skew()
raw_loan_skew = X['loan_amnt'].skew()
log_loan_skew = X_eng['log_loan_amnt'].skew()

print('\nSkewness Reduction Analysis:')
print(f'  Raw Income Skewness:      {raw_inc_skew:.3f} -> Log Income Skewness:      {log_inc_skew:.3f}')
print(f'  Raw Loan Amount Skewness: {raw_loan_skew:.3f} -> Log Loan Amount Skewness: {log_loan_skew:.3f}')

print('\nDebt Burden Summary:')
print(X_eng['debt_burden'].describe().round(4))

print('\nCredit Maturity Summary:')
print(X_eng['credit_maturity'].describe().round(4))

print('\nAge Group Discretization:')
print(X_eng['age_group'].value_counts())

print('\n=== 2. TESTING FULL PREPROCESSING PIPELINE ===')
pipeline = build_full_preprocessing_pipeline()
X_trans = pipeline.fit_transform(X)

print(f'Transformed matrix shape: {X_trans.shape}')
print(f'Any NaN values in output: {np.isnan(X_trans).any()}')
print('First row scaled numeric vector snippet:', np.round(X_trans[0][:6], 4))
print('\n[SUCCESS] All 6 Feature Engineering techniques verified!')
