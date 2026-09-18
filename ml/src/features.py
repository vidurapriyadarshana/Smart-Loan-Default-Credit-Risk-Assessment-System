"""
Feature Engineering & Transformation Module
Author: Member 2 (Feature Engineer)
Project: Smart Loan Default & Credit Risk Assessment System
Description: Implements the 6 mandatory feature engineering techniques:
  1. Debt Burden Ratio (Interaction)
  2. Credit History Maturity Ratio (Interaction)
  3. Non-Linear Log Transformations (np.log1p)
  4. IQR Winsorization Outlier Treatment (Custom Transformer)
  5. Age Discretization / Binning
  6. Dual Categorical Encoding (Ordinal for loan_grade, One-Hot for nominals)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DomainFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates financial interaction and domain features:
    - Debt Burden Ratio: (loan_amnt * (1 + loan_int_rate / 100)) / (person_income + 1)
    - Credit Maturity Ratio: cb_person_cred_hist_length / (person_age - 17)
    - Age Group Binning: Categorizes age into 4 life stages.
    - Log Transforms: log1p(person_income), log1p(loan_amnt)
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, np.ndarray):
            raise ValueError("DomainFeatureCreator expects a pandas DataFrame with named columns.")

        # 1. Technique 1: Debt Burden Ratio (Total repayable relative to income)
        X['debt_burden'] = (X['loan_amnt'] * (1.0 + X['loan_int_rate'] / 100.0)) / (X['person_income'] + 1.0)

        # 2. Technique 2: Credit Maturity Ratio (Proportion of adult life with credit)
        adult_years = np.maximum(X['person_age'] - 17, 1)
        X['credit_maturity'] = (X['cb_person_cred_hist_length'] / adult_years).clip(0.0, 1.0)

        # 3. Technique 3: Log Transformation on skewed financials
        X['log_income'] = np.log1p(np.maximum(X['person_income'], 0.0))
        X['log_loan_amnt'] = np.log1p(np.maximum(X['loan_amnt'], 0.0))

        # 4. Technique 5: Age Discretization
        X['age_group'] = pd.cut(
            X['person_age'],
            bins=[17, 25, 35, 50, 100],
            labels=['18-25', '26-35', '36-50', '50+']
        ).astype(str)

        return X


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """
    Technique 4: Outlier Treatment via IQR Capping / Winsorization.
    Calculates Q1, Q3, and IQR on the training set, clipping extreme tails:
      Lower = Q1 - 1.5 * IQR
      Upper = Q3 + 1.5 * IQR
    Guarantees no data leakage by fitting bounds strictly on training data.
    """
    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = X.copy()
        cols = self.columns if self.columns is not None else X_df.select_dtypes(include=[np.number]).columns
        for col in cols:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_df = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=lower, upper=upper)
        return X_df


def build_full_preprocessing_pipeline() -> Pipeline:
    """
    Constructs the end-to-end ColumnTransformer and Feature Pipeline:
    - Custom domain interactions and log transforms
    - Outlier capping via IQR
    - Ordinal encoding on hierarchical loan_grade (A -> 0, G -> 6)
    - One-hot encoding on nominal features (home ownership, loan intent, default on file, age group)
    - Standard scaling on numerical vectors
    """
    numeric_features = [
        'log_income', 'log_loan_amnt', 'person_emp_length',
        'loan_int_rate', 'loan_percent_income', 'debt_burden',
        'credit_maturity', 'person_age', 'cb_person_cred_hist_length'
    ]

    grade_categories = [['A', 'B', 'C', 'D', 'E', 'F', 'G']]
    nominal_features = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file', 'age_group']

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('winsorizer', IQRWinsorizer(columns=numeric_features)),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('ord', OrdinalEncoder(categories=grade_categories), ['loan_grade']),
            ('nom', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), nominal_features)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline([
        ('domain_features', DomainFeatureCreator()),
        ('preprocessor', column_transformer)
    ])

    return full_pipeline
