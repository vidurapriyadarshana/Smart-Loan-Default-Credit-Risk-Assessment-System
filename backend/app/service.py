"""
Inference Service and Model Management Engine
Author: Member 4 (Backend Engineer)
Project: Smart Loan Default & Credit Risk Assessment System
Description: 
  - Implements Singleton pattern for zero-overhead in-memory model inference.
  - Generates risk tier classifications and rule-based diagnostic explanations.
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, List
from backend.app.schemas import ApplicantRequest, PredictionResponse

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "models", "loan_risk_pipeline.joblib")


class CreditRiskService:
    """Singleton ML prediction engine managing model lifecycle and inference."""
    _instance = None
    _pipeline = None

    @classmethod
    def load_pipeline(cls, model_path: str = MODEL_PATH):
        """Loads serialized pipeline once into memory at application startup."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model pipeline not found at {model_path}")
        cls._pipeline = joblib.load(model_path)
        print(f"[Backend ML Service] Pipeline successfully loaded from: {model_path}")

    @classmethod
    def is_ready(cls) -> bool:
        """Returns True if the pipeline is in memory and ready for inference."""
        return cls._pipeline is not None

    @classmethod
    def evaluate_applicant(cls, applicant: ApplicantRequest) -> PredictionResponse:
        """Evaluates a single applicant, computing probabilities, risk tier, and explanations."""
        if cls._pipeline is None:
            raise RuntimeError("Model pipeline is not loaded in memory. Call load_pipeline() first.")

        data_dict = applicant.model_dump()

        # Compute derived raw loan_percent_income if needed
        data_dict['loan_percent_income'] = round(data_dict['loan_amnt'] / (data_dict['person_income'] + 1.0), 4)

        df_input = pd.DataFrame([data_dict])

        # Run inference through Member 2 & 3's master pipeline
        prob_default = float(cls._pipeline.predict_proba(df_input)[0][1])
        prob_approval = float(round(1.0 - prob_default, 4))
        prob_default = float(round(prob_default, 4))

        # Risk tiering calibration:
        # P(default) < 0.25 -> Low Risk (Approved)
        # 0.25 <= P(default) < 0.50 -> Medium Risk (Manual Underwriter Review)
        # P(default) >= 0.50 -> High Risk (Declined)
        if prob_default < 0.25:
            status = "Approved"
            tier = "Low Risk"
        elif prob_default < 0.50:
            status = "Review Required"
            tier = "Medium Risk"
        else:
            status = "Declined"
            tier = "High Risk"

        # Diagnostic Risk Factor Explanations
        factors: List[str] = []
        dti = data_dict['loan_percent_income']

        if dti > 0.35:
            factors.append(f"Critical loan-to-income ratio ({dti:.1%}) exceeds recommended 35% safe borrowing ceiling.")
        elif dti > 0.22:
            factors.append(f"Moderate debt burden ({dti:.1%}) requires elevated discretionary cash flow.")

        if data_dict['loan_grade'] in ['E', 'F', 'G']:
            factors.append(f"Subprime credit rating (Grade '{data_dict['loan_grade']}') indicates high historical loss frequency.")
        elif data_dict['loan_grade'] in ['C', 'D']:
            factors.append(f"Fair credit rating (Grade '{data_dict['loan_grade']}') warrants manual underwriter verification.")
        else:
            factors.append(f"Prime credit rating (Grade '{data_dict['loan_grade']}') strongly supports creditworthiness.")

        if data_dict['cb_person_default_on_file'] == 'Y':
            factors.append("Historical credit bureau default / charge-off flag found on file.")

        if data_dict['loan_int_rate'] > 15.0:
            factors.append(f"High interest rate ({data_dict['loan_int_rate']}%) accelerates total interest liability.")

        if data_dict['person_home_ownership'] == 'RENT' and dti > 0.25:
            factors.append("Rental tenure combined with elevated borrowing increases default susceptibility.")

        if not factors or len(factors) == 1 and "Prime" in factors[0]:
            factors.append("Strong demographic and financial parameters indicate robust repayment capacity.")

        confidence = round(max(prob_default, prob_approval) * 100, 1)

        return PredictionResponse(
            loan_status=status,
            risk_tier=tier,
            default_probability=prob_default,
            approval_probability=prob_approval,
            confidence_score=confidence,
            debt_to_income_ratio=round(dti * 100, 2),
            risk_factors=factors
        )
