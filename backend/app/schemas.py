"""
Pydantic Validation Schemas for Credit Risk API
Author: Member 4 (Backend Engineer)
Project: Smart Loan Default & Credit Risk Assessment System
Description: Defines strict boundary contracts for single and batch applicant evaluations.
"""

from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class ApplicantRequest(BaseModel):
    """Input contract for a single loan applicant."""
    person_age: int = Field(
        ..., ge=18, le=100,
        description="Applicant age in years (18 to 100)",
        examples=[28]
    )
    person_income: float = Field(
        ..., gt=0.0,
        description="Gross annual income in USD",
        examples=[65000.0]
    )
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = Field(
        ...,
        description="Housing tenure status",
        examples=["RENT"]
    )
    person_emp_length: float = Field(
        ..., ge=0.0, le=60.0,
        description="Employment duration in years",
        examples=[4.0]
    )
    loan_intent: Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"] = Field(
        ...,
        description="Stated purpose of the loan",
        examples=["PERSONAL"]
    )
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(
        ...,
        description="Assigned credit risk tier",
        examples=["B"]
    )
    loan_amnt: float = Field(
        ..., ge=500.0, le=1000000.0,
        description="Requested loan principal amount in USD",
        examples=[10000.0]
    )
    loan_int_rate: float = Field(
        ..., ge=1.0, le=35.0,
        description="Annual interest rate percentage",
        examples=[11.2]
    )
    cb_person_default_on_file: Literal["Y", "N"] = Field(
        ...,
        description="Historical default record with credit bureaus",
        examples=["N"]
    )
    cb_person_cred_hist_length: int = Field(
        ..., ge=0, le=60,
        description="Length of active credit history in years",
        examples=[5]
    )


class PredictionResponse(BaseModel):
    """Output prediction and diagnostic explanation schema."""
    loan_status: Literal["Approved", "Review Required", "Declined"]
    risk_tier: Literal["Low Risk", "Medium Risk", "High Risk"]
    default_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted likelihood of default")
    approval_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted likelihood of safe repayment")
    confidence_score: float = Field(..., description="Normalized confidence metric")
    debt_to_income_ratio: float = Field(..., description="Computed raw debt-to-income metric")
    risk_factors: List[str] = Field(..., description="Actionable explanatory factors driving the decision")


class BatchApplicantRequest(BaseModel):
    """Batch applicant container schema."""
    applicants: List[ApplicantRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Batch evaluation response schema."""
    total_processed: int
    approved_count: int
    review_count: int
    declined_count: int
    predictions: List[PredictionResponse]


class HealthCheckResponse(BaseModel):
    """System health check and readiness response."""
    status: str
    model_loaded: bool
    model_name: str
    version: str
    uptime_status: str
