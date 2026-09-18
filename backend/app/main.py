"""
FastAPI REST API Server for Credit Risk Assessment
Author: Member 4 (Backend Engineer)
Project: Smart Loan Default & Credit Risk Assessment System
Description: Exposes high-speed asynchronous endpoints for real-time credit scoring and batch analysis.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import (
    ApplicantRequest, PredictionResponse,
    BatchApplicantRequest, BatchPredictionResponse,
    HealthCheckResponse
)
from backend.app.service import CreditRiskService

APP_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads ML model into memory on startup and cleans up on shutdown."""
    print("[FastAPI Lifespan] Initializing Credit Risk ML Inference Engine...")
    try:
        CreditRiskService.load_pipeline()
    except Exception as e:
        print(f"[FastAPI Lifespan ERROR] Could not load model pipeline: {e}")
    yield
    print("[FastAPI Lifespan] Shutting down Credit Risk Service.")


app = FastAPI(
    title="Smart Loan Default & Credit Risk Assessment API",
    description="Machine Learning REST API predicting loan default probabilities and automated risk tiering.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Frontend communication (Member 5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root():
    """Root endpoint welcoming clients and linking to Swagger UI."""
    return {
        "service": "Smart Loan Default & Credit Risk Assessment API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "online"
    }


@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["System"])
def health_check():
    """System health check and model readiness monitor."""
    uptime_seconds = int(time.time() - APP_START_TIME)
    return HealthCheckResponse(
        status="healthy" if CreditRiskService.is_ready() else "degraded",
        model_loaded=CreditRiskService.is_ready(),
        model_name="XGBoost Champion Pipeline",
        version="1.0.0",
        uptime_status=f"{uptime_seconds} seconds"
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single_applicant(applicant: ApplicantRequest):
    """
    Evaluates a single loan applicant.
    Returns default probability, approval likelihood, risk tier, and diagnostic factors.
    """
    try:
        response = CreditRiskService.evaluate_applicant(applicant)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(e)}"
        )


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_applicants(batch: BatchApplicantRequest):
    """
    Evaluates an array of up to 500 applicants simultaneously.
    Returns individual evaluations and summary counts.
    """
    predictions = []
    approved = 0
    review = 0
    declined = 0

    for applicant in batch.applicants:
        res = CreditRiskService.evaluate_applicant(applicant)
        predictions.append(res)
        if res.loan_status == "Approved":
            approved += 1
        elif res.loan_status == "Review Required":
            review += 1
        else:
            declined += 1

    return BatchPredictionResponse(
        total_processed=len(predictions),
        approved_count=approved,
        review_count=review,
        declined_count=declined,
        predictions=predictions
    )
