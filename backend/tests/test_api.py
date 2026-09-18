"""
Automated Integration Test Suite for Credit Risk FastAPI Server
Author: Member 4 (Backend Engineer)
Project: Smart Loan Default & Credit Risk Assessment System
Description: Validates endpoint response codes, Pydantic boundary contracts, and inference accuracy.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.service import CreditRiskService

# Initialize TestClient with lifespan context
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure pipeline is loaded for tests."""
    CreditRiskService.load_pipeline()


def test_root_endpoint():
    """Verify root endpoint returns HTTP 200 and system metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["documentation"] == "/docs"


def test_health_check_endpoint():
    """Verify health check reports model loaded and healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "XGBoost Champion Pipeline"


def test_predict_prime_applicant_approved():
    """Verify prime creditworthy applicant receives Low Risk / Approved."""
    payload = {
        "person_age": 35,
        "person_income": 95000.0,
        "person_home_ownership": "OWN",
        "person_emp_length": 8.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "A",
        "loan_amnt": 5000.0,
        "loan_int_rate": 7.5,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 10
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["loan_status"] == "Approved"
    assert data["risk_tier"] == "Low Risk"
    assert data["default_probability"] < 0.25
    assert data["approval_probability"] > 0.75
    assert len(data["risk_factors"]) > 0


def test_predict_distressed_applicant_declined():
    """Verify high-risk distressed applicant receives High Risk / Declined."""
    payload = {
        "person_age": 21,
        "person_income": 18000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 0.5,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "F",
        "loan_amnt": 28000.0,
        "loan_int_rate": 21.5,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["loan_status"] == "Declined"
    assert data["risk_tier"] == "High Risk"
    assert data["default_probability"] >= 0.50
    assert any("subprime" in f.lower() or "debt-to-income" in f.lower() or "default" in f.lower() for f in data["risk_factors"])


def test_validation_error_on_invalid_age():
    """Verify age under 18 is intercepted with HTTP 422."""
    payload = {
        "person_age": 14,  # Under 18 boundary
        "person_income": 50000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 2.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "loan_amnt": 5000.0,
        "loan_int_rate": 10.0,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 2
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_batch_prediction_endpoint():
    """Verify batch endpoint evaluates multiple applicants."""
    batch_payload = {
        "applicants": [
            {
                "person_age": 30,
                "person_income": 70000.0,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": 5.0,
                "loan_intent": "HOMEIMPROVEMENT",
                "loan_grade": "A",
                "loan_amnt": 8000.0,
                "loan_int_rate": 7.9,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6
            },
            {
                "person_age": 22,
                "person_income": 20000.0,
                "person_home_ownership": "RENT",
                "person_emp_length": 1.0,
                "loan_intent": "VENTURE",
                "loan_grade": "E",
                "loan_amnt": 18000.0,
                "loan_int_rate": 18.2,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 2
            }
        ]
    }
    response = client.post("/api/v1/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
