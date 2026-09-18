# Smart Loan Default & Credit Risk Assessment System
**End-to-End Machine Learning Development & Full-Stack Application**

---

## 📌 1. Project Overview & Problem Statement
Evaluating credit risk is a critical task for financial institutions. Traditional loan approval processes rely on manual reviews or rigid rule sets that fail to detect non-linear insolvency patterns, suffer from high default rates, or produce excessive false rejections.

This project implements an end-to-end **Smart Loan Default & Credit Risk Assessment System**. It takes raw applicant demographic and financial metrics, processes them through an automated feature engineering pipeline, predicts the likelihood of loan default ($P(\text{Default})$), categorizes applicants into risk tiers (`Low Risk`, `Medium Risk`, `High Risk`), and delivers actionable, explainable diagnostic risk factors to underwriters.

The solution integrates a production-trained **XGBoost machine learning pipeline** with an asynchronous **FastAPI REST API** and an interactive **Vite + React + TailwindCSS** web application.

---

## 📊 2. Dataset Information & Data Quality Audit

### 2.1 Dataset Source
* **Origin**: Credit Risk Dataset (sourced from Kaggle and Lending Club historical loan archives).
* **Location**: `ml/data/raw/credit_risk_dataset.csv`
* **Raw Dimensions**: 32,581 records $\times$ 12 columns (11 predictive features + 1 binary target).
* **Cleaned Dimensions**: 30,786 records $\times$ 12 columns (exported to `ml/data/processed/cleaned_credit_data.csv`).

### 2.2 Feature Definitions & Schema

| Column Name | Data Type | Type Category | Missing Rate | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`person_age`** | `int64` | Continuous | 0.0% | 20 to 85 yrs | Age of the loan applicant in years (filtered anomalies > 85). |
| **`person_income`** | `int64` | Continuous | 0.0% | $4,000 to $6,000,000 | Gross annual income in USD (heavily right-skewed). |
| **`person_home_ownership`** | `object` | Categorical | 0.0% | `RENT`, `OWN`, `MORTGAGE`, `OTHER` | Housing tenure status of the applicant. |
| **`person_emp_length`** | `float64` | Continuous | 2.75% (895 rows) | 0.0 to 45.0 yrs | Length of continuous employment history in years. |
| **`loan_intent`** | `object` | Categorical | 0.0% | 6 unique purposes | Stated purpose (`PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`). |
| **`loan_grade`** | `object` | Ordinal | 0.0% | `A` to `G` (7 grades) | Credit rating tier calibrated to risk frequency. |
| **`loan_amnt`** | `int64` | Continuous | 0.0% | $500 to $35,000 | Principal loan amount requested in USD. |
| **`loan_int_rate`** | `float64` | Continuous | 9.56% (3,116 rows)| 5.42% to 23.22% | Annual loan interest rate percentage. |
| **`loan_percent_income`** | `float64` | Ratio | 0.0% | 0.00 to 0.83 | Requested loan amount relative to gross annual income. |
| **`cb_person_default_on_file`** | `object` | Binary Flag | 0.0% | `Y` or `N` | Historical default/charge-off recorded with credit bureaus. |
| **`cb_person_cred_hist_length`** | `int64` | Discrete | 0.0% | 2 to 30 yrs | Length of active credit history registered with bureaus. |
| **`loan_status` (Target)** | `int64` | Binary Target | 0.0% | `0` or `1` | **0 = Non-Default / Approved (78.2%)**<br>**1 = Default / Charged-off (21.8%)** |

### 2.3 Data Cleaning & Quality Audit Summary
1. **Deduplication**: Identified and eliminated **165 duplicate records** to prevent artificial cross-validation bias.
2. **Biological Anomaly Filtering**: Removed records with input anomalies (e.g., recorded ages $> 100$, employment length exceeding legal working life $\text{emp\_length} > \text{age} - 16$).
3. **Domain-Informed Imputation**:
   * Missing `person_emp_length` (~2.75%) imputed using **median employment length grouped by applicant age brackets**.
   * Missing `loan_int_rate` (~9.56%) imputed using **median interest rate grouped by `loan_grade`** (preserving the risk-based pricing curve).

---

## ⚙️ 3. Feature Engineering (The 6 Mandatory Techniques)

To satisfy the core machine learning requirements, 6 distinct feature engineering techniques were designed and implemented:

1. **Technique 1 — Debt Burden Ratio (Domain Interaction)**:
   $$\text{debt\_burden} = \frac{\text{loan\_amnt} \times \left(1 + \frac{\text{loan\_int\_rate}}{100}\right)}{\text{person\_income} + 1}$$
   *Captures the total debt repayment liability relative to income rather than just the raw loan principal.*
2. **Technique 2 — Credit History Maturity Ratio (Temporal Ratio)**:
   $$\text{credit\_maturity} = \frac{\text{cb\_person\_cred\_hist\_length}}{\max(\text{person\_age} - 17, 1)}$$
   *Measures the proportion of an applicant's adult life spent actively managing formal credit.*
3. **Technique 3 — Non-Linear Log Transformations ($\log(1+x)$)**:
   * Applied to heavily right-skewed variables: `person_income` and `loan_amnt`.
   * **Result**: Reduced `person_income` skewness from **+9.892 down to +0.183** (approximating a normal Gaussian distribution).
4. **Technique 4 — IQR Winsorization (Outlier Boundary Clipping)**:
   * Calculates $Q1 - 1.5 \times \text{IQR}$ and $Q3 + 1.5 \times \text{IQR}$ **strictly on the training split** and caps extreme tail noise without discarding valid production inference requests.
5. **Technique 5 — Life-Stage Age Discretization (Binning)**:
   * Discretizes continuous age into risk-segmented career stages: `18-25` (Entry), `26-35` (Early Career), `36-50` (Mid Career), and `50+` (Mature).
6. **Technique 6 — Dual Categorical Encoding & Feature Scaling**:
   * **Ordinal Encoding**: For `loan_grade` (`A: 0` through `G: 6`), preserving the monotonic default rate relationship.
   * **One-Hot Encoding**: For nominal categories (`person_home_ownership`, `loan_intent`, `cb_person_default_on_file`, `age_group`) with `drop='first'` to prevent multicollinearity.
   * **StandardScaler**: Normalizes the final 22-dimensional feature space to zero mean and unit variance.

---

## 🧠 4. Model Training, Evaluation & Benchmarking

### 4.1 Candidate Model Benchmarking (Held-Out Test Set: 6,158 Rows)

| Model Name | Accuracy | Precision (Class 1) | Recall (Default - Class 1) | F1-Score | ROC-AUC | Primary Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (L2)** | 86.72% | 76.49% | 55.71% | 0.6447 | 0.8835 | Baseline (high false negatives) |
| **Random Forest (Balanced)** | **91.70%** | **82.61%** | 78.08% | **0.8028** | 0.9360 | Strong ensemble model |
| **XGBoost Classifier (Champion)** | 91.21% | 79.23% | **80.48%** | 0.7985 | **0.9472** | 🏆 **Champion Model (Highest Default Recall & ROC-AUC)** |

### 4.2 Class Imbalance & Metric Strategy
* **Imbalance Ratio**: Approximately **3.62 : 1** (78.2% non-default vs. 21.8% default).
* **Cost Asymmetry**: A False Negative (approving a borrower who defaults) loses ~100% of the loan principal, whereas a False Positive (rejecting a safe borrower) costs only foregone interest (~8–10%).
* **Optimization**: XGBoost was configured with `scale_pos_weight = 3.62` to heavily penalize False Negatives, capturing **80.48% of all defaulting borrowers** in the unseen test set.
* **Stratified 5-Fold Cross-Validation**: Achieved a mean ROC-AUC of **0.9453** ($\sigma = 0.0032$), demonstrating robust generalization across folds.

### 4.3 Feature Importance Breakdown
Analysis of Gini gain feature importances in the champion model:
1. `loan_grade`: 17.96%
2. `debt_burden` (Engineered Feature): **12.80%** *(#2 most predictive feature overall)*
3. `person_home_ownership_RENT`: 11.01%
4. `loan_percent_income`: 7.07%
5. `loan_intent_HOMEIMPROVEMENT`: 6.11%

---

## 🏗️ 5. Full-Stack Application Architecture

```
[ User / Loan Underwriter ]
         │
         ▼
[ Frontend Web Application ]
  • Tech: Vite, React 18, TailwindCSS, Lucide Icons
  • URL: http://localhost:3000
  • Features: Real-time range sliders, live Debt-to-Income (DTI) meter,
              animated circular SVG risk score gauge, demo quick-fill profiles
         │
         │ HTTP POST /api/v1/predict (JSON Payload)
         ▼
[ Backend REST API Service ]
  • Tech: FastAPI, Uvicorn, Pydantic V2
  • URL: http://localhost:8000 (Swagger UI at /docs)
  • Features: Pydantic validation firewall, CORS middleware,
              singleton in-memory model loading (sub-15ms latency), batch inference
         │
         │ Feature Vector (1x22 numerical dimensions)
         ▼
[ Production ML Pipeline Bundle ]
  • File: ml/models/loan_risk_pipeline.joblib (523 KB)
  • Pipeline: Custom Feature Transformers -> ColumnTransformer -> Tuned XGBoost
         │
         ▼
[ Decision Engine: Risk Tier (Low/Medium/High) + Explainable Factors ]
```

---

## 📁 6. Project Directory Structure

```text
Machine Learning Module - Group Project Assignment/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI server & route handlers
│   │   ├── schemas.py                 # Pydantic V2 request & response contracts
│   │   └── service.py                 # Singleton ML inference & risk tiering
│   ├── tests/
│   │   └── test_api.py                # Automated pytest integration test suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── MetricCard.jsx         # Metric display pill component
│   │   │   ├── RiskFactorsList.jsx    # Explainable AI risk factors list
│   │   │   └── RiskGauge.jsx          # Animated SVG circular risk gauge
│   │   ├── App.jsx                    # Root React dashboard & form simulator
│   │   ├── index.css                  # TailwindCSS directives & custom styles
│   │   └── main.jsx                   # React 18 entrypoint
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   └── vite.config.js
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   │   └── credit_risk_dataset.csv       # Raw sourced dataset (1.8 MB)
│   │   └── processed/
│   │       ├── cleaned_credit_data.csv       # Cleaned dataset (30,786 rows)
│   │       ├── train_data.csv                # 80% train split (24,628 rows)
│   │       └── test_data.csv                 # 20% test split (6,158 rows)
│   ├── models/
│   │   ├── preprocessor.joblib               # Fitted feature engineering pipeline
│   │   └── loan_risk_pipeline.joblib         # Serialized production pipeline (523 KB)
│   ├── notebooks/
│   │   ├── 01_eda_and_data_cleaning.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   └── 03_model_training_and_tuning.ipynb
│   ├── reports/
│   │   └── figures/                          # Visual evaluation figures (ROC, CM, etc.)
│   └── src/
│       ├── data_loader.py                    # Data cleaning & anomaly filtering
│       ├── export_preprocessor.py            # Preprocessor fitting & export
│       ├── features.py                       # The 6 feature engineering transformers
│       ├── generate_eda_plots.py             # EDA visualization generator
│       ├── test_features.py                  # Feature pipeline assertion tests
│       ├── train.py                          # Multi-model benchmarking script
│       └── tune_and_export_champion.py       # Cross-validation & model serialization
├── .gitignore
├── Assignment.md
└── README.md
```

---

## 🚀 7. Setup & Execution Instructions

### Prerequisites
* **Python**: 3.10 or higher
* **Node.js**: v18.0 or higher (with npm)

---

### Step 1: Install Python Dependencies
From the project root:
```bash
pip install -r backend/requirements.txt
pip install matplotlib seaborn
```

---

### Step 2: Start the Backend REST API
Run the FastAPI application server:
```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```
* **API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

### Step 3: Start the Frontend Web Application
In a separate terminal, navigate to the `frontend/` directory:
```bash
cd frontend
npm install
npm run dev
```
* **Frontend Portal**: Open [http://localhost:3000](http://localhost:3000) in your web browser.

---

### Step 4: Run Automated Verification Tests
```bash
# 1. Run Backend Integration Test Suite (6 tests, 100% Passed)
python -m pytest backend/tests/test_api.py -v

# 2. Run Feature Engineering Assertion Tests
python -m ml.src.test_features
```

---

## 📡 8. REST API Reference

### Single Applicant Prediction Endpoint
* **Method**: `POST`
* **Route**: `/api/v1/predict`
* **Request Payload Example**:
```json
{
  "person_age": 28,
  "person_income": 65000.0,
  "person_home_ownership": "RENT",
  "person_emp_length": 4.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000.0,
  "loan_int_rate": 11.2,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

* **Response Payload Example**:
```json
{
  "loan_status": "Approved",
  "risk_tier": "Low Risk",
  "default_probability": 0.0964,
  "approval_probability": 0.9036,
  "confidence_score": 90.4,
  "debt_to_income_ratio": 15.38,
  "risk_factors": [
    "Prime credit rating (Grade 'B') strongly supports creditworthiness."
  ]
}
```
