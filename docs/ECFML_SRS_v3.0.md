# SRS v3.0 — Hybrid ML + LangGraph Architecture
## ECFML — NW Cameroon

**UNIVERSITY OF BAMENDA**  
Faculty of Science — Department of Mathematics and Computer Science  
Academic Year 2025/2026  

# SOFTWARE REQUIREMENTS SPECIFICATION

**Version 3.0 | Hybrid ML + LangGraph Agent Architecture**  
(Random Forest · SVR · LangGraph v1.1 · FastAPI · Next.js 16 · Prisma 7 · Clerk · Recharts)

## Electricity Consumption Forecasting Using Machine Learning Techniques
### A Case Study of the North West Region of Cameroon

---

## Prepared by
**Sanda Elvis Toge (UBa22S1297)**

## Supervisor
**Prof. Fautso Gaetan**

## Level
**Level 500, Year 4**

## SRS Version
**3.0 — Hybrid Architecture (May 2026)**

## Supersedes
**SRS v2.0 (pure traditional ML stack)**

## Status
**Draft — Pending Supervisor Approval**

---

# 1. Introduction

## 1.1 Purpose

This document — SRS version 3.0 — supersedes SRS v2.0 and reflects a significant architectural revision to the Electricity Consumption Forecasting System using Machine Learning (ECFML).

Following evaluation of development complexity, timeline constraints, and academic research objectives, the system has been redesigned as a Hybrid ML + LangGraph Agent architecture.

This approach combines two classical owned ML models (Random Forest and SVR) with a LangGraph-orchestrated multi-agent pipeline that leverages state-of-the-art large language models (GPT-5.4, Gemini 3.1 Pro Preview, Claude Sonnet 4.6) for contextual forecasting, analysis, and natural language interaction.

---

## 1.2 What Changed from v2.0 and Why

| Aspect | v2.0 (Deprecated) | v3.0 (Current) |
|---|---|---|
| ML Approach | ANN + LSTM + RF + SVR | RF + SVR + LangGraph Agent |
| ANN | Required | Removed |
| LSTM | Required | Removed |
| Random Forest | Required | Retained |
| SVR | Required | Retained |
| LLM Forecasting | Low Priority | Core Feature |
| Agent Framework | None | LangGraph v1.1 |
| Academic Contribution | Compare 4 ML Models | Compare ML vs LLM-Agent Forecasting |
| Difficulty Rating | 7/10 | 4/10 |
| GPU Required | Yes | No |

---

## 1.3 Academic Framing

The research contribution of this project under the hybrid architecture is:

> A comparative study of traditional machine learning models (Random Forest, SVR) against LLM-agent-based forecasting for short- and medium-term electricity consumption prediction in the North West Region of Cameroon.

---

## 1.4 Scope

The ECFML v3.0 system will:

1. Provide a Next.js 16 web dashboard.
2. Expose a FastAPI backend.
3. Train and evaluate RF and SVR models.
4. Run a LangGraph multi-agent forecasting pipeline.
5. Compare forecasting performance across all engines.
6. Present results through Recharts dashboards.
7. Stream agent execution progress using SSE.

### Out of Scope
- ANN
- LSTM
- Real-time SCADA integration
- Billing systems
- Electricity generation planning

---

# 2. Overall Description

## 2.1 System Architecture Overview

The system consists of:

1. **Next.js 16 Frontend**
2. **FastAPI ML + Agent Backend**
3. **Neon Postgres Database**

The LangGraph agent pipeline runs entirely inside the FastAPI backend.

---

## 2.2 The Three Forecasting Engines

| Engine | Type | Training Required | Key Characteristic |
|---|---|---|---|
| Random Forest | Traditional ML | Yes | Explainable |
| SVR | Traditional ML | Yes | Strong baseline |
| LangGraph Agent | LLM-Based | No | Contextual reasoning |

---

## 2.3 LangGraph Agent Architecture

### Nodes
1. `data_preparation_node`
2. `forecasting_node`
3. `validation_node`
4. `revision_node`

### Features
- Structured forecasting
- Validation loops
- Conditional revision cycles
- SSE streaming
- MemorySaver checkpointing

---

# 3. Functional Requirements

## 3.1 Authentication

| ID | Name | Description |
|---|---|---|
| FR-AU-01 | Sign-Up / Sign-In | Clerk-powered authentication |
| FR-AU-02 | Route Protection | Protect dashboard routes |
| FR-AU-03 | JWT Forwarding | JWT forwarded to FastAPI |
| FR-AU-04 | FastAPI JWT Validation | Validate JWT using Clerk JWKS |

---

## 3.2 Data Management

| ID | Name | Description |
|---|---|---|
| FR-DM-01 | Dataset Upload | Upload CSV/Excel datasets |
| FR-DM-02 | Weather Upload | Upload weather datasets |
| FR-DM-03 | Dataset Validation | Validate required columns |
| FR-DM-04 | Dataset Preview | Preview uploaded data |
| FR-DM-05 | Dataset Management | List and soft-delete datasets |

---

## 3.3 Data Pre-processing

Key features include:

- Missing value imputation
- Outlier detection
- Feature engineering
- Normalisation
- Train/validation/test splitting
- EDA report generation

---

## 3.4 Traditional ML Training

### Supported Models
- Random Forest
- Support Vector Regression (SVR)

### Features
- Background training tasks
- Model persistence
- Hyperparameter tuning

---

## 3.5 LangGraph Agent Pipeline

### Features
- Graph compilation at startup
- Structured forecasting prompts
- Validation and revision loops
- SSE progress streaming
- Structured JSON output

---

## 3.6 Model Evaluation

Metrics computed:

- RMSE
- MAE
- MAPE
- R²

Visualisations:
- Comparison tables
- Actual vs Predicted charts
- RF Feature Importance charts

---

## 3.7 Forecasting Module

### Features
- Engine selection
- Short-term forecasting
- Medium-term forecasting
- Forecast export
- Live SSE updates

---

# 4. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Performance | Forecast generation under required thresholds |
| Accuracy | Best ML model should achieve MAPE < 10% |
| Reliability | Retry handling for LLM API failures |
| Security | JWT-protected endpoints and hidden API keys |
| Maintainability | Pure LangGraph node functions |
| Cost | Max 8,000 tokens per forecast request |

---

# 5. Use Cases

## UC-01: Authenticate and Access Dashboard

### Flow
1. User visits dashboard
2. Redirected to sign-in
3. Clerk authenticates user
4. JWT issued
5. Dashboard access granted

---

## UC-02: Upload and Preprocess Dataset

### Flow
1. Upload dataset
2. Validate columns
3. Run preprocessing
4. Generate EDA charts

---

## UC-03: Train RF and SVR Models

### Flow
1. Configure model
2. Start training
3. Save trained model
4. Persist results

---

## UC-04: Run LangGraph Agent Forecast

### Flow
1. Select LangGraph Agent
2. Start forecast
3. Receive SSE updates
4. Validate predictions
5. Render charts and reasoning

---

## UC-05: Compare Forecasting Engines

### Flow
1. View evaluation page
2. Compare RF, SVR, and Agent
3. Highlight best-performing engine
4. Visualize results

---

# 6. Data Requirements

## Input Datasets

| Dataset | Source | Format |
|---|---|---|
| Electricity Consumption | ENEO Cameroon | CSV / Excel |
| Temperature | ERA5 | CSV |
| Humidity | ERA5 | CSV |
| Rainfall | ERA5 | CSV |
| Public Holidays | Government Calendar | CSV |

---

# 7. System Constraints and Assumptions

## Constraints
- LangGraph >= 1.1.0
- Python >= 3.11
- ANN and LSTM are out of scope
- CPU-only execution

## Assumptions
- Historical ENEO data available
- Sufficient LLM API quotas
- LangSmith free tier sufficient

---

# 8. Appendices

## 8.1 Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 |
| Styling | Tailwind CSS + shadcn/ui |
| Charts | Recharts |
| Auth | Clerk |
| ORM | Prisma 7 |
| Database | Neon Postgres |
| ML Backend | FastAPI |
| Traditional ML | scikit-learn |
| Agent Framework | LangGraph v1.1 |
| Deployment | Vercel + Render |

---

## 8.2 Agile Sprint Plan

| Sprint | Scope |
|---|---|
| Sprint 1 | Project scaffolding |
| Sprint 2 | Data module |
| Sprint 3 | Preprocessing |
| Sprint 4 | RF & SVR training |
| Sprint 5 | LangGraph agent |
| Sprint 6 | SSE streaming |
| Sprint 7 | Evaluation dashboard |
| Sprint 8 | Optimization & write-up |

---

## 8.3 Revision History

| Version | Description |
|---|---|
| 1.0 | Flask + Chart.js |
| 2.0 | Next.js + FastAPI + ANN/LSTM |
| 3.0 | Hybrid ML + LangGraph Agent |

---

# End of SRS v3.0 — ECFML Hybrid Architecture
University of Bamenda — Faculty of Science — 2025/2026
