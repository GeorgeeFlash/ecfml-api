# ECFML API Design Document (v2.0.0)

This document outlines the API endpoints for the Electricity Consumption Forecasting (ECFML) backend. This is intended for use by the frontend development team to integrate with the forecasting system.

## General Information

- **Base URL**: `/api/v1` (unless otherwise specified)
- **Version**: 2.0.0
- **Format**: All request and response bodies are in JSON.

## Authentication

The API uses **Clerk** for authentication. All protected routes require a Bearer token in the `Authorization` header.

```http
Authorization: Bearer <clerk_jwt_token>
```

---

## 1. Datasets

Endpoints for dataset management, validation, and previewing.

### List Datasets
Retrieves all datasets for the authenticated user.

- **URL**: `GET /datasets`
- **Response**: `list[DatasetRead]`
  ```json
  [
    {
      "id": "string",
      "name": "string",
      "file_url": "string",
      "uploadthing_key": "string",
      "user_id": "string",
      "validation_status": "PENDING" | "VALID" | "INVALID" | "WARNING",
      "created_at": "string",
      "row_count": number
    }
  ]
  ```

### Create Dataset Record
Registers a new dataset (after it has been uploaded to storage).

- **URL**: `POST /datasets`
- **Request Body**: `DatasetCreate`
  ```json
  {
    "id": "string",
    "name": "string",
    "file_url": "string",
    "uploadthing_key": "string"
  }
  ```
- **Response**: `DatasetRead` (same as above)

### Delete Dataset
Soft-deletes a dataset.

- **URL**: `DELETE /datasets/{dataset_id}`
- **Response**:
  ```json
  {
    "status": "deleted"
  }
  ```

### Validate Dataset
Validates a dataset file against required columns and format.

- **URL**: `POST /datasets/{dataset_id}/validate`
- **Request Body**:
  ```json
  {
    "file_url": "string",
    "required_columns": ["string"]
  }
  ```
- **Response**: `DatasetValidationReport`
  ```json
  {
    "status": "PENDING" | "VALID" | "INVALID" | "WARNING",
    "missing_columns": ["string"],
    "columns": ["string"],
    "row_count": number,
    "warnings": ["string"],
    "details": {}
  }
  ```

### Preview Dataset
Retrieves a preview (first N rows) of the dataset.

- **URL**: `GET /datasets/{dataset_id}/preview`
- **Query Parameters**:
  - `file_url` (string, required): URL of the dataset.
  - `rows` (number, default: 100): Number of rows to return (max 500).
- **Response**: `DatasetPreviewResponse`
  ```json
  {
    "columns": ["string"],
    "rows": [{}],
    "row_count": number
  }
  ```

### List Weather Datasets
Retrieves all weather datasets for the authenticated user.

- **URL**: `GET /datasets/weather`
- **Response**: `list[WeatherDataset]`
  ```json
  [
    {
      "id": "string",
      "dataset_id": "string",
      "file_url": "string",
      "uploadthing_key": "string",
      "created_at": "string"
    }
  ]
  ```

### Create Weather Dataset Record
Registers a new weather dataset linked to a consumption dataset.

- **URL**: `POST /datasets/weather`
- **Request Body**: `WeatherDatasetCreate`
  ```json
  {
    "id": "string",
    "dataset_id": "string",
    "file_url": "string",
    "uploadthing_key": "string"
  }
  ```
- **Response**: `WeatherDataset` (same as above)

---

## 2. Preprocessing

Endpoints for managing data preprocessing jobs.

### Run Preprocessing
Starts a preprocessing job to prepare data for training.

- **URL**: `POST /preprocessing/run`
- **Request Body**:
  ```json
  {
    "job_id": "string",
    "dataset_url": "string",
    "weather_url": "string" (optional),
    "splits": {
      "train": 0.7,
      "val": 0.15,
      "test": 0.15
    } (optional)
  }
  ```
- **Response**: `PreprocessingStatusResponse`
  ```json
  {
    "job_id": "string",
    "status": "PENDING" | "RUNNING" | "COMPLETE" | "FAILED",
    "progress": number,
    "error": "string" (optional),
    "processed_file_path": "string" (optional),
    "result_summary": {},
    "eda_charts": {}
  }
  ```

### Get Preprocessing Status
Retrieves the status and result of a preprocessing job.

- **URL**: `GET /preprocessing/{job_id}/status`
- **Response**: `PreprocessingStatusResponse` (same as above)

---

## 3. Models

Endpoints for training, evaluating, and managing machine learning models.

### Train Model
Starts a training job for a specific model type.

- **URL**: `POST /models/train`
- **Request Body**:
  ```json
  {
    "job_id": "string",
    "preprocess_job_id": "string",
    "model_type": "RANDOM_FOREST" | "SVR",
    "hyperparams": {},
    "processed_file_path": "string" (optional)
  }
  ```
- **Response**: `ModelTrainResponse`
  ```json
  {
    "job_id": "string",
    "status": "PENDING" | "RUNNING" | "COMPLETE" | "FAILED",
    "model_file_path": "string",
    "scaler_file_path": "string",
    "training_time_secs": number,
    "error": "string"
  }
  ```

### Get Training Status
Retrieves the status of a training job.

- **URL**: `GET /models/jobs/{job_id}/status`
- **Response**: `ModelTrainResponse` (same as above)

### Evaluate Model
Calculates performance metrics for a trained model on a test set.

- **URL**: `POST /models/{model_id}/evaluate`
- **Request Body**:
  ```json
  {
    "model_run_id": "string",
    "model_file_path": "string",
    "processed_file_path": "string"
  }
  ```
- **Response**: `ModelEvaluateResponse`
  ```json
  {
    "rmse": number,
    "mae": number,
    "mape": number,
    "r2": number,
    "test_set_size": number,
    "actual": [{}],
    "predicted": [{}],
    "feature_importance": [{}]
  }
  ```

---

## 4. Forecast

Endpoints for generating and streaming predictions.

### Create Forecast
Generates a forecast using a trained model or the AI Agent.

- **URL**: `POST /forecast`
- **Request Body**:
  ```json
  {
    "engine": "RF" | "SVR" | "AGENT",
    "preprocess_job_id": "string",
    "model_run_id": "string" (optional),
    "start_date": "string" (YYYY-MM-DD),
    "horizon_days": number,
    "resolution": "HOURLY" | "DAILY" | "WEEKLY",
    "model_override": "string" (optional),
    "processed_file_path": "string" (optional)
  }
  ```
- **Response**: `ForecastResponse`
  ```json
  {
    "forecast_id": "string",
    "engine": "RF" | "SVR" | "AGENT",
    "predictions": [{}],
    "agent_reasoning": "string",
    "confidence": "string",
    "agent_run_id": "string"
  }
  ```

### Stream Agent Forecast (SSE)
Streams real-time updates from the AI Agent during forecast generation.

- **URL**: `GET /forecast/stream/{agent_run_id}`
- **Response**: `text/event-stream`
  - Each event is a JSON string: `data: {"type": "step", "step": "string", "details": "string"}`

### Get Agent Status
Retrieves the final result or status of an agentic run.

- **URL**: `GET /agents/{agent_run_id}/status`
- **Response**:
  ```json
  {
    "status": "PENDING" | "COMPLETE",
    "result": {} (only if COMPLETE)
  }
  ```

---

## 5. System

### Health Check
System health and version information.

- **URL**: `GET /health` (No `/api/v1` prefix)
- **Response**:
  ```json
  {
    "status": "ok",
    "version": "2.0.0"
  }
  ```
