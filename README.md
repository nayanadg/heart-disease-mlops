# Heart Disease Classification – End-to-End MLOps Pipeline

## 1. Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for Heart Disease Classification using the UCI Heart Disease dataset.

The project covers the complete ML lifecycle:

- Data acquisition and exploratory data analysis
- Data preprocessing and feature engineering
- Model training and comparison
- Hyperparameter tuning and cross-validation
- Experiment tracking using MLflow
- Model packaging and reproducible inference
- Automated testing using Pytest
- CI/CD using GitHub Actions
- REST API development using FastAPI
- Docker containerization
- Kubernetes deployment
- Monitoring using Prometheus metrics
- API request logging

The final selected model is a **Logistic Regression classifier**, packaged together with its preprocessing pipeline.

---

## 2. Dataset

The project uses the **Heart Disease UCI dataset**.

Dataset characteristics:

- Total records: **303**
- Input features: **13**
- Target variable: Heart Disease
- Binary classification:
  - `0` – No Heart Disease
  - `1` – Heart Disease

The original target values were converted into a binary classification problem:

- Original target `0` → No Heart Disease
- Original targets `1–4` → Heart Disease

After conversion:

| Class | Samples | Percentage |
|---|---:|---:|
| No Heart Disease | 164 | 54.13% |
| Heart Disease | 139 | 45.87% |

Missing values were identified in:

- `ca`: 4 missing values
- `thal`: 2 missing values

Missing-value handling is performed inside the Scikit-learn preprocessing pipeline to avoid data leakage.

---

## 3. Exploratory Data Analysis

EDA is implemented in:

```text
src/eda.py
```

The analysis includes:

- Dataset structure and statistics
- Missing-value analysis
- Duplicate detection
- Class distribution
- Feature histograms
- Age distribution
- Cholesterol distribution
- Resting blood pressure analysis
- Chest pain analysis
- Maximum heart-rate analysis
- Correlation heatmap
- Feature-to-target correlations

Generated EDA artifacts are stored under:

```text
artifacts/eda/
```

The cleaned dataset is stored at:

```text
data/processed/heart_disease_cleaned.csv
```

---

## 4. Preprocessing and Feature Engineering

A Scikit-learn `ColumnTransformer` is used to create a reproducible preprocessing pipeline.

### Numerical Features

- age
- trestbps
- chol
- thalach
- oldpeak

Numerical preprocessing:

```text
Median Imputation → StandardScaler
```

### Categorical Features

- sex
- cp
- fbs
- restecg
- exang
- slope
- ca
- thal

Categorical preprocessing:

```text
Most-Frequent Imputation → OneHotEncoder
```

The preprocessing pipeline is packaged together with the final classifier, ensuring identical preprocessing during training and inference.

---

## 5. Model Development

Three classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The dataset was split using an **80/20 stratified train-test split**.

Model evaluation used **5-fold Stratified Cross-Validation**.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Hyperparameter tuning was performed for Logistic Regression and Random Forest.

---

## 6. Model Performance

### Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8471 | 0.8768 | 0.7834 | 0.8245 | **0.9025** |
| Random Forest | 0.8139 | 0.8167 | 0.7735 | 0.7894 | 0.8990 |
| Gradient Boosting | 0.8017 | 0.7789 | 0.7925 | 0.7844 | 0.8567 |

### Test Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.8852** | **0.8387** | **0.9286** | **0.8814** | **0.9665** |
| Random Forest | 0.8689 | 0.8333 | 0.8929 | 0.8621 | 0.9448 |
| Gradient Boosting | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9470 |

### Selected Model

**Logistic Regression**

Final Test Accuracy:

```text
88.52%
```

Final Test ROC-AUC:

```text
96.65%
```

The complete trained pipeline is saved as:

```text
models/best_model.joblib
```

Model metadata is stored in:

```text
models/model_metadata.json
```

---

## 7. MLflow Experiment Tracking

MLflow is used to track model experiments.

Experiment name:

```text
Heart-Disease-Classification
```

For each model, MLflow records:

- Model parameters
- Cross-validation metrics
- Test metrics
- Classification reports
- Confusion matrices
- ROC curves
- Trained model artifacts

Start the MLflow UI using:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 8. Automated Testing

Automated tests are implemented using **Pytest**.

Tests cover:

- Dataset availability
- Dataset structure
- Target validation
- Duplicate validation
- Preprocessing
- Missing-value handling
- Model loading
- Prediction output
- Prediction probabilities
- FastAPI endpoints
- API input validation

Run all tests using:

```bash
pytest tests/ -v
```

The project currently contains **20 automated tests**.

---

## 9. CI/CD Pipeline

GitHub Actions is used for Continuous Integration.

Workflow:

```text
.github/workflows/ml_pipeline.yml
```

The pipeline automatically performs:

```text
Source Code
    ↓
Install Dependencies
    ↓
Flake8 Linting
    ↓
Pytest Unit Tests
    ↓
Model Training
    ↓
Inference Verification
    ↓
Upload Model & Evaluation Artifacts
```

The workflow runs automatically on pushes and pull requests to the main branch.

---

## 10. FastAPI Prediction Service

The trained model is exposed through a REST API using **FastAPI**.

API implementation:

```text
api/main.py
```

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Model prediction |
| `/metrics` | GET | Prometheus metrics |

Start the API locally:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Example Prediction Request

```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```

Example response:

```json
{
  "prediction": 0,
  "prediction_label": "No Heart Disease",
  "heart_disease_probability": 0.1655,
  "confidence": 0.8345,
  "confidence_percentage": 83.45
}
```

This API is intended for educational demonstration and is not a medical diagnostic system.

---

## 11. Docker Containerization

The FastAPI application is containerized using Docker.

Build the image:

```bash
docker build -t heart-disease-api:1.0 .
```

Run the container:

```bash
docker run --name heart-disease-api \
  -p 8000:8000 \
  heart-disease-api:1.0
```

Test the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

The model environment uses the same pinned Scikit-learn version used to serialize the trained model, ensuring reproducible model loading.

---

## 12. Kubernetes Deployment

The Dockerized application is deployed to a local Kubernetes cluster using Docker Desktop Kubernetes.

Kubernetes manifests:

```text
deployment/k8s-deployment.yaml
deployment/k8s-service.yaml
```

The deployment uses:

- **2 API replicas**
- Readiness probe
- Liveness probe
- CPU and memory resource requests/limits
- LoadBalancer service

Deploy:

```bash
kubectl apply -f deployment/k8s-deployment.yaml
kubectl apply -f deployment/k8s-service.yaml
```

Check deployment:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

The expected deployment state is:

```text
Deployment: 2/2 Ready
Pod 1:      1/1 Running
Pod 2:      1/1 Running
Service:    LoadBalancer
```

---

## 13. Monitoring and Logging

Prometheus-compatible metrics are exposed through:

```text
/metrics
```

Custom metrics include:

### API Request Count

```text
api_requests_total
```

### API Request Latency

```text
api_request_latency_seconds
```

### Model Prediction Count

```text
model_predictions_total
```

API requests are also logged with:

- HTTP method
- Endpoint
- HTTP status
- Request duration

Example:

```text
POST /predict - Status: 200 - Duration: 0.0085s
```

Monitoring was verified both locally and through the Kubernetes-deployed application.

---

## 14. Project Structure

```text
heart-disease-mlops/
│
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── artifacts/
│   ├── eda/
│   └── modeling/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── deployment/
│   ├── k8s-deployment.yaml
│   └── k8s-service.yaml
│
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
│
├── screenshots/
│
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── train.py
│   ├── train_mlflow.py
│   └── predict.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

---

## 15. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/nayanadg/heart-disease-mlops/
cd heart-disease-mlops
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 16. Running the Pipeline

### Data Acquisition and EDA

```bash
python src/eda.py
```

### Train Models

```bash
python src/train.py
```

### MLflow Experiments

```bash
python src/train_mlflow.py
```

### Standalone Inference

```bash
python src/predict.py
```

### Run Tests

```bash
pytest tests/ -v
```

### Run API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

---

## 17. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- MLflow
- Joblib
- FastAPI
- Uvicorn
- Pytest
- Flake8
- GitHub Actions
- Docker
- Kubernetes
- Prometheus Client

---

## 18. MLOps Workflow Summary

The complete workflow implemented in this project is:

```text
UCI Heart Disease Dataset
          ↓
Data Acquisition & EDA
          ↓
Preprocessing Pipeline
          ↓
Model Training & Cross-Validation
          ↓
Hyperparameter Tuning
          ↓
MLflow Experiment Tracking
          ↓
Best Model Selection
          ↓
Model Packaging
          ↓
Automated Testing
          ↓
GitHub Actions CI
          ↓
FastAPI Model Serving
          ↓
Docker Containerization
          ↓
Kubernetes Deployment
          ↓
Prometheus Metrics & API Logging
```

---

## 19. Evidence

Screenshots demonstrating the MLOps pipeline are available in the:

```text
screenshots/
```

directory.

They include evidence of:

- MLflow experiment tracking
- Automated test execution
- Successful GitHub Actions pipeline
- FastAPI Swagger documentation
- API prediction
- Docker image/container execution
- Kubernetes deployment
- Kubernetes prediction
- Prometheus metrics
- API request logging
- Monitoring from Kubernetes

---

## 20. Reproducibility

The project provides:

- Pinned core dependencies
- Complete preprocessing pipeline
- Saved model artifact
- Training scripts
- Automated tests
- Docker environment
- Kubernetes manifests
- CI pipeline
- Experiment tracking

These components allow the ML workflow to be reproduced from data preparation through model serving and deployment.
