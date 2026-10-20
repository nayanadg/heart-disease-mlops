"""FastAPI service for Heart Disease prediction."""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import logging
import time
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


MODEL_PATH = Path("models/best_model.joblib")

model = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["prediction"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the API starts."""
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run python src/train.py first."
        )

    model = joblib.load(MODEL_PATH)

    print(f"Model loaded successfully from {MODEL_PATH}")

    yield

    model = None


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Machine learning API for heart disease risk prediction. "
        "This API is for educational demonstration purposes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Log requests and collect Prometheus metrics."""

    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    logger.info(
        "%s %s - Status: %s - Duration: %.4fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


class PatientData(BaseModel):
    """Input schema expected by the trained model."""

    age: float = Field(..., gt=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., gt=0)
    chol: float = Field(..., gt=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., gt=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0)
    slope: int = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=3)
    thal: float = Field(..., ge=3, le=7)


@app.get("/")
def root():
    """API root endpoint."""

    return {
        "message": "Heart Disease Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Health-check endpoint."""

    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict")
def predict(patient: PatientData):
    """Predict heart disease from patient features."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded.",
        )

    try:
        input_df = pd.DataFrame(
            [patient.model_dump()]
        )

        prediction = int(
            model.predict(input_df)[0]
        )

        heart_disease_probability = float(
            model.predict_proba(input_df)[0][1]
        )

        prediction_confidence = (
            heart_disease_probability
            if prediction == 1
            else 1 - heart_disease_probability
        )

        PREDICTION_COUNT.labels(
            prediction=str(prediction)
        ).inc()

        return {
            "prediction": prediction,
            "prediction_label": (
                "Heart Disease"
                if prediction == 1
                else "No Heart Disease"
            ),
            "heart_disease_probability": round(
                heart_disease_probability,
                4,
            ),
            "confidence": round(
                prediction_confidence,
                4,
            ),
            "confidence_percentage": round(
                prediction_confidence * 100,
                2,
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
