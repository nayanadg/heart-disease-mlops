from fastapi.testclient import TestClient

from api.main import app


SAMPLE_PATIENT = {
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
    "thal": 6,
}


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=SAMPLE_PATIENT,
        )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["heart_disease_probability"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_missing_field():
    invalid_patient = SAMPLE_PATIENT.copy()
    invalid_patient.pop("age")

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_patient,
        )

    assert response.status_code == 422
