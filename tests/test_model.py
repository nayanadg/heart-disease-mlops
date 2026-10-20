from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/best_model.joblib")


def create_sample_patient():
    """Create a valid sample patient for inference."""

    return pd.DataFrame(
        [
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
                "thal": 6,
            }
        ]
    )


def test_model_exists():
    """Verify that the trained model artifact exists."""

    assert MODEL_PATH.exists()


def test_model_loads():
    """Verify that the serialized model loads successfully."""

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_contains_pipeline_steps():
    """Verify preprocessing and classifier are packaged together."""

    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "named_steps")
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_model_prediction():
    """Verify that the model produces a valid binary prediction."""

    model = joblib.load(MODEL_PATH)

    patient = create_sample_patient()

    prediction = model.predict(patient)[0]

    assert prediction in [0, 1]


def test_model_probability():
    """Verify that the model produces a valid probability."""

    model = joblib.load(MODEL_PATH)

    patient = create_sample_patient()

    probability = model.predict_proba(patient)[0][1]

    assert 0.0 <= probability <= 1.0


def test_prediction_probability_shape():
    """Verify probability output contains both classes."""

    model = joblib.load(MODEL_PATH)

    patient = create_sample_patient()

    probabilities = model.predict_proba(patient)

    assert probabilities.shape == (1, 2)
