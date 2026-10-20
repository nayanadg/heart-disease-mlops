"""
Standalone inference script for Heart Disease prediction.

Loads the complete saved sklearn pipeline containing:
- Missing-value imputation
- Numerical scaling
- Categorical encoding
- Logistic Regression classifier

No retraining is required for inference.
"""

from pathlib import Path
import json

import joblib
import pandas as pd


MODEL_PATH = Path("models/best_model.joblib")
METADATA_PATH = Path("models/model_metadata.json")


REQUIRED_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def load_model():
    """Load the serialized preprocessing + model pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run python src/train.py first."
        )

    model = joblib.load(MODEL_PATH)

    print(f"Model loaded successfully from: {MODEL_PATH}")

    return model


def load_metadata():
    """Load model metadata if available."""

    if not METADATA_PATH.exists():
        return None

    with open(METADATA_PATH, "r") as file:
        return json.load(file)


def validate_input(patient_data):
    """Validate required model input features."""

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in patient_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    return True


def predict_heart_disease(model, patient_data):
    """
    Generate heart-disease prediction and probability.

    Returns:
        prediction:
            0 = No heart disease
            1 = Heart disease

        probability:
            Probability of heart disease.
    """

    validate_input(patient_data)

    input_df = pd.DataFrame(
        [patient_data],
        columns=REQUIRED_FEATURES,
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
        "prediction_confidence": round(
            prediction_confidence,
            4,
        ),
        "confidence_percentage": round(
            prediction_confidence * 100,
            2,
        ),
    }


def main():

    print("\n" + "=" * 65)
    print(" HEART DISEASE - SAVED MODEL INFERENCE TEST")
    print("=" * 65)

    model = load_model()

    metadata = load_metadata()

    if metadata:

        print("\nModel metadata:")

        print(
            f"Model name       : "
            f"{metadata.get('model_name')}"
        )

        print(
            f"Selection metric : "
            f"{metadata.get('selection_metric')}"
        )

        print(
            f"Test ROC-AUC     : "
            f"{metadata.get('test_roc_auc')}"
        )

    # Example patient
    patient = {
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

    print("\nInput patient:")

    for key, value in patient.items():
        print(f"{key:12s}: {value}")

    result = predict_heart_disease(
        model,
        patient,
    )

    print("\nPrediction result:")

    print(
        f"Prediction          : "
        f"{result['prediction']}"
    )

    print(
        f"Prediction label    : "
        f"{result['prediction_label']}"
    )

    print(
        f"Disease probability : "
        f"{result['heart_disease_probability']}"
    )

    print(
        f"Confidence          : "
        f"{result['confidence_percentage']}%"
    )

    print("\nInference completed successfully.")


if __name__ == "__main__":
    main()
