import numpy as np
import pandas as pd

from src.train import build_preprocessor


def create_sample_data():
    """Create sample records including missing values."""

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
                "ca": np.nan,
                "thal": np.nan,
            },
            {
                "age": 41,
                "sex": 0,
                "cp": 2,
                "trestbps": 130,
                "chol": 204,
                "fbs": 0,
                "restecg": 2,
                "thalach": 172,
                "exang": 0,
                "oldpeak": 1.4,
                "slope": 1,
                "ca": 0,
                "thal": 3,
            },
        ]
    )


def test_preprocessor_builds():
    """Verify that preprocessing pipeline can be created."""

    preprocessor = build_preprocessor()

    assert preprocessor is not None


def test_preprocessor_handles_missing_values():
    """Verify that preprocessing handles missing values."""

    sample = create_sample_data()

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(sample)

    assert transformed.shape[0] == 2


def test_preprocessor_output_has_no_nan():
    """Verify that preprocessing removes missing values."""

    sample = create_sample_data()

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(sample)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert not np.isnan(transformed).any()
