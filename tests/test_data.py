from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/heart_disease_cleaned.csv")

EXPECTED_COLUMNS = {
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
    "target",
}


def load_dataset():
    """Load the processed heart disease dataset."""
    return pd.read_csv(DATA_PATH)


def test_dataset_exists():
    """Verify that the processed dataset exists."""
    assert DATA_PATH.exists()


def test_dataset_not_empty():
    """Verify that the dataset contains records."""
    df = load_dataset()

    assert not df.empty
    assert len(df) > 0


def test_expected_columns():
    """Verify that all required features are present."""
    df = load_dataset()

    assert set(df.columns) == EXPECTED_COLUMNS


def test_target_has_no_missing_values():
    """Verify that target values are complete."""
    df = load_dataset()

    assert df["target"].isnull().sum() == 0


def test_binary_target():
    """Verify that the target is binary."""
    df = load_dataset()

    assert set(df["target"].unique()).issubset({0, 1})


def test_both_target_classes_exist():
    """Verify that both target classes are represented."""
    df = load_dataset()

    assert set(df["target"].unique()) == {0, 1}


def test_no_duplicate_rows():
    """Verify that processed data has no duplicate rows."""
    df = load_dataset()

    assert df.duplicated().sum() == 0
