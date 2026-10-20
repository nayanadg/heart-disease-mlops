"""
Model Training Pipeline - UCI Heart Disease Dataset
MLOps Assignment 01

This script:
1. Loads the processed Heart Disease dataset
2. Separates features and target
3. Performs stratified train/test split
4. Builds reproducible preprocessing pipelines
5. Handles missing numerical/categorical values
6. Scales numerical features
7. One-hot encodes categorical features
8. Trains multiple classification models
9. Performs 5-fold stratified cross-validation
10. Tunes model hyperparameters
11. Evaluates final models
12. Generates evaluation artifacts
13. Selects the best model using ROC-AUC
14. Saves the complete preprocessing + model pipeline
"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

DATA_PATH = Path("data/processed/heart_disease_cleaned.csv")
MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts/modeling")

BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
RESULTS_PATH = ARTIFACT_DIR / "model_comparison.csv"


# ============================================================
# Feature definitions
# ============================================================

# Continuous numerical features
NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

# Features represented numerically in the UCI dataset but
# conceptually categorical/discrete.
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


# ============================================================
# Create directories
# ============================================================

def create_directories():
    """Create directories required for model artifacts."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load data
# ============================================================

def load_data():
    """Load processed Heart Disease dataset."""

    print("\n" + "=" * 70)
    print("1. LOADING PROCESSED DATASET")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nProcessed dataset not found at {DATA_PATH}.\n"
            "Run the EDA pipeline first:\n"
            "python src/eda.py"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset loaded from: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")

    if "target" not in df.columns:
        raise ValueError("Target column is missing from dataset.")

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    return df


# ============================================================
# Prepare features
# ============================================================

def prepare_features(df):
    """Separate independent variables and target."""

    print("\n" + "=" * 70)
    print("2. FEATURE PREPARATION")
    print("=" * 70)

    X = df.drop(columns=["target"])
    y = df["target"]

    expected_features = set(
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    actual_features = set(X.columns)

    missing_features = expected_features - actual_features

    if missing_features:
        raise ValueError(
            f"Expected features missing: {missing_features}"
        )

    print("\nNumerical features:")
    for feature in NUMERIC_FEATURES:
        print(f"  - {feature}")

    print("\nCategorical/discrete features:")
    for feature in CATEGORICAL_FEATURES:
        print(f"  - {feature}")

    print(f"\nTotal input features: {X.shape[1]}")

    return X, y


# ============================================================
# Train/Test Split
# ============================================================

def split_data(X, y):
    """
    Split data into training and test sets.

    Stratification preserves the target-class distribution.
    """

    print("\n" + "=" * 70)
    print("3. STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True).round(3))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True).round(3))

    return X_train, X_test, y_train, y_test


# ============================================================
# Preprocessing Pipeline
# ============================================================

def build_preprocessor():
    """
    Build reusable preprocessing pipeline.

    Numerical:
        Median imputation -> StandardScaler

    Categorical:
        Most-frequent imputation -> OneHotEncoder
    """

    print("\n" + "=" * 70)
    print("4. BUILDING PREPROCESSING PIPELINE")
    print("=" * 70)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    print(
        "\nNumerical preprocessing:"
        "\n  Median imputation"
        "\n  Standard scaling"
    )

    print(
        "\nCategorical preprocessing:"
        "\n  Most-frequent imputation"
        "\n  One-hot encoding"
    )

    return preprocessor


# ============================================================
# Define candidate models
# ============================================================

def get_models():
    """Return candidate classification algorithms."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# Cross-validation
# ============================================================

def perform_cross_validation(
    preprocessor,
    models,
    X_train,
    y_train,
):
    """Evaluate candidate models using stratified 5-fold CV."""

    print("\n" + "=" * 70)
    print("5. 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = []

    for model_name, model in models.items():

        print(f"\nEvaluating: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        result = {
            "Model": model_name,
            "CV Accuracy": scores[
                "test_accuracy"
            ].mean(),
            "CV Precision": scores[
                "test_precision"
            ].mean(),
            "CV Recall": scores[
                "test_recall"
            ].mean(),
            "CV F1": scores[
                "test_f1"
            ].mean(),
            "CV ROC-AUC": scores[
                "test_roc_auc"
            ].mean(),
            "CV ROC-AUC Std": scores[
                "test_roc_auc"
            ].std(),
        }

        cv_results.append(result)

        print(
            f"Accuracy  : "
            f"{result['CV Accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{result['CV Precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{result['CV Recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{result['CV F1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{result['CV ROC-AUC']:.4f}"
            f" ± {result['CV ROC-AUC Std']:.4f}"
        )

    cv_df = pd.DataFrame(cv_results)

    cv_df.to_csv(
        ARTIFACT_DIR / "cross_validation_results.csv",
        index=False,
    )

    return cv_df


# ============================================================
# Hyperparameter tuning
# ============================================================

def tune_models(
    preprocessor,
    X_train,
    y_train,
):
    """
    Tune Logistic Regression and Random Forest using GridSearchCV.

    ROC-AUC is used as the optimization metric.
    """

    print("\n" + "=" * 70)
    print("6. HYPERPARAMETER TUNING")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search_spaces = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "classifier__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ]
            },
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "classifier__n_estimators": [
                    100,
                    200,
                    300,
                ],
                "classifier__max_depth": [
                    None,
                    5,
                    10,
                ],
                "classifier__min_samples_split": [
                    2,
                    5,
                ],
            },
        },
    }

    tuned_models = {}
    tuning_summary = {}

    for model_name, config in search_spaces.items():

        print(
            f"\nTuning {model_name}..."
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    config["model"],
                ),
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        grid_search.fit(
            X_train,
            y_train,
        )

        tuned_models[model_name] = (
            grid_search.best_estimator_
        )

        tuning_summary[model_name] = {
            "best_params": grid_search.best_params_,
            "best_cv_roc_auc": float(
                grid_search.best_score_
            ),
        }

        print(
            f"Best parameters: "
            f"{grid_search.best_params_}"
        )

        print(
            f"Best CV ROC-AUC: "
            f"{grid_search.best_score_:.4f}"
        )

    with open(
        ARTIFACT_DIR / "hyperparameter_tuning.json",
        "w",
    ) as file:
        json.dump(
            tuning_summary,
            file,
            indent=4,
        )

    return tuned_models


# ============================================================
# Add Gradient Boosting
# ============================================================

def train_gradient_boosting(
    preprocessor,
    X_train,
    y_train,
):
    """Train Gradient Boosting candidate pipeline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE
                ),
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model_name,
    pipeline,
    X_test,
    y_test,
):
    """Evaluate fitted model against held-out test data."""

    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Heart Disease",
            "Heart Disease",
        ],
        zero_division=0,
    )

    print(report_text)

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    with open(
        ARTIFACT_DIR
        / f"{safe_name}_classification_report.txt",
        "w",
    ) as file:
        file.write(report_text)

    # Confusion matrix

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Disease",
            "Disease",
        ],
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        ARTIFACT_DIR
        / f"{safe_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ROC curve

    RocCurveDisplay.from_predictions(
        y_test,
        y_probability,
        name=model_name,
    )

    plt.title(
        f"{model_name} - ROC Curve"
    )

    plt.tight_layout()

    plt.savefig(
        ARTIFACT_DIR
        / f"{safe_name}_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
    }


# ============================================================
# Model Comparison Plot
# ============================================================

def create_model_comparison_plot(results_df):
    """Create final model-performance comparison plot."""

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]

    plot_df = results_df.set_index(
        "Model"
    )[metrics]

    plot_df.plot(
        kind="bar",
        figsize=(12, 7),
    )

    plt.title(
        "Final Model Performance Comparison"
    )

    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)

    plt.legend(
        title="Evaluation Metric",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    plt.savefig(
        ARTIFACT_DIR
        / "model_performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Main training workflow
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(" HEART DISEASE - MODEL DEVELOPMENT PIPELINE")
    print("=" * 70)

    create_directories()

    # Load data
    df = load_data()

    # Features/target
    X, y = prepare_features(df)

    # Split before fitting preprocessing
    X_train, X_test, y_train, y_test = (
        split_data(X, y)
    )

    # Preprocessor
    preprocessor = build_preprocessor()

    # --------------------------------------------------------
    # Baseline cross-validation
    # --------------------------------------------------------

    candidate_models = get_models()

    cv_results = perform_cross_validation(
        preprocessor,
        candidate_models,
        X_train,
        y_train,
    )

    print("\nCross-validation summary:")
    print(
        cv_results.round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Hyperparameter tuning
    # --------------------------------------------------------

    tuned_models = tune_models(
        preprocessor,
        X_train,
        y_train,
    )

    # Add Gradient Boosting as additional candidate
    tuned_models["Gradient Boosting"] = (
        train_gradient_boosting(
            preprocessor,
            X_train,
            y_train,
        )
    )

    # --------------------------------------------------------
    # Final held-out test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("7. FINAL HELD-OUT TEST EVALUATION")
    print("=" * 70)

    final_results = []

    for model_name, pipeline in tuned_models.items():

        result = evaluate_model(
            model_name,
            pipeline,
            X_test,
            y_test,
        )

        final_results.append(result)

    results_df = pd.DataFrame(
        final_results
    )

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("8. FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        "\n"
        + results_df.round(4).to_string(
            index=False
        )
    )

    create_model_comparison_plot(
        results_df
    )

    # --------------------------------------------------------
    # Best model selection
    # --------------------------------------------------------

    best_model_name = results_df.iloc[0][
        "Model"
    ]

    best_model = tuned_models[
        best_model_name
    ]

    best_roc_auc = results_df.iloc[0][
        "ROC-AUC"
    ]

    print("\n" + "=" * 70)
    print("9. BEST MODEL SELECTION")
    print("=" * 70)

    print(
        f"\nBest model : {best_model_name}"
    )

    print(
        f"Test ROC-AUC: "
        f"{best_roc_auc:.4f}"
    )

    # Save complete pipeline:
    # preprocessing + classifier
    joblib.dump(
        best_model,
        BEST_MODEL_PATH,
    )

    print(
        f"\nComplete model pipeline saved to:"
        f"\n{BEST_MODEL_PATH}"
    )

    # Save metadata useful later for API/model tracking

    metadata = {
        "model_name": best_model_name,
        "selection_metric": "ROC-AUC",
        "test_roc_auc": float(
            best_roc_auc
        ),
        "random_state": RANDOM_STATE,
        "training_samples": len(
            X_train
        ),
        "test_samples": len(
            X_test
        ),
        "numeric_features": (
            NUMERIC_FEATURES
        ),
        "categorical_features": (
            CATEGORICAL_FEATURES
        ),
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"""
Artifacts generated:

{ARTIFACT_DIR}/
    cross_validation_results.csv
    hyperparameter_tuning.json
    model_comparison.csv
    model_performance_comparison.png
    *_classification_report.txt
    *_confusion_matrix.png
    *_roc_curve.png

{MODEL_DIR}/
    best_model.joblib
    model_metadata.json

Next phase:
    MLflow experiment tracking
"""
    )


if __name__ == "__main__":
    main()
