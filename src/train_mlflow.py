"""
MLflow Experiment Tracking
UCI Heart Disease - MLOps Assignment 01

Logs:
- Model parameters
- Cross-validation metrics
- Test metrics
- Confusion matrices
- ROC curves
- Classification reports
- Complete sklearn preprocessing/model pipelines
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from mlflow.models import infer_signature
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

DATA_PATH = Path(
    "data/processed/heart_disease_cleaned.csv"
)

ARTIFACT_DIR = Path(
    "artifacts/mlflow"
)

EXPERIMENT_NAME = (
    "Heart-Disease-Classification"
)

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

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
# Load data
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist. "
            "Run python src/eda.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=["target"]
    )

    y = df["target"]

    return X, y


# ============================================================
# Preprocessor
# ============================================================

def build_preprocessor():

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
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
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# ============================================================
# Models
# ============================================================

def get_models():

    return {
        "Logistic Regression":
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                random_state=RANDOM_STATE,
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
    }


# ============================================================
# Cross Validation
# ============================================================

def calculate_cv_metrics(
    pipeline,
    X_train,
    y_train,
):

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

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    return {
        "cv_accuracy":
            scores["test_accuracy"].mean(),

        "cv_precision":
            scores["test_precision"].mean(),

        "cv_recall":
            scores["test_recall"].mean(),

        "cv_f1":
            scores["test_f1"].mean(),

        "cv_roc_auc":
            scores["test_roc_auc"].mean(),

        "cv_roc_auc_std":
            scores["test_roc_auc"].std(),
    }


# ============================================================
# Generate artifacts
# ============================================================

def create_artifacts(
    model_name,
    pipeline,
    X_test,
    y_test,
    y_pred,
    y_probability,
):

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    model_artifact_dir = (
        ARTIFACT_DIR / safe_name
    )

    model_artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------
    # Classification Report
    # ----------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Heart Disease",
            "Heart Disease",
        ],
        zero_division=0,
    )

    report_path = (
        model_artifact_dir
        / "classification_report.txt"
    )

    with open(
        report_path,
        "w",
    ) as file:
        file.write(report)

    # ----------------------------------------
    # Confusion Matrix
    # ----------------------------------------

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=[
            "No Disease",
            "Disease",
        ],
        cmap="Blues",
        values_format="d",
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    confusion_path = (
        model_artifact_dir
        / "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ----------------------------------------
    # ROC Curve
    # ----------------------------------------

    RocCurveDisplay.from_predictions(
        y_test,
        y_probability,
        name=model_name,
    )

    plt.title(
        f"{model_name} - ROC Curve"
    )

    plt.tight_layout()

    roc_path = (
        model_artifact_dir
        / "roc_curve.png"
    )

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return [
        report_path,
        confusion_path,
        roc_path,
    ]


# ============================================================
# Run one MLflow experiment
# ============================================================

def run_experiment(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):

    print("\n" + "=" * 70)
    print(f"MLFLOW RUN: {model_name}")
    print("=" * 70)

    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                model,
            ),
        ]
    )

    # ----------------------------------------
    # Cross validation
    # ----------------------------------------

    cv_metrics = calculate_cv_metrics(
        pipeline,
        X_train,
        y_train,
    )

    # ----------------------------------------
    # Final training
    # ----------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    # ----------------------------------------
    # Test metrics
    # ----------------------------------------

    test_metrics = {
        "test_accuracy":
            accuracy_score(
                y_test,
                y_pred,
            ),

        "test_precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),

        "test_recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),

        "test_f1":
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),

        "test_roc_auc":
            roc_auc_score(
                y_test,
                y_probability,
            ),
    }

    # ----------------------------------------
    # MLflow run
    # ----------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ):

        # General parameters

        mlflow.log_param(
            "model_name",
            model_name,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_param(
            "train_samples",
            len(X_train),
        )

        mlflow.log_param(
            "test_samples",
            len(X_test),
        )

        mlflow.log_param(
            "cv_folds",
            5,
        )

        mlflow.log_param(
            "numeric_imputation",
            "median",
        )

        mlflow.log_param(
            "categorical_imputation",
            "most_frequent",
        )

        mlflow.log_param(
            "categorical_encoding",
            "one_hot",
        )

        mlflow.log_param(
            "numeric_scaling",
            "standard_scaler",
        )

        # Log classifier-specific parameters

        classifier_params = (
            model.get_params()
        )

        for key, value in (
            classifier_params.items()
        ):

            # MLflow parameters should remain
            # reasonably compact.
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            ):
                mlflow.log_param(
                    f"classifier_{key}",
                    value,
                )

        # Metrics

        mlflow.log_metrics(
            cv_metrics
        )

        mlflow.log_metrics(
            test_metrics
        )

        # Artifacts

        artifact_paths = (
            create_artifacts(
                model_name,
                pipeline,
                X_test,
                y_test,
                y_pred,
                y_probability,
            )
        )

        for artifact in artifact_paths:
            mlflow.log_artifact(
                str(artifact),
                artifact_path="evaluation",
            )

        # Create an input example for reproducible model serving
        input_example = X_train.head(5).copy()

        # Infer MLflow model input/output signature
        signature = infer_signature(
            input_example,
            pipeline.predict(input_example)
        )

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        # Tags

        mlflow.set_tag(
            "project",
            "MLOps Assignment 01",
        )

        mlflow.set_tag(
            "dataset",
            "UCI Heart Disease",
        )

        mlflow.set_tag(
            "task",
            "Binary Classification",
        )

        print("\nCross-validation metrics:")

        for key, value in (
            cv_metrics.items()
        ):
            print(
                f"{key:20s}: "
                f"{value:.4f}"
            )

        print("\nTest metrics:")

        for key, value in (
            test_metrics.items()
        ):
            print(
                f"{key:20s}: "
                f"{value:.4f}"
            )

    return {
        "Model": model_name,
        **cv_metrics,
        **test_metrics,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        " HEART DISEASE - MLFLOW EXPERIMENT TRACKING"
    )
    print("=" * 70)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------
    # Configure local MLflow tracking
    # ----------------------------------------

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"\nMLflow experiment: "
        f"{EXPERIMENT_NAME}"
    )

    # ----------------------------------------
    # Load data
    # ----------------------------------------

    X, y = load_data()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test)}"
    )

    # ----------------------------------------
    # Run experiments
    # ----------------------------------------

    models = get_models()

    experiment_results = []

    for model_name, model in models.items():

        result = run_experiment(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        experiment_results.append(
            result
        )

    # ----------------------------------------
    # Comparison
    # ----------------------------------------

    results_df = pd.DataFrame(
        experiment_results
    )

    results_df = results_df.sort_values(
        by="test_roc_auc",
        ascending=False,
    )

    comparison_path = (
        ARTIFACT_DIR
        / "mlflow_experiment_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("MLFLOW EXPERIMENT SUMMARY")
    print("=" * 70)

    print(
        "\n"
        + results_df.round(4).to_string(
            index=False
        )
    )

    print(
        "\nBest experiment:"
        f"\n{results_df.iloc[0]['Model']}"
    )

    print(
        "\nExperiment tracking completed successfully."
    )

    print(
        "\nStart the MLflow UI using:"
        "\nmlflow ui "
        "--backend-store-uri sqlite:///mlflow.db"
    )

    print(
        "\nThen open:"
        "\nhttp://127.0.0.1:5000"
    )


if __name__ == "__main__":
    main()
