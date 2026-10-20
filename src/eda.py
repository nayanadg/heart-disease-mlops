"""
Exploratory Data Analysis - UCI Heart Disease Dataset

MLOps Assignment 01

This script:
1. Downloads the UCI Heart Disease dataset
2. Combines features and target
3. Performs basic data inspection
4. Checks missing values
5. Checks duplicate records
6. Converts the target to binary classification
7. Analyses class balance
8. Generates professional EDA visualizations
9. Performs correlation analysis
10. Saves plots as artifacts
11. Saves the cleaned dataset
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo


# ============================================================
# Configuration
# ============================================================

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
EDA_OUTPUT_DIR = Path("artifacts/eda")

RAW_DATA_PATH = RAW_DATA_DIR / "heart_disease.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "heart_disease_cleaned.csv"

sns.set_theme(style="whitegrid")


# ============================================================
# Create required directories
# ============================================================

def create_directories():
    """Create directories required by the EDA pipeline."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

def load_dataset():
    """
    Download the Heart Disease dataset from the
    UCI Machine Learning Repository.
    """

    print("\n" + "=" * 60)
    print("1. DOWNLOADING UCI HEART DISEASE DATASET")
    print("=" * 60)

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    df = X.copy()
    df["target"] = y.iloc[:, 0]

    df.to_csv(RAW_DATA_PATH, index=False)

    print("\nDataset downloaded successfully.")
    print("Dataset shape: {df.shape}")
    print("Raw dataset saved to: {RAW_DATA_PATH}")

    return df


# ============================================================
# Basic Dataset Information
# ============================================================

def inspect_dataset(df):
    """Display basic information about the dataset."""

    print("\n" + "=" * 60)
    print("2. DATASET OVERVIEW")
    print("=" * 60)

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset dimensions:")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nStatistical summary:")
    print(df.describe(include="all").T)


# ============================================================
# Missing Value Analysis
# ============================================================

def analyze_missing_values(df):
    """Analyse missing values in the dataset."""

    print("\n" + "=" * 60)
    print("3. MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    missing_table = pd.DataFrame(
        {
            "Missing Values": missing_count,
            "Percentage": missing_percentage.round(2),
        }
    )

    missing_table = missing_table.sort_values(
        by="Missing Values",
        ascending=False
    )

    print("\nMissing value summary:")
    print(missing_table)

    missing_table.to_csv(
        EDA_OUTPUT_DIR / "missing_values.csv"
    )

    missing_nonzero = missing_count[missing_count > 0]

    if not missing_nonzero.empty:

        plt.figure(figsize=(10, 6))

        missing_nonzero.sort_values(
            ascending=False
        ).plot(kind="bar")

        plt.title("Missing Values by Feature")
        plt.xlabel("Features")
        plt.ylabel("Number of Missing Values")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            EDA_OUTPUT_DIR / "01_missing_values.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("\nMissing-value plot saved.")

    else:
        print("\nNo missing values detected.")

    return missing_table


# ============================================================
# Duplicate Analysis
# ============================================================

def handle_duplicates(df):
    """Check and remove duplicate records."""

    print("\n" + "=" * 60)
    print("4. DUPLICATE RECORD ANALYSIS")
    print("=" * 60)

    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate records found: {duplicate_count}")

    if duplicate_count > 0:

        df = df.drop_duplicates().reset_index(drop=True)

        print(
            f"{duplicate_count} duplicate records were removed."
        )

    else:
        print("No duplicate records were found.")

    print(f"Dataset shape after duplicate check: {df.shape}")

    return df


# ============================================================
# Target Transformation
# ============================================================

def transform_target(df):
    """
    Convert the original heart disease target into binary form.

    0 = No heart disease
    1 = Presence of heart disease
    """

    print("\n" + "=" * 60)
    print("5. TARGET TRANSFORMATION")
    print("=" * 60)

    print("\nOriginal target distribution:")
    print(df["target"].value_counts().sort_index())

    df["target"] = (df["target"] > 0).astype(int)

    print("\nBinary target distribution:")
    print(df["target"].value_counts().sort_index())

    print(
        "\nTarget definition:"
        "\n0 = No heart disease"
        "\n1 = Presence of heart disease"
    )

    return df


# ============================================================
# Class Balance Analysis
# ============================================================

def analyze_class_balance(df):
    """Analyse distribution of target classes."""

    print("\n" + "=" * 60)
    print("6. CLASS BALANCE ANALYSIS")
    print("=" * 60)

    class_counts = df["target"].value_counts().sort_index()

    class_percentages = (
        df["target"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    class_summary = pd.DataFrame(
        {
            "Count": class_counts,
            "Percentage": class_percentages.round(2),
        }
    )

    print("\nClass distribution:")
    print(class_summary)

    class_summary.to_csv(
        EDA_OUTPUT_DIR / "class_distribution.csv"
    )

    plt.figure(figsize=(7, 5))

    ax = sns.countplot(
        data=df,
        x="target"
    )

    plt.title("Heart Disease Target Distribution")
    plt.xlabel("Heart Disease Status")
    plt.ylabel("Number of Patients")
    plt.xticks(
        [0, 1],
        ["No Heart Disease", "Heart Disease"]
    )

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()

    plt.savefig(
        EDA_OUTPUT_DIR / "02_class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nClass distribution plot saved.")


# ============================================================
# Feature Distribution
# ============================================================

def plot_feature_distributions(df):
    """Generate histograms for numerical features."""

    print("\n" + "=" * 60)
    print("7. FEATURE DISTRIBUTIONS")
    print("=" * 60)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    df[numeric_columns].hist(
        figsize=(16, 12),
        bins=20,
        edgecolor="black"
    )

    plt.suptitle(
        "Distribution of Heart Disease Dataset Features",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        EDA_OUTPUT_DIR / "03_feature_distributions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nFeature distribution plot saved.")


# ============================================================
# Age Analysis
# ============================================================

def analyze_age(df):
    """Analyse age distribution by heart disease status."""

    plt.figure(figsize=(9, 6))

    sns.histplot(
        data=df,
        x="age",
        hue="target",
        kde=True,
        multiple="layer"
    )

    plt.title(
        "Age Distribution by Heart Disease Status"
    )

    plt.xlabel("Age")
    plt.ylabel("Number of Patients")

    plt.tight_layout()

    plt.savefig(
        EDA_OUTPUT_DIR / "04_age_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Age distribution plot saved.")


# ============================================================
# Cholesterol Analysis
# ============================================================

def analyze_cholesterol(df):
    """Analyse cholesterol against target."""

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="target",
        y="chol"
    )

    plt.title(
        "Cholesterol Distribution by Heart Disease Status"
    )

    plt.xlabel("Heart Disease Status")
    plt.ylabel("Cholesterol")

    plt.xticks(
        [0, 1],
        ["No Heart Disease", "Heart Disease"]
    )

    plt.tight_layout()

    plt.savefig(
        EDA_OUTPUT_DIR / "05_cholesterol_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Cholesterol analysis plot saved.")


# ============================================================
# Blood Pressure Analysis
# ============================================================

def analyze_blood_pressure(df):
    """Analyse resting blood pressure against target."""

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="target",
        y="trestbps"
    )

    plt.title(
        "Resting Blood Pressure by Heart Disease Status"
    )

    plt.xlabel("Heart Disease Status")
    plt.ylabel("Resting Blood Pressure")

    plt.xticks(
        [0, 1],
        ["No Heart Disease", "Heart Disease"]
    )

    plt.tight_layout()

    plt.savefig(
        EDA_OUTPUT_DIR / "06_blood_pressure_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Blood pressure analysis plot saved.")


# ============================================================
# Chest Pain Analysis
# ============================================================

def analyze_chest_pain(df):
    """Analyse chest pain type against heart disease."""

    plt.figure(figsize=(9, 6))

    sns.countplot(
        data=df,
        x="cp",
        hue="target"
    )

    plt.title(
        "Chest Pain Type vs Heart Disease"
    )

    plt.xlabel("Chest Pain Type")
    plt.ylabel("Number of Patients")

    plt.legend(
        title="Heart Disease",
        labels=["No", "Yes"]
    )

    plt.tight_layout()

    plt.savefig(
        EDA_OUTPUT_DIR / "07_chest_pain_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Chest pain analysis plot saved.")


# ============================================================
# Maximum Heart Rate Analysis
# ============================================================

def analyze_heart_rate(df):
    """Analyse maximum heart rate against target."""

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="target",
        y="thalach"
    )

    plt.title(
        "Maximum Heart Rate by Heart Disease Status"
    )

    plt.xlabel("Heart Disease Status")
    plt.ylabel("Maximum Heart Rate")

    plt.xticks(
        [0, 1],
        ["No Heart Disease", "Heart Disease"]
    )

    plt.tight_layout()

    plt.savefig(
        EDA_OUTPUT_DIR / "08_heart_rate_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Heart-rate analysis plot saved.")


# ============================================================
# Correlation Analysis
# ============================================================

def correlation_analysis(df):
    """Calculate and visualize feature correlations."""

    print("\n" + "=" * 60)
    print("8. CORRELATION ANALYSIS")
    print("=" * 60)

    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(15, 11))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title(
        "Heart Disease Feature Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_OUTPUT_DIR / "09_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    target_correlation = (
        correlation_matrix["target"]
        .drop("target")
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    print("\nFeature correlation with target:")
    print(target_correlation)

    target_correlation.to_csv(
        EDA_OUTPUT_DIR / "target_correlations.csv",
        header=["Correlation"]
    )

    # Plot target correlations

    plt.figure(figsize=(10, 6))

    target_correlation.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Feature Correlation with Heart Disease Target"
    )

    plt.xlabel("Correlation")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        EDA_OUTPUT_DIR / "10_target_correlations.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nCorrelation plots saved.")


# ============================================================
# Save Processed Dataset
# ============================================================

def save_processed_dataset(df):
    """Save cleaned dataset for model development."""

    print("\n" + "=" * 60)
    print("9. SAVING PROCESSED DATASET")
    print("=" * 60)

    df.to_csv(
        CLEAN_DATA_PATH,
        index=False
    )

    print(
        f"\nCleaned dataset saved to:"
        f"\n{CLEAN_DATA_PATH}"
    )

    print(f"\nFinal dataset shape: {df.shape}")


# ============================================================
# EDA Summary
# ============================================================

def print_summary(df):
    """Print final EDA summary."""

    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)

    print(f"""
Total observations : {df.shape[0]}
Total features     : {df.shape[1] - 1}

Target:
    0 = No heart disease
    1 = Presence of heart disease

EDA performed:
    - Dataset inspection
    - Missing-value analysis
    - Duplicate analysis
    - Binary target transformation
    - Class-balance analysis
    - Feature distributions
    - Age analysis
    - Cholesterol analysis
    - Blood-pressure analysis
    - Chest-pain analysis
    - Maximum-heart-rate analysis
    - Correlation analysis

EDA artifacts:
    {EDA_OUTPUT_DIR}

Processed dataset:
    {CLEAN_DATA_PATH}

NOTE:
Missing feature values have intentionally not been filled during
EDA. Imputation will be incorporated into the Scikit-learn
preprocessing pipeline during model development. This prevents
data leakage and ensures that identical preprocessing can be
applied during both training and inference.
""")


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print(" UCI HEART DISEASE - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    create_directories()

    # Load dataset
    df = load_dataset()

    # Dataset inspection
    inspect_dataset(df)

    # Data-quality checks
    analyze_missing_values(df)
    df = handle_duplicates(df)

    # Target preprocessing
    df = transform_target(df)

    # EDA
    analyze_class_balance(df)
    plot_feature_distributions(df)

    print("\nGenerating feature-level visualizations...")

    analyze_age(df)
    analyze_cholesterol(df)
    analyze_blood_pressure(df)
    analyze_chest_pain(df)
    analyze_heart_rate(df)

    # Correlation analysis
    correlation_analysis(df)

    # Save final dataset
    save_processed_dataset(df)

    # Summary
    print_summary(df)

    print("\nEDA completed successfully.\n")


if __name__ == "__main__":
    main()
