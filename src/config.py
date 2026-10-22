"""
Configuration file for the customer churn prediction project.
"""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODEL_DIR = ROOT_DIR / "models"

# Reports directory
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data file paths
RAW_DATA_PATH = RAW_DATA_DIR / "telecom_churn.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_churn.csv"

# Model file paths
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
PIPELINE_PATH = MODEL_DIR / "churn_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Categorical features (based on your actual dataset)
CATEGORICAL_FEATURES = [
    "State",
    "International plan",
    "Voice mail plan"
]

# Numerical features (based on your actual dataset)
NUMERICAL_FEATURES = [
    "Account length",
    "Area code",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls"
]

# Features added by create_features (all numeric)
ENGINEERED_FEATURES = [
    "Total minutes",
    "Total calls",
    "Total charge",
    "Avg charge per minute",
    "Avg calls per day",
    "Has intl plan",
    "Has vmail plan",
    "High usage",
    "High CS calls"
]

# Target variable
TARGET = "Churn"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_FOLDS = 5

# Business assumptions for the impact estimate (not measured from data; adjust to real figures)
AVG_CUSTOMER_VALUE = 1000       # Lifetime value of a retained customer ($)
RETENTION_COST = 100            # Cost of one retention effort ($)
RETENTION_SUCCESS_RATE = 0.3    # Share of contacted churners who accept the offer and stay

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
