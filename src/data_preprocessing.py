"""
Data preprocessing module for customer churn prediction.
Handles data loading, cleaning, feature engineering, and transformation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple
import joblib

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    SCALER_PATH
)


def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Load the telecom churn dataset.

    Args:
        file_path: Path to the CSV file. If None, uses default path from config.

    Returns:
        DataFrame containing the raw data.
    """
    if file_path is None:
        file_path = RAW_DATA_PATH

    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and data type conversions.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    # Convert Churn to string if it's boolean
    if df_clean[TARGET].dtype == bool:
        df_clean[TARGET] = df_clean[TARGET].map({True: 'Yes', False: 'No'})

    # Handle any missing values
    if df_clean.isnull().sum().sum() > 0:
        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    print(f"Data cleaned. Missing values: {df_clean.isnull().sum().sum()}")
    return df_clean


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for better prediction.

    Args:
        df: Cleaned DataFrame

    Returns:
        DataFrame with additional features
    """
    df_feat = df.copy()

    # Create total minutes (sum of all call minutes)
    df_feat['Total minutes'] = (
        df_feat['Total day minutes'] +
        df_feat['Total eve minutes'] +
        df_feat['Total night minutes'] +
        df_feat['Total intl minutes']
    )

    # Create total calls (sum of all calls)
    df_feat['Total calls'] = (
        df_feat['Total day calls'] +
        df_feat['Total eve calls'] +
        df_feat['Total night calls'] +
        df_feat['Total intl calls']
    )

    # Create total charge (sum of all charges)
    df_feat['Total charge'] = (
        df_feat['Total day charge'] +
        df_feat['Total eve charge'] +
        df_feat['Total night charge'] +
        df_feat['Total intl charge']
    )

    # Average charge per minute
    df_feat['Avg charge per minute'] = df_feat['Total charge'] / (df_feat['Total minutes'] + 1)

    # Average calls per day (assuming account length is in days)
    df_feat['Avg calls per day'] = df_feat['Total calls'] / (df_feat['Account length'] + 1)

    # Binary features
    df_feat['Has intl plan'] = (df_feat['International plan'] == 'Yes').astype(int)
    df_feat['Has vmail plan'] = (df_feat['Voice mail plan'] == 'Yes').astype(int)

    # High usage flag (top 25% of total minutes)
    df_feat['High usage'] = (df_feat['Total minutes'] > df_feat['Total minutes'].quantile(0.75)).astype(int)

    # High customer service calls flag (3 or more calls)
    df_feat['High CS calls'] = (df_feat['Customer service calls'] >= 3).astype(int)

    print(f"Features created. New shape: {df_feat.shape}")
    return df_feat


def encode_features(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding.

    Args:
        df: DataFrame with features
        drop_first: Whether to drop first dummy variable to avoid multicollinearity

    Returns:
        DataFrame with encoded features
    """
    df_encoded = df.copy()

    # Get categorical columns to encode (excluding already numeric ones)
    cols_to_encode = [col for col in CATEGORICAL_FEATURES if col in df_encoded.columns]

    # Add target to encoding if present
    if TARGET in df_encoded.columns:
        cols_to_encode.append(TARGET)

    # Add tenure_group if it exists
    if 'tenure_group' in df_encoded.columns:
        cols_to_encode.append('tenure_group')

    # One-hot encoding
    df_encoded = pd.get_dummies(
        df_encoded,
        columns=cols_to_encode,
        drop_first=drop_first
    )

    print(f"Encoding complete. Shape: {df_encoded.shape}")
    return df_encoded


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame = None,
    save_scaler: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale numerical features using StandardScaler.

    Args:
        X_train: Training features
        X_test: Test features (optional)
        save_scaler: Whether to save the scaler for later use

    Returns:
        Tuple of (scaled_train, scaled_test) or just scaled_train if X_test is None
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if save_scaler:
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler saved to {SCALER_PATH}")

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    return X_train_scaled


def prepare_data(
    df: pd.DataFrame,
    target_col: str = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data by separating features and target.

    Args:
        df: Processed DataFrame
        target_col: Name of target column

    Returns:
        Tuple of (X, y)
    """
    if target_col is None:
        # Find the target column (should end with _Yes after encoding)
        target_cols = [col for col in df.columns if col.startswith('Churn_')]
        if target_cols:
            target_col = target_cols[0]
        else:
            raise ValueError("Target column not found in DataFrame")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    print(f"Data prepared. X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")

    return X, y


def preprocess_pipeline(file_path: str = None, save_processed: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete preprocessing pipeline from raw data to features and target.

    Args:
        file_path: Path to raw data file
        save_processed: Whether to save processed data

    Returns:
        Tuple of (X, y)
    """
    print("=" * 50)
    print("Starting preprocessing pipeline...")
    print("=" * 50)

    # Load data
    df = load_data(file_path)

    # Clean data
    df_clean = clean_data(df)

    # Create features
    df_feat = create_features(df_clean)

    # Encode features
    df_encoded = encode_features(df_feat)

    # Save processed data if requested
    if save_processed:
        df_encoded.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Processed data saved to {PROCESSED_DATA_PATH}")

    # Prepare X and y
    X, y = prepare_data(df_encoded)

    print("=" * 50)
    print("Preprocessing pipeline complete!")
    print("=" * 50)

    return X, y


if __name__ == "__main__":
    # Test the preprocessing pipeline
    X, y = preprocess_pipeline()
    print(f"\nFinal X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    print(f"\nSample features:\n{X.head()}")