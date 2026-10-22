"""
Tests for data cleaning, feature engineering and the preprocessing pipeline.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import TARGET
from src.data_preprocessing import (
    load_data,
    load_raw_features,
    clean_data,
    FeatureEngineer,
    build_preprocessor
)
from src.model_training import build_pipeline


@pytest.fixture(scope="module")
def raw_features():
    return load_raw_features()


def test_clean_data_fills_missing_values():
    df = load_data().head(50)
    df.loc[0, 'Total day minutes'] = np.nan
    df.loc[1, 'International plan'] = np.nan

    cleaned = clean_data(df)

    assert cleaned.isnull().sum().sum() == 0
    assert cleaned.loc[0, 'Total day minutes'] == df['Total day minutes'].median()


def test_clean_data_maps_boolean_target():
    cleaned = clean_data(load_data().head(20))
    assert set(cleaned[TARGET]) <= {'Yes', 'No'}


def test_feature_engineer_uses_threshold_learned_in_fit(raw_features):
    X, _ = raw_features
    engineer = FeatureEngineer().fit(X)

    # A single customer is flagged against the training threshold, not against itself
    heavy_user = X.head(1).copy()
    heavy_user['Total day minutes'] = 400.0
    light_user = X.head(1).copy()
    for col in ['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']:
        light_user[col] = 0.0

    assert engineer.transform(heavy_user)['High usage'].iloc[0] == 1
    assert engineer.transform(light_user)['High usage'].iloc[0] == 0


def test_preprocessor_ignores_unseen_categories(raw_features):
    X, _ = raw_features
    preprocessor = build_preprocessor().fit(X)

    customer = X.head(1).copy()
    customer['State'] = 'ZZ'

    transformed = preprocessor.transform(customer)
    assert transformed.shape == (1, preprocessor.transform(X.head(1)).shape[1])
    assert np.isfinite(transformed).all()


def test_pipeline_scores_a_raw_customer_record(raw_features):
    X, y = raw_features
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)

    probability = pipeline.predict_proba(X.head(1))[0, 1]
    assert 0.0 <= probability <= 1.0
