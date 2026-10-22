"""
Shared data loading, colours and helpers for the dashboard pages.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.config import PIPELINE_PATH, METRICS_PATH, TEST_SIZE, RANDOM_STATE, TARGET
from src.data_preprocessing import load_data, load_raw_features
from src.explain import typical_customer, driver_contributions, best_actions

# Colours used by every chart: teal for customers who stay or lower risk,
# orange for churn or higher risk
TEAL = "#0d9488"
ORANGE = "#f97316"
GRAY = "#94a3b8"
CHURN_COLORS = {"No": TEAL, "Yes": ORANGE}


@st.cache_data
def load_customers() -> pd.DataFrame:
    """Raw dataset with churn as Yes/No, for exploration."""
    df = load_data()
    if df[TARGET].dtype == bool:
        df[TARGET] = df[TARGET].map({True: 'Yes', False: 'No'})
    return df


@st.cache_resource
def load_model():
    """Trained pipeline (preprocessing + calibrated model), or None if not trained yet."""
    if not PIPELINE_PATH.exists():
        return None
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_metrics():
    """Evaluation results saved by the latest training run, or None if not trained yet."""
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding='utf-8'))


@st.cache_data
def load_reference() -> pd.Series:
    """Typical customer that explanations are compared against."""
    X, _ = load_raw_features()
    return typical_customer(X)


def require_model():
    """Return (model, metrics), or stop the page with instructions if the model is missing."""
    model, metrics = load_model(), load_metrics()
    if model is None or metrics is None:
        st.warning("No trained model found.", icon=":material/warning:")
        st.info("Run `python -m src.model_training` from the project root to train the model.")
        st.stop()
    return model, metrics


@st.cache_data(show_spinner="Scoring customers...")
def score_test_customers() -> pd.DataFrame:
    """
    Score the held-out test customers with the deployed model.

    Returns:
        Raw inputs plus customer_id, churned (actual outcome), probability,
        main_driver, best_action and probability_after_action
    """
    model = load_model()
    X, y = load_raw_features()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    contributions = driver_contributions(model, X_test, load_reference())
    actions = best_actions(model, X_test)

    scored = X_test.reset_index(names='customer_id')
    scored['churned'] = y_test.to_numpy()
    scored['probability'] = model.predict_proba(X_test)[:, 1]
    scored['main_driver'] = contributions.idxmax(axis=1)
    scored['best_action'] = actions['action']
    scored['probability_after_action'] = actions['new_probability']
    return scored
