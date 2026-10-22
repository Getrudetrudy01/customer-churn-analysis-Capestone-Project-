"""
Tests for the committed model artifacts and the dashboard that serves them.
These catch a saved model that no longer loads or predicts with the pinned library versions.
"""
import json

import joblib
import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT_DIR, PIPELINE_PATH, METRICS_PATH
from src.data_preprocessing import load_raw_features

pytestmark = pytest.mark.skipif(
    not (PIPELINE_PATH.exists() and METRICS_PATH.exists()),
    reason="Run `python -m src.model_training` to create the model first"
)

PAGES = [" Home", " Data Exploration", " Model Performance", " Customer Prediction", " Business Impact"]


def test_saved_pipeline_predicts_valid_probabilities():
    X, _ = load_raw_features()
    pipeline = joblib.load(PIPELINE_PATH)

    probabilities = pipeline.predict_proba(X.head(20))[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_saved_metrics_match_dashboard_needs():
    metrics = json.loads(METRICS_PATH.read_text(encoding='utf-8'))

    assert 0 < metrics['threshold'] < 1
    assert metrics['results'][0]['model'] == metrics['best_model'].removesuffix(' (Tuned)') \
        or metrics['best_model'].endswith('(Tuned)')
    assert sum(metrics['confusion_matrix'].values()) == metrics['test_size']
    for key in ['accuracy', 'precision', 'recall', 'f1_score', 'pr_auc']:
        assert 0 <= metrics['final_model'][key] <= 1


@pytest.mark.parametrize("page", PAGES)
def test_dashboard_page_renders(page):
    app = AppTest.from_file(str(ROOT_DIR / "app" / "streamlit_app.py"), default_timeout=60)
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    if page == " Customer Prediction":
        app.button[0].click().run()

    assert not app.exception
