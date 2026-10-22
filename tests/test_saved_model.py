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

APP_PATH = str(ROOT_DIR / "app" / "streamlit_app.py")
PAGES = ["views/overview.py", "views/explore.py", "views/performance.py",
         "views/predict.py", "views/campaign.py"]


def open_page(page: str) -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.run()
    app.switch_page(page).run()
    return app


def metric_value(app: AppTest, label: str) -> str:
    return next(m.value for m in app.metric if m.label == label)


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
    assert len(metrics['feature_importance']) == len(load_raw_features()[0].columns)


@pytest.mark.parametrize("page", PAGES)
def test_dashboard_page_renders(page):
    app = open_page(page)
    assert not app.exception


def test_prediction_responds_to_example_customers():
    app = open_page("views/predict.py")
    typical_risk = float(metric_value(app, "Churn risk").rstrip('%'))

    app.selectbox(key='example').set_value('Frequent support caller').run()
    caller_risk = float(metric_value(app, "Churn risk").rstrip('%'))

    assert not app.exception
    assert typical_risk < 10 < 90 < caller_risk


def test_campaign_contacts_fewer_customers_when_offers_cost_more():
    app = open_page("views/campaign.py")
    cheap = int(metric_value(app, "Customers to contact").replace(',', ''))

    app.number_input[1].set_value(250).run()  # cost per retention offer
    expensive = int(metric_value(app, "Customers to contact").replace(',', ''))

    assert not app.exception
    assert 0 < expensive < cheap
