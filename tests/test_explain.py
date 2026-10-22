"""
Tests for per-customer explanations, retention actions and campaign maths.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import load_raw_features
from src.explain import (
    DRIVER_GROUPS,
    add_charges,
    typical_customer,
    explain_customer,
    action_probabilities,
    best_actions
)
from src.model_training import build_pipeline
from src.utils import break_even_probability, profit_curve


@pytest.fixture(scope="module")
def data():
    X, y = load_raw_features()
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    return X, model, typical_customer(X)


def test_add_charges_matches_dataset_prices(data):
    X, _, _ = data
    customer = X.iloc[0].to_dict()
    recomputed = add_charges(customer)
    for period in ['day', 'eve', 'night', 'intl']:
        assert recomputed[f'Total {period} charge'] == pytest.approx(customer[f'Total {period} charge'], abs=0.01)


def test_typical_customer_has_every_input(data):
    X, _, reference = data
    assert list(reference.index) == list(X.columns)
    assert reference['International plan'] == 'No'


def test_explanation_covers_every_driver(data):
    X, model, reference = data
    explanation = explain_customer(model, X.head(1), reference)

    assert set(explanation['driver']) == set(DRIVER_GROUPS)
    assert explanation['contribution'].abs().is_monotonic_decreasing


def test_typical_customer_has_no_contributions(data):
    X, model, reference = data
    customer = reference.to_frame().T.astype(X.dtypes.to_dict())

    explanation = explain_customer(model, customer, reference)
    assert np.allclose(explanation['contribution'], 0)


def test_actions_only_apply_where_relevant(data):
    X, model, _ = data
    customers = X.head(50)
    scenarios = action_probabilities(model, customers)

    no_intl_plan = (customers['International plan'] == 'No').to_numpy()
    assert scenarios.loc[no_intl_plan, 'Drop international plan'].isna().all()


def test_best_action_never_raises_risk(data):
    X, model, _ = data
    customers = X.head(200)
    base = model.predict_proba(customers)[:, 1]

    best = best_actions(model, customers)
    assert (best['new_probability'].to_numpy() <= base + 1e-12).all()
    assert best.loc[best['action'].isna(), 'new_probability'].to_numpy() == pytest.approx(
        base[best['action'].isna().to_numpy()])


def test_break_even_probability():
    assert break_even_probability(1000, 100, 0.3) == pytest.approx(1 / 3)


def test_profit_curve_contacts_highest_risk_first():
    probabilities = np.array([0.1, 0.9, 0.5])
    churned = np.array([0, 1, 0])

    curve = profit_curve(probabilities, churned, avg_customer_value=1000,
                         retention_cost=100, retention_success_rate=0.5)

    # Contact order: 0.9 (churner, +400), 0.5 (-100), 0.1 (-100)
    assert curve['net_benefit'].tolist() == [400, 300, 200]
    assert curve['min_probability'].tolist() == [0.9, 0.5, 0.1]
