"""
Tests for the business impact estimate and risk categories.
"""
import pytest

from src.utils import calculate_business_impact, get_churn_risk_category


def test_business_impact_compares_against_no_campaign():
    impact = calculate_business_impact(
        tp=120, fp=4, fn=25, tn=851,
        avg_customer_value=1000, retention_cost=100, retention_success_rate=0.3
    )

    assert impact['cost_of_retention'] == 12_400       # 124 contacts x $100
    assert impact['customers_saved'] == pytest.approx(36)
    assert impact['value_retained'] == pytest.approx(36_000)
    assert impact['net_benefit'] == pytest.approx(23_600)
    assert impact['roi'] == pytest.approx(23_600 / 12_400 * 100)


def test_missed_churners_are_reported_but_not_charged_to_the_model():
    few_misses = calculate_business_impact(tp=100, fp=0, fn=1, tn=900)
    many_misses = calculate_business_impact(tp=100, fp=0, fn=500, tn=401)

    assert few_misses['net_benefit'] == many_misses['net_benefit']
    assert many_misses['cost_of_missed_churners'] > few_misses['cost_of_missed_churners']


def test_business_impact_with_no_contacts_has_zero_roi():
    impact = calculate_business_impact(tp=0, fp=0, fn=10, tn=90)
    assert impact['roi'] == 0
    assert impact['net_benefit'] == 0


@pytest.mark.parametrize("probability, category", [
    (0.0, "Low Risk"),
    (0.29, "Low Risk"),
    (0.3, "Medium Risk"),
    (0.6, "High Risk"),
    (0.8, "Very High Risk"),
    (1.0, "Very High Risk"),
])
def test_churn_risk_category(probability, category):
    assert get_churn_risk_category(probability) == category
