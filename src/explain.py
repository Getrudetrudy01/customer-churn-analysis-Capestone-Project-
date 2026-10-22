"""
Explain individual churn predictions and estimate the effect of retention actions.

Both work by changing a customer's inputs and re-scoring them with the deployed model,
so they describe the calibrated model itself, in probability terms. They show what the
model has learned from historical data, not proven causal effects.
"""
import numpy as np
import pandas as pd
from typing import Any

from src.config import CHARGE_PER_MINUTE

# Raw inputs grouped into drivers a business user recognises.
# Minutes and charges always move together, so they are one driver.
DRIVER_GROUPS = {
    'Daytime usage': ['Total day minutes', 'Total day charge'],
    'Evening usage': ['Total eve minutes', 'Total eve charge'],
    'Night usage': ['Total night minutes', 'Total night charge'],
    'International usage': ['Total intl minutes', 'Total intl charge', 'Total intl calls'],
    'Customer service calls': ['Customer service calls'],
    'International plan': ['International plan'],
    'Voice mail plan': ['Voice mail plan', 'Number vmail messages'],
    'Account length': ['Account length'],
    'Number of calls': ['Total day calls', 'Total eve calls', 'Total night calls'],
    'Location': ['State', 'Area code']
}

# Actions a retention team can take: (description, applies to customer?, input changes)
RETENTION_ACTIONS = [
    (
        'Resolve service issues',
        lambda customers: customers['Customer service calls'] > 1,
        {'Customer service calls': 1}
    ),
    (
        'Drop international plan',
        lambda customers: customers['International plan'] == 'Yes',
        {'International plan': 'No'}
    ),
    (
        'Add voice mail plan',
        lambda customers: customers['Voice mail plan'] == 'No',
        {'Voice mail plan': 'Yes', 'Number vmail messages': 29}  # median for plan holders
    )
]


def add_charges(customer: dict) -> dict:
    """
    Fill in each charge column from its minutes, using the dataset's per-minute prices.

    Args:
        customer: Raw customer inputs including the four minutes columns

    Returns:
        Copy of the inputs with the four charge columns added
    """
    customer = dict(customer)
    for period, rate in CHARGE_PER_MINUTE.items():
        customer[f'Total {period} charge'] = round(customer[f'Total {period} minutes'] * rate, 2)
    return customer


def typical_customer(X: pd.DataFrame) -> pd.Series:
    """
    Build a reference customer: the median of each numeric input and
    the most common value of each categorical one.

    Args:
        X: Raw features

    Returns:
        Series with one value per raw input column
    """
    numeric = X.select_dtypes('number').median()
    categorical = X.drop(columns=numeric.index).mode().iloc[0]
    return pd.concat([numeric, categorical])[X.columns]


def driver_contributions(model: Any, customers: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """
    Estimate how much each driver moves each customer's churn probability.

    Each driver is reset to the reference customer's value and the customer is re-scored;
    the contribution is the change this causes. A positive value means the driver raises
    the customer's risk compared with a typical customer. Contributions are not additive,
    because drivers interact.

    Args:
        model: Fitted pipeline with predict_proba
        customers: Raw features of one or more customers
        reference: Reference customer from typical_customer

    Returns:
        DataFrame with one row per customer and one column per driver (probability change)
    """
    customers = customers.reset_index(drop=True)
    base = model.predict_proba(customers)[:, 1]

    contributions = {}
    for driver, columns in DRIVER_GROUPS.items():
        variant = customers.copy()
        for column in columns:
            variant[column] = reference[column]
        contributions[driver] = base - model.predict_proba(variant)[:, 1]

    return pd.DataFrame(contributions)


def explain_customer(model: Any, customer: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """
    Explain one customer's prediction, largest effects first.

    Args:
        model: Fitted pipeline with predict_proba
        customer: Raw features of a single customer
        reference: Reference customer from typical_customer

    Returns:
        DataFrame with columns driver and contribution (probability change)
    """
    contributions = driver_contributions(model, customer, reference).iloc[0]
    return (
        contributions.rename('contribution')
        .rename_axis('driver')
        .reset_index()
        .sort_values('contribution', key=np.abs, ascending=False, ignore_index=True)
    )


def action_probabilities(model: Any, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Re-score customers as if each retention action had been taken.

    Args:
        model: Fitted pipeline with predict_proba
        customers: Raw features of one or more customers

    Returns:
        DataFrame with one row per customer and one column per action holding the new
        churn probability, or NaN where the action does not apply to that customer
    """
    customers = customers.reset_index(drop=True)
    results = pd.DataFrame(index=customers.index)

    for action, applies, changes in RETENTION_ACTIONS:
        results[action] = np.nan
        mask = applies(customers)
        if mask.any():
            variant = customers[mask].copy()
            for column, value in changes.items():
                variant[column] = value
            results.loc[mask, action] = model.predict_proba(variant)[:, 1]

    return results


def best_actions(model: Any, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Find the action that lowers each customer's churn probability the most.

    Args:
        model: Fitted pipeline with predict_proba
        customers: Raw features of one or more customers

    Returns:
        DataFrame with columns action (None if nothing helps) and new_probability
    """
    customers = customers.reset_index(drop=True)
    base = model.predict_proba(customers)[:, 1]
    scenarios = action_probabilities(model, customers)

    helpful = scenarios.where(scenarios.lt(base, axis=0))
    has_helpful = helpful.notna().any(axis=1)

    result = pd.DataFrame({'action': None, 'new_probability': base}, index=customers.index)
    result.loc[has_helpful, 'action'] = helpful[has_helpful].idxmin(axis=1)
    result.loc[has_helpful, 'new_probability'] = helpful[has_helpful].min(axis=1)
    return result
