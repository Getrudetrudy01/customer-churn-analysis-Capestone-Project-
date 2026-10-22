"""
Prediction page: score one customer, explain the score and test retention actions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.common import TEAL, ORANGE, load_customers, load_reference, require_model
from src.explain import add_charges, explain_customer, action_probabilities
from src.utils import get_churn_risk_category

model, metrics = require_model()
reference = load_reference()
customers = load_customers()
threshold = metrics['threshold']

USAGE_PERIODS = [('day', 'Daytime'), ('eve', 'Evening'), ('night', 'Night'), ('intl', 'International')]
INTEGER_INPUTS = [
    'Account length', 'Area code', 'Number vmail messages', 'Customer service calls',
    'Total day calls', 'Total eve calls', 'Total night calls', 'Total intl calls'
]
FLOAT_INPUTS = ['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']
TEXT_INPUTS = ['State', 'International plan', 'Voice mail plan']

EXAMPLES = {
    'Typical customer': {},
    'Frequent support caller': {'Customer service calls': 5, 'Total day minutes': 120.0},
    'Heavy daytime user': {'Total day minutes': 300.0, 'Total eve minutes': 250.0},
    'International plan, few calls': {
        'International plan': 'Yes', 'Total intl minutes': 13.0, 'Total intl calls': 2
    }
}


def load_example():
    """Fill every input from the selected example, starting from the typical customer."""
    values = {column: int(reference[column]) for column in INTEGER_INPUTS}
    values.update({column: float(reference[column]) for column in FLOAT_INPUTS})
    values.update({column: str(reference[column]) for column in TEXT_INPUTS})
    values.update(EXAMPLES[st.session_state.example])
    for column, value in values.items():
        st.session_state[column] = value


if 'State' not in st.session_state:
    st.session_state.example = 'Typical customer'
    load_example()

st.title("Predict a Customer")
st.markdown("Change any input and the prediction updates immediately.")

# Result first, so it stays visible on phones where the columns stack
result_col, form_col = st.columns([2, 3], gap="large")

with form_col:
    st.selectbox("Start from an example", list(EXAMPLES), key='example', on_change=load_example)

    with st.container(border=True):
        st.markdown("**Account**")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("State", sorted(customers['State'].unique()), key='State')
        c2.selectbox("Area code", sorted(customers['Area code'].unique().tolist()), key='Area code')
        c3.number_input("Account length (days)", 0, 300, step=1, key='Account length')

    with st.container(border=True):
        st.markdown("**Plans**")
        c1, c2, c3 = st.columns(3)
        c1.radio("International plan", ["No", "Yes"], horizontal=True, key='International plan')
        c2.radio("Voice mail plan", ["No", "Yes"], horizontal=True, key='Voice mail plan')
        c3.number_input("Voicemail messages", 0, 60, step=1, key='Number vmail messages')

    with st.container(border=True):
        st.markdown("**Usage this period**")
        for period, label in USAGE_PERIODS:
            c1, c2 = st.columns(2)
            max_minutes = 30.0 if period == 'intl' else 400.0
            c1.number_input(f"{label} minutes", 0.0, max_minutes, step=5.0, key=f'Total {period} minutes')
            c2.number_input(f"{label} calls", 0, 200, step=1, key=f'Total {period} calls')

    with st.container(border=True):
        st.slider("Customer service calls", 0, 9, key='Customer service calls')

inputs = {column: st.session_state[column] for column in INTEGER_INPUTS + FLOAT_INPUTS + TEXT_INPUTS}
customer = pd.DataFrame([add_charges(inputs)])
probability = float(model.predict_proba(customer)[0, 1])
flagged = probability >= threshold

with result_col:
    with st.container(border=True):
        st.metric("Churn risk", f"{probability:.0%}", help="Calibrated probability that this customer churns")
        st.caption(get_churn_risk_category(probability))

        fig = go.Figure(go.Indicator(
            mode="gauge", value=probability * 100,
            gauge={
                'axis': {'range': [0, 100], 'ticksuffix': '%'},
                'bar': {'color': ORANGE if flagged else TEAL, 'thickness': 0.6},
                'bgcolor': 'rgba(148, 163, 184, 0.15)',
                'borderwidth': 0,
                'threshold': {'line': {'color': '#64748b', 'width': 3}, 'thickness': 0.8,
                              'value': threshold * 100}
            }
        ))
        fig.update_layout(height=200, margin=dict(l=40, r=40, t=20, b=0))
        st.plotly_chart(fig, width='stretch')

        if flagged:
            st.error(f"**Flag for retention.** Risk is above the model's {threshold:.0%} threshold "
                     "(grey line).", icon=":material/flag:")
        else:
            st.success(f"**No action needed.** Risk is below the model's {threshold:.0%} threshold "
                       "(grey line).", icon=":material/check_circle:")

explain_col, action_col = st.columns(2, gap="large")

with explain_col:
    st.subheader("Why this score")
    explanation = explain_customer(model, customer, reference)
    explanation = explanation[explanation['contribution'].abs() >= 0.005].head(6)

    if explanation.empty:
        st.info("Nothing stands out: this customer's risk is close to a typical customer's.")
    else:
        explanation['points'] = explanation['contribution'] * 100
        explanation['effect'] = explanation['points'].map(lambda p: 'Raises risk' if p > 0 else 'Lowers risk')
        fig = px.bar(
            explanation.iloc[::-1], x='points', y='driver', orientation='h', color='effect',
            color_discrete_map={'Raises risk': ORANGE, 'Lowers risk': TEAL},
            labels={'points': 'Change in churn risk (percentage points)', 'driver': '', 'effect': ''}
        )
        fig.update_layout(
            height=140 + 45 * len(explanation), margin=dict(t=10, b=0),
            legend=dict(orientation='h', y=-0.3), bargap=0.35
        )
        st.plotly_chart(fig, width='stretch')
    st.caption(
        "Each bar shows how much the risk changes when that factor is reset to a typical "
        f"customer's value ({int(reference['Customer service calls'])} service call, "
        f"{reference['Total day minutes']:.0f} daytime minutes, no international or voice mail plan). "
        "Factors interact, so the bars do not add up to the total."
    )

with action_col:
    st.subheader("What would lower the risk")
    scenarios = action_probabilities(model, customer).iloc[0].dropna()
    scenarios = scenarios[scenarios <= probability - 0.01]  # keep actions that help

    if scenarios.empty:
        st.info("None of the retention actions would lower this customer's risk.")
    else:
        actions = pd.DataFrame({
            'Action': scenarios.index,
            'Risk after': scenarios.to_numpy(),
            'Change': (scenarios.to_numpy() - probability) * 100
        }).sort_values('Change')
        st.dataframe(
            actions, hide_index=True, width='stretch',
            column_config={
                'Risk after': st.column_config.ProgressColumn(
                    'Risk after', format='percent', min_value=0, max_value=1),
                'Change': st.column_config.NumberColumn('Change (points)', format='%+.0f')
            }
        )
    st.caption(
        "Estimated by re-scoring the customer with the change applied: support calls back to 1, "
        "no international plan, or a voice mail plan with typical usage. These are patterns the "
        "model learned from historical data, not proven effects: test them before rolling out."
    )
