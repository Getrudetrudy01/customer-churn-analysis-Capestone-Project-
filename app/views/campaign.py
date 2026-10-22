"""
Campaign page: decide who is worth a retention offer and produce a call list.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.common import TEAL, ORANGE, GRAY, require_model, score_test_customers
from src.config import AVG_CUSTOMER_VALUE, RETENTION_COST, RETENTION_SUCCESS_RATE
from src.utils import break_even_probability, profit_curve

model, metrics = require_model()
scored = score_test_customers()

st.title("Plan a Retention Campaign")
st.markdown(
    f"Treat the {len(scored):,} held-out customers as this month's customer base. "
    "Set your business figures, and the page works out who is worth contacting."
)

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    customer_value = c1.number_input(
        "Customer lifetime value ($)", 100, 10000, AVG_CUSTOMER_VALUE, step=100)
    contact_cost = c2.number_input(
        "Cost per retention offer ($)", 10, 1000, RETENTION_COST, step=10)
    success_rate = c3.slider(
        "Churners who accept the offer (%)", 5, 100, int(RETENTION_SUCCESS_RATE * 100), step=5,
        help="An assumption: measure it with a pilot campaign before relying on these numbers"
    ) / 100

break_even = break_even_probability(customer_value, contact_cost, success_rate)
expected_value = scored['probability'] * success_rate * customer_value - contact_cost
contact = scored['probability'] >= break_even
n_contact = int(contact.sum())

if break_even >= 1:
    st.warning("At these figures no customer is worth contacting: even a certain churner "
               "is worth less than the cost of the offer.", icon=":material/money_off:")
    st.stop()

st.info(
    f"**Contact every customer whose churn risk is at least {break_even:.0%}.** Above that, "
    f"the expected value saved ({success_rate:.0%} × \\${customer_value:,} × risk) is more than "
    f"the \\${contact_cost:,} offer costs. This simple rule works because the model's "
    "probabilities are calibrated.",
    icon=":material/lightbulb:"
)

curve = profit_curve(scored['probability'], scored['churned'], customer_value, contact_cost, success_rate)
measured = curve['net_benefit'].iloc[n_contact - 1] if n_contact else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers to contact", f"{n_contact:,}", border=True)
col2.metric("Expected churners among them", f"{scored.loc[contact, 'probability'].sum():.0f}", border=True)
col3.metric("Expected net benefit", f"${expected_value[contact].sum():,.0f}", border=True,
            help="Predicted before knowing the outcome, from the calibrated probabilities")
col4.metric("Measured net benefit", f"${measured:,.0f}", border=True,
            help="What the campaign would have returned given who actually churned")

st.subheader("How many customers to contact")
best = curve.loc[curve['net_benefit'].idxmax()]
fig = go.Figure()
fig.add_scatter(x=curve['customers_contacted'], y=curve['net_benefit'], line_color=TEAL,
                name='Net benefit', hovertemplate='%{x} customers: $%{y:,.0f}<extra></extra>')
fig.add_hline(y=0, line_color=GRAY, line_width=1)
if n_contact:
    fig.add_scatter(x=[n_contact], y=[measured], mode='markers', marker=dict(size=12, color=ORANGE),
                    name=f'Break-even rule ({n_contact} customers)')
fig.update_layout(
    height=360, margin=dict(t=10, b=0), xaxis_title='Customers contacted, highest risk first',
    yaxis_title='Net benefit ($)', yaxis_tickprefix='$', legend=dict(orientation='h', y=-0.25)
)
st.plotly_chart(fig, width='stretch')
st.caption(
    f"Measured on actual outcomes. With hindsight, the best possible campaign contacts "
    f"{int(best['customers_contacted'])} customers for \\${best['net_benefit']:,.0f}; the "
    f"break-even rule, chosen without hindsight, reaches \\${measured:,.0f}."
)

st.subheader("Call list")
if n_contact == 0:
    st.info("No customer is above the break-even risk.")
else:
    call_list = scored[contact].assign(expected_value=expected_value[contact]).sort_values(
        'probability', ascending=False)
    call_list = call_list[[
        'customer_id', 'State', 'probability', 'main_driver', 'best_action',
        'probability_after_action', 'expected_value'
    ]]
    st.dataframe(
        call_list, hide_index=True, width='stretch', height=400,
        column_config={
            'customer_id': 'Customer',
            'probability': st.column_config.ProgressColumn(
                'Churn risk', format='percent', min_value=0, max_value=1),
            'main_driver': 'Main risk factor',
            'best_action': 'Suggested action',
            'probability_after_action': st.column_config.ProgressColumn(
                'Risk after action', format='percent', min_value=0, max_value=1),
            'expected_value': st.column_config.NumberColumn('Expected value', format='$%.0f')
        }
    )
    st.download_button(
        "Download call list (CSV)", call_list.to_csv(index=False), file_name="call_list.csv",
        mime="text/csv", icon=":material/download:"
    )
    st.caption(
        "Main risk factor and suggested action come from the same re-scoring method as the "
        "Predict a Customer page. Suggested actions are model estimates, not proven effects."
    )
