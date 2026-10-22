"""
Explore page: how churn varies with each customer attribute.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px
import streamlit as st

from app.common import CHURN_COLORS, ORANGE, GRAY, load_customers

df = load_customers()
churn_rate = (df['Churn'] == 'Yes').mean()

st.title("Explore the Data")
st.markdown(
    f"{len(df):,} customers, {df.shape[1] - 1} attributes each. "
    f"**{churn_rate:.1%}** of them churned."
)

NUMERIC = {
    'Customer service calls': 'Customer service calls',
    'Daytime minutes': 'Total day minutes',
    'Evening minutes': 'Total eve minutes',
    'Night minutes': 'Total night minutes',
    'International minutes': 'Total intl minutes',
    'Voicemail messages': 'Number vmail messages',
    'Account length (days)': 'Account length'
}
CATEGORICAL = {
    'International plan': 'International plan',
    'Voice mail plan': 'Voice mail plan',
    'Area code': 'Area code',
    'State': 'State'
}

choice = st.selectbox("Attribute", list(NUMERIC) + list(CATEGORICAL))

if choice in NUMERIC:
    column = NUMERIC[choice]
    left, right = st.columns(2, gap="large")

    with left:
        fig = px.histogram(
            df, x=column, color='Churn', barmode='group', histnorm='percent',
            color_discrete_map=CHURN_COLORS,
            labels={column: choice}, title="Distribution (each group sums to 100%)"
        )
        fig.update_layout(yaxis_title='% of customers', margin=dict(t=50))
        st.plotly_chart(fig, width='stretch')

    with right:
        fig = px.box(
            df, x='Churn', y=column, color='Churn', color_discrete_map=CHURN_COLORS,
            labels={column: choice}, title="Spread by churn status"
        )
        fig.update_layout(showlegend=False, margin=dict(t=50))
        st.plotly_chart(fig, width='stretch')
else:
    column = CATEGORICAL[choice]
    rates = (
        df.assign(churned=df['Churn'] == 'Yes')
        .groupby(column)['churned'].agg(['mean', 'size'])
        .rename(columns={'mean': 'churn_rate', 'size': 'customers'})
        .reset_index()
        .sort_values('churn_rate', ascending=False)
    )
    rates[column] = rates[column].astype(str)
    fig = px.bar(
        rates, x=column, y='churn_rate', hover_data=['customers'],
        labels={'churn_rate': 'Churn rate', column: choice},
        title=f"Churn rate by {choice.lower()}"
    )
    fig.update_traces(marker_color=ORANGE)
    fig.add_hline(y=churn_rate, line_dash='dot', line_color=GRAY,
                  annotation_text=f"All customers: {churn_rate:.1%}")
    fig.update_layout(yaxis_tickformat='.0%', margin=dict(t=50))
    st.plotly_chart(fig, width='stretch')

with st.expander("Raw data and summary statistics"):
    st.dataframe(df, width='stretch', height=300)
    st.dataframe(df.describe().round(2), width='stretch')
