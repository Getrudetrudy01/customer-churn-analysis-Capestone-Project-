"""
Overview page: what the project does, headline results and where to go next.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.common import TEAL, ORANGE, GRAY, load_customers, require_model

model, metrics = require_model()
df = load_customers()
final = metrics['final_model']
churned = df['Churn'] == 'Yes'

st.title("Customer Churn Prediction")
st.markdown(
    "Find the telecom customers most likely to leave, understand why, "
    "and decide who is worth a retention offer."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers analysed", f"{len(df):,}", border=True)
col2.metric("Churn rate", f"{churned.mean():.1%}", border=True)
col3.metric("Churners the model catches", f"{final['recall']:.0%}", border=True,
            help="Recall on 1,000 held-out customers")
col4.metric("Flagged customers who churn", f"{final['precision']:.0%}", border=True,
            help="Precision on 1,000 held-out customers")

left, right = st.columns(2, gap="large")

with left:
    st.subheader("What the model relies on")
    importance = pd.DataFrame(metrics['feature_importance']).head(6)
    fig = px.bar(
        importance.sort_values('importance'), x='importance', y='feature', orientation='h',
        labels={'importance': 'Drop in PR-AUC when shuffled', 'feature': ''}
    )
    fig.update_traces(marker_color=TEAL)
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("Where churn concentrates")
    segments = pd.DataFrame({
        'segment': ['4+ service calls', 'International plan', 'All customers'],
        'churn_rate': [
            churned[df['Customer service calls'] >= 4].mean(),
            churned[df['International plan'] == 'Yes'].mean(),
            churned.mean()
        ]
    })
    # Horizontal bars are drawn bottom-up, so reverse to read top-down
    fig = px.bar(
        segments.iloc[::-1], x='churn_rate', y='segment', orientation='h', text_auto='.0%',
        labels={'churn_rate': 'Churn rate', 'segment': ''}
    )
    fig.update_traces(marker_color=[GRAY, ORANGE, ORANGE])
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis_tickformat='.0%')
    st.plotly_chart(fig, width='stretch')

st.subheader("Explore the dashboard")
cards = [
    ("views/predict.py", "Predict a Customer", ":material/person_search:",
     "Score one customer, see why, and test which actions lower their risk."),
    ("views/campaign.py", "Plan a Campaign", ":material/campaign:",
     "Decide who is worth contacting and get a ranked call list."),
    ("views/performance.py", "Model Performance", ":material/insights:",
     "How the model was chosen and how well it works on unseen customers."),
    ("views/explore.py", "Explore the Data", ":material/query_stats:",
     "How churn varies with usage, plans and service calls."),
]
for column, (page, title, icon, description) in zip(st.columns(4), cards):
    with column.container(border=True, height="stretch"):
        st.page_link(page, label=title, icon=icon)
        st.caption(description)
