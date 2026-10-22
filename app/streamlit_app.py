"""
Customer Churn Prediction Dashboard
Entry point: page setup and navigation. Each page lives in app/views/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.common import ROOT, load_metrics

LOGO = str(ROOT / "app" / "assets" / "CustomerDesign.png")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon=LOGO,
    layout="wide",
    initial_sidebar_state="auto"
)
st.logo(LOGO, size="large")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/explore.py", title="Explore the Data", icon=":material/query_stats:"),
    st.Page("views/performance.py", title="Model Performance", icon=":material/insights:"),
    st.Page("views/predict.py", title="Predict a Customer", icon=":material/person_search:"),
    st.Page("views/campaign.py", title="Plan a Campaign", icon=":material/campaign:"),
]
navigation = st.navigation(pages)

with st.sidebar:
    metrics = load_metrics()
    if metrics:
        st.caption(f"Model: calibrated {metrics['best_model']}, trained on 3,333 telecom customers.")

navigation.run()
