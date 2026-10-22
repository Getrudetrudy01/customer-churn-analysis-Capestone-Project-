"""
Model performance page: how the model was chosen and how well it works on unseen customers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from app.common import TEAL, ORANGE, GRAY, require_model, score_test_customers

model, metrics = require_model()
final = metrics['final_model']
threshold = metrics['threshold']

st.title("Model Performance")
st.markdown(
    f"The deployed model is a **calibrated {metrics['best_model']}**. It flags a customer "
    f"when their churn probability is at least **{threshold:.0%}**. All numbers below come "
    f"from {metrics['test_size']:,} customers the model never saw during training or selection."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", f"{final['precision']:.1%}", border=True,
            help="Share of flagged customers who really churned")
col2.metric("Recall", f"{final['recall']:.1%}", border=True,
            help="Share of churners the model flagged")
col3.metric("F1-score", f"{final['f1_score']:.3f}", border=True)
col4.metric("PR-AUC", f"{final['pr_auc']:.3f}", border=True,
            help="Ranking quality across all thresholds; random guessing scores about 0.145")

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Predictions vs. outcomes")
    cm = metrics['confusion_matrix']
    matrix = np.array([
        [cm['true_negatives'], cm['false_positives']],
        [cm['false_negatives'], cm['true_positives']]
    ])
    fig = px.imshow(
        matrix, text_auto=True, color_continuous_scale=[[0, '#f1f5f9'], [1, TEAL]],
        x=['Predicted stay', 'Predicted churn'], y=['Stayed', 'Churned'],
        labels={'color': 'Customers'}
    )
    fig.update_layout(coloraxis_showscale=False, height=340, margin=dict(t=10, b=0))
    fig.update_traces(textfont_size=18)
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("Choosing the threshold")
    scored = score_test_customers()
    precision, recall, thresholds = precision_recall_curve(scored['churned'], scored['probability'])
    curve = pd.DataFrame({'Threshold': thresholds, 'Precision': precision[:-1], 'Recall': recall[:-1]})
    fig = go.Figure()
    fig.add_scatter(x=curve['Threshold'], y=curve['Precision'], name='Precision', line_color=TEAL)
    fig.add_scatter(x=curve['Threshold'], y=curve['Recall'], name='Recall', line_color=ORANGE)
    fig.add_vline(x=threshold, line_dash='dot', line_color=GRAY,
                  annotation_text=f"Chosen: {threshold:.2f}")
    fig.update_layout(
        height=340, margin=dict(t=10, b=0), xaxis_title='Threshold',
        yaxis_tickformat='.0%', legend=dict(orientation='h', y=-0.25)
    )
    st.plotly_chart(fig, width='stretch')
    st.caption("The threshold was set to maximise F1 on training data, before the test set was used.")

st.subheader("How the model was chosen")
st.markdown(
    "Eight models were compared with 5-fold cross-validation on the training set, ranked by "
    "PR-AUC, which suits data where only 14.5% of customers churn. Error bars show the "
    "spread across folds: the top three tree-based models are effectively tied."
)

results = pd.DataFrame(metrics['results']).sort_values('cv_pr_auc')
fig = px.bar(
    results, x='cv_pr_auc', y='model', orientation='h', error_x='cv_pr_auc_std',
    labels={'cv_pr_auc': 'Cross-validated PR-AUC', 'model': ''}
)
fig.update_traces(marker_color=[ORANGE if m == metrics['best_model'] else GRAY for m in results['model']])
fig.update_layout(height=360, margin=dict(t=10, b=0), xaxis_range=[0, 1])
st.plotly_chart(fig, width='stretch')

with st.expander("Full comparison table"):
    table = pd.DataFrame(metrics['results'])[
        ['model', 'cv_pr_auc', 'pr_auc', 'precision', 'recall', 'f1_score', 'roc_auc']
    ]
    st.dataframe(
        table, hide_index=True, width='stretch',
        column_config={
            'model': 'Model',
            'cv_pr_auc': st.column_config.ProgressColumn(
                'CV PR-AUC', format='%.3f', min_value=0, max_value=1),
            'pr_auc': st.column_config.NumberColumn('Test PR-AUC', format='%.3f'),
            'precision': st.column_config.NumberColumn('Precision', format='%.3f'),
            'recall': st.column_config.NumberColumn('Recall', format='%.3f'),
            'f1_score': st.column_config.NumberColumn('F1', format='%.3f'),
            'roc_auc': st.column_config.NumberColumn('ROC-AUC', format='%.3f'),
        }
    )
    st.caption("Test-set columns use a 0.5 threshold and uncalibrated models, so they differ "
               "slightly from the deployed model's numbers above.")
