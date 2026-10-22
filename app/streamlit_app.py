"""
Customer Churn Prediction Dashboard
Interactive Streamlit application for churn analysis and prediction.
"""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.config import (
    RAW_DATA_PATH, PIPELINE_PATH, METRICS_PATH,
    AVG_CUSTOMER_VALUE, RETENTION_COST, RETENTION_SUCCESS_RATE,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES
)
from src.data_preprocessing import load_data
from src.utils import get_churn_risk_category, calculate_business_impact

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_app_data():
    """Load data for the app."""
    try:
        df = load_data()
        # Convert boolean Churn to Yes/No
        if df['Churn'].dtype == bool:
            df['Churn'] = df['Churn'].map({True: 'Yes', False: 'No'})
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def load_pipeline():
    """Load the trained pipeline (preprocessing + model)."""
    try:
        if not PIPELINE_PATH.exists():
            return None

        return joblib.load(PIPELINE_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_metrics():
    """Load evaluation results saved by the latest training run."""
    try:
        if not METRICS_PATH.exists():
            return None

        with open(METRICS_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None


def show_missing_metrics():
    """Explain how to generate metrics when none are saved."""
    st.warning(" No model results found. Please train the model first.")
    st.info("Run `python -m src.model_training` from the project root to train the model and save its results.")


def show_home():
    """Home page with overview."""
    st.markdown('<h1 class="main-header"> Customer Churn Prediction Dashboard</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    ## Welcome to the Customer Churn Analysis Platform

    This interactive dashboard helps telecom companies predict and prevent customer churn using
    advanced machine learning models.

    ###  Key Features:
    - **Real-time Churn Prediction**: Predict individual customer churn probability
    - **Interactive Visualizations**: Explore data patterns and trends
    - **Model Performance**: View detailed model metrics and comparisons
    - **Business Insights**: Understand factors driving churn
    - **Risk Assessment**: Identify high-risk customers for targeted retention

    ###  Project Highlights:
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Size", "3,333 customers")
    with col2:
        st.metric("Features", "20+ variables")
    with col3:
        metrics = load_metrics()
        best_f1 = f"{metrics['final_model']['f1_score']:.1%}" if metrics else "Not trained"
        st.metric("Best Model F1-Score", best_f1)
    with col4:
        st.metric("Algorithms", "8 models")

    st.markdown("""
    ---
    ###  Getting Started:
    1. **Explore Data**: Navigate to 'Data Exploration' to understand the dataset
    2. **View Model Performance**: Check 'Model Performance' for detailed metrics
    3. **Make Predictions**: Use 'Customer Prediction' to predict churn for individual customers
    4. **Business Insights**: Review 'Business Impact' for actionable recommendations

    ###  Technology Stack:
    - **ML Algorithms**: XGBoost, LightGBM, Random Forest, Logistic Regression, SVM
    - **Explainability**: SHAP values for model interpretation
    - **Framework**: Streamlit, Scikit-learn, Plotly
    - **Languages**: Python 3.11+
    """)


def show_data_exploration():
    """Data exploration page."""
    st.header(" Data Exploration")

    df = load_app_data()
    if df is None:
        return

    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.2f}%")
    with col3:
        st.metric("Features", df.shape[1])

    # Data preview
    st.subheader("Data Sample")
    st.dataframe(df.head(100), use_container_width=True)

    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # Churn distribution
    st.subheader("Churn Distribution")
    col1, col2 = st.columns(2)

    with col1:
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Churn Distribution",
            color_discrete_sequence=['#14b8a6', '#8b5cf6']
        )
        fig.update_layout(
            font=dict(family="Poppins, sans-serif", size=14),
            title_font=dict(size=20, color='#14b8a6', family="Poppins")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            x=churn_counts.index,
            y=churn_counts.values,
            title="Churn Count",
            labels={'x': 'Churn Status', 'y': 'Count'},
            color=churn_counts.index,
            color_discrete_sequence=['#14b8a6', '#8b5cf6']
        )
        fig.update_layout(
            font=dict(family="Poppins, sans-serif", size=14),
            title_font=dict(size=20, color='#14b8a6', family="Poppins")
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature analysis
    st.subheader("Feature Analysis")

    # Select feature to analyze
    feature = st.selectbox(
        "Select a feature to analyze:",
        ['International plan', 'Voice mail plan', 'Customer service calls',
         'Total day minutes', 'Total eve minutes', 'Account length']
    )

    if feature in ['Customer service calls', 'Total day minutes', 'Total eve minutes', 'Account length']:
        # Numerical feature analysis
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df, x=feature, color='Churn',
                title=f"{feature} Distribution by Churn",
                marginal="box",
                color_discrete_sequence=['#14b8a6', '#8b5cf6']
            )
            fig.update_layout(
                font=dict(family="Poppins, sans-serif", size=14),
                title_font=dict(size=20, color='#14b8a6', family="Poppins")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                df, x='Churn', y=feature,
                title=f"{feature} by Churn Status",
                color='Churn',
                color_discrete_sequence=['#14b8a6', '#8b5cf6']
            )
            fig.update_layout(
                font=dict(family="Poppins, sans-serif", size=14),
                title_font=dict(size=20, color='#14b8a6', family="Poppins")
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Categorical feature analysis
        churn_by_feature = pd.crosstab(df[feature], df['Churn'], normalize='index') * 100

        fig = px.bar(
            churn_by_feature,
            title=f"Churn Rate by {feature}",
            labels={'value': 'Percentage (%)', 'variable': 'Churn Status'},
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_model_performance():
    """Model performance page."""
    st.header(" Model Performance")

    st.markdown("""
    This section displays the performance metrics of various machine learning models
    trained on the customer churn dataset.
    """)

    metrics = load_metrics()
    if metrics is None:
        show_missing_metrics()
        return

    results_df = pd.DataFrame(metrics['results']).rename(columns={
        'model': 'Model',
        'cv_pr_auc': 'CV PR-AUC',
        'pr_auc': 'PR-AUC',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'roc_auc': 'ROC-AUC'
    })[['Model', 'CV PR-AUC', 'PR-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]

    st.subheader("Model Comparison")
    st.caption("Models are ranked by PR-AUC from 5-fold cross-validation on the training set, "
               "so the test set plays no part in choosing the model. The other columns are "
               f"measured on a held-out test set of {metrics['test_size']:,} customers at a 0.5 threshold.")
    metric_columns = ['CV PR-AUC', 'PR-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    st.dataframe(results_df.style.highlight_max(axis=0, subset=metric_columns),
                 use_container_width=True)

    # Visualize metrics
    st.subheader("Performance Metrics Comparison")

    metric_choice = st.selectbox(
        "Select metric to visualize:",
        metric_columns
    )

    fig = px.bar(
        results_df.sort_values(metric_choice, ascending=False),
        x='Model', y=metric_choice,
        title=f"{metric_choice} Comparison Across Models",
        color=metric_choice,
        color_continuous_scale='Teal'
    )
    fig.update_layout(
        font=dict(family="Poppins, sans-serif", size=14),
        title_font=dict(size=20, color='#14b8a6', family="Poppins")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Best model details
    final = metrics['final_model']
    st.subheader(f" Deployed Model: {metrics['best_model']} (calibrated)")
    st.caption(f"Probabilities are calibrated, and customers at or above {metrics['threshold']:.2f} are "
               "flagged as churners. Both were set on training data only; these numbers are "
               "the model's single evaluation on the test set.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{final['accuracy']:.1%}")
    with col2:
        st.metric("Precision", f"{final['precision']:.1%}")
    with col3:
        st.metric("Recall", f"{final['recall']:.1%}")
    with col4:
        st.metric("F1-Score", f"{final['f1_score']:.1%}")

    st.markdown(f"""
    ### Key Insights:
    - **{metrics['best_model']}** has the highest cross-validated PR-AUC; the top tree-based models are within noise of each other
    - It catches **{final['recall']:.0%}** of churners; when it flags a customer, it is right **{final['precision']:.0%}** of the time
    - **Daytime and evening charges**, **customer service calls** and **international plan** matter most to its predictions
    - Customers with **4+ customer service calls** churn at about 52%, versus 11% for everyone else
    - **International plan** customers churn at about 42%, versus 11% without one
    - **PR-AUC of {final['pr_auc']:.3f}** (vs. 0.145 for random guessing) shows strong ability to find churners
    """)


def show_customer_prediction():
    """Customer prediction page."""
    st.header(" Customer Churn Prediction")

    pipeline = load_pipeline()

    if pipeline is None:
        st.warning(" Model not loaded. Please train the model first.")
        st.info("Run `python -m src.model_training` from the project root to train and save the model.")
        return

    metrics = load_metrics()
    threshold = metrics['threshold'] if metrics else 0.5

    st.markdown("""
    Enter customer information below to predict churn probability and risk level.
    """)

    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Account Information")
        state = st.selectbox("State", ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                                       "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                                       "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                                       "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                                       "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"])
        account_length = st.slider("Account Length (days)", 0, 250, 100)
        area_code = st.selectbox("Area Code", [408, 415, 510])

        st.subheader("Services")
        international_plan = st.selectbox("International Plan", ["No", "Yes"])
        voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
        number_vmail_messages = st.slider("Number of Voicemail Messages", 0, 50, 0)

    with col2:
        st.subheader("Usage Information")
        total_day_minutes = st.number_input("Total Day Minutes", 0.0, 400.0, 150.0)
        total_day_calls = st.slider("Total Day Calls", 0, 200, 100)

        total_eve_minutes = st.number_input("Total Evening Minutes", 0.0, 400.0, 180.0)
        total_eve_calls = st.slider("Total Evening Calls", 0, 200, 100)

        total_night_minutes = st.number_input("Total Night Minutes", 0.0, 400.0, 200.0)
        total_night_calls = st.slider("Total Night Calls", 0, 200, 100)

        total_intl_minutes = st.number_input("Total International Minutes", 0.0, 30.0, 10.0)
        total_intl_calls = st.slider("Total International Calls", 0, 20, 3)

        customer_service_calls = st.slider("Customer Service Calls", 0, 10, 1)

    if st.button(" Predict Churn", type="primary"):
        # Calculate charges (typically charge = minutes * rate)
        total_day_charge = total_day_minutes * 0.17
        total_eve_charge = total_eve_minutes * 0.085
        total_night_charge = total_night_minutes * 0.045
        total_intl_charge = total_intl_minutes * 0.27

        # Create customer data dictionary
        customer_data = {
            'State': state,
            'Account length': account_length,
            'Area code': area_code,
            'International plan': international_plan,
            'Voice mail plan': voice_mail_plan,
            'Number vmail messages': number_vmail_messages,
            'Total day minutes': total_day_minutes,
            'Total day calls': total_day_calls,
            'Total day charge': total_day_charge,
            'Total eve minutes': total_eve_minutes,
            'Total eve calls': total_eve_calls,
            'Total eve charge': total_eve_charge,
            'Total night minutes': total_night_minutes,
            'Total night calls': total_night_calls,
            'Total night charge': total_night_charge,
            'Total intl minutes': total_intl_minutes,
            'Total intl calls': total_intl_calls,
            'Total intl charge': total_intl_charge,
            'Customer service calls': customer_service_calls
        }

        # The pipeline handles feature engineering, scaling and encoding
        churn_probability = float(pipeline.predict_proba(pd.DataFrame([customer_data]))[0, 1])
        risk_level = get_churn_risk_category(churn_probability)

        st.success(" Prediction generated successfully!")

        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Churn Probability", f"{churn_probability*100:.1f}%")
        with col2:
            risk_color = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🟠", "Very High Risk": "🔴"}
            st.metric("Risk Level", f"{risk_color.get(risk_level, '')} {risk_level}")
        with col3:
            recommendation = "Immediate Retention Action" if churn_probability >= threshold else "Monitor"
            st.metric("Recommendation", recommendation)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_probability * 100,
            title={'text': "Churn Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"The red line marks the model's decision threshold ({threshold:.0%}). "
                   "Customers at or above it are flagged for retention.")

        # Key factors
        st.subheader(" Key Risk Factors")
        if customer_service_calls >= 3:
            st.error(f" High customer service calls ({customer_service_calls}) - Strong churn indicator")
        if international_plan == "Yes":
            st.warning(" International plan active - Higher churn risk")
        if voice_mail_plan == "No":
            st.info("ℹ No voicemail plan - Consider offering value-added services")

        # Recommendations
        st.subheader(" Retention Recommendations")
        if churn_probability >= threshold:
            st.error(" HIGH RISK CUSTOMER")
            st.markdown("""
            **Immediate Actions:**
            - Contact customer within 24 hours
            - Offer personalized retention package
            - Investigate recent service issues
            - Consider special discount or plan upgrade
            - Assign dedicated account manager
            """)
        elif churn_probability > 0.3:
            st.warning(" MEDIUM RISK CUSTOMER")
            st.markdown("""
            **Preventive Actions:**
            - Schedule proactive check-in call
            - Offer loyalty rewards or discounts
            - Promote additional services (voicemail, etc.)
            - Monitor usage patterns
            - Send satisfaction survey
            """)
        else:
            st.success(" LOW RISK CUSTOMER")
            st.markdown("""
            **Maintenance Actions:**
            - Continue regular service
            - Offer upgrade opportunities
            - Request feedback and reviews
            - Maintain service quality
            """)


def show_business_impact():
    """Business impact page."""
    st.header(" Business Impact Analysis")

    st.markdown("""
    Understanding the business impact of churn prediction helps justify investment
    in retention programs and demonstrates ROI.
    """)

    # Business metrics inputs
    st.subheader(" Configure Business Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_customer_value = st.number_input(
            "Average Customer Lifetime Value ($)",
            value=AVG_CUSTOMER_VALUE, min_value=100, max_value=10000, step=100
        )
    with col2:
        retention_cost = st.number_input(
            "Cost per Retention Effort ($)",
            value=RETENTION_COST, min_value=10, max_value=500, step=10
        )
    with col3:
        retention_success_rate = st.slider(
            "Retention Success Rate (%)", 5, 100, int(RETENTION_SUCCESS_RATE * 100), step=5,
            help="Share of contacted churners who accept the offer and stay"
        ) / 100

    metrics = load_metrics()
    if metrics is None:
        show_missing_metrics()
        return

    # Confusion matrix of the best model on the test set
    confusion = metrics['confusion_matrix']
    tp = confusion['true_positives']
    fp = confusion['false_positives']
    fn = confusion['false_negatives']
    tn = confusion['true_negatives']
    st.caption(f"Based on the {metrics['best_model']} model's predictions for "
               f"{metrics['test_size']:,} test-set customers. Net benefit is compared with "
               "running no retention campaign: every flagged customer is contacted, and only "
               "the chosen share of true churners stays.")

    impact = calculate_business_impact(
        tp, fp, fn, tn,
        avg_customer_value=avg_customer_value,
        retention_cost=retention_cost,
        retention_success_rate=retention_success_rate
    )

    # Display metrics
    st.subheader(" Financial Impact")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Customers Contacted", tp + fp)
        st.metric("Cost of Retention", f"${impact['cost_of_retention']:,.0f}")

    with col2:
        st.metric("Customers Saved", f"{impact['customers_saved']:.0f}")
        st.metric("Value Retained", f"${impact['value_retained']:,.0f}")

    with col3:
        st.metric("Missed Churners", fn)
        st.metric("Revenue Still Lost", f"${impact['cost_of_missed_churners']:,.0f}")

    with col4:
        st.metric("Net Benefit", f"${impact['net_benefit']:,.0f}")
        st.metric("ROI", f"{impact['roi']:.1f}%")

    # Visualizations
    st.subheader(" Financial Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        # Where churners end up
        fig = go.Figure(data=[
            go.Bar(x=['Saved by Campaign', 'Contacted but Left', 'Missed by Model'],
                   y=[impact['customers_saved'], tp - impact['customers_saved'], fn],
                   marker_color=['#14b8a6', 'lightsalmon', 'indianred'])
        ])
        fig.update_layout(title='Outcome for Actual Churners', yaxis_title='Customers')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROI waterfall
        fig = go.Figure(go.Waterfall(
            x=['Value Retained', 'Retention Cost', 'Net Benefit'],
            measure=['relative', 'relative', 'total'],
            y=[impact['value_retained'], -impact['cost_of_retention'], 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(title='ROI Waterfall', yaxis_title='Amount ($)')
        st.plotly_chart(fig, use_container_width=True)

    # Success rate at which the campaign pays for itself
    break_even_rate = impact['cost_of_retention'] / (tp * avg_customer_value) if tp > 0 else float('inf')

    # Key insights
    st.subheader(" Key Insights")

    st.markdown(f"""
    ### Model Impact Summary:
    - **Correctly Identified Churners**: {tp} of {tp + fn} ({tp / (tp + fn):.0%})
    - **False Alarms**: {fp} customers contacted unnecessarily
    - **Total Value Protected**: ${impact['value_retained']:,.0f}
    - **Return on Investment**: {impact['roi']:.1f}%
    - **Break-even**: the campaign pays for itself if at least {break_even_rate:.0%} of contacted churners stay

    ### Recommendations:
    1. **Prioritize High-Risk Customers**: Focus retention on customers with 4+ service calls
    2. **International Plan Review**: Investigate why intl plan customers churn more
    3. **Service Quality**: Reduce customer service calls through proactive support
    """)


def main():
    """Main application."""
    # Sidebar
    with st.sidebar:
        st.image("app/assets/CustomerDesign.png", width=80)
        st.title("Navigation")

        page = st.radio(
            "Go to",
            [" Home", " Data Exploration", " Model Performance",
             " Customer Prediction", " Business Impact"]
        )

        st.markdown("---")
        st.markdown("""
        ### About
        This dashboard provides comprehensive
        customer churn analysis and prediction
        capabilities for telecom companies.

        **Version**: 1.0.0
        **Dataset**: 3,333 customers
        """)

    # Main content
    if page == " Home":
        show_home()
    elif page == " Data Exploration":
        show_data_exploration()
    elif page == " Model Performance":
        show_model_performance()
    elif page == " Customer Prediction":
        show_customer_prediction()
    elif page == " Business Impact":
        show_business_impact()


if __name__ == "__main__":
    main()