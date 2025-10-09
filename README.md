# Customer Churn Prediction - Telecom Industry

A comprehensive machine learning project for predicting customer churn in the telecom industry, featuring advanced ML algorithms, model explainability with SHAP, and an interactive Streamlit dashboard for business insights.


## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Dashboard Preview](#dashboard-preview)
- [Notebooks](#notebooks)


---

## Overview

Customer churn (customer attrition) is a critical business metric that directly impacts revenue and growth. This project leverages machine learning to predict which customers are likely to churn, enabling proactive retention strategies.

### Problem Statement

Telecom companies face significant challenges with customer retention. The cost of acquiring new customers is typically 5-7 times higher than retaining existing ones. This project aims to:

- **Identify** customers at risk of churning
- **Understand** key factors driving churn
- **Provide** actionable insights for retention strategies
- **Quantify** business impact and ROI of retention efforts

### Solution

We developed an end-to-end machine learning pipeline that:
1. Processes and analyzes customer data
2. Engineers relevant features for prediction
3. Trains and evaluates multiple ML models
4. Provides model explainability using SHAP
5. Delivers predictions through an interactive dashboard

---

## Key Features

### Machine Learning
- **8 ML Algorithms**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, KNN
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV
- **Cross-Validation**: 5-fold stratified cross-validation
- **Model Explainability**: SHAP (SHapley Additive exPlanations) values
- **Feature Engineering**: Advanced feature creation and selection

### Data Analysis
- Comprehensive exploratory data analysis (EDA)
- Advanced visualizations with Matplotlib, Seaborn, and Plotly
- Correlation analysis and feature importance
- Business impact analysis

### Interactive Dashboard
- **Real-time Predictions**: Individual customer churn probability
- **Data Exploration**: Interactive visualizations
- **Model Performance**: Detailed metrics and comparisons
- **Business Insights**: ROI calculations and recommendations
- **Risk Assessment**: Customer segmentation by churn risk

### Production-Ready Code
- Modular, reusable Python modules
- Comprehensive documentation
- Unit tests (framework ready)
- Configuration management
- Error handling and logging


## Technologies Used

### Core Technologies
- **Python 3.11+**: Programming language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning framework

### Advanced ML
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Fast gradient boosting
- **SHAP**: Model interpretability

### Visualization
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts

### Web Framework
- **Streamlit**: Interactive dashboard

### Development Tools
- **Jupyter**: Notebook environment
- **Git**: Version control
- **Virtual Environment**: Dependency isolation

---

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Getrudetrudy01/customer-churn-analysis-Capestone-Project-.git
   cd customer-churn-analysis-Capestone-Project-
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import sklearn; import xgboost; import lightgbm; import streamlit; print('All packages installed successfully!')"
   ```


## Dashboard Preview

### Home Page
- Project overview and key metrics
- Quick navigation to all features

### Data Exploration
- Interactive visualizations
- Statistical summaries
- Feature distributions by churn status

### Model Performance
- Model comparison charts
- Detailed performance metrics
- Best model insights

### Customer Prediction
- Real-time churn probability prediction
- Risk level assessment
- Personalized retention recommendations

### Business Impact
- ROI calculator
- Financial impact analysis
- Cost-benefit visualizations

---

## Notebooks

### 01_exploratory_data_analysis.ipynb
- Dataset overview and quality assessment
- Univariate, bivariate, and multivariate analysis
- Correlation analysis
- Key insights and findings

### 02_feature_engineering.ipynb
- Data cleaning and transformation
- Feature creation (tenure groups, service count, etc.)
- Feature encoding and scaling
- Feature selection and importance

### 03_modeling.ipynb
- Multiple model training
- Hyperparameter tuning
- Model evaluation and comparison
- SHAP explainability analysis
- Business impact calculation


