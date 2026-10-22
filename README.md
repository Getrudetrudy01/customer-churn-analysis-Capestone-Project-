# Customer Churn Prediction - Telecom Industry

A machine learning project that predicts which telecom customers are likely to churn, explains what drives churn, and serves predictions through an interactive Streamlit dashboard.


## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dashboard Preview](#dashboard-preview)
- [Notebooks](#notebooks)
- [Limitations](#limitations)


---

## Overview

Customer churn (customer attrition) is a critical business metric that directly impacts revenue and growth. This project uses machine learning to predict which customers are likely to churn, enabling proactive retention strategies.

### Problem Statement

Telecom companies face significant challenges with customer retention. The cost of acquiring new customers is typically 5-7 times higher than retaining existing ones. This project aims to:

- **Identify** customers at risk of churning
- **Understand** key factors driving churn
- **Provide** actionable insights for retention strategies
- **Estimate** the business impact of a retention campaign

### Dataset

3,333 telecom customers with 19 features (account details, plan subscriptions, call minutes, charges and customer service calls by time of day). 14.5% of customers churned, so the classes are imbalanced.

### Solution

An end-to-end machine learning pipeline that:
1. Cleans the data and engineers usage features
2. Compares 8 models, chosen by cross-validated performance on the training data
3. Calibrates the chosen model's probabilities and tunes its decision threshold
4. Explains predictions with permutation importance and SHAP
5. Serves predictions through an interactive dashboard that explains each customer's score, tests retention actions and plans a campaign

---

## Results

The selected model is **LightGBM with calibrated probabilities**, evaluated once on a held-out test set of 1,000 customers:

| Metric | Value |
|---|---|
| Precision (flagged customers who really churn) | 96.8% |
| Recall (churners the model catches) | 82.8% |
| F1-score | 0.892 |
| PR-AUC (random guessing: 0.145) | 0.877 |

Gradient Boosting and XGBoost performed almost identically in cross-validation, so the top three tree-based models are effectively tied. Logistic Regression, SVM and KNN performed far worse. The full comparison is in [reports/model_report.txt](reports/model_report.txt), and both it and the dashboard are regenerated every time the model is trained.

**What drives churn:**
- **Daytime and evening charges**: heavy users are much more likely to leave
- **Customer service calls**: customers with 4+ calls churn at 52%, versus 11% for everyone else
- **International plan**: subscribers churn at 42%, versus 11% without one
- **Account length** has almost no relationship with churn

**Business impact (illustrative):** assuming a $1,000 customer value, a $100 offer and 30% of contacted churners staying, a customer is worth contacting when their churn risk is above 33%. On the test set that means 127 calls and about $23,600 net benefit (190% ROI). The best result possible with hindsight is $24,200, so this simple rule, chosen without seeing outcomes, gets within 3% of it; it works because the model's probabilities are calibrated. The business figures are assumptions, not measurements, and the dashboard lets you change them.

---

## Key Features

### Machine Learning
- **8 ML Algorithms**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, KNN
- **Leak-free evaluation**: every model is a single scikit-learn Pipeline, so preprocessing is fitted on training folds only
- **Model selection**: 5-fold stratified cross-validated PR-AUC on the training set; the test set is used once, for the final evaluation
- **Calibration and threshold tuning**: calibrated probabilities and an F1-optimal threshold, both set on out-of-fold training predictions
- **Optional hyperparameter tuning**: RandomizedSearchCV for XGBoost and LightGBM, adopted only if it beats the cross-validated baseline
- **Model Explainability**: permutation importance and SHAP values

### Interactive Dashboard
- **Live predictions with explanations**: each customer's churn risk, the factors raising or lowering it, and which retention actions would lower it most
- **Campaign planner**: a break-even rule for who to contact, a profit curve, and a ranked call list you can download
- **Model performance**: metrics, confusion matrix and threshold trade-off from the latest training run
- **Data exploration**: churn rates by usage, plans and service calls
- Light and dark mode, and a layout that works on phones

### Code Quality
- Modular Python package (`src/`) with configuration in one place
- Automated tests for preprocessing, explanations, the business calculations, the saved model and every dashboard page
- Pinned dependencies, so the saved model loads with the library versions it was trained with


## Technologies Used

- **Python**, **Pandas**, **NumPy**, **Scikit-learn**
- **XGBoost** and **LightGBM** for gradient boosting
- **SHAP** for model interpretability
- **Matplotlib**, **Seaborn** and **Plotly** for visualization
- **Streamlit** for the dashboard
- **pytest** and **Jupyter** for development

---

## Installation

### Prerequisites
- Python 3.13 (the pinned requirements were tested with it)
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
   pip install -r requirements.txt        # to run the dashboard
   pip install -r requirements-dev.txt    # also for tests and notebooks
   ```

---

## Usage

Run all commands from the project root.

**Launch the dashboard** (uses the trained model in `models/`):
```bash
streamlit run app/streamlit_app.py
```

**Retrain the model** (updates `models/churn_pipeline.pkl`, `models/metrics.json` and `reports/model_report.txt`):
```bash
python -m src.model_training
```

**Run the tests:**
```bash
pytest
```

**Deploying to Streamlit Cloud:** choose Python 3.13 in the app's advanced settings so the pinned requirements install and the saved model loads.

---

## Project Structure

```
├── .streamlit/config.toml      # Dashboard theme
├── app/
│   ├── streamlit_app.py        # Dashboard entry point and navigation
│   ├── common.py               # Shared loading, scoring and colours
│   ├── views/                  # One file per dashboard page
│   └── assets/
├── data/
│   ├── raw/telecom_churn.csv   # Original dataset
│   └── processed/              # Encoded data written by notebook 02
├── models/
│   ├── churn_pipeline.pkl      # Preprocessing + calibrated model
│   └── metrics.json            # Results read by the dashboard
├── notebooks/                  # EDA, feature engineering, modeling
├── reports/                    # Model report and figures
├── src/
│   ├── config.py               # Paths, feature lists, business assumptions
│   ├── data_preprocessing.py   # Cleaning, feature engineering, preprocessing pipeline
│   ├── explain.py              # Per-customer explanations and retention actions
│   ├── model_training.py       # Model comparison, calibration, threshold tuning
│   ├── utils.py                # Business impact, SHAP, reporting
│   └── visualization.py        # Plotting helpers
└── tests/
```

---

## Dashboard Preview

### Overview
- Headline results: churn rate, share of churners caught, accuracy of flags
- What the model relies on and where churn concentrates

### Explore the Data
- Distributions and churn rates for each attribute
- Raw data and summary statistics on demand

### Model Performance
- Precision, recall, F1 and PR-AUC of the deployed model
- Confusion matrix and how precision and recall change with the threshold
- How the eight models compared in cross-validation

### Predict a Customer
- Churn risk that updates as you change any input, with example customers to start from
- **Why this score**: how much each factor raises or lowers the risk compared with a typical customer
- **What would lower the risk**: the model's estimate for actions such as resolving service issues

### Plan a Campaign
- Adjustable customer value, offer cost and acceptance rate
- Break-even risk: who is worth contacting
- Profit curve and a downloadable call list with each customer's main risk factor and suggested action

---

## Notebooks

### 01_exploratory_data_analysis.ipynb
- Dataset overview and quality assessment
- Univariate, bivariate, and multivariate analysis
- Correlation analysis
- Key insights and findings

### 02_feature_engineering.ipynb
- Data cleaning and transformation
- Feature creation (total usage and charges, charge per minute, high-usage and service-call flags)
- Feature encoding and scaling
- Correlation and multicollinearity checks

### 03_modeling.ipynb
- Model training, selection, calibration and threshold tuning (same code as `python -m src.model_training`)
- Model comparison charts, confusion matrix and ROC curve
- Permutation importance and SHAP explainability
- Business impact estimate

---

## Limitations

- The dataset has no dates, so the model cannot be checked on newer customers than it was trained on; performance on future customers may be lower.
- The business impact depends on assumed customer value, contact cost and retention success rate, which should be replaced with real figures.
- Explanations and suggested actions show patterns the model learned from historical data, not proven causal effects. Measuring what an offer actually changes would need data on who was contacted, which this dataset does not have.
- Several features are near-duplicates (each charge column is a fixed multiple of its minutes column), and `State` adds 51 sparse columns; a simpler feature set would likely perform just as well.
