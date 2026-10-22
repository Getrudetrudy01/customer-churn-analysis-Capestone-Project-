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
5. Serves predictions through an interactive dashboard

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

**Business impact (illustrative):** if 30% of contacted churners accept a retention offer, contacting every customer the model flags returns about $23,600 net on the test set (190% ROI), assuming a $1,000 customer value and a $100 cost per contact. These figures are assumptions, not measurements; the dashboard lets you change them, and the campaign breaks even at roughly a 10% success rate.

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
- **Real-time Predictions**: churn probability for an individual customer from the trained model
- **Data Exploration**: interactive visualizations
- **Model Performance**: metrics from the latest training run
- **Business Impact**: adjustable ROI estimate based on the model's test-set results

### Code Quality
- Modular Python package (`src/`) with configuration in one place
- Automated tests for preprocessing, the business calculations, the saved model and every dashboard page
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
├── app/
│   ├── streamlit_app.py        # Dashboard
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
│   ├── model_training.py       # Model comparison, calibration, threshold tuning
│   ├── utils.py                # Business impact, SHAP, reporting
│   └── visualization.py        # Plotting helpers
└── tests/
```

---

## Dashboard Preview

### Home Page
- Project overview and key metrics
- Quick navigation to all features

### Data Exploration
- Interactive visualizations
- Statistical summaries
- Feature distributions by churn status

### Model Performance
- Model comparison table and charts
- Performance of the deployed model
- Key churn drivers

### Customer Prediction
- Churn probability from the trained model
- Risk level and the model's decision threshold
- Retention recommendations

### Business Impact
- Adjustable customer value, contact cost and success rate
- Net benefit and ROI compared with no campaign
- Break-even success rate

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
- Several features are near-duplicates (each charge column is a fixed multiple of its minutes column), and `State` adds 51 sparse columns; a simpler feature set would likely perform just as well.
