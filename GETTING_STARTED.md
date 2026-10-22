# 🚀 Getting Started Guide

## Quick Start (5 Minutes)

### Step 1: Set Up the Environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Step 2: Launch the Dashboard
The trained model is already in `models/`, so the dashboard works right away:
```bash
streamlit run app/streamlit_app.py
```

### Step 3: Explore the Notebooks
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

### Step 4: Retrain and Test
```bash
python -m src.model_training   # retrain; updates the model, dashboard metrics and report
pytest                         # check everything still works
```

---

## 📚 What's in the Project

### Structure
```
✓ data/raw/ - Original dataset
✓ data/processed/ - Encoded data from notebook 02
✓ notebooks/ - EDA, feature engineering, modeling
✓ src/ - Preprocessing, training, utilities, plotting
✓ app/ - Streamlit dashboard
✓ models/ - Trained pipeline and its metrics
✓ reports/ - Model report and figures
✓ tests/ - Automated tests
```

### Machine Learning Pipeline
- **8 ML Algorithms**: Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, KNN
- **One Pipeline per model**: preprocessing is fitted inside cross-validation, so nothing leaks from validation data
- **Model selection**: 5-fold cross-validated PR-AUC on the training set
- **Calibration and threshold tuning**: on out-of-fold training predictions
- **Model Explainability**: permutation importance and SHAP
- **Business Metrics**: net benefit and ROI of a retention campaign

---

## 📊 Expected Results

### After Training (`python -m src.model_training` or notebook 03)
- **Selected model**: LightGBM with calibrated probabilities (Gradient Boosting and XGBoost are effectively tied)
- **Test set**: 96.8% precision, 82.8% recall, F1 0.892, PR-AUC 0.877
- **Saved files**:
  - `models/churn_pipeline.pkl`
  - `models/metrics.json`
  - `reports/model_report.txt`

Training takes a few minutes without hyperparameter tuning.

---

## 🐛 Troubleshooting

### Issue: Import Errors
```bash
# Run commands from the project root
cd customer-churn-analysis-Capestone-Project-

# Reinstall requirements
pip install -r requirements-dev.txt
```

### Issue: Model Fails to Load After Upgrading Packages
The saved model only loads reliably with the library versions in `requirements.txt`. After upgrading, retrain:
```bash
python -m src.model_training
```

### Issue: Model Not Found in Dashboard
```bash
python -m src.model_training
```

### Issue: Streamlit Port Already in Use
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

---

## 💡 Tips for Portfolio

### Highlight These Points:
1. **End-to-End Pipeline**: from raw data to a dashboard that serves the trained model
2. **Sound evaluation**: model selection, calibration and threshold tuning without touching the test set
3. **Business Focus**: ROI estimate with stated assumptions and a break-even point
4. **Explainability**: permutation importance and SHAP
5. **Tested code**: automated tests, including the saved model and every dashboard page

### Talking Points in Interviews:
- "Compared 8 algorithms and selected by cross-validated PR-AUC, because only 14.5% of customers churn and accuracy hides that"
- "The model catches 83% of churners, and 97% of the customers it flags really do churn"
- "Calibrated the probabilities: customers scored under 10% churned at 2.4% against 2.3% predicted"
- "Customers with 4+ service calls churn at 52% versus 11%, which points to a concrete intervention"
- "With a 30% retention success rate, a campaign returns about 190% ROI; it breaks even at about 10%"

Be ready to explain the limitations too: there is no time dimension in the data, and the business figures are assumptions.

---

## 📞 Need Help?

### Resources:
- **Scikit-Learn Docs**: https://scikit-learn.org/
- **XGBoost Docs**: https://xgboost.readthedocs.io/
- **SHAP Docs**: https://shap.readthedocs.io/
- **Streamlit Docs**: https://docs.streamlit.io/

### Common Questions:
**Q: Can I use my own data?**
A: Yes, if it has the same columns. Replace `data/raw/telecom_churn.csv`, update the feature lists in `src/config.py` if needed, and retrain.

**Q: How do I deploy this?**
A: Streamlit Cloud (free). Choose Python 3.13 in the app's advanced settings so the pinned requirements install.

**Q: Is this production-ready?**
A: Not yet. Production use would need monitoring for data drift, scheduled retraining, logging and real business figures for the ROI estimate.
