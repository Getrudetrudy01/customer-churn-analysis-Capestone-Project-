"""
Utility functions for customer churn prediction project.
"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Any, List
from src.config import (
    FIGURES_DIR,
    AVG_CUSTOMER_VALUE,
    RETENTION_COST,
    RETENTION_SUCCESS_RATE
)


def explain_model_shap(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    plot_type: str = 'summary',
    max_display: int = 20,
    save: bool = False
) -> shap.Explainer:
    """
    Generate SHAP explanations for model predictions.

    Args:
        model: Trained model
        X_train: Training data for background
        X_test: Test data to explain
        feature_names: List of feature names
        plot_type: Type of SHAP plot ('summary', 'waterfall', 'force', 'bar')
        max_display: Maximum number of features to display
        save: Whether to save the plot

    Returns:
        SHAP explainer object
    """
    print("Generating SHAP explanations...")

    # Create explainer
    try:
        # For tree-based models
        explainer = shap.TreeExplainer(model)
    except:
        # For other models
        explainer = shap.Explainer(model, X_train)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)

    # Handle multi-output (for binary classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Get positive class

    # Generate plots
    if plot_type == 'summary':
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        if save:
            plt.savefig(FIGURES_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

    elif plot_type == 'bar':
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        if save:
            plt.savefig(FIGURES_DIR / 'shap_bar.png', dpi=300, bbox_inches='tight')
        plt.show()

    elif plot_type == 'waterfall':
        # Waterfall plot for first instance
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                data=X_test[0],
                feature_names=feature_names
            ),
            max_display=max_display,
            show=False
        )
        if save:
            plt.savefig(FIGURES_DIR / 'shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.show()

    print("SHAP explanations generated successfully!")
    return explainer


def get_shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Calculate feature importance from SHAP values.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance
    """
    # Calculate mean absolute SHAP values
    importance = np.abs(shap_values).mean(axis=0)

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)

    return importance_df


def calculate_churn_probability(
    model: Any,
    customer_data: pd.DataFrame,
    feature_names: List[str] = None
) -> float:
    """
    Calculate churn probability for a single customer.

    Args:
        model: Trained model
        customer_data: Customer data (single row)
        feature_names: Expected feature names (for validation)

    Returns:
        Churn probability
    """
    if feature_names is not None:
        # Ensure all features are present
        missing_features = set(feature_names) - set(customer_data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

    # Get probability
    churn_prob = model.predict_proba(customer_data)[0, 1]

    return churn_prob


def get_churn_risk_category(probability: float) -> str:
    """
    Categorize churn risk based on probability.

    Args:
        probability: Churn probability (0-1)

    Returns:
        Risk category string
    """
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    elif probability < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"


def calculate_business_impact(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    avg_customer_value: float = AVG_CUSTOMER_VALUE,
    retention_cost: float = RETENTION_COST,
    retention_success_rate: float = RETENTION_SUCCESS_RATE
) -> dict:
    """
    Estimate the net benefit of contacting every customer the model flags,
    compared with running no retention campaign at all.

    Every flagged customer (TP + FP) costs one retention effort. Only true churners
    can be saved, and only retention_success_rate of them accept the offer.
    Churners the model misses are lost with or without the model, so their revenue
    is reported but not counted as a cost of the model.

    Args:
        tp, fp, fn, tn: Confusion matrix counts
        avg_customer_value: Lifetime value of a retained customer
        retention_cost: Cost of one retention effort
        retention_success_rate: Share of contacted churners who stay

    Returns:
        Dictionary with business metrics
    """
    customers_saved = tp * retention_success_rate
    cost_of_retention = (tp + fp) * retention_cost
    value_retained = customers_saved * avg_customer_value
    net_benefit = value_retained - cost_of_retention

    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'retention_success_rate': retention_success_rate,
        'customers_saved': customers_saved,
        'cost_of_missed_churners': fn * avg_customer_value,
        'cost_of_retention': cost_of_retention,
        'value_retained': value_retained,
        'net_benefit': net_benefit,
        'roi': (net_benefit / cost_of_retention * 100) if cost_of_retention > 0 else 0
    }


def format_business_metrics(metrics: dict) -> str:
    """
    Format business metrics for display.

    Args:
        metrics: Dictionary of business metrics

    Returns:
        Formatted string
    """
    formatted = []
    formatted.append("=" * 50)
    formatted.append("BUSINESS IMPACT METRICS")
    formatted.append("=" * 50)

    formatted.append("\nConfusion Matrix Results:")
    formatted.append(f"  [+] True Positives (Correctly identified churners): {metrics['true_positives']}")
    formatted.append(f"  [-] False Positives (False alarms): {metrics['false_positives']}")
    formatted.append(f"  [!] False Negatives (Missed churners): {metrics['false_negatives']}")
    formatted.append(f"  [*] True Negatives (Correctly identified non-churners): {metrics['true_negatives']}")

    formatted.append(f"\nFinancial Impact (assuming {metrics['retention_success_rate']:.0%} of contacted churners stay):")
    formatted.append(f"  Customers Saved: {metrics['customers_saved']:.1f}")
    formatted.append(f"  Cost of Retention Efforts: ${metrics['cost_of_retention']:,.2f}")
    formatted.append(f"  Value of Retained Customers: ${metrics['value_retained']:,.2f}")
    formatted.append(f"  Net Benefit vs. No Campaign: ${metrics['net_benefit']:,.2f}")
    formatted.append(f"  ROI: {metrics['roi']:.2f}%")
    formatted.append(f"  Revenue Still Lost to Missed Churners: ${metrics['cost_of_missed_churners']:,.2f}")

    formatted.append("\n" + "=" * 50)

    return "\n".join(formatted)


def print_model_summary(results_df: pd.DataFrame) -> None:
    """
    Print a formatted summary of model results.

    Args:
        results_df: DataFrame with model results
    """
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE SUMMARY".center(80))
    print("=" * 80)

    for idx, row in results_df.iterrows():
        print(f"\n{row['model']}:")
        print(f"  Accuracy:  {row['accuracy']:.4f}")
        print(f"  Precision: {row['precision']:.4f}")
        print(f"  Recall:    {row['recall']:.4f}")
        print(f"  F1-Score:  {row['f1_score']:.4f}")
        if 'roc_auc' in row:
            print(f"  ROC-AUC:   {row['roc_auc']:.4f}")
        if 'cv_pr_auc' in row:
            print(f"  CV PR-AUC: {row['cv_pr_auc']:.4f} (+/- {row['cv_pr_auc_std']:.4f})")

    print("\n" + "=" * 80)
    print(f"🏆 BEST MODEL: {results_df.iloc[0]['model']} (CV PR-AUC: {results_df.iloc[0]['cv_pr_auc']:.4f})")
    print("=" * 80)


def generate_model_report(
    results_df: pd.DataFrame,
    business_metrics: dict,
    output_file: str = 'model_report.txt',
    final_metrics: dict = None,
    threshold: float = 0.5
) -> None:
    """
    Generate a comprehensive model report.

    Args:
        results_df: DataFrame with model results
        business_metrics: Dictionary of business metrics
        output_file: Output filename
        final_metrics: Test-set metrics of the calibrated final model at its threshold
        threshold: Decision threshold of the final model
    """
    from src.config import REPORTS_DIR

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CUSTOMER CHURN PREDICTION - MODEL REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("MODEL PERFORMANCE COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append("Ranked by cross-validated PR-AUC on the training set (cv_pr_auc).")
    report_lines.append("Test-set columns use the default 0.5 threshold.")
    report_lines.append(results_df.to_string(index=False))
    report_lines.append("")

    if final_metrics is not None:
        report_lines.append("FINAL MODEL (calibrated, test set)")
        report_lines.append("-" * 80)
        report_lines.append(f"  Model: {final_metrics['model']}")
        report_lines.append(f"  Decision threshold: {threshold:.3f}")
        for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']:
            report_lines.append(f"  {key}: {final_metrics[key]:.4f}")
        report_lines.append("")

    report_lines.append(format_business_metrics(business_metrics))

    report_lines.append("\n" + "=" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("=" * 80)

    best_model = results_df.iloc[0]['model']
    report_lines.append(f"1. Deploy {best_model} for production use")
    report_lines.append(f"2. Contact customers whose churn probability is at least {threshold:.2f}")
    report_lines.append("3. Monitor model performance monthly and retrain as needed")
    report_lines.append("4. Implement A/B testing to validate retention strategies")

    report = "\n".join(report_lines)

    # Save to file with UTF-8 encoding to handle emojis
    output_path = REPORTS_DIR / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    print("Utility functions loaded successfully!")