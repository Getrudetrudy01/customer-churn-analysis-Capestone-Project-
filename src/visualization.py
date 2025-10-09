"""
Visualization module for customer churn analysis.
Contains reusable plotting functions for EDA and model evaluation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from typing import List, Tuple
import plotly.express as px
import plotly.graph_objects as go

from src.config import FIGURES_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_numerical_distribution(
    df: pd.DataFrame,
    numerical_cols: List[str],
    target_col: str = 'Churn',
    save: bool = False
) -> None:
    """
    Plot distribution of numerical features by target variable.

    Args:
        df: DataFrame containing the data
        numerical_cols: List of numerical column names
        target_col: Target column name
        save: Whether to save the plot
    """
    fig, axes = plt.subplots(1, len(numerical_cols), figsize=(16, 4))

    if len(numerical_cols) == 1:
        axes = [axes]

    for idx, col in enumerate(numerical_cols):
        df[df[target_col] == 'No'][col].hist(bins=30, color='blue', alpha=0.5, ax=axes[idx], label='No Churn')
        df[df[target_col] == 'Yes'][col].hist(bins=30, color='red', alpha=0.5, ax=axes[idx], label='Churn')
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'numerical_distribution.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_categorical_distribution(
    df: pd.DataFrame,
    categorical_cols: List[str],
    rows: int = 4,
    cols: int = 4,
    save: bool = False
) -> None:
    """
    Plot distribution of categorical features.

    Args:
        df: DataFrame containing the data
        categorical_cols: List of categorical column names
        rows: Number of rows in subplot grid
        cols: Number of columns in subplot grid
        save: Whether to save the plot
    """
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.flatten()

    for idx, col in enumerate(categorical_cols[:rows*cols]):
        if col in df.columns:
            df[col].value_counts().plot(kind='bar', ax=axes[idx], color='steelblue')
            axes[idx].set_title(f'{col} Distribution', fontsize=12)
            axes[idx].set_xlabel('')
            axes[idx].tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for idx in range(len(categorical_cols), rows*cols):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'categorical_distribution.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
    save: bool = False
) -> None:
    """
    Plot correlation heatmap for numerical features.

    Args:
        df: DataFrame containing the data
        figsize: Figure size
        save: Whether to save the plot
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=figsize)
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1
    )
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    save: bool = False
) -> None:
    """
    Plot target variable distribution.

    Args:
        df: DataFrame containing the data
        target_col: Target column name
        save: Whether to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    df[target_col].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
    axes[0].set_title(f'{target_col} Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)

    # Add count labels
    for container in axes[0].containers:
        axes[0].bar_label(container)

    # Pie chart
    churn_counts = df[target_col].value_counts()
    axes[1].pie(
        churn_counts,
        labels=churn_counts.index,
        autopct='%1.1f%%',
        colors=['steelblue', 'coral'],
        startangle=90
    )
    axes[1].set_title(f'{target_col} Distribution (Percentage)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / 'target_distribution.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    title: str = 'Confusion Matrix',
    save: bool = False,
    save_name: str = 'confusion_matrix.png'
) -> None:
    """
    Plot confusion matrix with labels.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        save: Whether to save the plot
        save_name: Filename for saved plot
    """
    if labels is None:
        labels = ['Not Churned', 'Churned']

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('Actual Class', fontsize=12)
    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches='tight')

    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = 'Model',
    save: bool = False,
    save_name: str = 'roc_curve.png'
) -> None:
    """
    Plot ROC curve with AUC score.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        save: Whether to save the plot
        save_name: Filename for saved plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches='tight')

    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = 'Feature Importance',
    save: bool = False,
    save_name: str = 'feature_importance.png'
) -> None:
    """
    Plot feature importance from tree-based models.

    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to display
        title: Plot title
        save: Whether to save the plot
        save_name: Filename for saved plot
    """
    # Create DataFrame and sort
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp_df, x='importance', y='feature', palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches='tight')

    plt.show()


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = 'accuracy',
    save: bool = False
) -> None:
    """
    Plot comparison of multiple models.

    Args:
        results_df: DataFrame with model results
        metric: Metric to compare
        save: Whether to save the plot
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='model', y=metric, palette='Set2')
    plt.title(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.3f')

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / f'model_comparison_{metric}.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_churn_by_feature(
    df: pd.DataFrame,
    feature: str,
    target_col: str = 'Churn',
    save: bool = False
) -> None:
    """
    Plot churn rate by a specific feature.

    Args:
        df: DataFrame containing the data
        feature: Feature to analyze
        target_col: Target column name
        save: Whether to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    pd.crosstab(df[feature], df[target_col]).plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
    axes[0].set_title(f'{target_col} by {feature}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Count')
    axes[0].legend(title=target_col)
    axes[0].tick_params(axis='x', rotation=45)

    # Percentage plot
    churn_rate = df.groupby(feature)[target_col].apply(lambda x: (x == 'Yes').mean() * 100)
    churn_rate.plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title(f'{target_col} Rate by {feature}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Churn Rate (%)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / f'churn_by_{feature}.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    print("Visualization module loaded successfully!")