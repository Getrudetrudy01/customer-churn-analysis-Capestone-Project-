"""
Model training module for customer churn prediction.
Includes multiple ML algorithms, hyperparameter tuning, and evaluation.
"""
import json
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, Any
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_validate, cross_val_predict
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb

from src.config import MODEL_DIR, PIPELINE_PATH, METRICS_PATH, RANDOM_STATE, TEST_SIZE, CV_FOLDS
from src.data_preprocessing import build_preprocessor
from src.utils import generate_model_report, calculate_business_impact, format_business_metrics


def build_pipeline(model: Any) -> Pipeline:
    """
    Chain preprocessing and a classifier so raw customer records can be scored directly.

    Args:
        model: Unfitted classifier

    Returns:
        Unfitted Pipeline
    """
    return Pipeline([
        ('preprocess', build_preprocessor()),
        ('classifier', model)
    ])


class ChurnModelTrainer:
    """
    Class for training and evaluating customer churn prediction models.
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the model trainer.

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.final_model = None
        self.threshold = 0.5

    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models with default parameters.

        Returns:
            Dictionary of model instances
        """
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1
            ),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True
            ),
            'KNN': KNeighborsClassifier()
        }

        self.models = models
        return models

    def train_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Any:
        """
        Train a single model.

        Args:
            model: Model instance
            X_train: Training features
            y_train: Training target

        Returns:
            Trained model
        """
        model.fit(X_train, y_train)
        return model

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = 'Model',
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            threshold: Probability at or above which a customer is predicted to churn

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        return {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }

    def cv_splitter(self) -> StratifiedKFold:
        """Stratified folds shared by model selection, calibration and threshold tuning."""
        return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=self.random_state)

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Train and evaluate all models. Models are ranked by cross-validated PR-AUC
        on the training set, so the test set plays no part in choosing the best model.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            DataFrame with results for all models
        """
        if not self.models:
            self.initialize_models()

        results = []

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Cross-validation on the training set (used for model selection)
            cv_scores = cross_validate(
                model, X_train, y_train,
                cv=self.cv_splitter(), scoring=['average_precision', 'f1']
            )

            # Train model
            trained_model = self.train_model(model, X_train, y_train)

            # Evaluate model on the test set (reported only)
            metrics = self.evaluate_model(trained_model, X_test, y_test, name)
            metrics['cv_pr_auc'] = cv_scores['test_average_precision'].mean()
            metrics['cv_pr_auc_std'] = cv_scores['test_average_precision'].std()
            metrics['cv_f1'] = cv_scores['test_f1'].mean()

            results.append(metrics)

            print(f"{name} - CV PR-AUC: {metrics['cv_pr_auc']:.4f}, "
                  f"CV F1: {metrics['cv_f1']:.4f}")

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('cv_pr_auc', ascending=False)

        self.results = results_df

        # Store best model
        best_model_name = results_df.iloc[0]['model']
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]

        return results_df

    def build_final_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[Any, float]:
        """
        Calibrate the selected model's probabilities and choose its decision threshold.
        The threshold maximizes F1 on out-of-fold training predictions, so the test set
        is not used.

        Args:
            model: Selected model (fitted or not; it is cloned)
            X_train: Training features
            y_train: Training target

        Returns:
            Tuple of (fitted calibrated model, threshold)
        """
        calibrated = CalibratedClassifierCV(clone(model), method='isotonic', cv=self.cv_splitter())

        oof_proba = cross_val_predict(
            calibrated, X_train, y_train,
            cv=self.cv_splitter(), method='predict_proba'
        )[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_train, oof_proba)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        threshold = float(thresholds[np.argmax(f1[:-1])])

        calibrated.fit(X_train, y_train)
        print(f"Calibrated model built. Decision threshold: {threshold:.3f}")
        return calibrated, threshold

    def hyperparameter_tuning_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = 'random'
    ) -> Tuple[Any, Dict, float]:
        """
        Perform hyperparameter tuning for XGBoost.

        Args:
            X_train: Raw training features
            y_train: Training target
            method: 'grid' or 'random' search

        Returns:
            Tuple of (best_pipeline, best_params, best_cv_pr_auc)
        """
        print("\nPerforming hyperparameter tuning for XGBoost...")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }

        xgb_model = build_pipeline(xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss'
        ))
        param_grid = {f'classifier__{name}': values for name, values in param_grid.items()}

        if method == 'grid':
            search = GridSearchCV(
                xgb_model,
                param_grid,
                cv=self.cv_splitter(),
                scoring='average_precision',
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                xgb_model,
                param_grid,
                n_iter=50,
                cv=self.cv_splitter(),
                scoring='average_precision',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )

        search.fit(X_train, y_train)

        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best cross-validated PR-AUC: {search.best_score_:.4f}")

        return search.best_estimator_, search.best_params_, search.best_score_

    def hyperparameter_tuning_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = 'random'
    ) -> Tuple[Any, Dict, float]:
        """
        Perform hyperparameter tuning for LightGBM.

        Args:
            X_train: Raw training features
            y_train: Training target
            method: 'grid' or 'random' search

        Returns:
            Tuple of (best_pipeline, best_params, best_cv_pr_auc)
        """
        print("\nPerforming hyperparameter tuning for LightGBM...")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [15, 31, 63, 127],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_samples': [10, 20, 30]
        }

        lgb_model = build_pipeline(lgb.LGBMClassifier(
            random_state=self.random_state,
            verbose=-1
        ))
        param_grid = {f'classifier__{name}': values for name, values in param_grid.items()}

        if method == 'grid':
            search = GridSearchCV(
                lgb_model,
                param_grid,
                cv=self.cv_splitter(),
                scoring='average_precision',
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                lgb_model,
                param_grid,
                n_iter=50,
                cv=self.cv_splitter(),
                scoring='average_precision',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )

        search.fit(X_train, y_train)

        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best cross-validated PR-AUC: {search.best_score_:.4f}")

        return search.best_estimator_, search.best_params_, search.best_score_

    def calculate_business_impact(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        **assumptions: float
    ) -> Dict[str, float]:
        """
        Calculate business impact of the churn prediction model.

        Args:
            y_test: True labels
            y_pred: Predicted labels
            **assumptions: Overrides for avg_customer_value, retention_cost
                and retention_success_rate (see utils.calculate_business_impact)

        Returns:
            Dictionary with business metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return calculate_business_impact(tp, fp, fn, tn, **assumptions)

    def save_model(self, model: Any, filename: str) -> None:
        """
        Save a trained model to disk.

        Args:
            model: Trained model
            filename: Filename for the saved model
        """
        filepath = MODEL_DIR / filename
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filename: str) -> Any:
        """
        Load a trained model from disk.

        Args:
            filename: Filename of the saved model

        Returns:
            Loaded model
        """
        filepath = MODEL_DIR / filename
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


def save_metrics(
    results_df: pd.DataFrame,
    best_model_name: str,
    business_impact: Dict[str, float],
    test_size: int,
    final_metrics: Dict[str, float],
    threshold: float,
    feature_importance: pd.Series
) -> None:
    """
    Save evaluation results so the dashboard shows the numbers from the latest training run.

    Args:
        results_df: DataFrame with results for all models
        best_model_name: Name of the selected model
        business_impact: Output of calculate_business_impact for the final model
        test_size: Number of customers in the test set
        final_metrics: Test-set metrics of the calibrated final model at its threshold
        threshold: Decision threshold of the final model
        feature_importance: Permutation importance of each raw input, highest first
    """
    metrics = {
        'best_model': best_model_name,
        'test_size': test_size,
        'threshold': threshold,
        'final_model': final_metrics,
        'feature_importance': [
            {'feature': feature, 'importance': float(value)}
            for feature, value in feature_importance.items()
        ],
        'results': results_df.to_dict(orient='records'),
        'confusion_matrix': {
            key: business_impact[key]
            for key in ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']
        }
    }

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


def train_and_evaluate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    tune_hyperparameters: bool = False
) -> Tuple[ChurnModelTrainer, pd.DataFrame]:
    """
    Complete training and evaluation pipeline. Every model is wrapped with the
    preprocessing steps. The model with the best cross-validated PR-AUC is calibrated,
    given a tuned decision threshold, evaluated once on the test set and saved.

    Args:
        X: Raw features (from load_raw_features)
        y: Target (0/1)
        test_size: Proportion of data for testing
        tune_hyperparameters: Whether to perform hyperparameter tuning

    Returns:
        Tuple of (trainer, results_dataframe)
    """
    print("=" * 50)
    print("Starting model training pipeline...")
    print("=" * 50)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Initialize trainer with every model wrapped in the preprocessing pipeline
    trainer = ChurnModelTrainer()
    trainer.models = {
        name: build_pipeline(model)
        for name, model in trainer.initialize_models().items()
    }

    # Train all models
    results_df = trainer.train_all_models(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 50)
    print("Model Comparison Results:")
    print("=" * 50)
    print(results_df.to_string(index=False))

    # Hyperparameter tuning if requested
    if tune_hyperparameters:
        print("\n" + "=" * 50)
        print("Starting hyperparameter tuning...")
        print("=" * 50)

        # Tuned models replace the best model only if their CV PR-AUC is higher
        tuned_candidates = [
            ('XGBoost (Tuned)', trainer.hyperparameter_tuning_xgboost(X_train, y_train, method='random')),
            ('LightGBM (Tuned)', trainer.hyperparameter_tuning_lightgbm(X_train, y_train, method='random'))
        ]
        best_cv_pr_auc = results_df.iloc[0]['cv_pr_auc']
        for name, (tuned_model, _, cv_pr_auc) in tuned_candidates:
            if cv_pr_auc > best_cv_pr_auc:
                best_cv_pr_auc = cv_pr_auc
                trainer.best_model_name = name
                trainer.best_model = tuned_model
        print(f"\nSelected model after tuning: {trainer.best_model_name}")

    # Calibrate the selected model and choose its threshold on training data only
    print("\n" + "=" * 50)
    print(f"Calibrating {trainer.best_model_name}...")
    print("=" * 50)
    final_model, threshold = trainer.build_final_model(trainer.best_model, X_train, y_train)

    # The only time the final model sees the test set
    final_metrics = trainer.evaluate_model(
        final_model, X_test, y_test, f"{trainer.best_model_name} (calibrated)", threshold
    )
    y_pred_final = (final_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    business_impact = trainer.calculate_business_impact(y_test, y_pred_final)

    print(f"\nFinal model on test set - Precision: {final_metrics['precision']:.4f}, "
          f"Recall: {final_metrics['recall']:.4f}, F1: {final_metrics['f1_score']:.4f}, "
          f"PR-AUC: {final_metrics['pr_auc']:.4f}")
    print(format_business_metrics(business_impact))

    # Which raw inputs the final model relies on: drop in test-set PR-AUC when shuffled
    importance = permutation_importance(
        final_model, X_test, y_test,
        scoring='average_precision', n_repeats=5, random_state=RANDOM_STATE
    )
    feature_importance = pd.Series(
        importance.importances_mean, index=X_test.columns
    ).sort_values(ascending=False)

    # Save final model together with its preprocessing
    trainer.save_model(final_model, PIPELINE_PATH.name)

    # Save results for the dashboard and the text report
    save_metrics(
        results_df, trainer.best_model_name, business_impact,
        len(y_test), final_metrics, threshold, feature_importance
    )
    generate_model_report(
        results_df, business_impact,
        final_metrics=final_metrics, threshold=threshold
    )

    trainer.final_model = final_model
    trainer.threshold = threshold

    print("\n" + "=" * 50)
    print("Training pipeline complete!")
    print("=" * 50)

    return trainer, results_df


if __name__ == "__main__":
    from src.data_preprocessing import load_raw_features

    # Load raw features; preprocessing happens inside each model's pipeline
    X, y = load_raw_features()

    # Train and evaluate models
    trainer, results = train_and_evaluate_pipeline(X, y, tune_hyperparameters=False)