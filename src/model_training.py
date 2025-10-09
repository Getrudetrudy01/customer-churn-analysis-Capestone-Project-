"""
Model training module for customer churn prediction.
Includes multiple ML algorithms, hyperparameter tuning, and evaluation.
"""
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import shap

from src.config import MODEL_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS
from src.data_preprocessing import scale_features


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
                eval_metric='logloss',
                use_label_encoder=False
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
        model_name: str = 'Model'
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        return metrics

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Train and evaluate all models.

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

            # Train model
            trained_model = self.train_model(model, X_train, y_train)

            # Evaluate model
            metrics = self.evaluate_model(trained_model, X_test, y_test, name)

            # Cross-validation score
            cv_scores = cross_val_score(
                trained_model, X_train, y_train,
                cv=CV_FOLDS, scoring='accuracy'
            )
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()

            results.append(metrics)

            print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                  f"F1: {metrics['f1_score']:.4f}, "
                  f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1_score', ascending=False)

        self.results = results_df

        # Store best model
        best_model_name = results_df.iloc[0]['model']
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]

        return results_df

    def hyperparameter_tuning_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = 'random'
    ) -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning for XGBoost.

        Args:
            X_train: Training features
            y_train: Training target
            method: 'grid' or 'random' search

        Returns:
            Tuple of (best_model, best_params)
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

        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )

        if method == 'grid':
            search = GridSearchCV(
                xgb_model,
                param_grid,
                cv=CV_FOLDS,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                xgb_model,
                param_grid,
                n_iter=50,
                cv=CV_FOLDS,
                scoring='f1',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )

        search.fit(X_train, y_train)

        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")

        return search.best_estimator_, search.best_params_

    def hyperparameter_tuning_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = 'random'
    ) -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning for LightGBM.

        Args:
            X_train: Training features
            y_train: Training target
            method: 'grid' or 'random' search

        Returns:
            Tuple of (best_model, best_params)
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

        lgb_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            verbose=-1
        )

        if method == 'grid':
            search = GridSearchCV(
                lgb_model,
                param_grid,
                cv=CV_FOLDS,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                lgb_model,
                param_grid,
                n_iter=50,
                cv=CV_FOLDS,
                scoring='f1',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )

        search.fit(X_train, y_train)

        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")

        return search.best_estimator_, search.best_params_

    def calculate_business_impact(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        avg_customer_value: float = 1000,
        retention_cost: float = 100
    ) -> Dict[str, float]:
        """
        Calculate business impact of the churn prediction model.

        Args:
            y_test: True labels
            y_pred: Predicted labels
            avg_customer_value: Average customer lifetime value
            retention_cost: Cost to retain a customer

        Returns:
            Dictionary with business metrics
        """
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate costs and savings
        cost_of_false_negatives = fn * avg_customer_value
        cost_of_retention_efforts = (tp + fp) * retention_cost
        value_of_retained_customers = tp * (avg_customer_value - retention_cost)

        total_cost = cost_of_false_negatives + cost_of_retention_efforts
        total_revenue_saved = value_of_retained_customers

        net_benefit = total_revenue_saved - total_cost

        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'cost_of_missed_churners': cost_of_false_negatives,
            'cost_of_retention': cost_of_retention_efforts,
            'value_retained': value_of_retained_customers,
            'net_benefit': net_benefit,
            'roi': (net_benefit / cost_of_retention_efforts * 100) if cost_of_retention_efforts > 0 else 0
        }

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


def train_and_evaluate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    tune_hyperparameters: bool = False
) -> Tuple[ChurnModelTrainer, pd.DataFrame]:
    """
    Complete training and evaluation pipeline.

    Args:
        X: Features
        y: Target
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

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Initialize trainer
    trainer = ChurnModelTrainer()

    # Train all models
    results_df = trainer.train_all_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    print("\n" + "=" * 50)
    print("Model Comparison Results:")
    print("=" * 50)
    print(results_df.to_string(index=False))

    # Hyperparameter tuning if requested
    if tune_hyperparameters:
        print("\n" + "=" * 50)
        print("Starting hyperparameter tuning...")
        print("=" * 50)

        # Tune XGBoost
        best_xgb, xgb_params = trainer.hyperparameter_tuning_xgboost(
            X_train_scaled, y_train, method='random'
        )

        # Tune LightGBM
        best_lgb, lgb_params = trainer.hyperparameter_tuning_lightgbm(
            X_train_scaled, y_train, method='random'
        )

        # Evaluate tuned models
        print("\nEvaluating tuned models...")
        tuned_xgb_metrics = trainer.evaluate_model(
            best_xgb, X_test_scaled, y_test, 'XGBoost (Tuned)'
        )
        tuned_lgb_metrics = trainer.evaluate_model(
            best_lgb, X_test_scaled, y_test, 'LightGBM (Tuned)'
        )

        print(f"\nXGBoost (Tuned) - Accuracy: {tuned_xgb_metrics['accuracy']:.4f}, "
              f"F1: {tuned_xgb_metrics['f1_score']:.4f}")
        print(f"LightGBM (Tuned) - Accuracy: {tuned_lgb_metrics['accuracy']:.4f}, "
              f"F1: {tuned_lgb_metrics['f1_score']:.4f}")

    # Calculate business impact for best model
    y_pred_best = trainer.best_model.predict(X_test_scaled)
    business_impact = trainer.calculate_business_impact(y_test, y_pred_best)

    print("\n" + "=" * 50)
    print("Business Impact Analysis:")
    print("=" * 50)
    for key, value in business_impact.items():
        if 'cost' in key or 'value' in key or 'benefit' in key:
            print(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
        elif 'roi' in key:
            print(f"{key.upper()}: {value:.2f}%")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")

    # Save best model
    trainer.save_model(trainer.best_model, 'best_model.pkl')

    print("\n" + "=" * 50)
    print("Training pipeline complete!")
    print("=" * 50)

    return trainer, results_df


if __name__ == "__main__":
    from src.data_preprocessing import preprocess_pipeline

    # Load and preprocess data
    X, y = preprocess_pipeline()

    # Train and evaluate models
    trainer, results = train_and_evaluate_pipeline(X, y, tune_hyperparameters=False)