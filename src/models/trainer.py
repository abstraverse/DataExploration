"""Model training and evaluation."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from typing import Dict, Tuple

# Import model comparison
from .model_comparison import ModelComparator, StatisticalBaselineModel


class ModelTrainer:
    """Trains and evaluates ML models for chess game prediction."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'result') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features
            target_column: Name of target column
        
        Returns:
            X (features), y (target)
        """
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Encode categorical variables
        X = pd.get_dummies(X, columns=['time_control', 'opening'], drop_first=True)
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        n_estimators: int = 100,
        max_depth: int = 10
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            n_estimators: Number of trees (for Random Forest)
            max_depth: Maximum tree depth
        
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def train_and_compare_models(
        self,
        df: pd.DataFrame,
        target_column: str = 'result',
        test_size: float = 0.2,
        n_estimators: int = 100,
        max_depth: int = 10,
        logistic_regression_params: Dict = None,
        parallel: bool = True
    ) -> ModelComparator:
        """
        Train and compare multiple models: statistical baseline, logistic regression, and random forest.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            test_size: Proportion of data for testing
            n_estimators: Number of trees for Random Forest
            max_depth: Maximum tree depth for Random Forest
            logistic_regression_params: Parameters for Logistic Regression
        
        Returns:
            ModelComparator with trained models and results
        """
        # Prepare data
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Encode categorical variables
        categorical_cols = [col for col in ['time_control', 'opening'] if col in X.columns]
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Initialize comparator
        comparator = ModelComparator(random_state=self.random_state)
        
        # Prepare train/test split
        X_train, y_train, X_test, y_test = comparator.prepare_data(
            df, target_column=target_column, test_size=test_size
        )
        
        # Add Statistical Baseline Model
        baseline = StatisticalBaselineModel()
        comparator.add_model(
            "Statistical Baseline (Rating-based)",
            baseline,
            baseline.description
        )
        
        # Add Logistic Regression
        lr_params = logistic_regression_params or {
            'max_iter': 1000,
            'random_state': self.random_state,
            'multi_class': 'multinomial',
            'solver': 'lbfgs'
        }
        lr_model = LogisticRegression(**lr_params)
        comparator.add_model(
            "Logistic Regression",
            lr_model,
            (
                "A classical statistical model that uses logistic function to model the probability "
                "of game outcomes. It's interpretable and works well for binary and multi-class "
                "classification problems. Good baseline for understanding feature importance."
            )
        )
        
        # Add Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        comparator.add_model(
            "Random Forest Classifier",
            rf_model,
            (
                "An ensemble learning method that combines multiple decision trees. "
                "It's robust to overfitting, handles non-linear relationships well, "
                "and can capture complex patterns in the data. Generally performs well "
                "on structured data like chess game features."
            )
        )
        
        # Train and evaluate all models (in parallel by default)
        comparator.train_all_models(
            X_train, y_train, X_test, y_test,
            parallel=parallel
        )
        
        # Print comparison
        comparator.print_comparison()
        
        return comparator
    
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to load model from
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.model_type = data['model_type']
        print(f"Model loaded from {filepath}")
