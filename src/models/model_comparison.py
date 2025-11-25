"""Model comparison system for evaluating multiple models."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings('ignore')


class StatisticalBaselineModel:
    """
    Statistical baseline model based on rating difference (Elo formula).
    Predicts outcome probability using the standard logistic distribution.
    """

    def __init__(self):
        """Initialize the statistical baseline model."""
        self.name = "Statistical Baseline (Rating-based)"
        self.description = (
            "A simple statistical model that predicts game outcomes based solely on "
            "the rating difference between players using the standard Elo formula. "
            "It serves as a specialized benchmark for chess engines."
        )
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model (stores target classes to align probabilities correctly).
        """
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels (who wins?)."""
        probs = self.predict_proba(X)
        # Wybieramy indeks z najwyższym prawdopodobieństwem
        max_indices = np.argmax(probs, axis=1)
        return self.classes_[max_indices]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using Elo formula.
        P(White) = 1 / (1 + 10^(-diff/400))
        """
        # Obliczamy różnicę, jeśli jej nie ma w danych, liczymy w locie
        if 'rating_diff' in X.columns:
            diff = X['rating_diff']
        else:
            diff = X['white_rating'] - X['black_rating']

        # Wzór Elo na szansę wygranej Białych (standard w szachach)
        # Zwraca wartość od 0.0 do 1.0
        expected_score_white = 1 / (1 + 10 ** (-diff / 400))

        # Prawdopodobieństwo remisu (szacunkowe, stałe dla uproszczenia baseline)
        # W prawdziwym życiu zależy od poziomu graczy, ale tu przyjmijmy 10%
        prob_draw = 0.10

        # P(W) + P(L) = 0.9
        prob_white = expected_score_white * (1.0 - prob_draw)
        prob_black = (1.0 - expected_score_white) * (1.0 - prob_draw)

        proba_matrix = []

        # Mapowanie prawdopodobieństw do odpowiednich kolumn
        probs_map = {
            'white_wins': prob_white,
            'black_wins': prob_black,
            'draw': pd.Series([prob_draw] * len(diff), index=diff.index),
            '1-0': prob_white,
            '0-1': prob_black,
            '1/2-1/2': prob_draw
        }

        columns = []
        for class_label in self.classes_:
            if str(class_label) in probs_map:
                columns.append(probs_map[str(class_label)])
            else:
                columns.append(pd.Series([0.0] * len(diff), index=diff.index))

        return np.column_stack(columns)

class ModelComparator:
    """Compares multiple ML models with the same metrics."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model comparator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_columns = None
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'result',
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare and split data for training and testing.
        
        Args:
            df: DataFrame with features
            target_column: Name of target column
            test_size: Proportion of data for testing
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Encode categorical variables
        categorical_cols = [col for col in ['time_control', 'opening'] if col in X.columns]
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, y_train, X_test, y_test
    
    def add_model(self, name: str, model: Any, description: str = ""):
        """
        Add a model to compare.
        
        Args:
            name: Model name
            model: Model object (must have fit, predict, and optionally predict_proba methods)
            description: Model description
        """
        if hasattr(model, 'name'):
            model.name = name
        if hasattr(model, 'description') and description:
            model.description = description
        
        self.models[name] = model
    
    def _train_single_model(
        self,
        name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[str, Dict]:
        """
        Train and evaluate a single model.
        
        Args:
            name: Model name
            model: Model object
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        
        Returns:
            Tuple of (model_name, metrics_dict)
        """
        start_time = time.time()
        
        print(f"[{name}] Starting training...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        print(f"[{name}] Making predictions...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, model, X_test)
        training_time = time.time() - start_time
        metrics['training_time'] = training_time
        
        print(f"[{name}] Completed in {training_time:.2f} seconds")
        
        return name, metrics
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        parallel: bool = True,
        max_workers: int = None
    ) -> Dict[str, Dict]:
        """
        Train all models and evaluate them (in parallel by default).
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            parallel: Whether to train models in parallel (default: True)
            max_workers: Maximum number of parallel workers (default: number of models)
        
        Returns:
            Dictionary with results for each model
        """
        print("=" * 80)
        print("Training and Evaluating Models")
        if parallel:
            print(f"Running {len(self.models)} models in parallel...")
        else:
            print(f"Running {len(self.models)} models sequentially...")
        print("=" * 80)
        
        results = {}
        start_time = time.time()
        
        if parallel and len(self.models) > 1:
            # Train models in parallel
            max_workers = max_workers or min(len(self.models), 4)  # Limit to 4 workers by default
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all training tasks
                future_to_model = {
                    executor.submit(
                        self._train_single_model,
                        name, model, X_train, y_train, X_test, y_test
                    ): name
                    for name, model in self.models.items()
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_model):
                    name, metrics = future.result()
                    results[name] = metrics
                    completed += 1
                    print(f"\n[{name}] Training completed ({completed}/{len(self.models)})")
                    self._print_model_summary(name, metrics)
        else:
            # Train models sequentially (for debugging or when parallel=False)
            for name, model in self.models.items():
                print(f"\n{'='*80}")
                print(f"Model: {name}")
                if hasattr(model, 'description'):
                    print(f"Description: {model.description}")
                print(f"{'='*80}")
                
                name, metrics = self._train_single_model(
                    name, model, X_train, y_train, X_test, y_test
                )
                results[name] = metrics
                self._print_model_summary(name, metrics)
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"All models completed in {total_time:.2f} seconds")
        print(f"{'='*80}")
        
        self.results = results
        return results
    
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model: Any,
        X_test: pd.DataFrame
    ) -> Dict:
        """
        Calculate comprehensive metrics for a model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model: Model object
            X_test: Test features (for probability predictions)
        
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1 (macro and weighted averages)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # ROC AUC (if model supports predict_proba)
        roc_auc = None
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                # Convert to binary if needed, or use multi-class ROC AUC
                if len(np.unique(y_true)) == 2:
                    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class ROC AUC
                    from sklearn.preprocessing import LabelBinarizer
                    lb = LabelBinarizer()
                    y_true_bin = lb.fit_transform(y_true)
                    if y_proba.shape[1] == y_true_bin.shape[1]:
                        roc_auc = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        except Exception as e:
            print(f"  Warning: Could not calculate ROC AUC: {e}")
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
    
    def _print_model_summary(self, name: str, metrics: Dict):
        """Print summary of model performance."""
        print(f"\n{name} Performance Summary:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
        print(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        if 'training_time' in metrics:
            print(f"  Training Time: {metrics['training_time']:.2f} seconds")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create a comparison table of all models.
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.results:
            raise ValueError("No results available. Run train_all_models() first.")
        
        comparison_data = []
        for name, metrics in self.results.items():
            row = {
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Precision (Weighted)': metrics['precision_weighted'],
                'Recall (Macro)': metrics['recall_macro'],
                'Recall (Weighted)': metrics['recall_weighted'],
                'F1 Score (Macro)': metrics['f1_macro'],
                'F1 Score (Weighted)': metrics['f1_weighted'],
                'ROC AUC': metrics['roc_auc'] if metrics['roc_auc'] is not None else np.nan
            }
            if 'training_time' in metrics:
                row['Training Time (s)'] = metrics['training_time']
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def select_best_model(self, metric: str = 'accuracy') -> Tuple[str, Dict]:
        """
        Select the best model based on a specified metric.
        
        Args:
            metric: Metric to use for selection ('accuracy', 'f1_macro', 'f1_weighted', etc.)
        
        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        if not self.results:
            raise ValueError("No results available. Run train_all_models() first.")
        
        best_name = None
        best_score = -np.inf
        
        for name, metrics in self.results.items():
            score = metrics.get(metric, -np.inf)
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_name is None:
            raise ValueError(f"Could not find best model using metric: {metric}")
        
        return best_name, self.results[best_name]
    
    def print_comparison(self):
        """Print a formatted comparison of all models."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        comparison_df = self.compare_models()
        print("\n" + comparison_df.to_string(index=False))
        
        # Print best model
        best_name, best_metrics = self.select_best_model('f1_weighted')
        print(f"\n{'='*80}")
        print(f"BEST MODEL (based on F1 Weighted Score): {best_name}")
        print(f"{'='*80}")
        if hasattr(self.models[best_name], 'description'):
            print(f"Description: {self.models[best_name].description}")
        print(f"\nBest Model Metrics:")
        for key, value in best_metrics.items():
            if key not in ['confusion_matrix', 'classification_report']:
                if value is not None:
                    print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")

