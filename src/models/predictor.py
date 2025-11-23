"""Model prediction interface."""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List
from .trainer import ModelTrainer


class GamePredictor:
    """Predicts chess game results."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model file (best_model.pkl from comparison)
        """
        self.model = None
        self.feature_columns = None
        self.model_name = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model from file.
        
        Args:
            model_path: Path to model file
        """
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_columns = data.get('feature_columns', [])
        self.model_name = data.get('model_name', 'Unknown')
        print(f"Loaded model: {self.model_name} from {model_path}")
    
    def predict(
        self,
        moves: List[str],
        white_rating: int,
        black_rating: int,
        time_control: str = 'blitz',
        num_moves: int = 20
    ) -> Dict:
        """
        Predict game result from first moves.
        
        Args:
            moves: List of moves (first 20)
            white_rating: White player rating
            black_rating: Black player rating
            time_control: Time control category
            num_moves: Number of moves to analyze
        
        Returns:
            Dictionary with prediction and probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a trained model first.")
        
        # Extract features
        from ..features.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor(num_moves=num_moves)
        game = {
            'moves': ' '.join(moves[:num_moves]),
            'white_rating': white_rating,
            'black_rating': black_rating,
            'time_control': time_control
        }
        
        features = extractor.extract_features(game)
        features_df = pd.DataFrame([features])
        
        # Prepare features to match training data
        categorical_cols = [col for col in ['time_control', 'opening'] if col in features_df.columns]
        if categorical_cols:
            features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)
        
        # Align columns with training data
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[self.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(features_df)[0]
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(features_df)[0]
            except:
                pass
        
        # Map prediction to result
        result_map = {0: 'black_wins', 1: 'white_wins', 2: 'draw'}
        if hasattr(self.model, 'classes_'):
            result_map = {i: cls for i, cls in enumerate(self.model.classes_)}
        
        result = {
            'prediction': result_map.get(prediction, prediction),
            'model_name': self.model_name
        }
        
        if probabilities is not None:
            result['probabilities'] = {
                result_map.get(i, f'class_{i}'): float(prob)
                for i, prob in enumerate(probabilities)
            }
            result['confidence'] = float(np.max(probabilities))
        
        return result

