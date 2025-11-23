"""Feature extraction from chess games."""

import pandas as pd
import numpy as np
from typing import List, Dict
from .position_evaluator import PositionEvaluator


class FeatureExtractor:
    """Extracts features from chess games for ML model."""
    
    def __init__(self, num_moves: int = 20, fullmoves: int = None):
        """
        Initialize the feature extractor.
        
        Args:
            num_moves: Number of half-moves (plies) to analyze (deprecated, use fullmoves)
            fullmoves: Number of full moves to analyze (1 full move = white + black turn)
                      If provided, overrides num_moves. 10 fullmoves = 20 half-moves.
        """
        if fullmoves is not None:
            # Convert full moves to half-moves (plies)
            # 1 full move = 2 half-moves (white move + black move)
            self.num_moves = fullmoves * 2
            self.fullmoves = fullmoves
        else:
            # Legacy: num_moves is in half-moves
            self.num_moves = num_moves
            self.fullmoves = num_moves // 2  # Approximate
        
        self.position_evaluator = PositionEvaluator()
    
    def extract_features(self, game: Dict) -> Dict:
        """
        Extract features from a single game.
        
        Args:
            game: Game dictionary with moves, ratings, etc.
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Player rating features
        features['white_rating'] = game.get('white_rating', 1500)
        features['black_rating'] = game.get('black_rating', 1500)
        features['rating_diff'] = features['white_rating'] - features['black_rating']
        features['avg_rating'] = (features['white_rating'] + features['black_rating']) / 2
        
        # Move-based features
        moves = game.get('moves', '').split()[:self.num_moves]
        features['num_moves'] = len(moves)
        
        # Position evaluation features
        if moves:
            fens = self.position_evaluator.get_position_after_moves(moves, self.num_moves)
            evaluations = [self.position_evaluator.evaluate_position(fen) for fen in fens]
            
            if evaluations:
                features['eval_mean'] = np.mean(evaluations)
                features['eval_std'] = np.std(evaluations)
                features['eval_max'] = np.max(evaluations)
                features['eval_min'] = np.min(evaluations)
                features['eval_final'] = evaluations[-1] if evaluations else 0.0
                features['eval_trend'] = evaluations[-1] - evaluations[0] if len(evaluations) > 1 else 0.0
            else:
                features['eval_mean'] = 0.0
                features['eval_std'] = 0.0
                features['eval_max'] = 0.0
                features['eval_min'] = 0.0
                features['eval_final'] = 0.0
                features['eval_trend'] = 0.0
        
        # Time control features
        time_control = game.get('time_control', '')
        features['time_control'] = self._parse_time_control(time_control)
        
        # Opening features (simplified)
        if moves and len(moves) >= 2:
            features['opening'] = self._identify_opening(moves[:10])
        else:
            features['opening'] = 'unknown'
        
        return features
    
    def extract_features_batch(self, games: List[Dict]) -> pd.DataFrame:
        """
        Extract features from multiple games.
        
        Args:
            games: List of game dictionaries
        
        Returns:
            DataFrame with extracted features
        """
        features_list = [self.extract_features(game) for game in games]
        return pd.DataFrame(features_list)
    
    def _parse_time_control(self, time_control: str) -> str:
        """
        Parse time control string.
        
        Args:
            time_control: Time control string from game
        
        Returns:
            Categorized time control
        """
        if not time_control:
            return 'unknown'
        
        time_control_lower = time_control.lower()
        if 'blitz' in time_control_lower or '+' in time_control:
            # Parse actual time control if needed
            return 'blitz'
        elif 'rapid' in time_control_lower:
            return 'rapid'
        elif 'classical' in time_control_lower:
            return 'classical'
        else:
            return 'other'
    
    def _identify_opening(self, moves: List[str]) -> str:
        """
        Identify opening from first moves (simplified).
        
        Args:
            moves: List of first moves
        
        Returns:
            Opening name
        """
        # TODO: Implement proper opening detection
        # This is a simplified version
        if len(moves) < 2:
            return 'unknown'
        
        # Very basic opening detection
        first_move = moves[0].lower() if moves else ''
        second_move = moves[1].lower() if len(moves) > 1 else ''
        
        if 'e4' in first_move:
            return 'e4_opening'
        elif 'd4' in first_move:
            return 'd4_opening'
        else:
            return 'other'

