"""Feature extraction from chess games."""

import pandas as pd
import numpy as np
from typing import List, Dict
from .position_evaluator import PositionEvaluator
from src.utils.openings import get_opening_name

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
            self.num_moves = fullmoves * 2
            self.fullmoves = fullmoves
        else:
            self.num_moves = num_moves
            self.fullmoves = num_moves // 2

        self.position_evaluator = PositionEvaluator()

    def extract_features(self, game: Dict) -> Dict:
        """
        Extract features from a single game.
        """
        features = {}

        features['white_rating'] = game.get('white_rating', 1500)
        features['black_rating'] = game.get('black_rating', 1500)
        features['rating_diff'] = features['white_rating'] - features['black_rating']
        features['avg_rating'] = (features['white_rating'] + features['black_rating']) / 2

        moves = game.get('moves', '').split()[:self.num_moves]
        features['num_moves'] = len(moves)

        if moves:
            try:
                if hasattr(self.position_evaluator, 'get_position_after_moves'):
                    fens = self.position_evaluator.get_position_after_moves(moves, self.num_moves)
                else:
                    fens = self.position_evaluator.get_fens_from_moves(moves) # Fallback do starej nazwy

                evaluations = [self.position_evaluator.evaluate_position(fen) for fen in fens]

                if evaluations:
                    features['eval_mean'] = np.mean(evaluations)
                    features['eval_std'] = np.std(evaluations)
                    features['eval_max'] = np.max(evaluations)
                    features['eval_min'] = np.min(evaluations)
                    features['eval_final'] = evaluations[-1] if evaluations else 0.0
                    features['eval_trend'] = evaluations[-1] - evaluations[0] if len(evaluations) > 1 else 0.0
                else:
                    self._set_empty_evals(features)
            except Exception:
                self._set_empty_evals(features)
        else:
            self._set_empty_evals(features)

        time_control = game.get('time_control', '')
        features['time_control'] = self._parse_time_control(time_control)

        raw_moves = game.get('moves', '').split()
        features['opening'] = self._identify_opening(raw_moves)

        return features

    def _set_empty_evals(self, features):
        """Helper to set empty evaluation features."""
        features['eval_mean'] = 0.0
        features['eval_std'] = 0.0
        features['eval_max'] = 0.0
        features['eval_min'] = 0.0
        features['eval_final'] = 0.0
        features['eval_trend'] = 0.0

    def extract_features_batch(self, games: List[Dict]) -> pd.DataFrame:
        """Extract features from multiple games."""
        features_list = [self.extract_features(game) for game in games]
        return pd.DataFrame(features_list)

    def _parse_time_control(self, time_control: str) -> str:
        """Parse time control string."""
        if not time_control:
            return 'unknown'

        time_control_lower = time_control.lower()
        if 'blitz' in time_control_lower or '+' in time_control:
            return 'blitz'
        elif 'rapid' in time_control_lower:
            return 'rapid'
        elif 'classical' in time_control_lower:
            return 'classical'
        else:
            return 'other'

    def _identify_opening(self, moves: List[str]) -> str:
        """
        Identify opening using the comprehensive database.

        Args:
            moves: List of moves in the game

        Returns:
            Opening name (e.g. "Sicilian Defense") or "Unknown"
        """
        return get_opening_name(moves)