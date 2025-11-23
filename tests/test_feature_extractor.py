"""Tests for feature extraction."""

import pytest
from src.features.feature_extractor import FeatureExtractor


def test_feature_extraction():
    """Test basic feature extraction."""
    extractor = FeatureExtractor(num_moves=20)
    
    game = {
        'moves': 'e2e4 e7e5 g1f3 b8c6 f1b5 a7a6',
        'white_rating': 1800,
        'black_rating': 1750,
        'time_control': 'blitz',
        'result': '1-0'
    }
    
    features = extractor.extract_features(game)
    
    assert 'white_rating' in features
    assert 'black_rating' in features
    assert 'rating_diff' in features
    assert features['rating_diff'] == 50

