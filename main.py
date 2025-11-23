"""Main entry point for Chess Game Result Predictor."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.lichess_api import LichessDataCollector
from src.preprocessing.data_cleaner import DataCleaner
from src.features.feature_extractor import FeatureExtractor
from src.models.trainer import ModelTrainer
from src.utils.helpers import load_config, ensure_dir, parse_result


def collect_data(config):
    """Collect data from Lichess API."""
    print("=" * 50)
    print("Step 1: Collecting data from Lichess")
    print("=" * 50)
    
    collector = LichessDataCollector(
        base_url=config['lichess']['base_url'],
        rate_limit_delay=config['lichess']['rate_limit_delay']
    )
    
    ensure_dir(config['paths']['raw_data'])
    
    games = collector.fetch_games(
        num_games=config['data_collection']['num_games'],
        min_rating=config['data_collection']['min_rating'],
        max_rating=config['data_collection']['max_rating'],
        time_controls=config['data_collection']['time_controls'],
        usernames=config['lichess'].get('usernames', []),
        api_token=config['lichess'].get('api_token', None)
    )
    
    raw_data_path = Path(config['paths']['raw_data']) / 'games.json'
    collector.save_games(games, str(raw_data_path))
    
    return games


def preprocess_data(config, games):
    """Preprocess and clean data."""
    print("\n" + "=" * 50)
    print("Step 2: Preprocessing data")
    print("=" * 50)
    
    cleaner = DataCleaner(
        min_rating=config['data_collection']['min_rating'],
        max_rating=config['data_collection']['max_rating']
    )
    
    df = cleaner.clean_games(games)
    
    ensure_dir(config['paths']['processed_data'])
    processed_path = Path(config['paths']['processed_data']) / 'cleaned_games.csv'
    df.to_csv(processed_path, index=False)
    
    print(f"Cleaned {len(df)} games")
    return df


def extract_features(config, df):
    """Extract features from games."""
    print("\n" + "=" * 50)
    print("Step 3: Extracting features")
    print("=" * 50)
    
    # Support both num_moves (legacy) and fullmoves
    fullmoves = config['features'].get('fullmoves', None)
    if fullmoves is not None:
        extractor = FeatureExtractor(fullmoves=fullmoves)
    else:
        # Legacy: use num_moves (half-moves)
        extractor = FeatureExtractor(num_moves=config['features'].get('num_moves', 20))
    
    # Convert DataFrame back to list of dicts for feature extraction
    games_list = df.to_dict('records')
    features_df = extractor.extract_features_batch(games_list)
    
    # Add result column
    if 'result' in df.columns:
        features_df['result'] = df['result'].apply(parse_result)
    
    features_path = Path(config['paths']['processed_data']) / 'features.csv'
    features_df.to_csv(features_path, index=False)
    
    print(f"Extracted features for {len(features_df)} games")
    return features_df


def train_model(config, features_df):
    """Train and compare multiple models."""
    print("\n" + "=" * 50)
    print("Step 4: Training and Comparing Models")
    print("=" * 50)
    
    trainer = ModelTrainer(
        model_type=config['model']['type'],
        random_state=config['model']['random_state']
    )
    
    # Train and compare all models (in parallel by default)
    parallel_training = config['model'].get('parallel_training', True)
    comparator = trainer.train_and_compare_models(
        features_df,
        target_column='result',
        test_size=config['model']['test_size'],
        n_estimators=config['model'].get('n_estimators', 100),
        max_depth=config['model'].get('max_depth', 10),
        parallel=parallel_training
    )
    
    # Save the best model
    best_model_name, best_metrics = comparator.select_best_model('f1_weighted')
    print(f"\n{'='*80}")
    print(f"Saving best model: {best_model_name}")
    print(f"{'='*80}")
    
    ensure_dir(config['paths']['models'])
    
    # Save comparison results
    comparison_df = comparator.compare_models()
    comparison_path = Path(config['paths']['models']) / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Model comparison saved to {comparison_path}")
    
    # Save best model (if it's one of the sklearn models)
    best_model = comparator.models[best_model_name]
    if hasattr(best_model, 'predict_proba'):  # Skip baseline model
        model_path = Path(config['paths']['models']) / 'best_model.pkl'
        import joblib
        joblib.dump({
            'model': best_model,
            'model_name': best_model_name,
            'feature_columns': trainer.feature_columns,
            'metrics': best_metrics
        }, str(model_path))
        print(f"Best model saved to {model_path}")
    
    return comparator, best_model_name, best_metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Chess Game Result Predictor')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['collect', 'preprocess', 'features', 'train', 'all'],
        default='all',
        help='Pipeline mode to run'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'lichess': {'base_url': 'https://lichess.org/api', 'rate_limit_delay': 1.0},
            'data_collection': {'num_games': 1000, 'min_rating': 1500, 'max_rating': 3000},
            'features': {'num_moves': 20},
            'model': {'type': 'random_forest', 'test_size': 0.2, 'random_state': 42},
            'paths': {'raw_data': 'data/raw', 'processed_data': 'data/processed', 'models': 'models'}
        }
    
    if args.mode in ['collect', 'all']:
        games = collect_data(config)
        # If no games collected (API not implemented), try to load existing data
        if not games or len(games) == 0:
            raw_data_path = Path(config['paths']['raw_data']) / 'games.json'
            if raw_data_path.exists():
                print(f"No games collected from API. Loading existing data from {raw_data_path}...")
                from src.data_collection.lichess_api import LichessDataCollector
                collector = LichessDataCollector()
                games = collector.load_games(str(raw_data_path))
            else:
                print("\n⚠️  ERROR: No games data found!")
                print("Please either:")
                print("  1. Generate sample data: python generate_sample_data.py")
                print("  2. Implement Lichess API in src/data_collection/lichess_api.py")
                print("  3. Place your games.json file in data/raw/")
                return
    else:
        # Load existing data
        from src.data_collection.lichess_api import LichessDataCollector
        collector = LichessDataCollector()
        raw_data_path = Path(config['paths']['raw_data']) / 'games.json'
        if not raw_data_path.exists():
            print(f"\n⚠️  ERROR: Data file not found: {raw_data_path}")
            print("Please generate sample data first: python generate_sample_data.py")
            return
        games = collector.load_games(str(raw_data_path))
    
    if args.mode in ['preprocess', 'all']:
        df = preprocess_data(config, games)
    else:
        import pandas as pd
        processed_path = Path(config['paths']['processed_data']) / 'cleaned_games.csv'
        df = pd.read_csv(processed_path)
    
    if args.mode in ['features', 'all']:
        features_df = extract_features(config, df)
    else:
        import pandas as pd
        features_path = Path(config['paths']['processed_data']) / 'features.csv'
        features_df = pd.read_csv(features_path)
    
    if args.mode in ['train', 'all']:
        comparator, best_model_name, best_metrics = train_model(config, features_df)
        print("\n" + "=" * 80)
        print("Training and Comparison Completed!")
        print("=" * 80)
        print(f"\nBest Model: {best_model_name}")
        print(f"Best F1 Weighted Score: {best_metrics['f1_weighted']:.4f}")
        print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")


if __name__ == '__main__':
    main()

