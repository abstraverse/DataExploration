"""Experiment: Compare model performance across different numbers of full moves."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.lichess_api import LichessDataCollector
from src.preprocessing.data_cleaner import DataCleaner
from src.features.feature_extractor import FeatureExtractor
from src.models.trainer import ModelTrainer
from src.utils.helpers import load_config, ensure_dir, parse_result


def run_experiment_for_fullmoves(
    config: dict,
    fullmoves: int,
    games: list = None,
    df: pd.DataFrame = None
) -> dict:
    """
    Run the full pipeline for a specific number of full moves.
    
    Args:
        config: Configuration dictionary
        fullmoves: Number of full moves to analyze
        games: Pre-loaded games (optional)
        df: Pre-loaded cleaned DataFrame (optional)
    
    Returns:
        Dictionary with results for all models
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {fullmoves} FULL MOVES")
    print(f"{'='*80}\n")
    
    # Step 1: Load or preprocess data
    if df is None:
        if games is None:
            # Load games
            collector = LichessDataCollector()
            raw_data_path = Path(config['paths']['raw_data']) / 'games.json'
            if not raw_data_path.exists():
                raise FileNotFoundError(f"No games data found at {raw_data_path}")
            games = collector.load_games(str(raw_data_path))
        
        # Preprocess
        cleaner = DataCleaner(
            min_rating=config['data_collection']['min_rating'],
            max_rating=config['data_collection']['max_rating']
        )
        df = cleaner.clean_games(games)
    
    # Step 2: Extract features with specified fullmoves
    print(f"Extracting features for {fullmoves} full moves...")
    extractor = FeatureExtractor(fullmoves=fullmoves)
    
    games_list = df.to_dict('records')
    features_df = extractor.extract_features_batch(games_list)
    
    # Add result column
    if 'result' in df.columns:
        features_df['result'] = df['result'].apply(parse_result)
    else:
        # Try to get from games
        features_df['result'] = [parse_result(g.get('result', 'draw')) for g in games_list]
    
    # Filter out games with insufficient moves
    min_halfmoves = fullmoves * 2
    features_df = features_df[features_df['num_moves'] >= min_halfmoves]
    
    print(f"Games with sufficient moves: {len(features_df)}")
    
    if len(features_df) < 100:
        print(f"WARNING: Only {len(features_df)} games have {fullmoves} full moves. Results may be unreliable.")
    
    # Step 3: Train and compare models
    trainer = ModelTrainer(
        model_type=config['model']['type'],
        random_state=config['model']['random_state']
    )
    
    comparator = trainer.train_and_compare_models(
        features_df,
        target_column='result',
        test_size=config['model']['test_size'],
        n_estimators=config['model'].get('n_estimators', 100),
        max_depth=config['model'].get('max_depth', 10),
        parallel=config['model'].get('parallel_training', True)
    )
    
    # Extract results
    results = {}
    for model_name, metrics in comparator.results.items():
        results[model_name] = {
            'fullmoves': fullmoves,
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'roc_auc': metrics['roc_auc'] if metrics['roc_auc'] is not None else np.nan,
            'num_games': len(features_df)
        }
    
    return results


def run_fullmoves_experiment(
    config: dict,
    fullmoves_list: list = [10, 20, 30],
    output_dir: str = "experiments"
):
    """
    Run experiment for multiple fullmoves values.
    
    Args:
        config: Configuration dictionary
        fullmoves_list: List of fullmoves values to test
        output_dir: Directory to save results
    """
    print("="*80)
    print("FULL MOVES EXPERIMENT")
    print("="*80)
    print(f"Testing fullmoves: {fullmoves_list}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    ensure_dir(output_dir)
    
    # Load games once (reused for all experiments)
    print("\nLoading games...")
    collector = LichessDataCollector()
    raw_data_path = Path(config['paths']['raw_data']) / 'games.json'
    
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"No games data found at {raw_data_path}\n"
            "Please run data collection first: python main.py --mode collect"
        )
    
    games = collector.load_games(str(raw_data_path))
    print(f"Loaded {len(games)} games")
    
    # Preprocess once (reused for all experiments)
    print("\nPreprocessing games...")
    cleaner = DataCleaner(
        min_rating=config['data_collection']['min_rating'],
        max_rating=config['data_collection']['max_rating']
    )
    df = cleaner.clean_games(games)
    print(f"Cleaned games: {len(df)}")
    
    # Run experiments
    all_results = []
    
    for fullmoves in fullmoves_list:
        try:
            results = run_experiment_for_fullmoves(
                config=config,
                fullmoves=fullmoves,
                df=df  # Reuse preprocessed data
            )
            
            # Flatten results
            for model_name, metrics in results.items():
                all_results.append({
                    'model': model_name,
                    **metrics
                })
                
        except Exception as e:
            print(f"\nERROR running experiment for {fullmoves} fullmoves: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = Path(output_dir) / 'results_by_fullmoves.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*80}\n")
    
    # Create plots
    create_plots(results_df, output_dir)
    
    return results_df


def create_plots(results_df: pd.DataFrame, output_dir: str):
    """Create visualization plots comparing models across fullmoves."""
    print("Creating plots...")
    
    # Prepare data
    models = results_df['model'].unique()
    fullmoves_list = sorted(results_df['fullmoves'].unique())
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_to_plot = [
        ('accuracy', 'Accuracy', axes[0]),
        ('f1_weighted', 'F1 Score (Weighted)', axes[1]),
        ('roc_auc', 'ROC AUC', axes[2])
    ]
    
    for metric_name, metric_label, ax in metrics_to_plot:
        for model in models:
            model_data = results_df[results_df['model'] == model]
            model_data = model_data.sort_values('fullmoves')
            
            ax.plot(
                model_data['fullmoves'],
                model_data[metric_name],
                marker='o',
                label=model,
                linewidth=2,
                markersize=8
            )
        
        ax.set_xlabel('Number of Full Moves', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'{metric_label} vs Full Moves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(fullmoves_list)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'fullmoves_experiment_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Also create a summary table plot
    create_summary_table(results_df, output_dir)


def create_summary_table(results_df: pd.DataFrame, output_dir: str):
    """Create a summary table visualization."""
    # Pivot table for better visualization
    pivot_accuracy = results_df.pivot(index='fullmoves', columns='model', values='accuracy')
    pivot_f1 = results_df.pivot(index='fullmoves', columns='model', values='f1_weighted')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy table
    im1 = axes[0].imshow(pivot_accuracy.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(range(len(pivot_accuracy.columns)))
    axes[0].set_xticklabels(pivot_accuracy.columns, rotation=45, ha='right')
    axes[0].set_yticks(range(len(pivot_accuracy.index)))
    axes[0].set_yticklabels(pivot_accuracy.index)
    axes[0].set_title('Accuracy by Full Moves', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Full Moves', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    # Add text annotations
    for i in range(len(pivot_accuracy.index)):
        for j in range(len(pivot_accuracy.columns)):
            text = axes[0].text(j, i, f'{pivot_accuracy.iloc[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    # F1 table
    im2 = axes[1].imshow(pivot_f1.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(range(len(pivot_f1.columns)))
    axes[1].set_xticklabels(pivot_f1.columns, rotation=45, ha='right')
    axes[1].set_yticks(range(len(pivot_f1.index)))
    axes[1].set_yticklabels(pivot_f1.index)
    axes[1].set_title('F1 Score (Weighted) by Full Moves', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Full Moves', fontsize=12)
    plt.colorbar(im2, ax=axes[1])
    
    # Add text annotations
    for i in range(len(pivot_f1.index)):
        for j in range(len(pivot_f1.columns)):
            text = axes[1].text(j, i, f'{pivot_f1.iloc[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    
    table_path = Path(output_dir) / 'fullmoves_experiment_table.png'
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"Summary table saved to: {table_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Full Moves Experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--fullmoves',
        type=int,
        nargs='+',
        default=[10, 20, 30],
        help='List of fullmoves values to test (default: 10 20 30)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiments',
        help='Output directory for results (default: experiments)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'data_collection': {'min_rating': 1500, 'max_rating': 3000},
            'model': {'type': 'random_forest', 'test_size': 0.2, 'random_state': 42,
                     'n_estimators': 100, 'max_depth': 10, 'parallel_training': True},
            'paths': {'raw_data': 'data/raw', 'processed_data': 'data/processed', 'models': 'models'}
        }
    
    # Run experiment
    results_df = run_fullmoves_experiment(
        config=config,
        fullmoves_list=args.fullmoves,
        output_dir=args.output
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output}/results_by_fullmoves.csv")
    print(f"Plots saved to: {args.output}/fullmoves_experiment_results.png")
    print(f"Summary table saved to: {args.output}/fullmoves_experiment_table.png")
    print("\nResults summary:")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()

