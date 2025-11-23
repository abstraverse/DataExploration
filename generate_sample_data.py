"""Generate sample chess game data for testing."""

import json
import random
from pathlib import Path

# Sample chess moves (first 20 moves of common openings)
SAMPLE_OPENINGS = [
    # Ruy Lopez
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f6e4 d2d4 b7b5 a4b3 d7d5 d4e5 c8e6 c2c3 e4c5 b3c2",
    # Sicilian Defense
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 f1e2 f8g7 e1g1 e8g8 c1e3 b8c6",
    # Queen's Gambit
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c4d5 e6d5 c1f4 f8b4 e2e3 e8g8 f1d3 b8c6 g1f3 c8g4",
    # French Defense
    "e2e4 e7e6 d2d4 d7d5 b1c3 d5e4 c3e4 b8d7 g1f3 g8f6 e4f6 d7f6 f1d3 f8e7 e1g1 e8g8",
    # Caro-Kann
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 b8d7 g1f3 g8f6 e4f6 d7f6 f1c4 g7g6"
]

RESULTS = ['white_wins', 'black_wins', 'draw']
TIME_CONTROLS = ['blitz', 'rapid', 'classical']


def generate_sample_game(game_id: int) -> dict:
    """Generate a single sample game."""
    # Select random opening
    moves = random.choice(SAMPLE_OPENINGS)
    
    # Generate realistic ratings (1500-2500 range)
    white_rating = random.randint(1500, 2500)
    black_rating = random.randint(1500, 2500)
    
    # Result based on rating difference (with some randomness)
    rating_diff = white_rating - black_rating
    if rating_diff > 100:
        result = 'white_wins' if random.random() > 0.2 else random.choice(RESULTS)
    elif rating_diff < -100:
        result = 'black_wins' if random.random() > 0.2 else random.choice(RESULTS)
    else:
        result = random.choice(RESULTS)
    
    return {
        'id': f'sample_game_{game_id}',
        'moves': moves,
        'white_rating': white_rating,
        'black_rating': black_rating,
        'result': result,
        'time_control': random.choice(TIME_CONTROLS),
        'status': 'finished'  # Not abandoned
    }


def generate_sample_data(num_games: int = 1000, output_path: str = 'data/raw/games.json'):
    """Generate sample chess game data."""
    print(f"Generating {num_games} sample games...")
    
    games = []
    for i in range(num_games):
        games.append(generate_sample_game(i))
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_games} games...")
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(games, f, indent=2)
    
    print(f"\n✓ Generated {num_games} sample games")
    print(f"✓ Saved to {output_path}")
    
    # Print statistics
    results_count = {}
    for game in games:
        result = game['result']
        results_count[result] = results_count.get(result, 0) + 1
    
    print(f"\nResult distribution:")
    for result, count in results_count.items():
        print(f"  {result}: {count} ({count/num_games*100:.1f}%)")
    
    return games


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample chess game data')
    parser.add_argument(
        '--num-games',
        type=int,
        default=1000,
        help='Number of games to generate (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/games.json',
        help='Output file path (default: data/raw/games.json)'
    )
    
    args = parser.parse_args()
    
    generate_sample_data(args.num_games, args.output)

