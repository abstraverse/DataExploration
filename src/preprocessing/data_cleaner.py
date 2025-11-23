"""Data cleaning and validation for chess games."""

import pandas as pd
from typing import List, Dict, Optional


class DataCleaner:
    """Cleans and validates chess game data."""
    
    def __init__(self, min_rating: int = 1500, max_rating: int = 3000):
        """
        Initialize the data cleaner.
        
        Args:
            min_rating: Minimum acceptable player rating
            max_rating: Maximum acceptable player rating
        """
        self.min_rating = min_rating
        self.max_rating = max_rating
    
    def clean_games(self, games: List[Dict]) -> pd.DataFrame:
        """
        Clean and convert games to DataFrame.
        Filters out games that are:
        - Shorter than 20 moves
        - Missing player ratings
        - Abandoned games
        
        Args:
            games: List of raw game dictionaries
        
        Returns:
            Cleaned DataFrame
        """
        df = pd.DataFrame(games)
        
        initial_count = len(df)
        print(f"Initial games: {initial_count}")
        
        # Remove games with missing essential data
        required_columns = ['moves', 'result']
        for col in required_columns:
            if col in df.columns:
                before = len(df)
                df = df[df[col].notna()]
                after = len(df)
                if before != after:
                    print(f"Removed {before - after} games with missing {col}")
        
        # Remove games shorter than 20 moves
        if 'moves' in df.columns:
            before = len(df)
            df = df[df['moves'].str.split().apply(len) >= 20]
            after = len(df)
            if before != after:
                print(f"Removed {before - after} games with fewer than 20 moves")
        
        # Remove games without rankings (missing white_rating or black_rating)
        if 'white_rating' in df.columns and 'black_rating' in df.columns:
            before = len(df)
            df = df[
                df['white_rating'].notna() &
                df['black_rating'].notna() &
                (df['white_rating'] >= self.min_rating) &
                (df['white_rating'] <= self.max_rating) &
                (df['black_rating'] >= self.min_rating) &
                (df['black_rating'] <= self.max_rating)
            ]
            after = len(df)
            if before != after:
                print(f"Removed {before - after} games with missing or invalid ratings")
        else:
            # If rating columns don't exist, remove all games
            print("Warning: Rating columns not found. Removing all games.")
            df = df.iloc[0:0]
        
        # Remove abandoned games (result indicates abandonment)
        if 'result' in df.columns:
            before = len(df)
            # Abandoned games typically have results like "abandoned", "timeout", or similar
            abandoned_keywords = ['abandoned', 'timeout', 'disconnect', 'unknown']
            if df['result'].dtype == 'object':
                df = df[~df['result'].str.lower().isin(abandoned_keywords)]
            after = len(df)
            if before != after:
                print(f"Removed {before - after} abandoned games")
        
        # Also check status field if it exists
        if 'status' in df.columns:
            before = len(df)
            abandoned_statuses = ['abandoned', 'timeout', 'disconnect']
            if df['status'].dtype == 'object':
                df = df[~df['status'].str.lower().isin(abandoned_statuses)]
            after = len(df)
            if before != after:
                print(f"Removed {before - after} games with abandoned status")
        
        final_count = len(df)
        print(f"Final games after cleaning: {final_count} ({final_count/initial_count*100:.1f}% retained)")
        
        return df
    
    def validate_game(self, game: Dict) -> bool:
        """
        Validate a single game.
        Checks for:
        - At least 20 moves
        - Both player ratings present
        - Not abandoned
        
        Args:
            game: Game dictionary
        
        Returns:
            True if game is valid, False otherwise
        """
        required_fields = ['moves', 'white_rating', 'black_rating', 'result']
        
        # Check required fields exist
        for field in required_fields:
            if field not in game or game[field] is None:
                return False
        
        # Check ratings are valid numbers
        try:
            white_rating = float(game['white_rating'])
            black_rating = float(game['black_rating'])
        except (ValueError, TypeError):
            return False
        
        # Check rating ranges
        if not (self.min_rating <= white_rating <= self.max_rating):
            return False
        if not (self.min_rating <= black_rating <= self.max_rating):
            return False
        
        # Check moves exist and have at least 20 moves
        moves = game.get('moves', '')
        if not moves or len(moves.split()) < 20:
            return False
        
        # Check not abandoned
        result = str(game.get('result', '')).lower()
        status = str(game.get('status', '')).lower()
        abandoned_keywords = ['abandoned', 'timeout', 'disconnect', 'unknown']
        
        if result in abandoned_keywords or status in abandoned_keywords:
            return False
        
        return True

