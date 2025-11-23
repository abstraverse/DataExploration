"""Lichess API client for fetching public game data."""

import requests
import time
import json
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class LichessDataCollector:
    """Collects chess games from Lichess public API."""
    
    def __init__(self, base_url: str = "https://lichess.org/api", rate_limit_delay: float = 1.0):
        """
        Initialize the Lichess data collector.
        
        Args:
            base_url: Base URL for Lichess API
            rate_limit_delay: Delay between API requests in seconds
        """
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ChessPredictor/0.1.0'
        })
    
    def fetch_games(
        self,
        num_games: int = 1000,
        min_rating: int = 1500,
        max_rating: int = 3000,
        time_controls: Optional[List[str]] = None,
        usernames: Optional[List[str]] = None,
        api_token: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch games from Lichess database.
        
        Args:
            num_games: Number of games to fetch
            min_rating: Minimum player rating
            max_rating: Maximum player rating
            time_controls: List of time controls (e.g., ['blitz', 'rapid'])
            usernames: List of Lichess usernames to fetch games from (optional)
            api_token: Lichess API token (optional, but recommended for higher rate limits)
        
        Returns:
            List of game dictionaries
        """
        games = []
        time_controls = time_controls or ['blitz', 'rapid', 'classical']
        
        print(f"Fetching {num_games} games from Lichess...")
        
        # Set up authentication if token provided
        if api_token:
            self.session.headers.update({
                'Authorization': f'Bearer {api_token}'
            })
        
        # Method 1: Fetch games from specific users (if provided)
        if usernames:
            games = self._fetch_games_from_users(
                usernames, num_games, min_rating, max_rating, time_controls
            )
        
        # Method 2: Fetch games from Lichess export endpoint (public games)
        if len(games) < num_games:
            remaining = num_games - len(games)
            additional_games = self._fetch_games_from_export(
                remaining, min_rating, max_rating, time_controls
            )
            games.extend(additional_games)
        
        print(f"Fetched {len(games)} games from Lichess")
        return games[:num_games]  # Return exactly num_games
    
    def _fetch_games_from_users(
        self,
        usernames: List[str],
        num_games: int,
        min_rating: int,
        max_rating: int,
        time_controls: List[str]
    ) -> List[Dict]:
        """Fetch games from specific Lichess users."""
        games = []
        
        for username in usernames:
            if len(games) >= num_games:
                break
            
            try:
                print(f"Fetching games from user: {username}...")
                url = f"{self.base_url}/games/user/{username}"
                
                params = {
                    'max': min(100, num_games - len(games)),  # Lichess API limit
                    'rated': 'true',
                    'perfType': ','.join(time_controls)
                }
                
                # Set proper headers for NDJSON response
                headers = {'Accept': 'application/x-ndjson'}
                response = self.session.get(url, params=params, stream=True, headers=headers)
                response.raise_for_status()
                
                # Parse NDJSON (newline-delimited JSON) response
                batch_games = self._parse_ndjson_response(response, min_rating, max_rating)
                games.extend(batch_games)
                
                time.sleep(self.rate_limit_delay)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching games from {username}: {e}")
                continue
        
        return games
    
    def _fetch_games_from_export(
        self,
        num_games: int,
        min_rating: int,
        max_rating: int,
        time_controls: List[str]
    ) -> List[Dict]:
        """
        Fetch games from Lichess export endpoint.
        This uses the public game export API.
        """
        games = []
        
        # Lichess export endpoint for master games or recent games
        # Note: This is a simplified approach - you may need to adjust based on actual API
        try:
            # Try to fetch from master games database
            url = f"{self.base_url}/game/export"
            
            # Get recent game IDs from Lichess (this is a workaround)
            # In practice, you might need to use a different endpoint
            # or fetch games by specific criteria
            
            # Alternative: Use berserk library if available
            try:
                import berserk
                games = self._fetch_with_berserk(num_games, min_rating, max_rating, time_controls)
                if games:
                    return games
            except ImportError:
                pass  # berserk not installed, continue with direct API
            
            # Direct API approach - fetch games by IDs
            # This requires getting game IDs first, which is complex
            # For now, return empty and suggest using usernames or berserk
            
        except Exception as e:
            print(f"Error fetching from export endpoint: {e}")
        
        return games
    
    def _fetch_with_berserk(
        self,
        num_games: int,
        min_rating: int,
        max_rating: int,
        time_controls: List[str]
    ) -> List[Dict]:
        """Fetch games using berserk library (if installed)."""
        try:
            import berserk
            
            # This requires an API token
            # For now, return empty - user should provide token
            return []
            
        except ImportError:
            return []
    
    def _parse_ndjson_response(
        self,
        response: requests.Response,
        min_rating: int,
        max_rating: int
    ) -> List[Dict]:
        """Parse NDJSON (newline-delimited JSON) response from Lichess API."""
        games = []
        
        for line in response.iter_lines():
            if not line:
                continue
            
            try:
                game_data = json.loads(line)
                parsed_game = self._parse_lichess_game(game_data)
                
                if parsed_game:
                    # Filter by rating
                    if (min_rating <= parsed_game['white_rating'] <= max_rating and
                        min_rating <= parsed_game['black_rating'] <= max_rating):
                        games.append(parsed_game)
                        
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error parsing game: {e}")
                continue
        
        return games
    
    def _parse_lichess_game(self, game_data: Dict) -> Optional[Dict]:
        """
        Parse a Lichess game object into our format.
        
        Lichess API returns games in a specific format. This converts it to our format.
        """
        try:
            # Extract moves - Lichess returns moves as a space-separated string
            moves_str = game_data.get('moves', '')
            if not moves_str:
                return None
            
            # Split moves into list to count them
            moves = moves_str.split() if isinstance(moves_str, str) else moves_str
            
            # Filter games with less than 20 moves
            if len(moves) < 20:
                return None
            
            # Extract ratings from nested structure
            # Lichess format: players.white.rating and players.black.rating
            white_player = game_data.get('players', {}).get('white', {})
            black_player = game_data.get('players', {}).get('black', {})
            
            white_rating = white_player.get('rating', 1500)
            black_rating = black_player.get('rating', 1500)
            
            # Skip if ratings are missing or invalid
            if not white_rating or not black_rating:
                return None
            
            # Extract result
            winner = game_data.get('winner')
            if winner == 'white':
                result = 'white_wins'
            elif winner == 'black':
                result = 'black_wins'
            else:
                result = 'draw'
            
            # Check if game was abandoned
            status = game_data.get('status', '').lower()
            if status in ['abandoned', 'timeout', 'disconnect', 'stalemate']:
                # Note: 'resign' and 'mate' are valid finished games
                if status not in ['resign', 'mate', 'outoftime', 'draw']:
                    return None
            
            # Extract time control/perf type
            perf = game_data.get('perf', 'blitz')  # bullet, blitz, rapid, classical
            clock = game_data.get('clock', {})
            
            # Convert perf type to our time_control format
            time_control = perf  # Use perf type as time control
            
            return {
                'id': game_data.get('id', ''),
                'moves': moves_str,  # Already space-separated string
                'white_rating': int(white_rating),
                'black_rating': int(black_rating),
                'result': result,
                'time_control': time_control,
                'status': 'finished'
            }
            
        except Exception as e:
            print(f"Error parsing Lichess game: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def fetch_game_by_id(self, game_id: str, api_token: Optional[str] = None) -> Optional[Dict]:
        """
        Fetch a specific game by ID.
        
        Args:
            game_id: Lichess game ID
            api_token: Lichess API token (optional)
        
        Returns:
            Game dictionary or None if not found
        """
        try:
            if api_token:
                self.session.headers.update({
                    'Authorization': f'Bearer {api_token}'
                })
            
            url = f"{self.base_url}/game/export/{game_id}"
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse PGN or JSON response
            game_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else None
            
            if game_data:
                return self._parse_lichess_game(game_data)
            else:
                # Try parsing as PGN
                return self._parse_pgn_game(response.text, game_id)
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching game {game_id}: {e}")
            return None
    
    def _parse_pgn_game(self, pgn_text: str, game_id: str) -> Optional[Dict]:
        """Parse a PGN format game."""
        try:
            # Simple PGN parser - extract key information
            lines = pgn_text.split('\n')
            metadata = {}
            moves = []
            in_moves = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    # Metadata line like [White "username"] or [WhiteElo "1800"]
                    match = re.match(r'\[(\w+)\s+"([^"]+)"\]', line)
                    if match:
                        key, value = match.groups()
                        metadata[key] = value
                elif line and not in_moves:
                    # Start of moves section
                    in_moves = True
                    moves = line.split()
                elif in_moves:
                    moves.extend(line.split())
            
            # Extract ratings
            white_rating = int(metadata.get('WhiteElo', 1500))
            black_rating = int(metadata.get('BlackElo', 1500))
            
            # Extract result
            result_str = metadata.get('Result', '1/2-1/2')
            if result_str == '1-0':
                result = 'white_wins'
            elif result_str == '0-1':
                result = 'black_wins'
            else:
                result = 'draw'
            
            # Filter short games
            if len(moves) < 20:
                return None
            
            return {
                'id': game_id,
                'moves': ' '.join(moves),
                'white_rating': white_rating,
                'black_rating': black_rating,
                'result': result,
                'time_control': metadata.get('TimeControl', 'blitz'),
                'status': 'finished'
            }
            
        except Exception as e:
            print(f"Error parsing PGN game: {e}")
            return None
    
    def save_games(self, games: List[Dict], filepath: str):
        """
        Save games to JSON file.
        
        Args:
            games: List of game dictionaries
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(games, f, indent=2)
        print(f"Saved {len(games)} games to {filepath}")
    
    def load_games(self, filepath: str) -> List[Dict]:
        """
        Load games from JSON file.
        
        Args:
            filepath: Path to load file from
        
        Returns:
            List of game dictionaries
        """
        with open(filepath, 'r') as f:
            games = json.load(f)
        print(f"Loaded {len(games)} games from {filepath}")
        return games

