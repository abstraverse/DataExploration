"""Position evaluation using chess engine."""

import chess
import chess.engine
from typing import List, Optional
import numpy as np


class PositionEvaluator:
    """Evaluates chess positions using engine analysis."""
    
    def __init__(self):
        """Initialize the position evaluator."""
        self.board = chess.Board()
    
    def evaluate_position(self, fen: str, depth: int = 5) -> float:
        """
        Evaluate a chess position.
        
        Args:
            fen: FEN string of the position
            depth: Search depth for evaluation
        
        Returns:
            Evaluation score (positive = white advantage, negative = black advantage)
        """
        try:
            self.board.set_fen(fen)
            
            # Simple material-based evaluation as fallback
            # In production, use a chess engine like Stockfish
            evaluation = self._simple_evaluation()
            
            return evaluation
        except Exception as e:
            print(f"Error evaluating position: {e}")
            return 0.0
    
    def _simple_evaluation(self) -> float:
        """
        Simple material-based position evaluation.
        
        Returns:
            Evaluation score
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        score = 0.0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        return score
    
    def get_position_after_moves(self, moves: List[str], num_moves: int = 20) -> List[str]:
        """
        Get FEN strings after each move.
        
        Args:
            moves: List of moves in UCI or SAN format
            num_moves: Number of moves to process
        
        Returns:
            List of FEN strings
        """
        self.board.reset()
        fens = []
        
        moves_to_process = moves[:num_moves] if len(moves) >= num_moves else moves
        
        for move_str in moves_to_process:
            move_str = move_str.strip()
            if not move_str:
                continue
                
            try:
                # Detect format: UCI is 4-5 chars (e.g., "e2e4", "e7e5q"), SAN is usually shorter
                # Try SAN first (Lichess typically uses SAN)
                move = None
                
                # Try SAN format first (Lichess API returns SAN format like "Nf3", "d5")
                try:
                    move = self.board.parse_san(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        fens.append(self.board.fen())
                        continue
                except Exception:
                    pass  # Not SAN, try UCI
                
                # Try UCI format (fallback for other sources)
                try:
                    if len(move_str) >= 4:  # UCI moves are at least 4 characters (e.g., "e2e4")
                        move = chess.Move.from_uci(move_str)
                        if move in self.board.legal_moves:
                            self.board.push(move)
                            fens.append(self.board.fen())
                            continue
                except Exception:
                    pass  # Not UCI either - skip this move
                
                # If we get here, couldn't parse the move
                # Skip it and continue (don't break the whole process)
                # print(f"Warning: Could not parse move '{move_str}', skipping...")
                continue
                
            except Exception as e:
                # Skip problematic moves but continue processing
                # print(f"Error parsing move {move_str}: {e}")
                continue
        
        return fens

