"""
Database of chess openings mapped to SAN moves (Standard Algebraic Notation).
Format: "move1 move2 move3...": "Opening Name"
"""

COMMON_OPENINGS = {
    # e4 Openings (King's Pawn)
    "e4 c5 d6": "Sicilian (Classical)",
    "e4 c5 Nc6": "Sicilian (Open)",
    "e4 c5": "Sicilian Defense",

    "e4 e5 Nf3 Nc6 Bb5": "Ruy Lopez",
    "e4 e5 Nf3 Nc6 Bc4": "Italian Game",
    "e4 e5 f4": "King's Gambit",
    "e4 e5 Nf3 Nf6": "Petrov's Defense",
    "e4 e5 Nc3": "Vienna Game",
    "e4 e5": "King's Pawn Game",

    "e4 e6 d4 d5": "French Defense",
    "e4 e6": "French Defense",

    "e4 c6 d4 d5": "Caro-Kann Defense",
    "e4 c6": "Caro-Kann Defense",

    "e4 d6 d4 Nf6": "Pirc Defense",
    "e4 d6": "Pirc Defense",

    "e4 Nf6": "Alekhine Defense",
    "e4 Nc6": "Nimzowitsch Defense",

    # d4 Openings (Queen's Pawn)
    "d4 d5 c4 c6": "Slav Defense",
    "d4 d5 c4 e6": "Queen's Gambit Declined",
    "d4 d5 c4": "Queen's Gambit",
    "d4 d5 Bf4": "London System",
    "d4 d5": "Queen's Pawn Game",

    "d4 Nf6 c4 g6": "King's Indian / Grunfeld",
    "d4 Nf6 c4 e6": "Nimzo-Indian / Queen's Indian",
    "d4 f5": "Dutch Defense",

    # Flank Openings
    "c4 e5": "English (Reverse Sicilian)",
    "c4": "English Opening",
    "Nf3": "Reti Opening",
    "f4": "Bird's Opening",
    "b3": "Nimzo-Larsen Attack",
}

def get_opening_name(moves_list):
    """
    Matches a list of moves against the opening database.
    Supports SAN format (e.g. ['e4', 'c5']).
    """
    if not moves_list:
        return "Unknown"

    # Łączymy pierwsze ruchy w jeden ciąg
    # Sprawdzamy do 8 ruchów (półruchów) w głąb
    game_str = " ".join(moves_list[:8]).strip()

    best_match = "Unknown"
    longest_key_len = 0

    for sequence, name in COMMON_OPENINGS.items():
        if game_str.startswith(sequence):
            if len(sequence) > longest_key_len:
                best_match = name
                longest_key_len = len(sequence)

    # Fallback - jeśli nic nie znalazł w bazie, próbujemy ogólnej klasyfikacji
    if best_match == "Unknown" and moves_list:
        first_move = moves_list[0]
        if first_move == "e4":
            return "King's Pawn Game"
        elif first_move == "d4":
            return "Queen's Pawn Game"
        elif first_move == "c4":
            return "English Opening"
        elif first_move == "Nf3":
            return "Reti Opening"

    return best_match