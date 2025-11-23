"""Helper utility functions."""

import yaml
from typing import Dict, Any
import os


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def parse_result(result_str: str) -> str:
    """
    Parse game result string to standardized format.
    
    Args:
        result_str: Result string (e.g., "1-0", "0-1", "1/2-1/2")
    
    Returns:
        Standardized result: "white_wins", "black_wins", or "draw"
    """
    result_str = result_str.strip().lower()
    
    if result_str in ['1-0', 'white_wins', 'white']:
        return 'white_wins'
    elif result_str in ['0-1', 'black_wins', 'black']:
        return 'black_wins'
    elif result_str in ['1/2-1/2', 'draw', '0.5-0.5']:
        return 'draw'
    else:
        return 'unknown'

