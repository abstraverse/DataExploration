# Chess Game Result Predictor

A machine learning project that predicts chess game results based on the first 20 moves, position evaluations, and player ratings using public games from Lichess.com.

## Project Structure

```
PythonProject/
├── data/                   # Raw and processed data
│   ├── raw/               # Raw Lichess game data
│   └── processed/         # Processed features and datasets
├── models/                # Trained model files
├── src/                   # Source code
│   ├── data_collection/   # Lichess API integration
│   ├── preprocessing/     # Data cleaning and preparation
│   ├── features/          # Feature extraction (position eval, ratings)
│   ├── models/            # ML model definitions and training
│   └── utils/             # Helper functions
├── config/                # Configuration files
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── main.py               # Main entry point
```

## Features

- **Data Collection**: Fetches public games from Lichess.com API
- **Position Evaluation**: Analyzes chess positions using engine evaluation
- **Player Ratings**: Incorporates player ELO ratings as features
- **Move Analysis**: Extracts features from the first 20 moves
- **Result Prediction**: Predicts game outcome (win/loss/draw)

## Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the project (if needed):
   - Edit `config/config.yaml` for API settings and model parameters

4. Generate sample data (for testing):
```bash
python generate_sample_data.py --num-games 1000
```

5. Run the main script:
```bash
# Run full pipeline
python main.py --mode all

# Or run individual steps
python main.py --mode preprocess  # Only preprocess
python main.py --mode features    # Only extract features
python main.py --mode train        # Only train models
```

**Note**: Remember to activate the virtual environment (`source venv/bin/activate`) before running any scripts or installing packages.

## Quick Start

1. Activate venv: `source venv/bin/activate`
2. Generate sample data: `python generate_sample_data.py`
3. Run pipeline: `python main.py --mode all`
4. Check results in `models/model_comparison.csv`

## Usage

### Collecting Data
```python
from src.data_collection.lichess_api import LichessDataCollector

collector = LichessDataCollector()
games = collector.fetch_games(num_games=1000)
```

### Training Model
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train()
```

### Making Predictions
```python
from src.models.predictor import GamePredictor

predictor = GamePredictor()
result = predictor.predict(moves, white_rating, black_rating)
```

## Scientific Experiment: Full Moves Analysis

The project includes a key experiment investigating how the number of full moves analyzed affects prediction accuracy.

### Running the Experiment

```bash
# Run experiment with default fullmoves: 10, 20, 30
python experiment_fullmoves.py

# Custom fullmoves values
python experiment_fullmoves.py --fullmoves 5 10 15 20 25 30

# Custom output directory
python experiment_fullmoves.py --output my_experiments
```

### Experiment Output

After running, you'll find in the `experiments/` directory:

1. **`results_by_fullmoves.csv`**: Detailed results for all models and fullmoves values
2. **`fullmoves_experiment_results.png`**: Line plots comparing Accuracy, F1 Score, and ROC AUC
3. **`fullmoves_experiment_table.png`**: Heatmap tables for quick comparison

### What It Tests

- **10 full moves** (20 half-moves): Early game positions
- **20 full moves** (40 half-moves): Mid-game positions
- **30 full moves** (60 half-moves): More developed positions

The experiment trains all three models (Baseline, Logistic Regression, Random Forest) for each configuration and compares results.

See `EXPERIMENT_DESCRIPTION.md` for detailed methodology and interpretation.

## Data Sources

- **Lichess API**: Public game database
- **Position Evaluation**: python-chess library for position analysis

## License

MIT

