# Quick Command Reference

## 🚀 Fastest Way to Run

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Generate sample data (if not exists)
python generate_sample_data.py

# 3. Run everything
python main.py --mode all
```

Or use the script:
```bash
./run.sh
```

## 📋 Complete Command List

### Setup (One-time)
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Generate Sample Data
```bash
# Generate 1000 sample games
python generate_sample_data.py

# Generate custom amount
python generate_sample_data.py --num-games 5000
```

### Run Pipeline

**Full pipeline (recommended):**
```bash
python main.py --mode all
```

**Individual steps:**
```bash
python main.py --mode preprocess   # Clean data
python main.py --mode features    # Extract features  
python main.py --mode train       # Train models
```

### View Results
```bash
# View model comparison
cat models/model_comparison.csv

# Or open in spreadsheet
open models/model_comparison.csv
```

### Make Predictions
```python
from src.models.predictor import GamePredictor

predictor = GamePredictor(model_path='models/best_model.pkl')
result = predictor.predict(
    moves=['e2e4', 'e7e5', 'g1f3'],
    white_rating=1800,
    black_rating=1750
)
print(result['prediction'])
```

## 📁 Output Files

After running, you'll find:
- `data/raw/games.json` - Raw game data
- `data/processed/cleaned_games.csv` - Cleaned games
- `data/processed/features.csv` - Extracted features
- `models/model_comparison.csv` - Model comparison results
- `models/best_model.pkl` - Best trained model

## ⚙️ Configuration

Edit `config/config.yaml` to change:
- Number of games
- Rating ranges
- Model parameters
- Parallel training (on/off)

