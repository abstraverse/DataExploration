#!/bin/bash

# Quick run script for Chess Game Predictor

echo "=========================================="
echo "Chess Game Result Predictor"
echo "=========================================="
echo ""

# Check if venv is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if sample data exists
if [ ! -f "data/raw/games.json" ]; then
    echo "📊 Generating sample data..."
    python generate_sample_data.py --num-games 1000
    echo ""
fi

# Run the pipeline
echo "🚀 Running full pipeline..."
python main.py --mode all

echo ""
echo "=========================================="
echo "Done! Check results in:"
echo "  - models/model_comparison.csv"
echo "  - models/best_model.pkl"
echo "=========================================="

