import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import load_config
from src.models.model_comparison import StatisticalBaselineModel


def plot_and_save_matrix(model, X_test, y_test, model_name, output_dir):
    y_pred = model.predict(X_test)

    if hasattr(model, 'classes_'):
        labels = model.classes_
    else:
        labels = sorted(y_test.unique())

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    color_map = 'Blues'
    if 'Random' in model_name: color_map = 'Greens'
    if 'Baseline' in model_name: color_map = 'Oranges'

    disp.plot(cmap=color_map, ax=ax, values_format='d')
    plt.title(f"Confusion Matrix\n{model_name}")

    filename = output_dir / f"matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    config_path = project_root / 'config' / 'config.yaml'
    config = load_config(str(config_path))

    features_path = project_root / config['paths']['processed_data'] / 'features.csv'
    reports_dir = project_root / "reports"

    if not features_path.exists():
        print(f"File not found: {features_path}")
        return

    df = pd.read_csv(features_path)

    target_col = 'result'
    feature_cols = [c for c in df.columns if c not in [target_col, 'id', 'moves', 'status', 'opening_name']]

    X = df[feature_cols].select_dtypes(include=['number'])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = [
        ("Statistical Baseline", StatisticalBaselineModel()),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        plot_and_save_matrix(model, X_test, y_test, name, reports_dir)


if __name__ == "__main__":
    main()