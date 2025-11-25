# Przewidywanie Wyniku Partii Szachowej - Chess Game Result Predictor

Projekt uczenia maszynowego do przewidywania wyników partii szachowych na podstawie pozycji z wybranego momentu gry, oceny pozycji oraz rankingów graczy. Implementacja oparta na publikacji "Predicting the Outcome of a Chess Game by Statistical and Machine Learning Techniques".

**Język**: Python 3.10+  
**Środowisko**: Virtual environment (venv)  
**Dane**: Lichess.org API

## Struktura Projektu

```
PythonProject/
├── data/                   # Dane surowe i przetworzone
│   ├── raw/               # Surowe dane partii z Lichess
│   └── processed/         # Przetworzone cechy i zbiory danych
├── models/                # Pliki wytrenowanych modeli
├── src/                   # Kod źródłowy
│   ├── data_collection/   # Integracja z API Lichess
│   ├── preprocessing/     # Czyszczenie i przygotowanie danych
│   ├── features/          # Ekstrakcja cech (ocena pozycji, rankingi)
│   ├── models/            # Definicje i trenowanie modeli ML
│   └── utils/             # Funkcje pomocnicze
├── config/                # Pliki konfiguracyjne
├── notebooks/             # Notatniki Jupyter do eksploracji
├── reports/               # Wygenerowane wykresy wizualizacyjne
├── experiments/           # Wyniki eksperymentów
├── tests/                 # Testy jednostkowe
├── requirements.txt       # Zależności Pythona
└── main.py               # Główny punkt wejścia
```

## Funkcjonalności

- **Zbieranie Danych**: Pobiera publiczne partie z API Lichess.com
- **Ocena Pozycji**: Analizuje pozycje szachowe przy użyciu oceny silnika
- **Rankingi Graczy**: Wykorzystuje rankingi ELO graczy jako cechy
- **Analiza Ruchów**: Ekstrahuje cechy z pierwszych N pełnych ruchów
- **Przewidywanie Wyniku**: Przewiduje wynik partii (wygrana/remis)
- **Porównanie Modeli**: Automatycznie porównuje 3 modele (Baseline, Regresja Logistyczna, Random Forest)
- **Równoległe Trenowanie**: Modele trenują jednocześnie, co przyspiesza proces
- **Wizualizacja**: Generuje wykresy i tabele porównawcze wyników eksperymentów

## Instalacja i Uruchomienie

### Wymagania

- Python 3.10 lub nowszy
- pip (menedżer pakietów Pythona)
- Git (opcjonalnie, dla repozytorium)

### Krok 1: Przygotowanie Środowiska

1. Utwórz i aktywuj środowisko wirtualne:
```bash
# Utworzenie środowiska wirtualnego
python3 -m venv venv

# Aktywacja środowiska wirtualnego
# Na macOS/Linux:
source venv/bin/activate
# Na Windows:
# venv\Scripts\activate
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

### Krok 2: Konfiguracja (Opcjonalne)

Edytuj `config/config.yaml` jeśli chcesz zmienić:
- Liczbę partii do pobrania
- Zakres rankingów
- Parametry modeli

### Krok 3: Pobranie Danych

**Opcja A: Pobierz rzeczywiste dane z Lichess** (zalecane):
```bash
# Upewnij się, że masz usernames w config/config.yaml
python main.py --mode collect
```

**Opcja B: Wygeneruj przykładowe dane** (do testów):
```bash
python generate_sample_data.py --num-games 1000
```

### Krok 4: Uruchomienie Pipeline

**Pełny pipeline** (zbieranie danych → preprocessing → ekstrakcja cech → trenowanie):
```bash
python main.py --mode all
```

**Lub osobne kroki**:
```bash
python main.py --mode preprocess  # Tylko preprocessing
python main.py --mode features    # Tylko ekstrakcja cech
python main.py --mode train        # Tylko trenowanie modeli
```

**Uwaga**: Pamiętaj o aktywacji środowiska wirtualnego (`source venv/bin/activate`) przed uruchomieniem skryptów.

## Szybki Start

```bash
# 1. Aktywuj środowisko
source venv/bin/activate

# 2. Zainstaluj zależności (jeśli jeszcze nie)
pip install -r requirements.txt

# 3. Pobierz dane (lub wygeneruj przykładowe)
python main.py --mode collect
# LUB
python generate_sample_data.py

# 4. Uruchom pełny pipeline
python main.py --mode all

# 5. Sprawdź wyniki
cat models/model_comparison.csv
```

## Dokumentacja

- **`DESC.md`**: Szczegółowy opis projektu, metodologii i podejścia
- **`EXAMPLE.md`**: Przykład analizy na konkretnych danych z interpretacją wyników
- **`README.md`**: Ten plik - instrukcja uruchomienia

## Użycie

### Zbieranie Danych
```python
from src.data_collection.lichess_api import LichessDataCollector

collector = LichessDataCollector()
games = collector.fetch_games(num_games=1000)
```

### Trenowanie Modeli
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
# Automatycznie trenuje i porównuje wszystkie modele
comparator = trainer.train_and_compare_models(features_df)
```

### Przewidywanie Wyników
```python
from src.models.predictor import GamePredictor

# Załaduj najlepszy model
predictor = GamePredictor(model_path='models/best_model.pkl')

# Przewiduj wynik
result = predictor.predict(
    moves=['e2e4', 'e7e5', 'g1f3'],
    white_rating=1800,
    black_rating=1750,
    time_control='blitz'
)

print(f"Przewidywany wynik: {result['prediction']}")
print(f"Pewność: {result.get('confidence', 'N/A')}")
```


## Eksperyment Naukowy: Analiza Wpływu Liczby Pełnych Ruchów

Projekt zawiera kluczowy eksperyment badający, jak liczba przeanalizowanych pełnych ruchów wpływa na dokładność predykcji.

### Uruchomienie Eksperymentu

```bash
# Eksperyment z domyślnymi wartościami: 10, 20, 30 pełnych ruchów
python experiment_fullmoves.py

# Własne wartości pełnych ruchów
python experiment_fullmoves.py --fullmoves 5 10 15 20 25 30

# Własny katalog wyjściowy
python experiment_fullmoves.py --output my_experiments
```

### Wyniki Eksperymentu

Po uruchomieniu, w katalogu `experiments/` znajdziesz:

1. **`results_by_fullmoves.csv`**: Szczegółowe wyniki dla wszystkich modeli i wartości pełnych ruchów
2. **`fullmoves_experiment_results.png`**: Wykresy liniowe porównujące Accuracy, F1 Score i ROC AUC
3. **`fullmoves_experiment_table.png`**: Tabele heatmap do szybkiego porównania

### Co Testuje Eksperyment

- **10 pełnych ruchów** (20 półruchów): Pozycje wczesnej gry
- **20 pełnych ruchów** (40 półruchów): Pozycje środkowej gry
- **30 pełnych ruchów** (60 półruchów): Bardziej rozwinięte pozycje

Eksperyment trenuje wszystkie trzy modele (Baseline, Regresja Logistyczna, Random Forest) dla każdej konfiguracji i porównuje wyniki.

Zobacz `DESC.md` i `EXAMPLE.md` dla szczegółowej metodologii i interpretacji wyników.

## Analiza Wizualna

Zrozum, gdzie modele popełniają błędy, używając wygenerowanych macierzy pomyłek (Confusion Matrix).

### Generowanie Raportów
Uruchom skrypt wizualizacyjny, aby ocenić wszystkie wytrenowane modele:

```bash
python reports/visualize_results.py
```

To utworzy mapy cieplne PNG w katalogu **reports/**:

- matrix_statistical_baseline.png
- matrix_logistic_regression.png
- matrix_random_forest.png

### Jak czytać Macierz Pomyłek?

**Oś Y (True Label)**: Rzeczywisty wynik partii.

**Oś X (Predicted Label)**: To, co przewidział model.

**Przekątna**: Poprawne przewidywania (ciemniejszy kolor = lepiej).

**Poza przekątną**: Błędy (np. jeśli wiersz "Draw" jest jasny, model ma problemy z przewidywaniem remisów).

## Źródła Danych

- **Lichess API**: Publiczna baza danych partii szachowych
- **Ocena Pozycji**: Biblioteka python-chess do analizy pozycji


