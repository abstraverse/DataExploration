# Przewidywanie Wyniku Partii Szachowej na Podstawie Pozycji z Wybranego Momentu Gry

## Postawienie Problemu

Czy można przewidzieć wynik partii szachowej (wygrana białych, wygrana czarnych lub remis) na podstawie pozycji z wybranego momentu gry, bez konieczności symulowania całej partii do końca? To pytanie jest kluczowe dla zrozumienia, na ile wczesne fazy gry determinują jej ostateczny rezultat.

W praktyce, mamy dostęp do milionów partii szachowych zapisanych w formacie PGN. Z każdej partii możemy wyodrębnić pozycję z konkretnego momentu (np. po 10, 20 lub 30 pełnych ruchach) i na tej podstawie próbować przewidzieć końcowy wynik.

## Dotychczasowe Rozwiązania

### Tradycyjne Podejścia

1. **Symulacja całej partii**: Najbardziej oczywiste rozwiązanie to symulowanie partii do końca przy użyciu silników szachowych (np. Stockfish). Jednak jest to czasochłonne i wymaga dużej mocy obliczeniowej.

2. **Ocena pozycji przez silniki**: Silniki szachowe mogą ocenić pozycję, ale nie przewidują bezpośrednio wyniku partii. Ocena pozycji (np. +2.5 dla białych) wskazuje na przewagę, ale nie gwarantuje wygranej.

3. **Proste heurystyki**: Różnica w materiale, różnica w rankingach ELO - te proste wskaźniki mogą dawać pewne wskazówki, ale są zbyt uproszczone.

### Podejścia z Uczenia Maszynowego

Publikacja "Predicting the Outcome of a Chess Game by Statistical and Machine Learning Techniques" (Héctor Apolo Rosales Pulido) pokazuje, że:

- Modele statystyczne (regresja logistyczna) mogą osiągać przyzwoite wyniki
- Modele ML (drzewa decyzyjne, random forest) przewyższają proste metody statystyczne
- Najważniejsze cechy to: różnica rankingów ELO, ocena pozycji przez silnik, przewaga materiału
- Jakość predykcji rośnie wraz z liczbą przeanalizowanych ruchów

## Nowe Podejście - Nasza Implementacja

### Główne Założenia

Nasza implementacja opiera się na publikacji, ale wprowadza kilka ulepszeń:

1. **Analiza wpływu liczby pełnych ruchów**: Eksperymentujemy z różną liczbą pełnych ruchów (10, 20, 30), aby systematycznie zbadać, jak ilość informacji wpływa na jakość predykcji.

2. **Porównanie trzech modeli**:
   - **Model statystyczny bazowy**: Prosty model oparty wyłącznie na różnicy rankingów ELO
   - **Regresja logistyczna**: Klasyczny model statystyczny, interpretowalny i szybki
   - **Random Forest**: Zaawansowany model ML, który może uchwycić złożone zależności

3. **Ekstrakcja cech z pozycji**:
   - Ocena pozycji (material-based evaluation)
   - Różnica rankingów ELO
   - Średnia rankingów
   - Statystyki z oceny pozycji (średnia, odchylenie, trend)
   - Kontrola czasu (blitz/rapid/classical)
   - Identyfikacja debiutu

4. **Systematyczna walidacja**: Wszystkie modele są trenowane i testowane na tych samych danych, z użyciem tych samych metryk (accuracy, F1 score, ROC AUC).

### Architektura Rozwiązania

```
Dane wejściowe (Lichess API)
    ↓
Preprocessing (filtrowanie: ≥20 półruchów, rankingi, zakończone partie)
    ↓
Ekstrakcja cech (pozycja po N pełnych ruchach)
    ↓
Trenowanie modeli (równoległe)
    ↓
Porównanie i wybór najlepszego modelu
    ↓
Eksperyment z różną liczbą ruchów
    ↓
Wizualizacja wyników
```

### Kluczowe Cechy Implementacji

1. **Automatyczna ekstrakcja danych**: Integracja z Lichess API do pobierania rzeczywistych partii
2. **Filtrowanie danych**: Automatyczne usuwanie partii zbyt krótkich, bez rankingów lub porzuconych
3. **Równoległe trenowanie**: Modele trenują jednocześnie, co przyspiesza proces
4. **Eksperymentalna weryfikacja**: Systematyczne badanie wpływu liczby ruchów na jakość predykcji
5. **Wizualizacja wyników**: Automatyczne generowanie wykresów i tabel porównawczych

## Metodyka

### Zbiór Danych

- **Źródło**: Lichess.org - publiczna baza danych partii szachowych
- **Filtrowanie**: 
  - Partie z co najmniej 20 półruchami
  - Partie z ważnymi rankingami obu graczy
  - Tylko zakończone partie (bez porzuconych)
  - Rankingi w zakresie 1500-3000

### Ekstrakcja Cech

Dla każdej partii, po N pełnych ruchach, ekstrahujemy:

1. **Cechy rankingowe**:
   - Ranking białych
   - Ranking czarnych
   - Różnica rankingów
   - Średnia rankingów

2. **Cechy pozycyjne**:
   - Średnia ocena pozycji
   - Odchylenie standardowe ocen
   - Maksymalna/minimalna ocena
   - Ostateczna ocena pozycji
   - Trend oceny (zmiana w czasie)

3. **Cechy kontekstowe**:
   - Kontrola czasu (blitz/rapid/classical)
   - Identyfikacja debiutu

### Modele

1. **Model Statystyczny Bazowy**:
   - Przewiduje wynik wyłącznie na podstawie różnicy rankingów
   - Jeśli różnica > 50 punktów → wygrywa silniejszy gracz
   - W przeciwnym razie → remis

2. **Regresja Logistyczna**:
   - Klasyczny model statystyczny
   - Interpretowalne współczynniki
   - Dobra jako baseline dla złożonych modeli

3. **Random Forest**:
   - Ensemble method z wieloma drzewami decyzyjnymi
   - Może uchwycić nieliniowe zależności
   - Odporny na overfitting

### Eksperyment

Eksperyment polega na trenowaniu wszystkich trzech modeli dla różnych wartości liczby pełnych ruchów:
- 10 pełnych ruchów (20 półruchów)
- 20 pełnych ruchów (40 półruchów)
- 30 pełnych ruchów (60 półruchów)

Dla każdej konfiguracji mierzymy:
- Accuracy (dokładność)
- F1 Score (zbalansowana metryka)
- ROC AUC (gdy dostępne)
- Precision i Recall

## Oczekiwane Wyniki

Na podstawie publikacji i wstępnych testów, oczekujemy:

1. **Wzrost jakości z liczbą ruchów**: Im więcej ruchów przeanalizujemy, tym lepsza predykcja
2. **Random Forest jako najlepszy model**: Zaawansowane modele ML powinny przewyższać proste metody
3. **Różnica ELO jako kluczowy predyktor**: Rankingi graczy mają duży wpływ na wynik
4. **Ocena pozycji jako ważna cecha**: Pozycyjne cechy znacząco poprawiają predykcję

## Wnioski i Wkład

### Wkład Naukowy

1. **Systematyczna analiza wpływu liczby ruchów**: Badamy, jak ilość informacji wpływa na jakość predykcji
2. **Porównanie modeli na tych samych danych**: Sprawiedliwe porównanie różnych podejść
3. **Reprodukowalna metodologia**: Pełna implementacja dostępna publicznie

### Praktyczne Zastosowania

1. **Analiza partii**: Szybka ocena szans w trakcie partii
2. **Trening szachowy**: Identyfikacja momentów, w których partia się rozstrzyga
3. **Systemy rekomendacji**: Sugerowanie najlepszych ruchów na podstawie przewidywanego wyniku

## Referencje

- Rosales Pulido, H. A. "Predicting the Outcome of a Chess Game by Statistical and Machine Learning Techniques"
- Lichess API: https://lichess.org/api
- Python-Chess Library: https://python-chess.readthedocs.io/

## Technologie

- **Python 3.10+**
- **Scikit-learn**: Modele ML
- **Python-Chess**: Analiza pozycji szachowych
- **Pandas/NumPy**: Przetwarzanie danych
- **Matplotlib**: Wizualizacja wyników
- **Lichess API**: Pobieranie danych

