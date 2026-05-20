# Credibility Detector

MVP systemu oceny wiarygodnosci artykulow. Projekt analizuje wklejony tekst albo URL, zwraca `credibility_score` oraz uzasadnienie wyniku.

## Zakres MVP

- wejscie: tekst lub URL,
- ekstrakcja tresci z URL przez `trafilatura` z fallbackiem BeautifulSoup,
- modularny scoring:
  - `source_score`,
  - `linguistic_score`,
  - `fact_score`,
  - `consensus_score`,
  - `consistency_score`,
- finalny weighted ensemble,
- explainability,
- syntetyczny dataset w `data/datasets/synthetic_articles.csv`,
- API w FastAPI i demo w Streamlit.

MVP celowo nie obsluguje PDF, DOCX, OCR ani obrazow.

## Uruchomienie lokalne

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload
```

API bedzie dostepne pod adresem:

```text
http://127.0.0.1:8000/docs
```

Streamlit:

```bash
streamlit run app/ui/streamlit_app.py
```

Docker Compose:

```bash
docker compose up
```

## Endpointy

`POST /analyze/text`

```json
{
  "title": "Agency report",
  "content": "The public health agency published a report...",
  "url": "https://gov.pl/report",
  "author": "Agency desk",
  "publish_date": "2026-02-12",
  "source_links": ["https://gov.pl/dataset"]
}
```

`POST /analyze/url`

```json
{
  "url": "https://example.com/article"
}
```

## Trening modelu demonstracyjnego

Dataset syntetyczny ma 200 rekordow i mozna go odtworzyc komenda:

```bash
python -m ml.training.generate_synthetic_dataset
```

Nastepnie mozna wykorzystac go do treningu lekkiego regresora:

```bash
python -m ml.training.train_model
```

Model zostanie zapisany w `ml/models/credibility_regressor.joblib`. Obecny pipeline produkcyjny uzywa deterministycznego, wyjasnialnego scoringu, co jest stabilniejsze dla malego datasetu MVP.

Dodatkowo projekt ma model tekstowy `TF-IDF + Ridge`, ktory uczy sie wzorcow jezykowych bezposrednio z tresci:

```bash
python -m ml.training.train_text_model
```

Porownanie modelu na cechach, modelu tekstowego i prostego ensemble:

```bash
python -m ml.training.evaluate_models
```

## Testy

```bash
pytest
```
