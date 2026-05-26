# Credibility Detector

MVP systemu oceny wiarygodnosci artykulow. Projekt analizuje wklejony tekst albo URL, zwraca `credibility_score` oraz uzasadnienie wyniku.

## Zakres MVP

- wejscie: tekst, URL albo plik,
- ekstrakcja tresci z URL przez `trafilatura` z fallbackiem BeautifulSoup,
- modularny scoring:
  - `source_score`,
  - `linguistic_score`,
  - `fact_score`,
  - `consensus_score`,
  - `consistency_score`,
- finalny weighted ensemble z kara za rozjazd modulow,
- explainability,
- syntetyczny dataset w `data/datasets/synthetic_articles.csv` oraz skrypt do lokalnego zebrania seed datasetu realnych artykulow,
- API w FastAPI i demo w Streamlit.
- dodatkowy `profile_score` dla postow i screenshotow, gdy znane sa dane profilu autora.
- dynamiczne wagi zalezne od typu wejscia: `url`, `raw_text`, `document`, `screenshot`.

Obsluga plikow obejmuje PDF, DOCX, TXT oraz obrazy przez OCR. OCR korzysta z EasyOCR, wiec nie wymaga zainstalowanego programu Tesseract w systemie.

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

`POST /analyze/file`

Upload pliku przez formularz `multipart/form-data`, pole `file`.

Dla postow z mediow spolecznosciowych mozna dodac metadata profilu:

- `profile_name`
- `profile_url`
- `platform`
- `is_verified`
- `follower_count`
- `account_age_days`

To nie jest live scraping konta. System ocenia wiarygodnosc profilu na podstawie danych przekazanych przez uzytkownika albo uchwytu wykrytego w OCR.

Odpowiedz zawiera jeden finalny `credibility_score`. Pole `module_scores` pokazuje moduly uzyte w finalnym wyniku, a `diagnostic_scores` pokazuje sygnaly pomocnicze, ktore nie byly liczone dla danego typu wejscia.

## Dataset i trening

Dataset syntetyczny ma 200 rekordow i mozna go odtworzyc komenda:

```bash
python -m ml.training.generate_synthetic_dataset
```

Publiczne repo nie zawiera pelnego CSV z realnymi artykulami, bo taki plik moze zawierac skopiowane tresci chronione prawem autorskim. W repo zostaje tylko naglowek/szablon:

```text
data/datasets/real_articles_seed.csv
```

Pelny lokalny seed dataset mozna trzymac poza Gitem w:

```text
data/datasets/real_articles_seed.local.csv
```

Kod automatycznie uzyje pliku `.local.csv`, jesli istnieje.

Wymagane kolumny:

- `title`
- `content`
- `url`
- `source`
- `author`
- `publish_date`
- `source_links`
- `credibility_label`

Opcjonalne, ale zalecane:

- `label_reason`
- `dataset_source`

`credibility_label` jest liczba od `0.0` do `1.0`. Linki w `source_links` rozdzielamy znakiem `|`.

Walidacja realnego datasetu:

```bash
python -m ml.training.validate_real_dataset
```

Nie trzeba recznie wpisywac setek artykulow. Mozna wpisac URL-e do:

```text
data/datasets/real_article_urls.txt
```

i wygenerowac plik kandydatow:

```bash
python -m ml.training.collect_real_dataset_candidates
```

Skrypt zapisze `data/datasets/real_articles_candidates.csv` z pobrana trescia, metadanymi i sugerowana ocena. Ten plik jest ignorowany przez Git, zeby nie publikowac pelnych tresci artykulow. Te oceny sa tylko propozycjami: zaakceptowane/poprawione wiersze nalezy dopiero przeniesc do lokalnego `real_articles_seed.local.csv`.

Trening automatycznie laczy `synthetic_articles.csv` i `real_articles_seed.csv`. Realne rekordy maja domyslnie wage `1.0`, a syntetyczne `0.25`, zeby syntetyk uzupelnial dane, ale nie dominowal treningu.

Trening lekkiego regresora na cechach:

```bash
python -m ml.training.train_model
```

Model zostanie zapisany w `ml/models/credibility_regressor.joblib`. Obecny pipeline laczy deterministyczny, wyjasnialny scoring z lekkimi modelami ML trenowanymi na syntetycznym i realnym seed datasecie.

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
