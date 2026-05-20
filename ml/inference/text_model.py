from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import warnings

import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "datasets" / "synthetic_articles.csv"
MODEL_PATH = ROOT / "ml" / "models" / "text_credibility_regressor.joblib"
TEXT_MODEL_VERSION = "tfidf-ridge-v1"


def predict_text_ml_score(content: str) -> tuple[float, str]:
    model = _load_or_train_text_model()
    prediction = float(model.predict([content])[0])
    score = max(0.0, min(1.0, prediction))
    return score, "text_ml_score wyliczony przez TF-IDF + Ridge na tresci artykulu."


@lru_cache(maxsize=1)
def _load_or_train_text_model() -> Pipeline:
    if MODEL_PATH.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            artifact = joblib.load(MODEL_PATH)
        if (
            artifact.get("model_version") == TEXT_MODEL_VERSION
            and artifact.get("sklearn_version") == sklearn.__version__
        ):
            return artifact["model"]

    dataset = pd.read_csv(DATASET_PATH)
    model = build_text_model()
    model.fit(dataset["content"].astype(str), dataset["credibility_label"])
    return model


def build_text_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
