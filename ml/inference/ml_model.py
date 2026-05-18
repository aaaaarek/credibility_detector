from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import warnings

import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.ensemble import GradientBoostingRegressor

from ml.features.feature_builder import MODEL_FEATURE_COLUMNS, build_model_features


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "datasets" / "synthetic_articles.csv"
MODEL_PATH = ROOT / "ml" / "models" / "credibility_regressor.joblib"


def predict_ml_score(
    content: str,
    url: str | None = None,
    author: str | None = None,
    publish_date: str | None = None,
    source_links: list[str] | None = None,
) -> tuple[float, str]:
    model, feature_columns = _load_or_train_model()
    features = build_model_features(content, url, author, publish_date, source_links)
    frame = pd.DataFrame([{column: features.get(column, 0.0) for column in feature_columns}])
    prediction = float(model.predict(frame)[0])
    return max(0.0, min(1.0, prediction)), "ml_score wyliczony przez regresor trenowany na syntetycznym datasecie."


@lru_cache(maxsize=1)
def _load_or_train_model() -> tuple[GradientBoostingRegressor, list[str]]:
    if MODEL_PATH.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            artifact = joblib.load(MODEL_PATH)
        feature_columns = artifact.get("features", [])
        if feature_columns == MODEL_FEATURE_COLUMNS and artifact.get("sklearn_version") == sklearn.__version__:
            return artifact["model"], feature_columns

    dataset = pd.read_csv(DATASET_PATH)
    rows = []
    for _, row in dataset.iterrows():
        links = _parse_links(row.get("source_links", ""))
        rows.append(
            build_model_features(
                content=str(row["content"]),
                url=_optional(row.get("url")),
                author=_optional(row.get("author")),
                publish_date=_optional(row.get("publish_date")),
                source_links=links,
            )
        )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(pd.DataFrame(rows)[MODEL_FEATURE_COLUMNS], dataset["credibility_label"])
    return model, MODEL_FEATURE_COLUMNS


def _parse_links(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [link for link in str(value).split("|") if link]


def _optional(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
