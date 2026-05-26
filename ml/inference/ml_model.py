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
from ml.training.datasets import build_feature_frame, load_training_dataset


ROOT = Path(__file__).resolve().parents[2]
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
    return max(0.0, min(1.0, prediction)), "ml_score wyliczony przez regresor trenowany na mieszanym datasecie."


@lru_cache(maxsize=1)
def _load_or_train_model() -> tuple[GradientBoostingRegressor, list[str]]:
    if MODEL_PATH.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            artifact = joblib.load(MODEL_PATH)
        feature_columns = artifact.get("features", [])
        if feature_columns == MODEL_FEATURE_COLUMNS and artifact.get("sklearn_version") == sklearn.__version__:
            return artifact["model"], feature_columns

    dataset = load_training_dataset()

    model = GradientBoostingRegressor(random_state=42)
    model.fit(
        build_feature_frame(dataset),
        dataset["credibility_label"],
        sample_weight=dataset["sample_weight"],
    )
    return model, MODEL_FEATURE_COLUMNS
