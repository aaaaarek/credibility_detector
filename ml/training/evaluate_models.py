from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ml.features.feature_builder import MODEL_FEATURE_COLUMNS, build_model_features
from ml.inference.text_model import build_text_model


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "datasets" / "synthetic_articles.csv"


def main() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    train, test = train_test_split(dataset, test_size=0.25, random_state=42)

    feature_model = GradientBoostingRegressor(random_state=42)
    feature_model.fit(_features(train), train["credibility_label"])
    feature_predictions = feature_model.predict(_features(test))

    text_model = build_text_model()
    text_model.fit(train["content"].astype(str), train["credibility_label"])
    text_predictions = text_model.predict(test["content"].astype(str))

    ensemble_predictions = 0.55 * feature_predictions + 0.45 * text_predictions

    print("Model evaluation on synthetic holdout")
    print(f"feature_model_mae: {mean_absolute_error(test['credibility_label'], feature_predictions):.3f}")
    print(f"text_model_mae: {mean_absolute_error(test['credibility_label'], text_predictions):.3f}")
    print(f"simple_ensemble_mae: {mean_absolute_error(test['credibility_label'], ensemble_predictions):.3f}")


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        links = [] if pd.isna(row.get("source_links")) else [link for link in str(row.get("source_links", "")).split("|") if link]
        rows.append(
            build_model_features(
                content=str(row["content"]),
                url=None if pd.isna(row.get("url")) else str(row.get("url", "")),
                author=None if pd.isna(row.get("author")) else str(row.get("author", "")),
                publish_date=None if pd.isna(row.get("publish_date")) else str(row.get("publish_date", "")),
                source_links=links,
            )
        )
    return pd.DataFrame(rows)[MODEL_FEATURE_COLUMNS]


if __name__ == "__main__":
    main()
