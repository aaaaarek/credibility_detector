from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ml.features.source_features import extract_source_features
from ml.features.text_features import extract_text_features


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "datasets" / "synthetic_articles.csv"
MODEL_PATH = ROOT / "ml" / "models" / "credibility_regressor.joblib"


def main() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    features = pd.DataFrame([_row_features(row) for _, row in dataset.iterrows()])
    target = dataset["credibility_label"]

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(features.columns), "mae": mae}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Validation MAE: {mae:.3f}")


def _row_features(row: pd.Series) -> dict[str, float]:
    text = extract_text_features(str(row["content"])).as_dict()
    source = extract_source_features(
        url=str(row.get("url", "")),
        author=str(row.get("author", "")),
        publish_date=str(row.get("publish_date", "")),
        source_links=str(row.get("source_links", "")).split("|") if row.get("source_links") else [],
    ).as_dict()

    return {
        **text,
        "has_url": float(source["has_url"]),
        "uses_https": float(source["uses_https"]),
        "known_reputable_domain": float(source["known_reputable_domain"]),
        "suspicious_domain_hint": float(source["suspicious_domain_hint"]),
        "has_author": float(source["has_author"]),
        "has_publish_date": float(source["has_publish_date"]),
        "source_link_count": float(source["source_link_count"]),
    }


if __name__ == "__main__":
    main()
