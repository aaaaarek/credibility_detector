from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ml.features.feature_builder import MODEL_FEATURE_COLUMNS, build_model_features


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
    joblib.dump(
        {
            "model": model,
            "features": MODEL_FEATURE_COLUMNS,
            "mae": mae,
            "sklearn_version": sklearn.__version__,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")
    print(f"Validation MAE: {mae:.3f}")


def _row_features(row: pd.Series) -> dict[str, float]:
    links = [] if pd.isna(row.get("source_links")) else [link for link in str(row.get("source_links", "")).split("|") if link]
    return build_model_features(
        content=str(row["content"]),
        url=None if pd.isna(row.get("url")) else str(row.get("url", "")),
        author=None if pd.isna(row.get("author")) else str(row.get("author", "")),
        publish_date=None if pd.isna(row.get("publish_date")) else str(row.get("publish_date", "")),
        source_links=links,
    )


if __name__ == "__main__":
    main()
