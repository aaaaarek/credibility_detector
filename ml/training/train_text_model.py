from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ml.inference.text_model import TEXT_MODEL_VERSION, build_text_model


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "datasets" / "synthetic_articles.csv"
MODEL_PATH = ROOT / "ml" / "models" / "text_credibility_regressor.joblib"


def main() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset["content"].astype(str),
        dataset["credibility_label"],
        test_size=0.25,
        random_state=42,
    )

    model = build_text_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "mae": mae,
            "model_version": TEXT_MODEL_VERSION,
            "sklearn_version": sklearn.__version__,
        },
        MODEL_PATH,
    )
    print(f"Saved text model to {MODEL_PATH}")
    print(f"Validation MAE: {mae:.3f}")


if __name__ == "__main__":
    main()
