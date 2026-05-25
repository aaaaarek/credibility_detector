from __future__ import annotations

from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ml.inference.text_model import TEXT_MODEL_VERSION, build_text_model
from ml.training.datasets import load_training_dataset


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "ml" / "models" / "text_credibility_regressor.joblib"


def main() -> None:
    dataset = load_training_dataset()
    train, test = train_test_split(dataset, test_size=0.25, random_state=42)

    model = build_text_model()
    model.fit(
        train["content"].astype(str),
        train["credibility_label"],
        regressor__sample_weight=train["sample_weight"],
    )

    predictions = model.predict(test["content"].astype(str))
    mae = mean_absolute_error(test["credibility_label"], predictions)
    weighted_mae = mean_absolute_error(
        test["credibility_label"],
        predictions,
        sample_weight=test["sample_weight"],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "mae": mae,
            "weighted_mae": weighted_mae,
            "training_rows": len(train),
            "training_sources": train["dataset_source"].value_counts().to_dict(),
            "model_version": TEXT_MODEL_VERSION,
            "sklearn_version": sklearn.__version__,
        },
        MODEL_PATH,
    )
    print(f"Saved text model to {MODEL_PATH}")
    print(f"Validation MAE: {mae:.3f}")
    print(f"Weighted validation MAE: {weighted_mae:.3f}")
    print(f"Training sources: {train['dataset_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
