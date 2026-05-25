from __future__ import annotations

from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ml.features.feature_builder import MODEL_FEATURE_COLUMNS
from ml.training.datasets import build_feature_frame, load_training_dataset


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "ml" / "models" / "credibility_regressor.joblib"


def main() -> None:
    dataset = load_training_dataset()
    train, test = train_test_split(dataset, test_size=0.25, random_state=42)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(
        build_feature_frame(train),
        train["credibility_label"],
        sample_weight=train["sample_weight"],
    )

    predictions = model.predict(build_feature_frame(test))
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
            "features": MODEL_FEATURE_COLUMNS,
            "mae": mae,
            "weighted_mae": weighted_mae,
            "training_rows": len(train),
            "training_sources": train["dataset_source"].value_counts().to_dict(),
            "sklearn_version": sklearn.__version__,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")
    print(f"Validation MAE: {mae:.3f}")
    print(f"Weighted validation MAE: {weighted_mae:.3f}")
    print(f"Training sources: {train['dataset_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
