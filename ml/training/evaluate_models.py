from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ml.training.datasets import build_feature_frame, load_training_dataset
from ml.inference.text_model import build_text_model


def main() -> None:
    dataset = load_training_dataset()
    train, test = train_test_split(dataset, test_size=0.25, random_state=42)

    feature_model = GradientBoostingRegressor(random_state=42)
    feature_model.fit(
        build_feature_frame(train),
        train["credibility_label"],
        sample_weight=train["sample_weight"],
    )
    feature_predictions = feature_model.predict(build_feature_frame(test))

    text_model = build_text_model()
    text_model.fit(
        train["content"].astype(str),
        train["credibility_label"],
        regressor__sample_weight=train["sample_weight"],
    )
    text_predictions = text_model.predict(test["content"].astype(str))

    ensemble_predictions = 0.55 * feature_predictions + 0.45 * text_predictions

    print("Model evaluation on mixed holdout")
    print(f"dataset_sources: {dataset['dataset_source'].value_counts().to_dict()}")
    print(f"feature_model_mae: {mean_absolute_error(test['credibility_label'], feature_predictions):.3f}")
    print(
        "feature_model_weighted_mae: "
        f"{mean_absolute_error(test['credibility_label'], feature_predictions, sample_weight=test['sample_weight']):.3f}"
    )
    print(f"text_model_mae: {mean_absolute_error(test['credibility_label'], text_predictions):.3f}")
    print(
        "text_model_weighted_mae: "
        f"{mean_absolute_error(test['credibility_label'], text_predictions, sample_weight=test['sample_weight']):.3f}"
    )
    print(f"simple_ensemble_mae: {mean_absolute_error(test['credibility_label'], ensemble_predictions):.3f}")
    print(
        "simple_ensemble_weighted_mae: "
        f"{mean_absolute_error(test['credibility_label'], ensemble_predictions, sample_weight=test['sample_weight']):.3f}"
    )


if __name__ == "__main__":
    main()
