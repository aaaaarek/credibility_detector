from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.features.feature_builder import MODEL_FEATURE_COLUMNS, build_model_features


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DATASET_PATH = ROOT / "data" / "datasets" / "synthetic_articles.csv"
REAL_DATASET_PATH = ROOT / "data" / "datasets" / "real_articles_seed.csv"

REQUIRED_COLUMNS = [
    "title",
    "content",
    "url",
    "source",
    "author",
    "publish_date",
    "source_links",
    "credibility_label",
]

OPTIONAL_COLUMNS = [
    "label_reason",
    "dataset_source",
    "sample_weight",
]

SYNTHETIC_SAMPLE_WEIGHT = 0.25
REAL_SAMPLE_WEIGHT = 1.0


def load_training_dataset(
    synthetic_path: Path = SYNTHETIC_DATASET_PATH,
    real_path: Path = REAL_DATASET_PATH,
) -> pd.DataFrame:
    frames = [
        _prepare_dataset(
            pd.read_csv(synthetic_path),
            dataset_source="synthetic",
            sample_weight=SYNTHETIC_SAMPLE_WEIGHT,
            path=synthetic_path,
        )
    ]

    if real_path.exists():
        real = pd.read_csv(real_path)
        if not real.empty:
            frames.append(
                _prepare_dataset(
                    real,
                    dataset_source="real_seed",
                    sample_weight=REAL_SAMPLE_WEIGHT,
                    path=real_path,
                )
            )

    dataset = pd.concat(frames, ignore_index=True)
    return dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)


def validate_real_dataset(path: Path = REAL_DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    dataset = pd.read_csv(path)
    if dataset.empty:
        _ensure_required_columns(dataset, path)
        return dataset
    return _prepare_dataset(
        dataset,
        dataset_source="real_seed",
        sample_weight=REAL_SAMPLE_WEIGHT,
        path=path,
    )


def build_feature_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([row_features(row) for _, row in dataset.iterrows()])[MODEL_FEATURE_COLUMNS]


def row_features(row: pd.Series) -> dict[str, float]:
    return build_model_features(
        content=str(row["content"]),
        url=_optional_text(row.get("url")),
        author=_optional_text(row.get("author")),
        publish_date=_optional_text(row.get("publish_date")),
        source_links=_parse_links(row.get("source_links")),
    )


def _prepare_dataset(dataset: pd.DataFrame, dataset_source: str, sample_weight: float, path: Path) -> pd.DataFrame:
    _ensure_required_columns(dataset, path)
    prepared = dataset.copy()
    for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = ""

    prepared["credibility_label"] = pd.to_numeric(prepared["credibility_label"], errors="raise")
    invalid_labels = prepared[
        (prepared["credibility_label"] < 0.0) | (prepared["credibility_label"] > 1.0)
    ]
    if not invalid_labels.empty:
        raise ValueError(f"{path} contains credibility_label values outside 0.0-1.0.")

    if prepared["content"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{path} contains rows with empty content.")

    prepared["dataset_source"] = prepared["dataset_source"].replace("", pd.NA).fillna(dataset_source)
    if "sample_weight" not in dataset.columns or prepared["sample_weight"].astype(str).str.strip().eq("").all():
        prepared["sample_weight"] = sample_weight
    prepared["sample_weight"] = pd.to_numeric(prepared["sample_weight"], errors="raise")
    if (prepared["sample_weight"] <= 0).any():
        raise ValueError(f"{path} contains non-positive sample_weight values.")

    return prepared[REQUIRED_COLUMNS + OPTIONAL_COLUMNS]


def _ensure_required_columns(dataset: pd.DataFrame, path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _optional_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_links(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [link.strip() for link in str(value).split("|") if link.strip()]
