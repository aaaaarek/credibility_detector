from pathlib import Path

import pandas as pd
import pytest

from ml.features.feature_builder import MODEL_FEATURE_COLUMNS
from ml.training.datasets import (
    REAL_SAMPLE_WEIGHT,
    SYNTHETIC_SAMPLE_WEIGHT,
    build_feature_frame,
    load_training_dataset,
    validate_real_dataset,
)


def test_load_training_dataset_mixes_synthetic_and_real_rows(tmp_path: Path) -> None:
    synthetic_path = tmp_path / "synthetic.csv"
    real_path = tmp_path / "real.csv"
    _write_dataset(
        synthetic_path,
        [
            {
                "title": "Synthetic",
                "content": "Agency report with data and experts from a public registry.",
                "url": "https://gov.pl/report",
                "source": "synthetic",
                "author": "Desk",
                "publish_date": "2026-01-01",
                "source_links": "https://stat.gov.pl/data",
                "credibility_label": 0.80,
            }
        ],
    )
    _write_dataset(
        real_path,
        [
            {
                "title": "Real",
                "content": "Local report gives named sources and public data for verification.",
                "url": "https://example.com/report",
                "source": "real",
                "author": "Reporter",
                "publish_date": "2026-02-02",
                "source_links": "https://example.com/data",
                "credibility_label": 0.70,
                "label_reason": "Sourced local report.",
                "dataset_source": "manual_seed",
            }
        ],
    )

    dataset = load_training_dataset(synthetic_path=synthetic_path, real_path=real_path)

    assert len(dataset) == 2
    assert set(dataset["sample_weight"]) == {SYNTHETIC_SAMPLE_WEIGHT, REAL_SAMPLE_WEIGHT}
    assert "manual_seed" in set(dataset["dataset_source"])
    assert list(build_feature_frame(dataset).columns) == MODEL_FEATURE_COLUMNS


def test_validate_real_dataset_allows_header_only_seed_file(tmp_path: Path) -> None:
    real_path = tmp_path / "real.csv"
    real_path.write_text(
        "title,content,url,source,author,publish_date,source_links,credibility_label,label_reason,dataset_source\n",
        encoding="utf-8",
    )

    dataset = validate_real_dataset(real_path)

    assert dataset.empty


def test_validate_real_dataset_rejects_label_outside_range(tmp_path: Path) -> None:
    real_path = tmp_path / "real.csv"
    _write_dataset(
        real_path,
        [
            {
                "title": "Bad label",
                "content": "This row has enough text but an invalid numeric label.",
                "url": "https://example.com",
                "source": "manual",
                "author": "Reporter",
                "publish_date": "2026-03-03",
                "source_links": "",
                "credibility_label": 1.30,
            }
        ],
    )

    with pytest.raises(ValueError, match="outside 0.0-1.0"):
        validate_real_dataset(real_path)


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
