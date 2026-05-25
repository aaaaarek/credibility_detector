from __future__ import annotations

from pathlib import Path

from ml.training.datasets import REAL_DATASET_PATH, validate_real_dataset


def main() -> None:
    dataset = validate_real_dataset(REAL_DATASET_PATH)
    print(f"Validated dataset: {Path(REAL_DATASET_PATH)}")
    print(f"Rows: {len(dataset)}")
    if not dataset.empty:
        print(f"Label range: {dataset['credibility_label'].min():.2f}-{dataset['credibility_label'].max():.2f}")
        print(f"Sources: {dataset['dataset_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
