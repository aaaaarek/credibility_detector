from pathlib import Path

from app.services.article_fetcher import FetchedArticle
from ml.training.collect_real_dataset_candidates import _read_urls, candidate_row, suggest_label


def test_read_urls_skips_comments_blanks_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "urls.txt"
    path.write_text(
        "\n"
        "# comment\n"
        "https://example.com/a\n"
        "https://example.com/a\n"
        "https://example.com/b\n",
        encoding="utf-8",
    )

    assert _read_urls(path) == ["https://example.com/a", "https://example.com/b"]


def test_candidate_row_marks_auto_label_for_review() -> None:
    article = FetchedArticle(
        title="Agency report",
        content=(
            "The public agency report gives data from 120 hospitals. "
            "Experts explain the method and the dataset is available for verification."
        ),
        url="https://gov.pl/report",
        author="Agency desk",
        publish_date="2026-05-20",
        source_links=["https://stat.gov.pl/data"],
    )

    row = candidate_row(article)

    assert row["source"] == "gov.pl"
    assert row["needs_review"] == "true"
    assert str(row["label_reason"]).startswith("AUTO-SUGGESTION:")
    assert 0.0 <= float(row["credibility_label"]) <= 1.0


def test_suggest_label_scores_reputable_article_above_suspicious_article() -> None:
    reputable = FetchedArticle(
        title="Agency report",
        content=(
            "The public agency report gives data from 120 hospitals. "
            "Experts explain the method and the dataset is available for verification."
        ),
        url="https://gov.pl/report",
        author="Agency desk",
        publish_date="2026-05-20",
        source_links=["https://stat.gov.pl/data"],
    )
    suspicious = FetchedArticle(
        title="Secret cure",
        content=(
            "SHOCKING secret cure discovered! Doctors hate this trick. "
            "Share now before it is deleted and hidden from everyone!"
        ),
        url="http://prawda-secret.example.com/post",
        author=None,
        publish_date=None,
        source_links=[],
    )

    assert suggest_label(reputable).value > suggest_label(suspicious).value
