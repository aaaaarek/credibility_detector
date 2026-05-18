from ml.inference.pipeline import ArticleInput, analyze_article


def test_reputable_sourced_article_scores_higher_than_clickbait() -> None:
    reputable = analyze_article(
        ArticleInput(
            title="Agency report",
            content=(
                "The public health agency published a report on 2026-02-12. "
                "The report includes data from 120 hospitals, laboratory results and comments from named experts. "
                "It links to the official dataset and explains the method used by researchers."
            ),
            url="https://gov.pl/report",
            author="Agency desk",
            publish_date="2026-02-12",
            source_links=["https://gov.pl/dataset", "https://gov.pl/report"],
        )
    )
    clickbait = analyze_article(
        ArticleInput(
            title="Secret cure",
            content=(
                "SHOCKING secret cure discovered! Doctors hate this simple trick and they do not want you to know. "
                "Share now before it is deleted!!! No report, no named author and no source is provided."
            ),
            url="http://secret-cure.example.com/post",
        )
    )

    assert reputable.credibility_score > clickbait.credibility_score
    assert reputable.credibility_score >= 0.65
    assert clickbait.credibility_score <= 0.45


def test_short_text_receives_explainability_reason() -> None:
    result = analyze_article(
        ArticleInput(
            title="Short note",
            content="A short post says something happened yesterday without sources or details.",
        )
    )

    assert result.reasons
    assert any("krotki" in reason.lower() or "brak" in reason.lower() for reason in result.reasons)
