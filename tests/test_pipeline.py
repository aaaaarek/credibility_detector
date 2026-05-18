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


def test_pipeline_returns_ml_and_claim_scores() -> None:
    result = analyze_article(
        ArticleInput(
            title="Ministerstwo publikuje dane",
            content=(
                "Ministerstwo opublikowalo raport z danymi za 2026 rok. "
                "Wedlug raportu wskaznik wyniosl 4,8 procent i byl nizszy niz miesiac wczesniej. "
                "Artykul cytuje ekspertow, opisuje metode i linkuje do danych publicznych."
            ),
            url="https://gov.pl/report",
            author="Departament analiz",
            publish_date="2026-04-15",
            source_links=["https://gov.pl/dane", "https://stat.gov.pl/dane"],
        )
    )

    assert 0 <= result.module_scores.ml_score <= 1
    assert 0 <= result.module_scores.claim_score <= 1
    assert result.metadata["extracted_claims"]["claims"]


def test_polish_conspiracy_language_is_penalized() -> None:
    result = analyze_article(
        ArticleInput(
            title="Pilne",
            content=(
                "PILNE! Oni ukrywaja prawde i nikt nie chce o tym mowic. "
                "Tekst twierdzi, ze lekarze ukrywaja dowody, ale nie pokazuje badan ani raportow. "
                "Udostepnij teraz zanim zostanie usuniete!"
            ),
            url="http://prawda-secret.example.com/post",
        )
    )

    assert result.credibility_score < 0.45
    assert any("spisk" in reason.lower() or "pilnosci" in reason.lower() for reason in result.reasons)
