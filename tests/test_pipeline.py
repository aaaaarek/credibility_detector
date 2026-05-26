import pytest

from ml.features.profile_features import ProfileInput
from ml.inference.pipeline import ArticleInput, _disagreement_penalty, _level, _spread_weighted_score, analyze_article


def test_reputable_sourced_article_scores_higher_than_clickbait() -> None:
    reputable = analyze_article(
        ArticleInput(
            title="Agency report",
            input_type="url",
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
            input_type="url",
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


def test_raw_text_metadata_has_limited_influence_when_links_are_unrelated() -> None:
    content = (
        "The city office report says water tests from 24 sampling points were within legal limits. "
        "Laboratory data, named experts and the public registry explain the method and the result."
    )
    baseline = analyze_article(ArticleInput(title="Water report", content=content, input_type="raw_text"))
    with_metadata = analyze_article(
        ArticleInput(
            title="Water report",
            content=content,
            input_type="raw_text",
            author="City office",
            publish_date="2026-05-20",
            source_links=["https://nature.com/astronomy"],
        )
    )

    assert with_metadata.credibility_score <= baseline.credibility_score + 0.02
    assert with_metadata.metadata["source_features"]["relevant_source_link_count"] == 0
    assert with_metadata.metadata["source_features"]["unrelated_source_link_count"] == 1


def test_raw_text_source_url_can_inform_ml_score_when_provided() -> None:
    content = (
        "The city office report says water tests from 24 sampling points were within legal limits. "
        "Laboratory data, named experts and the public registry explain the method and the result."
    )
    baseline = analyze_article(ArticleInput(title="Water report", content=content, input_type="raw_text"))
    with_source_url = analyze_article(
        ArticleInput(
            title="Water report",
            content=content,
            input_type="raw_text",
            url="https://gov.pl/water-report",
            author="City office",
            publish_date="2026-05-20",
            source_links=["https://stat.gov.pl/water-data"],
        )
    )

    assert with_source_url.module_scores["ml_score"] > baseline.module_scores["ml_score"]
    assert with_source_url.credibility_score > baseline.credibility_score
    assert with_source_url.metadata["source_features"]["has_url"] is True


def test_raw_text_uses_only_relevant_source_links_for_consensus() -> None:
    content = (
        "The city office report says water tests from 24 sampling points were within legal limits. "
        "Laboratory water data and named experts explain the public registry result."
    )
    unrelated = analyze_article(
        ArticleInput(
            title="Water report",
            content=content,
            input_type="raw_text",
            source_links=["https://nature.com/astronomy"],
        )
    )
    relevant = analyze_article(
        ArticleInput(
            title="Water report",
            content=content,
            input_type="raw_text",
            source_links=["https://stat.gov.pl/water-data"],
        )
    )

    assert relevant.metadata["source_features"]["relevant_source_link_count"] == 1
    assert relevant.module_scores["consensus_score"] > unrelated.module_scores["consensus_score"]


def test_url_consensus_ignores_unrelated_links() -> None:
    result = analyze_article(
        ArticleInput(
            title="Water report",
            input_type="url",
            content=(
                "The city office report says water tests from 24 sampling points were within legal limits. "
                "Laboratory water data and named experts explain the public registry result."
            ),
            url="https://example.com/water-report",
            author="City desk",
            publish_date="2026-05-20",
            source_links=[
                "https://nature.com/astronomy",
                "https://science.org/space",
                "https://reuters.com/world/elections",
                "https://who.int/vaccines",
            ],
        )
    )

    assert result.metadata["source_features"]["relevant_source_link_count"] == 0
    assert result.module_scores["consensus_score"] <= 0.34


def test_fact_score_is_capped_for_high_risk_claims_without_strong_evidence() -> None:
    result = analyze_article(
        ArticleInput(
            title="Cudowny lek",
            input_type="raw_text",
            content=(
                "Raport twierdzi, ze cudowny preparat leczy wszystkie nowotwory w 24 godziny. "
                "Autor podaje 97 procent skutecznosci, 120 pacjentow i 3 kliniki, ale nie pokazuje "
                "recenzowanego badania ani publicznego zbioru danych."
            ),
            source_links=["https://nature.com/astronomy"],
        )
    )

    assert result.metadata["content_quality"]["high_risk_claim_count"] >= 1
    assert result.diagnostic_scores["fact_score"] <= 0.45


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

    assert 0 <= result.module_scores["ml_score"] <= 1
    assert 0 <= result.module_scores["text_ml_score"] <= 1
    assert 0 <= result.module_scores["claim_score"] <= 1
    assert result.metadata["score_calibration"]["spread_factor"] == 1.30
    assert "profile_score" not in result.diagnostic_scores
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


def test_text_ml_score_distinguishes_reliable_and_clickbait_language() -> None:
    reliable = analyze_article(
        ArticleInput(
            title="Court report",
            content=(
                "The court published a judgement with a case number and public register link. "
                "The article summarises the decision, quotes named officials and separates confirmed facts "
                "from reactions by the parties."
            ),
            url="https://reuters.com/world/court-report",
            author="Reuters staff",
            publish_date="2026-01-31",
            source_links=["https://court.example.gov/judgement"],
        )
    )
    clickbait = analyze_article(
        ArticleInput(
            title="Hidden secret",
            content=(
                "PILNE! Ukrywaja prawde i lekarze nie chca zebys wiedzial. "
                "Udostepnij teraz zanim usuna ten sekret. Gwarantowany cud bez raportu i bez danych."
            ),
            url="http://viral-secret.example.com/post",
        )
    )

    assert reliable.module_scores["text_ml_score"] > clickbait.module_scores["text_ml_score"]


def test_profile_context_affects_social_post_analysis() -> None:
    verified_profile = analyze_article(
        ArticleInput(
            title="Screenshot",
            input_type="screenshot",
            content=(
                "@cityoffice opublikował raport o jakości wody. Według danych laboratoryjnych "
                "wyniki z 24 punktów pomiarowych mieszczą się w normach."
            ),
            profile=ProfileInput(
                profile_name="@cityoffice",
                profile_url="https://x.com/cityoffice",
                is_verified=True,
                follower_count=120000,
                account_age_days=1800,
            ),
        )
    )
    suspicious_profile = analyze_article(
        ArticleInput(
            title="Screenshot",
            input_type="screenshot",
            content=(
                "@prawda_secret twierdzi, że władze ukrywają skażenie wody. "
                "Nie pokazuje raportów ani danych i prosi o natychmiastowe udostępnianie."
            ),
            profile=ProfileInput(
                profile_name="@prawda_secret",
                profile_url="https://x.com/prawda_secret",
                is_verified=False,
                follower_count=40,
                account_age_days=7,
            ),
        )
    )

    assert verified_profile.module_scores["profile_score"] > suspicious_profile.module_scores["profile_score"]
    assert verified_profile.metadata["profile_features"]["known_platform"] is True


def test_url_and_screenshot_use_different_active_modules() -> None:
    url_result = analyze_article(
        ArticleInput(
            title="Report",
            input_type="url",
            content="The agency published a report with data, named experts and source documents in 2026.",
            url="https://gov.pl/report",
            author="Agency desk",
            publish_date="2026-02-12",
            source_links=["https://gov.pl/data"],
        )
    )
    screenshot_result = analyze_article(
        ArticleInput(
            title="Screenshot",
            input_type="screenshot",
            content="@cityoffice published a report with data, named experts and source documents in 2026.",
            profile=ProfileInput(profile_name="@cityoffice", profile_url="https://x.com/cityoffice", is_verified=True),
        )
    )

    assert "source_score" in url_result.module_scores
    assert "profile_score" in screenshot_result.module_scores
    assert "profile_score" not in url_result.diagnostic_scores
    assert "source_score" in screenshot_result.diagnostic_scores


def test_strong_sourced_scientific_article_gets_positive_floor() -> None:
    result = analyze_article(
        ArticleInput(
            title="Researchers publish clinical review",
            input_type="url",
            content=(
                "Researchers from a university published a peer reviewed report on 2026-04-12. "
                "According to the study, data from 120 hospitals and 4800 patients showed a 14 percent reduction "
                "in complications. The article describes the methodology, cites named experts, links to the journal "
                "paper, dataset, public registry and ministry report, and explains limitations of the research."
            ),
            url="https://nature.com/articles/clinical-review",
            author="Science desk",
            publish_date="2026-04-12",
            source_links=[
                "https://nature.com/paper",
                "https://data.example.org/study",
                "https://gov.pl/report",
                "https://clinicaltrials.example.org/trial",
            ],
        )
    )

    assert result.credibility_score >= 0.75
    assert "profile_score" not in result.diagnostic_scores


def test_screenshot_with_unverified_handle_only_is_capped_lower() -> None:
    result = analyze_article(
        ArticleInput(
            title="Screenshot",
            input_type="screenshot",
            content=(
                "@prawda_secret says doctors hide the truth and asks everyone to share before it is deleted."
            ),
        )
    )

    assert result.credibility_score <= 0.38
    assert result.module_scores["profile_score"] <= 0.35


def test_repeated_character_gibberish_scores_very_low() -> None:
    result = analyze_article(
        ArticleInput(
            title=None,
            input_type="raw_text",
            content="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    )

    assert result.credibility_score <= 0.10
    assert "repeated_characters" in result.metadata["content_quality"]["flags"]


def test_random_low_quality_text_scores_low() -> None:
    result = analyze_article(
        ArticleInput(
            title=None,
            input_type="raw_text",
            content="asdf qwer zxcv asdf qwer zxcv asdf qwer zxcv",
        )
    )

    assert result.credibility_score <= 0.25
    assert result.metadata["content_quality"]["quality_score"] < 0.40


def test_repeated_lines_score_very_low() -> None:
    result = analyze_article(
        ArticleInput(
            title=None,
            input_type="raw_text",
            content=(
                "zamawiam modelke i wciskam enter\n"
                "zamawiam modelke i wciskam enter\n"
                "zamawiam modelke i wciskam enter\n"
                "zamawiam modelke i wciskam enter\n"
                "zamawiam modelke i wciskam enter"
            ),
        )
    )

    assert result.credibility_score <= 0.10
    assert "repeated_lines" in result.metadata["content_quality"]["flags"]


def test_credibility_level_boundaries_are_strict() -> None:
    assert _level(0.75) == "wysoka wiarygodnosc / silnie uzrodlowiony material"
    assert _level(0.65) == "normalny artykul z autorem, data i weryfikowalnymi zrodlami"
    assert _level(0.50) == "opinia, blog lub czesciowy kontekst z pewna wartoscia weryfikacyjna"
    assert _level(0.35) == "jednostronny, slabo uzrodlowiony lub clickbaitowy tekst"
    assert _level(0.34) == "wysokie ryzyko: scam, spisek, fabrykacja lub dezinformacja"


def test_weighted_score_is_spread_around_midpoint() -> None:
    assert _spread_weighted_score(0.50) == 0.50
    assert _spread_weighted_score(0.60) == pytest.approx(0.63)
    assert _spread_weighted_score(0.40) == pytest.approx(0.37)


def test_disagreement_penalty_only_applies_to_divergent_module_scores() -> None:
    weights = {"source_score": 1.0, "claim_score": 1.0, "text_ml_score": 1.0}

    aligned_penalty = _disagreement_penalty(
        {"source_score": 0.70, "claim_score": 0.72, "text_ml_score": 0.68},
        weights,
    )
    divergent_penalty = _disagreement_penalty(
        {"source_score": 0.90, "claim_score": 0.90, "text_ml_score": 0.30},
        weights,
    )

    assert aligned_penalty == 0.0
    assert divergent_penalty == pytest.approx(0.064, abs=0.001)


def test_high_risk_claim_without_evidence_is_capped() -> None:
    result = analyze_article(
        ArticleInput(
            title="Cudowny lek",
            input_type="raw_text",
            content=(
                "Cudowny preparat leczy wszystkie nowotwory w 24 godziny. "
                "Lekarze ukrywają prawdę i nie chcą, żeby pacjenci poznali ten sekret. "
                "Kup teraz, zanim informacja zostanie usunięta."
            ),
        )
    )

    assert result.credibility_score <= 0.20
    assert result.metadata["content_quality"]["high_risk_claim_count"] >= 1


def test_reliable_article_is_not_capped_by_quality_gate() -> None:
    result = analyze_article(
        ArticleInput(
            title="Agency report",
            input_type="url",
            content=(
                "The public health agency published a report on 2026-02-12. "
                "The report includes data from 120 hospitals, laboratory results and comments from named experts. "
                "It links to the official dataset and explains the method used by researchers."
            ),
            url="https://gov.pl/report",
            author="Agency desk",
            publish_date="2026-02-12",
            source_links=["https://gov.pl/dataset", "https://stat.gov.pl/report"],
        )
    )

    assert result.metadata["content_quality"]["quality_score"] >= 0.70
    assert result.credibility_score >= 0.65


def test_scientific_pdf_like_document_scores_high() -> None:
    content = (
        "ARTICLE IN PRESS Deep learning and radiomics models in patients with advanced non-small cell lung cancer "
        "treated with immunotherapy combined with stereotactic radiotherapy. Scientific Reports. "
        "Received: 22 September 2025 Accepted: 12 May 2026. "
        "Kothari G., Hardcastle N., Perera R. et al. "
        "https://doi.org/10.1038/s41598-026-53520-5 "
        "Abstract: This study analysed clinical data from 120 patients and radiomics features from imaging. "
        "Methods: researchers trained deep learning models and compared validation performance. "
        "Results: the model reported 0.81 AUC and confidence intervals across cross validation folds. "
        "Discussion: the article explains limitations, sample size and external validation requirements. "
        "References: Smith et al. 2024. Jones et al. 2025. Scientific Reports open access article."
    )

    result = analyze_article(
        ArticleInput(
            title="s41598-026-53520-5_reference.pdf",
            input_type="document",
            content=content,
        )
    )

    assert result.credibility_score >= 0.76
    assert result.metadata["document_features"]["has_doi"] is True
    assert result.metadata["document_features"]["scientific_document_score"] >= 0.70
    assert "fact_score" in result.module_scores
