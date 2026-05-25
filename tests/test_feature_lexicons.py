from ml.features.content_quality import analyze_content_quality
from ml.features.document_features import extract_document_features
from ml.features.profile_features import ProfileInput, extract_profile_features
from ml.features.source_features import extract_source_features
from ml.features.text_features import extract_text_features
from ml.inference.claims import analyze_claims


def test_reputable_domain_matching_includes_subdomains_and_trusted_suffixes() -> None:
    features = extract_source_features(
        url="https://report.nih.gov/article",
        source_links=[
            "https://data.cdc.gov/dataset",
            "https://research.example.edu/paper",
        ],
    )

    assert features.known_reputable_domain is True
    assert features.reputable_source_link_count == 2


def test_suspicious_domain_hints_are_token_based() -> None:
    neutral = extract_source_features(url="https://snow.example.com/story")
    suspicious = extract_source_features(url="https://prawda-secret.example.com/story")

    assert neutral.suspicious_domain_hint is False
    assert suspicious.suspicious_domain_hint is True


def test_expanded_text_lexicons_detect_language_signals() -> None:
    features = extract_text_features(
        "Mainstream media won't tell you this alarming story. "
        "Professor Smith said the official registry contains statistics. "
        "Share now, this is your last chance."
    )

    assert features.clickbait_phrase_count >= 1
    assert features.emotional_word_count >= 1
    assert features.authority_signal_count >= 1
    assert features.source_word_count >= 2
    assert features.urgency_word_count >= 2


def test_expanded_document_profile_claim_and_quality_markers() -> None:
    document = extract_document_features(
        "JAMA research article. Background. Methodology. Results. "
        "Conflict of interest. Data availability. ORCID 0000-0000."
    )
    profile = extract_profile_features(
        ProfileInput(profile_name="@breaking_news24", profile_url="https://www.threads.net/@agency")
    )
    claims = analyze_claims("The report warns that statistics show a 12 percent increase.")
    quality = analyze_content_quality("The election was stolen by fraud, but the post shows no evidence.")

    assert document.has_journal_marker is True
    assert document.has_authors_marker is True
    assert profile.known_platform is True
    assert profile.suspicious_handle_hint is True
    assert claims.evidence_marker_count >= 1
    assert "election_fraud" in quality.high_risk_claim_types
