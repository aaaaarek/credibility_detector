from __future__ import annotations

from ml.features.source_features import extract_source_features
from ml.features.text_features import extract_text_features


MODEL_FEATURE_COLUMNS = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "uppercase_ratio",
    "exclamation_count",
    "question_count",
    "url_count",
    "number_count",
    "date_count",
    "quote_count",
    "emotional_word_count",
    "clickbait_phrase_count",
    "hedging_word_count",
    "source_word_count",
    "manipulation_word_count",
    "conspiracy_word_count",
    "urgency_word_count",
    "authority_signal_count",
    "has_url",
    "uses_https",
    "known_reputable_domain",
    "suspicious_domain_hint",
    "has_author",
    "has_publish_date",
    "source_link_count",
    "unique_source_domain_count",
    "reputable_source_link_count",
    "relevant_source_link_count",
    "relevant_unique_source_domain_count",
    "relevant_reputable_source_link_count",
    "unrelated_source_link_count",
]


def build_model_features(
    content: str,
    url: str | None = None,
    author: str | None = None,
    publish_date: str | None = None,
    source_links: list[str] | None = None,
) -> dict[str, float]:
    text = extract_text_features(content).as_dict()
    source = extract_source_features(
        url=url,
        author=author,
        publish_date=publish_date,
        source_links=source_links,
        content=content,
    ).as_dict()
    features = {
        **text,
        "has_url": float(source["has_url"]),
        "uses_https": float(source["uses_https"]),
        "known_reputable_domain": float(source["known_reputable_domain"]),
        "suspicious_domain_hint": float(source["suspicious_domain_hint"]),
        "has_author": float(source["has_author"]),
        "has_publish_date": float(source["has_publish_date"]),
        "source_link_count": float(source["source_link_count"]),
        "unique_source_domain_count": float(source["unique_source_domain_count"]),
        "reputable_source_link_count": float(source["reputable_source_link_count"]),
        "relevant_source_link_count": float(source["relevant_source_link_count"]),
        "relevant_unique_source_domain_count": float(source["relevant_unique_source_domain_count"]),
        "relevant_reputable_source_link_count": float(source["relevant_reputable_source_link_count"]),
        "unrelated_source_link_count": float(source["unrelated_source_link_count"]),
    }
    return {column: float(features.get(column, 0.0)) for column in MODEL_FEATURE_COLUMNS}
