from __future__ import annotations

import re
from dataclasses import asdict, dataclass


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "i",
    "oraz",
    "ale",
    "bo",
    "czy",
    "dla",
    "do",
    "jest",
    "na",
    "nie",
    "o",
    "od",
    "po",
    "przez",
    "się",
    "to",
    "w",
    "z",
    "że",
}

HIGH_RISK_PATTERNS = {
    "miracle_cure": [
        r"\b(cud|cudown\w+|miracle)\b",
        r"\b(leczy|wyleczy|cure|cures)\b",
    ],
    "guaranteed_profit": [
        r"\b(gwarantowan\w+|guaranteed)\b",
        r"\b(zysk|profit|return)\b",
    ],
    "hidden_truth": [
        r"\b(ukrywaj\w+|hide|hidden|secret)\b",
        r"\b(prawd\w+|truth|dowod\w+|evidence)\b",
    ],
    "anti_medical_conspiracy": [
        r"\b(lekarz\w+|doctor\w+|medic\w+)\b",
        r"\b(ukrywaj\w+|hide|hate|nienawidz\w+)\b",
    ],
    "absolute_health_claim": [
        r"\b(leczy|wyleczy|cures?|cure)\b",
        r"\b(wszystkie|every|all)\b",
        r"\b(nowotwor\w+|cancer|chorob\w+|disease\w+)\b",
    ],
}


@dataclass(frozen=True)
class ContentQuality:
    quality_score: float
    flags: list[str]
    char_count: int
    word_count: int
    sentence_count: int
    unique_word_ratio: float
    max_char_run_ratio: float
    alphabetic_ratio: float
    punctuation_ratio: float
    average_token_length: float
    repeated_token_ratio: float
    repeated_line_ratio: float
    repeated_ngram_ratio: float
    stopword_ratio: float
    high_risk_claim_count: int
    high_risk_claim_types: list[str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_content_quality(text: str) -> ContentQuality:
    normalized = _normalize(text)
    tokens = re.findall(r"\b[\w'-]+\b", normalized)
    lines = [_normalize(line.strip()) for line in text.splitlines() if line.strip()]
    sentences = [sentence for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
    alpha_count = sum(1 for char in text if char.isalpha())
    punctuation_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    unique_tokens = set(tokens)
    repeated_tokens = len(tokens) - len(unique_tokens)
    high_risk_types = _high_risk_claim_types(normalized)

    metrics = {
        "char_count": len(text),
        "word_count": len(tokens),
        "sentence_count": len(sentences),
        "unique_word_ratio": len(unique_tokens) / max(len(tokens), 1),
        "max_char_run_ratio": _max_char_run(text) / max(len(text), 1),
        "alphabetic_ratio": alpha_count / max(len(text), 1),
        "punctuation_ratio": punctuation_count / max(len(text), 1),
        "average_token_length": sum(len(token) for token in tokens) / max(len(tokens), 1),
        "repeated_token_ratio": repeated_tokens / max(len(tokens), 1),
        "repeated_line_ratio": _repeated_item_ratio(lines),
        "repeated_ngram_ratio": _repeated_ngram_ratio(tokens, size=4),
        "stopword_ratio": sum(1 for token in tokens if token in STOPWORDS) / max(len(tokens), 1),
    }
    score, flags = _score_quality(metrics, high_risk_types)
    return ContentQuality(
        quality_score=round(score, 3),
        flags=flags,
        high_risk_claim_count=len(high_risk_types),
        high_risk_claim_types=high_risk_types,
        **metrics,
    )


def _score_quality(metrics: dict[str, float], high_risk_types: list[str]) -> tuple[float, list[str]]:
    score = 1.0
    flags: list[str] = []

    if metrics["word_count"] < 3:
        score -= 0.70
        flags.append("too_few_words")
    elif metrics["word_count"] < 12:
        score -= 0.35
        flags.append("short_text")
    if metrics["sentence_count"] == 0:
        score -= 0.20
        flags.append("no_sentence_structure")
    if metrics["max_char_run_ratio"] > 0.45:
        score -= 0.75
        flags.append("repeated_characters")
    if metrics["average_token_length"] > 24:
        score -= 0.45
        flags.append("abnormally_long_token")
    if metrics["alphabetic_ratio"] < 0.45:
        score -= 0.35
        flags.append("low_alphabetic_content")
    if metrics["punctuation_ratio"] > 0.35:
        score -= 0.25
        flags.append("too_much_punctuation")
    if metrics["word_count"] >= 5 and metrics["unique_word_ratio"] < 0.35:
        score -= 0.30
        flags.append("low_vocabulary_diversity")
    if metrics["repeated_token_ratio"] > 0.50:
        score -= 0.25
        flags.append("repeated_tokens")
    if metrics["repeated_line_ratio"] > 0.45:
        score -= 0.70
        flags.append("repeated_lines")
    if metrics["repeated_ngram_ratio"] > 0.45:
        score -= 0.45
        flags.append("repeated_phrases")
    if metrics["word_count"] >= 8 and metrics["stopword_ratio"] < 0.05:
        score -= 0.18
        flags.append("low_function_word_ratio")
    if high_risk_types:
        flags.append("high_risk_claims_detected")

    return max(0.0, min(1.0, score)), flags


def _high_risk_claim_types(text: str) -> list[str]:
    detected: list[str] = []
    for claim_type, patterns in HIGH_RISK_PATTERNS.items():
        if all(re.search(pattern, text) for pattern in patterns):
            detected.append(claim_type)
    return detected


def _max_char_run(text: str) -> int:
    if not text:
        return 0
    longest = 1
    current = 1
    previous = text[0]
    for char in text[1:]:
        if char == previous and not char.isspace():
            current += 1
            longest = max(longest, current)
        else:
            current = 1
            previous = char
    return longest


def _repeated_item_ratio(items: list[str]) -> float:
    if len(items) < 2:
        return 0.0
    normalized_items = [item for item in items if item]
    if not normalized_items:
        return 0.0
    return (len(normalized_items) - len(set(normalized_items))) / len(normalized_items)


def _repeated_ngram_ratio(tokens: list[str], size: int) -> float:
    if len(tokens) < size * 2:
        return 0.0
    ngrams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    return _repeated_item_ratio([" ".join(ngram) for ngram in ngrams])


def _normalize(text: str) -> str:
    return text.lower().translate(
        str.maketrans(
            {
                "\u0105": "a",
                "\u0107": "c",
                "\u0119": "e",
                "\u0142": "l",
                "\u0144": "n",
                "\u00f3": "o",
                "\u015b": "s",
                "\u017a": "z",
                "\u017c": "z",
            }
        )
    )
