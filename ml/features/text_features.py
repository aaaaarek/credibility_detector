from __future__ import annotations

import re
from dataclasses import asdict, dataclass


EMOTIONAL_WORDS = {
    "shocking",
    "scandal",
    "scandalous",
    "outrage",
    "terrifying",
    "disaster",
    "panic",
    "secret",
    "hidden",
    "miracle",
    "betrayal",
    "explosive",
    "breaking",
    "pilne",
    "szok",
    "skandal",
    "katastrofa",
    "panika",
    "tajemnica",
    "ukrywaja",
    "cud",
    "zdrada",
}

CLICKBAIT_PHRASES = {
    "you won't believe",
    "what happened next",
    "doctors hate",
    "the truth about",
    "they do not want you to know",
    "this changes everything",
    "nie uwierzysz",
    "co stalo sie potem",
    "lekarze nienawidza",
    "prawda o",
    "oni nie chca zebys wiedzial",
    "to zmienia wszystko",
}

HEDGING_WORDS = {
    "allegedly",
    "reportedly",
    "may",
    "might",
    "could",
    "according",
    "prawdopodobnie",
    "rzekomo",
    "wedlug",
    "mozliwe",
}

SOURCE_WORDS = {
    "study",
    "report",
    "data",
    "research",
    "university",
    "ministry",
    "agency",
    "journal",
    "badanie",
    "raport",
    "dane",
    "ministerstwo",
    "agencja",
    "uniwersytet",
    "czasopismo",
}


@dataclass(frozen=True)
class TextFeatures:
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    uppercase_ratio: float
    exclamation_count: int
    question_count: int
    url_count: int
    number_count: int
    date_count: int
    quote_count: int
    emotional_word_count: int
    clickbait_phrase_count: int
    hedging_word_count: int
    source_word_count: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def extract_text_features(text: str) -> TextFeatures:
    normalized = _normalize(text)
    words = re.findall(r"\b[\w'-]+\b", normalized)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    uppercase_letters = sum(1 for char in text if char.isupper())
    letters = sum(1 for char in text if char.isalpha())

    return TextFeatures(
        word_count=len(words),
        sentence_count=len(sentences),
        avg_sentence_length=(len(words) / max(len(sentences), 1)),
        uppercase_ratio=(uppercase_letters / max(letters, 1)),
        exclamation_count=text.count("!"),
        question_count=text.count("?"),
        url_count=len(re.findall(r"https?://\S+|www\.\S+", text, flags=re.IGNORECASE)),
        number_count=len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text)),
        date_count=len(re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b20\d{2}\b", text)),
        quote_count=text.count('"') // 2,
        emotional_word_count=_count_terms(normalized, EMOTIONAL_WORDS),
        clickbait_phrase_count=sum(1 for phrase in CLICKBAIT_PHRASES if phrase in normalized),
        hedging_word_count=_count_terms(normalized, HEDGING_WORDS),
        source_word_count=_count_terms(normalized, SOURCE_WORDS),
    )


def _count_terms(text: str, terms: set[str]) -> int:
    tokens = set(re.findall(r"\b[\w'-]+\b", text))
    return sum(1 for term in terms if term in tokens)


def _normalize(text: str) -> str:
    polish_ascii = str.maketrans(
        {
            "ą": "a",
            "ć": "c",
            "ę": "e",
            "ł": "l",
            "ń": "n",
            "ó": "o",
            "ś": "s",
            "ź": "z",
            "ż": "z",
        }
    )
    return text.lower().translate(polish_ascii)
