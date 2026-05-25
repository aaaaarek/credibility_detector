from __future__ import annotations

import re
from dataclasses import asdict, dataclass


EMOTIONAL_WORDS = {
    "alarming",
    "betrayal",
    "bombshell",
    "breaking",
    "catastrophe",
    "chaos",
    "devastating",
    "disaster",
    "explosive",
    "furious",
    "hidden",
    "horror",
    "insane",
    "miracle",
    "outrage",
    "panic",
    "scandal",
    "scandalous",
    "shocking",
    "terrifying",
    "unbelievable",
    "alarm",
    "cud",
    "dramat",
    "katastrofa",
    "koszmar",
    "masakra",
    "niewiarygodne",
    "panika",
    "pilne",
    "przerazajace",
    "sensacja",
    "skandal",
    "szok",
    "secret",
    "tajemnica",
    "ukrywaja",
    "zdrada",
}

CLICKBAIT_PHRASES = {
    "doctors hate",
    "jedna prosta sztuczka",
    "lekarze nienawidza",
    "mainstream media won't tell you",
    "media milcza",
    "nie uwierzysz",
    "oni nie chca zebys wiedzial",
    "one simple trick",
    "prawda o",
    "share before",
    "the truth about",
    "they do not want you to know",
    "this is not a joke",
    "this changes everything",
    "co stalo sie potem",
    "to zmienia wszystko",
    "udostepnij zanim usuna",
    "what happened next",
    "you need to see",
    "you won't believe",
    "zobacz zanim usuna",
}

HEDGING_WORDS = {
    "allegedly",
    "apparently",
    "according to",
    "according",
    "could",
    "indicates",
    "may",
    "might",
    "reportedly",
    "rumored",
    "suggests",
    "supposedly",
    "unconfirmed",
    "nieoficjalnie",
    "podobno",
    "mozliwe",
    "mozliwe ze",
    "prawdopodobnie",
    "rzekomo",
    "sugeruje",
    "wedlug",
}

SOURCE_WORDS = {
    "agency",
    "analysis",
    "court",
    "data",
    "dataset",
    "evidence",
    "journal",
    "ministry",
    "official",
    "peer-reviewed",
    "registry",
    "report",
    "research",
    "spokesperson",
    "statistics",
    "study",
    "university",
    "agencja",
    "analiza",
    "badanie",
    "badacze",
    "czasopismo",
    "dane",
    "dowody",
    "eksperci",
    "instytut",
    "ministerstwo",
    "naukowcy",
    "oficjalny",
    "raport",
    "rejestr",
    "rzecznik",
    "statystyki",
    "uniwersytet",
    "urzad",
    "zrodlo",
}

MANIPULATION_WORDS = {
    "agenda",
    "always",
    "corrupt",
    "guaranteed",
    "lie",
    "lies",
    "must",
    "never",
    "nobody",
    "proof",
    "scam",
    "traitor",
    "everyone",
    "wake",
    "dowod",
    "gwarantowane",
    "klamie",
    "musisz",
    "nigdy",
    "nikt",
    "oszustwo",
    "skorumpowani",
    "sprzedajni",
    "wszyscy",
    "zawsze",
    "zdrajcy",
}

CONSPIRACY_WORDS = {
    "agenda2030",
    "chemtrails",
    "coverup",
    "cover-up",
    "censored",
    "deepstate",
    "deep-state",
    "elites",
    "globalists",
    "greatreset",
    "hoax",
    "illuminati",
    "mainstream",
    "newworldorder",
    "plandemic",
    "cenzura",
    "elity",
    "globalisci",
    "klamstwo",
    "masoni",
    "plandemia",
    "spisek",
    "ukrywaja",
    "zamach",
}

URGENCY_WORDS = {
    "act",
    "alert",
    "before",
    "deleted",
    "hurry",
    "immediately",
    "last chance",
    "now",
    "share",
    "today",
    "urgent",
    "alarm",
    "teraz",
    "dzis",
    "natychmiast",
    "ostatnia",
    "pilne",
    "szybko",
    "szansa",
    "udostepnij",
    "zanim",
    "usuniete",
}

AUTHORITY_SIGNALS = {
    "analyst",
    "court",
    "doctor",
    "expert",
    "experts",
    "inspectorate",
    "institute",
    "official",
    "physician",
    "professor",
    "regulator",
    "scientist",
    "scientists",
    "researcher",
    "researchers",
    "spokesperson",
    "analityk",
    "ekspert",
    "eksperci",
    "badacz",
    "badacze",
    "inspektorat",
    "instytut",
    "lekarz",
    "naukowiec",
    "naukowcy",
    "oficjalny",
    "profesor",
    "regulator",
    "rzadowy",
    "rzecznik",
    "sad",
    "urzad",
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
    manipulation_word_count: int
    conspiracy_word_count: int
    urgency_word_count: int
    authority_signal_count: int

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
        manipulation_word_count=_count_terms(normalized, MANIPULATION_WORDS),
        conspiracy_word_count=_count_terms(normalized, CONSPIRACY_WORDS),
        urgency_word_count=_count_terms(normalized, URGENCY_WORDS),
        authority_signal_count=_count_terms(normalized, AUTHORITY_SIGNALS),
    )


def _count_terms(text: str, terms: set[str]) -> int:
    tokens = set(re.findall(r"\b[\w'-]+\b", text))
    return sum(1 for term in terms if (term in text if " " in term else term in tokens))


def _normalize(text: str) -> str:
    polish_ascii = str.maketrans(
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
    return text.lower().translate(polish_ascii)
