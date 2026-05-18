from __future__ import annotations

import re
from dataclasses import asdict, dataclass


CLAIM_VERBS = {
    "says",
    "said",
    "claims",
    "claimed",
    "reports",
    "reported",
    "shows",
    "showed",
    "found",
    "confirmed",
    "announced",
    "ostrzega",
    "twierdzi",
    "podaje",
    "informuje",
    "wykazuje",
    "potwierdza",
    "oglosil",
    "oglosila",
}

EVIDENCE_MARKERS = {
    "according to",
    "report",
    "study",
    "data",
    "dataset",
    "court",
    "ministry",
    "agency",
    "university",
    "wedlug",
    "raport",
    "badanie",
    "dane",
    "ministerstwo",
    "agencja",
    "uniwersytet",
    "sad",
}


@dataclass(frozen=True)
class ClaimAnalysis:
    claims: list[str]
    evidence_marker_count: int
    numeric_claim_count: int
    unsupported_claim_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_claims(text: str) -> ClaimAnalysis:
    sentences = _sentences(text)
    claims: list[str] = []
    unsupported = 0
    numeric = 0
    evidence_markers = 0

    for sentence in sentences:
        normalized = _normalize(sentence)
        has_claim_verb = any(verb in normalized for verb in CLAIM_VERBS)
        has_number = bool(re.search(r"\b\d+(?:[.,]\d+)?%?\b", sentence))
        has_evidence = any(marker in normalized for marker in EVIDENCE_MARKERS) or "http" in normalized

        if has_evidence:
            evidence_markers += 1
        if has_number:
            numeric += 1
        if has_claim_verb or has_number:
            claims.append(sentence.strip())
            if not has_evidence and not has_number:
                unsupported += 1

    return ClaimAnalysis(
        claims=claims[:6],
        evidence_marker_count=evidence_markers,
        numeric_claim_count=numeric,
        unsupported_claim_count=unsupported,
    )


def _sentences(text: str) -> list[str]:
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _normalize(text: str) -> str:
    return (
        text.lower()
        .translate(
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
    )
