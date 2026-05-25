from __future__ import annotations

import re
from dataclasses import asdict, dataclass


JOURNAL_MARKERS = {
    "acm",
    "arxiv",
    "biorxiv",
    "bmj",
    "cambridge core",
    "cell",
    "elsevier",
    "plos",
    "frontiers",
    "ieee",
    "jama",
    "medrxiv",
    "nature",
    "nejm",
    "oxford academic",
    "science",
    "scientific reports",
    "scopus",
    "springer nature",
    "the lancet",
    "wiley",
}

SECTION_MARKERS = {
    "abstract",
    "acknowledgements",
    "background",
    "conflict of interest",
    "conclusion",
    "data availability",
    "discussion",
    "funding",
    "introduction",
    "limitations",
    "materials and methods",
    "methodology",
    "methods",
    "objective",
    "results",
    "references",
    "supplementary material",
}


@dataclass(frozen=True)
class DocumentFeatures:
    doi: str | None
    has_doi: bool
    has_journal_marker: bool
    has_publication_dates: bool
    has_authors_marker: bool
    has_references: bool
    scientific_section_count: int
    scientific_document_score: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_document_features(text: str) -> DocumentFeatures:
    normalized = " ".join(text.lower().split())
    doi = _extract_doi(text)
    section_count = sum(1 for marker in SECTION_MARKERS if re.search(rf"\b{re.escape(marker)}\b", normalized))
    has_references = bool(re.search(r"\breferences\b", normalized)) or len(re.findall(r"\bet al\.|\bdoi\b", normalized)) >= 3

    raw_score = 0.0
    raw_score += 0.25 if doi else 0.0
    raw_score += 0.18 if any(marker in normalized for marker in JOURNAL_MARKERS) else 0.0
    raw_score += 0.16 if re.search(r"\b(received|revised|accepted|published|online)\b", normalized) else 0.0
    raw_score += 0.12 if re.search(r"\bet al\.|\bauthor\(s\)|\baffiliation|correspondence|orcid\b", normalized) else 0.0
    raw_score += 0.14 if has_references else 0.0
    raw_score += min(0.15, section_count * 0.03)

    return DocumentFeatures(
        doi=doi,
        has_doi=bool(doi),
        has_journal_marker=any(marker in normalized for marker in JOURNAL_MARKERS),
        has_publication_dates=bool(re.search(r"\b(received|revised|accepted|published|online)\b", normalized)),
        has_authors_marker=bool(re.search(r"\bet al\.|\bauthor\(s\)|\baffiliation|correspondence|orcid\b", normalized)),
        has_references=has_references,
        scientific_section_count=section_count,
        scientific_document_score=round(min(1.0, raw_score), 3),
    )


def _extract_doi(text: str) -> str | None:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, flags=re.IGNORECASE)
    return match.group(0).rstrip(".") if match else None
