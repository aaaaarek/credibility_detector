from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urlparse


HIGH_REPUTATION_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "npr.org",
    "nature.com",
    "science.org",
    "who.int",
    "gov.pl",
    "ec.europa.eu",
    "pap.pl",
}

LOW_REPUTATION_HINTS = {
    "truth",
    "secret",
    "patriot",
    "viral",
    "now",
    "dailybuzz",
    "uncensored",
    "click",
    "rumor",
}


@dataclass(frozen=True)
class SourceFeatures:
    domain: str | None
    has_url: bool
    uses_https: bool
    known_reputable_domain: bool
    suspicious_domain_hint: bool
    has_author: bool
    has_publish_date: bool
    source_link_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_source_features(
    url: str | None = None,
    author: str | None = None,
    publish_date: str | None = None,
    source_links: list[str] | None = None,
) -> SourceFeatures:
    parsed = urlparse(url or "")
    domain = _clean_domain(parsed.netloc) if parsed.netloc else None
    source_links = source_links or []

    return SourceFeatures(
        domain=domain,
        has_url=bool(url),
        uses_https=parsed.scheme == "https",
        known_reputable_domain=domain in HIGH_REPUTATION_DOMAINS if domain else False,
        suspicious_domain_hint=_has_suspicious_hint(domain),
        has_author=bool(author and author.strip()),
        has_publish_date=bool(publish_date and publish_date.strip()),
        source_link_count=len(source_links),
    )


def _clean_domain(netloc: str) -> str:
    domain = netloc.lower().split("@")[-1].split(":")[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _has_suspicious_hint(domain: str | None) -> bool:
    if not domain:
        return False
    return any(hint in domain for hint in LOW_REPUTATION_HINTS)
