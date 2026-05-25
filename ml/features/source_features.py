from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urlparse


HIGH_REPUTATION_DOMAINS = {
    "apnews.com",
    "bbc.co.uk",
    "bbc.com",
    "bmj.com",
    "cdc.gov",
    "cochranelibrary.com",
    "ec.europa.eu",
    "ecdc.europa.eu",
    "ema.europa.eu",
    "europarl.europa.eu",
    "eurostat.ec.europa.eu",
    "fda.gov",
    "ft.com",
    "gov.pl",
    "imf.org",
    "knf.gov.pl",
    "nature.com",
    "nbp.pl",
    "ncbi.nlm.nih.gov",
    "nejm.org",
    "nfz.gov.pl",
    "nih.gov",
    "npr.org",
    "oecd.org",
    "ourworldindata.org",
    "pap.pl",
    "pkw.gov.pl",
    "policja.pl",
    "reuters.com",
    "science.org",
    "sejm.gov.pl",
    "senat.gov.pl",
    "stat.gov.pl",
    "thelancet.com",
    "uokik.gov.pl",
    "un.org",
    "who.int",
    "worldbank.org",
    "zus.pl",
}

TRUSTED_DOMAIN_SUFFIXES = {
    ".ac.uk",
    ".edu",
    ".edu.pl",
    ".europa.eu",
    ".gov",
    ".gov.pl",
    ".int",
}

LOW_REPUTATION_HINTS = {
    "agenda",
    "alert",
    "awakening",
    "bez-cenzury",
    "bez_cenzury",
    "breaking",
    "click",
    "dailybuzz",
    "deepstate",
    "exposed",
    "fake",
    "freedom",
    "greatreset",
    "hoax",
    "illuminati",
    "leak",
    "matrix",
    "miracle",
    "news24",
    "now",
    "patriot",
    "plandemic",
    "plandemia",
    "prawda",
    "rumor",
    "scam",
    "secret",
    "sensacja",
    "spisek",
    "szok",
    "truth",
    "uncensored",
    "viral",
    "wolnosc",
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
    unique_source_domain_count: int
    reputable_source_link_count: int

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
    source_domains = [_clean_domain(urlparse(link).netloc) for link in source_links if urlparse(link).netloc]

    return SourceFeatures(
        domain=domain,
        has_url=bool(url),
        uses_https=parsed.scheme == "https",
        known_reputable_domain=_is_reputable_domain(domain),
        suspicious_domain_hint=_has_suspicious_hint(domain),
        has_author=bool(author and author.strip()),
        has_publish_date=bool(publish_date and publish_date.strip()),
        source_link_count=len(source_links),
        unique_source_domain_count=len(set(source_domains)),
        reputable_source_link_count=sum(1 for source_domain in source_domains if _is_reputable_domain(source_domain)),
    )


def _clean_domain(netloc: str) -> str:
    domain = netloc.lower().split("@")[-1].split(":")[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _has_suspicious_hint(domain: str | None) -> bool:
    if not domain:
        return False
    normalized_domain = domain.replace("_", "-")
    tokens = _domain_tokens(domain)
    for hint in LOW_REPUTATION_HINTS:
        normalized_hint = hint.replace("_", "-")
        if "-" in normalized_hint and normalized_hint in normalized_domain:
            return True
        if any(_token_matches_hint(token, normalized_hint) for token in tokens):
            return True
    return False


def _is_reputable_domain(domain: str | None) -> bool:
    if not domain:
        return False
    if any(domain == known or domain.endswith(f".{known}") for known in HIGH_REPUTATION_DOMAINS):
        return True
    return any(domain.endswith(suffix) for suffix in TRUSTED_DOMAIN_SUFFIXES)


def _domain_tokens(domain: str) -> list[str]:
    return [token for token in domain.replace("_", "-").replace(".", "-").split("-") if token]


def _token_matches_hint(token: str, hint: str) -> bool:
    if len(hint) <= 3:
        return token == hint
    return token == hint or token.startswith(hint) or token.endswith(hint)
