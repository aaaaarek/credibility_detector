from __future__ import annotations

import re
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
    relevant_source_link_count: int
    relevant_unique_source_domain_count: int
    relevant_reputable_source_link_count: int
    unrelated_source_link_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_source_features(
    url: str | None = None,
    author: str | None = None,
    publish_date: str | None = None,
    source_links: list[str] | None = None,
    content: str | None = None,
) -> SourceFeatures:
    parsed = urlparse(url or "")
    domain = _clean_domain(parsed.netloc) if parsed.netloc else None
    source_links = source_links or []
    source_domains = [_clean_domain(urlparse(link).netloc) for link in source_links if urlparse(link).netloc]
    relevant_source_links = filter_relevant_source_links(source_links, content)
    relevant_source_domains = [
        _clean_domain(urlparse(link).netloc) for link in relevant_source_links if urlparse(link).netloc
    ]

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
        relevant_source_link_count=len(relevant_source_links),
        relevant_unique_source_domain_count=len(set(relevant_source_domains)),
        relevant_reputable_source_link_count=sum(
            1 for source_domain in relevant_source_domains if _is_reputable_domain(source_domain)
        ),
        unrelated_source_link_count=max(0, len(source_links) - len(relevant_source_links)),
    )


def filter_relevant_source_links(source_links: list[str] | None, content: str | None) -> list[str]:
    source_links = source_links or []
    if not content or not content.strip():
        return source_links
    return [link for link in source_links if _is_relevant_source_link(link, content)]


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


def _is_relevant_source_link(link: str, content: str) -> bool:
    parsed = urlparse(link)
    if not parsed.netloc:
        return False

    normalized_content = _normalize_text(content)
    text_tokens = set(_text_tokens(normalized_content))
    link_domain = _clean_domain(parsed.netloc)
    if link_domain in normalized_content:
        return True

    link_tokens = _link_tokens(parsed.netloc, parsed.path, parsed.query)
    return any(_token_matches_text(token, text_tokens) for token in link_tokens)


def _link_tokens(netloc: str, path: str, query: str) -> list[str]:
    raw = " ".join([netloc, path, query]).replace("_", " ").replace("-", " ")
    tokens = _text_tokens(_normalize_text(raw))
    return [token for token in tokens if token not in GENERIC_LINK_TOKENS and len(token) >= 3]


def _text_tokens(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9][a-z0-9-]*\b", text)


def _token_matches_text(link_token: str, text_tokens: set[str]) -> bool:
    if link_token in text_tokens:
        return True
    if link_token == "dane":
        return any(text_token.startswith("dan") for text_token in text_tokens if len(text_token) >= 4)
    return False


def _normalize_text(text: str) -> str:
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


GENERIC_LINK_TOKENS = {
    "article",
    "articles",
    "blog",
    "com",
    "content",
    "data",
    "en",
    "gov",
    "html",
    "http",
    "https",
    "news",
    "page",
    "paper",
    "pl",
    "post",
    "report",
    "reports",
    "story",
    "www",
}
