from __future__ import annotations

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
import trafilatura


@dataclass(frozen=True)
class FetchedArticle:
    title: str | None
    content: str
    url: str
    author: str | None
    publish_date: str | None
    source_links: list[str]


def fetch_article(url: str, timeout: int = 12) -> FetchedArticle:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "credibility-detector/0.1 (+https://github.com/aaaaarek/credibility_detector)"},
    )
    response.raise_for_status()

    html = _decode_html(response)
    extracted = trafilatura.extract(
        html,
        include_links=False,
        include_comments=False,
        favor_precision=True,
    )
    soup = BeautifulSoup(html, "html.parser")

    content = extracted or _fallback_text(soup)
    if not content or len(content.split()) < 20:
        raise ValueError("Nie udalo sie wyciagnac wystarczajacej tresci artykulu z URL.")

    return FetchedArticle(
        title=_title(soup),
        content=content,
        url=url,
        author=_meta(soup, ["author", "article:author", "twitter:creator"]),
        publish_date=_meta(soup, ["article:published_time", "pubdate", "date", "datePublished"]),
        source_links=_links(soup),
    )


def _decode_html(response: requests.Response) -> str:
    detected = from_bytes(response.content).best()
    if detected is not None:
        return str(detected)

    if response.encoding:
        try:
            return response.content.decode(response.encoding)
        except (LookupError, UnicodeDecodeError):
            pass

    return response.content.decode("utf-8", errors="replace")


def _title(soup: BeautifulSoup) -> str | None:
    og_title = _meta(soup, ["og:title", "twitter:title"])
    if og_title:
        return og_title
    return soup.title.string.strip() if soup.title and soup.title.string else None


def _meta(soup: BeautifulSoup, names: list[str]) -> str | None:
    for name in names:
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def _links(soup: BeautifulSoup) -> list[str]:
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if href.startswith("http"):
            links.append(href)
    return list(dict.fromkeys(links))[:30]


def _fallback_text(soup: BeautifulSoup) -> str:
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n\n".join(p for p in paragraphs if p)
