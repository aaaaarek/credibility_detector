from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.services.article_fetcher import FetchedArticle, fetch_article
from ml.features.content_quality import analyze_content_quality
from ml.features.source_features import extract_source_features
from ml.features.text_features import extract_text_features
from ml.training.datasets import REQUIRED_COLUMNS, ROOT


URL_LIST_PATH = ROOT / "data" / "datasets" / "real_article_urls.txt"
CANDIDATES_PATH = ROOT / "data" / "datasets" / "real_articles_candidates.csv"
CANDIDATE_COLUMNS = REQUIRED_COLUMNS + [
    "label_reason",
    "dataset_source",
    "needs_review",
]


@dataclass(frozen=True)
class SuggestedLabel:
    value: float
    reasons: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real-article dataset candidates from a URL list.")
    parser.add_argument("--input", type=Path, default=URL_LIST_PATH)
    parser.add_argument("--output", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=12)
    args = parser.parse_args()

    urls = _read_urls(args.input)
    if args.limit is not None:
        urls = urls[: args.limit]

    rows = []
    failures = []
    for url in urls:
        try:
            article = fetch_article(url, timeout=args.timeout)
            rows.append(candidate_row(article))
        except Exception as exc:
            failures.append((url, str(exc)))

    _write_candidates(args.output, rows)
    print(f"Read URLs: {len(urls)}")
    print(f"Saved candidates: {len(rows)} -> {args.output}")
    if failures:
        print(f"Failed URLs: {len(failures)}")
        for url, error in failures[:10]:
            print(f"- {url}: {error}")


def candidate_row(article: FetchedArticle) -> dict[str, object]:
    suggestion = suggest_label(article)
    source_links = _candidate_source_links(article)
    return {
        "title": article.title or "",
        "content": article.content,
        "url": article.url,
        "source": _source_name(article.url),
        "author": article.author or "",
        "publish_date": article.publish_date or "",
        "source_links": "|".join(source_links),
        "credibility_label": f"{suggestion.value:.2f}",
        "label_reason": "AUTO-SUGGESTION: " + "; ".join(suggestion.reasons),
        "dataset_source": "url_candidate",
        "needs_review": "true",
    }


def suggest_label(article: FetchedArticle) -> SuggestedLabel:
    text = extract_text_features(article.content)
    quality = analyze_content_quality(article.content)
    source = extract_source_features(
        url=article.url,
        author=article.author,
        publish_date=article.publish_date,
        source_links=article.source_links,
        content=article.content,
    )
    score = 0.55
    reasons: list[str] = []

    if source.known_reputable_domain:
        score += 0.18
        reasons.append("known reputable domain")
    if source.suspicious_domain_hint:
        score -= 0.22
        reasons.append("suspicious domain hint")
    if source.uses_https:
        score += 0.04
    if source.has_author:
        score += 0.04
        reasons.append("author detected")
    if source.has_publish_date:
        score += 0.04
        reasons.append("publish date detected")
    if source.relevant_reputable_source_link_count >= 2:
        score += 0.10
        reasons.append("multiple relevant reputable links")
    elif source.relevant_source_link_count >= 1:
        score += 0.05
        reasons.append("relevant source link detected")
    if text.source_word_count >= 3 and text.number_count >= 2:
        score += 0.08
        reasons.append("data and source language in text")
    if text.clickbait_phrase_count:
        score -= min(0.20, text.clickbait_phrase_count * 0.08)
        reasons.append("clickbait phrase detected")
    if text.conspiracy_word_count:
        score -= min(0.20, text.conspiracy_word_count * 0.10)
        reasons.append("conspiracy vocabulary detected")
    if text.emotional_word_count >= 3 or text.exclamation_count >= 3:
        score -= 0.10
        reasons.append("emotional or sensational language")
    if quality.high_risk_claim_count:
        score -= 0.18
        reasons.append("high-risk claim pattern")
    if quality.quality_score < 0.40:
        score = min(score, 0.35)
        reasons.append("low content quality")
    if not reasons:
        reasons.append("neutral automatic estimate")

    return SuggestedLabel(value=max(0.0, min(1.0, score)), reasons=reasons)


def _candidate_source_links(article: FetchedArticle) -> list[str]:
    source = extract_source_features(source_links=article.source_links, content=article.content)
    if source.relevant_source_link_count == 0:
        return []
    relevant = []
    for link in article.source_links:
        link_source = extract_source_features(source_links=[link], content=article.content)
        if link_source.relevant_source_link_count:
            relevant.append(link)
    return relevant[:10]


def _read_urls(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"URL list does not exist: {path}")
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return list(dict.fromkeys(urls))


def _write_candidates(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _source_name(url: str) -> str:
    domain = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    return domain[4:] if domain.startswith("www.") else domain


if __name__ == "__main__":
    main()
