from __future__ import annotations

from dataclasses import asdict, dataclass

from ml.features.source_features import SourceFeatures, extract_source_features
from ml.features.text_features import TextFeatures, extract_text_features


@dataclass(frozen=True)
class ArticleInput:
    title: str | None
    content: str
    url: str | None = None
    author: str | None = None
    publish_date: str | None = None
    source_links: list[str] | None = None


@dataclass(frozen=True)
class ModuleScores:
    source_score: float
    linguistic_score: float
    fact_score: float
    consensus_score: float
    consistency_score: float


@dataclass(frozen=True)
class CredibilityResult:
    credibility_score: float
    credibility_level: str
    module_scores: ModuleScores
    reasons: list[str]
    metadata: dict[str, object]


def analyze_article(article: ArticleInput) -> CredibilityResult:
    text_features = extract_text_features(article.content)
    source_features = extract_source_features(
        url=article.url,
        author=article.author,
        publish_date=article.publish_date,
        source_links=article.source_links,
    )

    source_score, source_reasons = _score_source(source_features)
    linguistic_score, linguistic_reasons = _score_language(text_features)
    fact_score, fact_reasons = _score_fact_markers(text_features)
    consensus_score, consensus_reasons = _score_consensus(article, text_features)
    consistency_score, consistency_reasons = _score_internal_consistency(text_features)

    module_scores = ModuleScores(
        source_score=source_score,
        linguistic_score=linguistic_score,
        fact_score=fact_score,
        consensus_score=consensus_score,
        consistency_score=consistency_score,
    )
    final_score = _clamp(
        0.25 * source_score
        + 0.25 * linguistic_score
        + 0.20 * fact_score
        + 0.15 * consensus_score
        + 0.15 * consistency_score
    )

    reasons = source_reasons + linguistic_reasons + fact_reasons + consensus_reasons + consistency_reasons
    return CredibilityResult(
        credibility_score=round(final_score, 3),
        credibility_level=_level(final_score),
        module_scores=module_scores,
        reasons=reasons[:8],
        metadata={
            "title": article.title,
            "url": article.url,
            "domain": source_features.domain,
            "text_features": text_features.as_dict(),
            "source_features": source_features.as_dict(),
        },
    )


def result_to_dict(result: CredibilityResult) -> dict[str, object]:
    return {
        "credibility_score": result.credibility_score,
        "credibility_level": result.credibility_level,
        "module_scores": asdict(result.module_scores),
        "reasons": result.reasons,
        "metadata": result.metadata,
    }


def _score_source(features: SourceFeatures) -> tuple[float, list[str]]:
    score = 0.55
    reasons: list[str] = []

    if not features.has_url:
        reasons.append("Brak URL ogranicza ocene reputacji zrodla.")
    if features.uses_https:
        score += 0.08
    elif features.has_url:
        score -= 0.08
        reasons.append("Strona nie uzywa HTTPS.")
    if features.known_reputable_domain:
        score += 0.22
        reasons.append("Domena znajduje sie na liscie zrodel o wysokiej reputacji.")
    if features.suspicious_domain_hint:
        score -= 0.20
        reasons.append("Domena zawiera sygnaly typowe dla stron clickbaitowych lub spiskowych.")
    if features.has_author:
        score += 0.08
    else:
        score -= 0.05
        reasons.append("Brakuje rozpoznanego autora.")
    if features.has_publish_date:
        score += 0.07
    else:
        reasons.append("Brakuje daty publikacji.")
    if features.source_link_count >= 3:
        score += 0.10
    elif features.source_link_count == 0:
        score -= 0.05
        reasons.append("Nie wykryto linkow do zrodel zewnetrznych.")

    return _clamp(score), reasons


def _score_language(features: TextFeatures) -> tuple[float, list[str]]:
    score = 0.75
    reasons: list[str] = []
    emotional_rate = features.emotional_word_count / max(features.word_count, 1)

    if features.clickbait_phrase_count:
        score -= min(0.30, 0.12 * features.clickbait_phrase_count)
        reasons.append("Tekst zawiera frazy clickbaitowe.")
    if emotional_rate > 0.015 or features.exclamation_count >= 3:
        score -= 0.18
        reasons.append("Jezyk jest silnie emocjonalny lub sensacyjny.")
    if features.uppercase_ratio > 0.08:
        score -= 0.12
        reasons.append("Podwyzszony udzial wielkich liter moze wskazywac na perswazyjny styl.")
    if features.hedging_word_count >= 3:
        score -= 0.08
        reasons.append("Wiele sformulowan niepewnosciowych obniza jednoznacznosc przekazu.")
    if features.avg_sentence_length > 34:
        score -= 0.05
        reasons.append("Bardzo dlugie zdania moga utrudniac weryfikacje twierdzen.")
    if not reasons:
        reasons.append("Jezyk tekstu jest relatywnie neutralny.")

    return _clamp(score), reasons


def _score_fact_markers(features: TextFeatures) -> tuple[float, list[str]]:
    score = 0.45
    reasons: list[str] = []

    if features.source_word_count >= 2:
        score += 0.18
        reasons.append("Tekst odwoluje sie do raportow, danych lub instytucji.")
    if features.number_count >= 3:
        score += 0.12
        reasons.append("Obecne sa konkretne liczby wspierajace twierdzenia.")
    if features.quote_count >= 1:
        score += 0.08
        reasons.append("Wykryto cytaty lub wypowiedzi zrodlowe.")
    if features.url_count >= 1:
        score += 0.12
        reasons.append("Tekst zawiera linki, ktore mozna dalej zweryfikowac.")
    if features.word_count < 120:
        score -= 0.12
        reasons.append("Tekst jest krotki, wiec zawiera malo materialu do weryfikacji.")

    return _clamp(score), reasons


def _score_consensus(article: ArticleInput, features: TextFeatures) -> tuple[float, list[str]]:
    score = 0.50
    reasons: list[str] = []
    links = article.source_links or []

    if len(links) >= 4:
        score += 0.20
        reasons.append("Artykul linkuje do kilku zewnetrznych zrodel.")
    elif len(links) >= 1:
        score += 0.08
        reasons.append("Artykul zawiera przynajmniej jedno zewnetrzne zrodlo.")
    else:
        score -= 0.10
        reasons.append("Brak zewnetrznych linkow utrudnia cross-source verification.")
    if features.source_word_count >= 3:
        score += 0.10

    return _clamp(score), reasons


def _score_internal_consistency(features: TextFeatures) -> tuple[float, list[str]]:
    score = 0.70
    reasons: list[str] = []

    if features.question_count >= 4:
        score -= 0.10
        reasons.append("Wiele pytan retorycznych moze wskazywac na sugestywny styl.")
    if features.exclamation_count >= 5:
        score -= 0.12
    if features.date_count and features.number_count:
        score += 0.08
        reasons.append("Tekst zawiera daty i liczby, co pomaga oceniac spojnosc faktow.")
    if features.sentence_count < 3:
        score -= 0.10
        reasons.append("Za malo zdan, aby solidnie ocenic spojnosc wewnetrzna.")

    return _clamp(score), reasons


def _level(score: float) -> str:
    if score >= 0.75:
        return "wysoka wiarygodnosc"
    if score >= 0.55:
        return "umiarkowana wiarygodnosc"
    if score >= 0.35:
        return "niejednoznaczne / wymaga weryfikacji"
    return "wysokie ryzyko dezinformacji"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
