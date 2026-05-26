from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ml.features.content_quality import ContentQuality, analyze_content_quality
from ml.features.document_features import DocumentFeatures, extract_document_features
from ml.features.profile_features import ProfileFeatures, ProfileInput, extract_profile_features
from ml.features.source_features import SourceFeatures, extract_source_features, filter_relevant_source_links
from ml.features.text_features import TextFeatures, extract_text_features
from ml.inference.claims import ClaimAnalysis, analyze_claims
from ml.inference.ml_model import predict_ml_score
from ml.inference.text_model import predict_text_ml_score


@dataclass(frozen=True)
class ArticleInput:
    title: str | None
    content: str
    input_type: InputType = "raw_text"
    url: str | None = None
    author: str | None = None
    publish_date: str | None = None
    source_links: list[str] | None = None
    profile: ProfileInput | None = None


@dataclass(frozen=True)
class ModuleScores:
    source_score: float
    linguistic_score: float
    fact_score: float
    consensus_score: float
    consistency_score: float
    claim_score: float
    profile_score: float
    ml_score: float
    text_ml_score: float


@dataclass(frozen=True)
class CredibilityResult:
    credibility_score: float
    credibility_level: str
    module_scores: dict[str, float]
    diagnostic_scores: dict[str, float]
    reasons: list[str]
    metadata: dict[str, object]


def analyze_article(article: ArticleInput) -> CredibilityResult:
    text_features = extract_text_features(article.content)
    content_quality = analyze_content_quality(article.content)
    source_features = extract_source_features(
        url=article.url,
        author=article.author,
        publish_date=article.publish_date,
        source_links=article.source_links,
        content=article.content,
    )
    profile_features = extract_profile_features(article.profile, article.content)
    document_features = extract_document_features(article.content)
    claim_analysis = analyze_claims(article.content)

    source_score, source_reasons = _score_source(source_features, article.input_type)
    linguistic_score, linguistic_reasons = _score_language(text_features)
    fact_score, fact_reasons = _score_fact_markers(
        text_features,
        claim_analysis,
        source_features,
        content_quality,
    )
    consensus_score, consensus_reasons = _score_consensus(source_features, text_features, article.input_type)
    consistency_score, consistency_reasons = _score_internal_consistency(text_features)
    claim_score, claim_reasons = _score_claims(claim_analysis, text_features)
    profile_score, profile_reasons = _score_profile(profile_features)
    ml_url, ml_author, ml_publish_date, ml_source_links = _ml_source_inputs(article)
    ml_score, ml_reason = predict_ml_score(
        content=article.content,
        url=ml_url,
        author=ml_author,
        publish_date=ml_publish_date,
        source_links=ml_source_links,
    )
    text_ml_score, text_ml_reason = predict_text_ml_score(article.content)

    all_scores = ModuleScores(
        source_score=source_score,
        linguistic_score=linguistic_score,
        fact_score=fact_score,
        consensus_score=consensus_score,
        consistency_score=consistency_score,
        claim_score=claim_score,
        profile_score=profile_score,
        ml_score=round(ml_score, 3),
        text_ml_score=round(text_ml_score, 3),
    )
    all_scores_dict = _scores_to_dict(all_scores)
    weights = SCORE_WEIGHTS[article.input_type]
    module_scores = {name: all_scores_dict[name] for name in weights}
    diagnostic_scores = _diagnostic_scores(all_scores_dict, weights, article, profile_features)
    weighted_score = _weighted_score(module_scores, weights)
    disagreement_penalty = _disagreement_penalty(module_scores, weights)
    spread_score = _spread_weighted_score(weighted_score)
    disagreement_adjusted_score = _clamp(spread_score - disagreement_penalty)
    final_score = disagreement_adjusted_score
    final_score = _apply_calibration_caps(
        final_score,
        all_scores_dict,
        article.input_type,
        content_quality,
        document_features,
    )

    reason_groups = {
        "content_quality": _content_quality_reasons(content_quality),
        "source_score": source_reasons,
        "linguistic_score": linguistic_reasons,
        "fact_score": fact_reasons,
        "consensus_score": consensus_reasons,
        "consistency_score": consistency_reasons,
        "claim_score": claim_reasons,
        "profile_score": profile_reasons,
        "ml_score": [ml_reason],
        "text_ml_score": [text_ml_reason],
    }
    active_reasons = reason_groups["content_quality"] + [reason for name in module_scores for reason in reason_groups[name]]
    diagnostic_reasons = [reason for name in diagnostic_scores for reason in reason_groups[name]]
    return CredibilityResult(
        credibility_score=round(final_score, 3),
        credibility_level=_level(final_score),
        module_scores=module_scores,
        diagnostic_scores=diagnostic_scores,
        reasons=(active_reasons + diagnostic_reasons)[:8],
        metadata={
            "input_type": article.input_type,
            "weights": weights,
            "content_quality": content_quality.as_dict(),
            "document_features": document_features.as_dict(),
            "score_calibration": {
                "weighted_score": round(weighted_score, 3),
                "spread_factor": SCORE_SPREAD_FACTOR,
                "spread_score": round(spread_score, 3),
                "disagreement_penalty": round(disagreement_penalty, 3),
                "disagreement_adjusted_score": round(disagreement_adjusted_score, 3),
                "final_score": round(final_score, 3),
            },
            "title": article.title,
            "url": article.url,
            "domain": source_features.domain,
            "extracted_claims": claim_analysis.as_dict(),
            "profile_features": profile_features.as_dict(),
            "text_features": text_features.as_dict(),
            "source_features": source_features.as_dict(),
        },
    )


def result_to_dict(result: CredibilityResult) -> dict[str, object]:
    return {
        "credibility_score": result.credibility_score,
        "credibility_level": result.credibility_level,
        "module_scores": result.module_scores,
        "diagnostic_scores": result.diagnostic_scores,
        "reasons": result.reasons,
        "metadata": result.metadata,
    }


def _score_source(features: SourceFeatures, input_type: InputType) -> tuple[float, list[str]]:
    score = 0.50
    reasons: list[str] = []
    metadata_weight = RAW_TEXT_METADATA_WEIGHT if input_type == "raw_text" else 1.0
    source_link_count = _trusted_source_link_count(features, input_type)
    reputable_source_link_count = _trusted_reputable_source_link_count(features, input_type)

    if not features.has_url:
        score -= 0.12 * metadata_weight
        reasons.append("Brak URL ogranicza ocene reputacji zrodla.")
    if features.uses_https:
        score += 0.08 * metadata_weight
    elif features.has_url:
        score -= 0.08 * metadata_weight
        reasons.append("Strona nie uzywa HTTPS.")
    if features.known_reputable_domain:
        score += 0.22 * metadata_weight
        reasons.append("Domena znajduje sie na liscie zrodel o wysokiej reputacji.")
    if features.suspicious_domain_hint:
        score -= 0.20
        reasons.append("Domena zawiera sygnaly typowe dla stron clickbaitowych lub spiskowych.")
    if features.has_author:
        score += 0.08 * metadata_weight
    else:
        score -= 0.08 * metadata_weight
        reasons.append("Brakuje rozpoznanego autora.")
    if features.has_publish_date:
        score += 0.07 * metadata_weight
    else:
        score -= 0.05 * metadata_weight
        reasons.append("Brakuje daty publikacji.")
    if source_link_count >= 3:
        score += 0.10
    elif source_link_count == 0:
        score -= 0.10
        reasons.append("Nie wykryto linkow do zrodel zewnetrznych.")
    if reputable_source_link_count >= 2:
        score += 0.06
        reasons.append("Linki zrodlowe prowadza do kilku domen instytucjonalnych lub o wysokiej reputacji.")
    elif reputable_source_link_count == 1:
        score += 0.03
    if input_type == "raw_text" and features.unrelated_source_link_count:
        reasons.append("Pominieto linki zrodlowe bez widocznego dopasowania do tresci tekstu.")

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
    if features.manipulation_word_count >= 2:
        score -= 0.10
        reasons.append("Wykryto slowa wzmacniajace presje lub czarno-biale tezy.")
    if features.conspiracy_word_count >= 1:
        score -= 0.14
        reasons.append("Tekst zawiera slownictwo typowe dla narracji spiskowych.")
    if features.urgency_word_count >= 2:
        score -= 0.08
        reasons.append("Tekst buduje poczucie pilnosci zamiast spokojnej argumentacji.")
    if features.authority_signal_count >= 2 and features.source_word_count >= 1:
        score += 0.06
        reasons.append("Tekst uzywa sygnalow eksperckich lub instytucjonalnych.")
    if features.avg_sentence_length > 34:
        score -= 0.05
        reasons.append("Bardzo dlugie zdania moga utrudniac weryfikacje twierdzen.")
    if not reasons:
        reasons.append("Jezyk tekstu jest relatywnie neutralny.")

    return _clamp(score), reasons


def _score_fact_markers(
    features: TextFeatures,
    claims: ClaimAnalysis,
    source: SourceFeatures,
    quality: ContentQuality,
) -> tuple[float, list[str]]:
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
    if claims.evidence_marker_count >= 2:
        score += 0.10
        reasons.append("Czesc twierdzen ma jawne markery dowodow lub instytucji.")
    if claims.unsupported_claim_count >= 2:
        score -= 0.12
        reasons.append("Wykryto kilka twierdzen bez widocznego wsparcia zrodlowego.")
    if quality.high_risk_claim_count:
        if source.relevant_reputable_source_link_count == 0 and claims.evidence_marker_count < 2:
            score -= 0.28
            reasons.append("Twierdzenia wysokiego ryzyka nie maja mocnego wsparcia w zrodlach.")
        else:
            score -= 0.10
            reasons.append("Twierdzenia wysokiego ryzyka obnizaja ocene markerow faktograficznych.")
    if features.conspiracy_word_count and claims.evidence_marker_count < 2:
        score -= 0.14
        reasons.append("Slownictwo spiskowe pojawia sie bez wystarczajacych markerow dowodowych.")
    if source.source_link_count and source.relevant_source_link_count == 0:
        score -= 0.10
        reasons.append("Podane linki nie maja widocznego dopasowania do tresci.")
    if features.word_count < 120:
        score -= 0.12
        reasons.append("Tekst jest krotki, wiec zawiera malo materialu do weryfikacji.")

    if quality.high_risk_claim_count and source.relevant_reputable_source_link_count == 0:
        score = min(score, 0.45)
    if features.conspiracy_word_count and claims.evidence_marker_count < 2:
        score = min(score, 0.50)

    return _clamp(score), reasons


def _score_consensus(source: SourceFeatures, features: TextFeatures, input_type: InputType) -> tuple[float, list[str]]:
    score = 0.45
    reasons: list[str] = []
    source_link_count = _trusted_source_link_count(source, input_type)
    unique_source_domain_count = _trusted_unique_source_domain_count(source, input_type)
    reputable_source_link_count = _trusted_reputable_source_link_count(source, input_type)

    if source_link_count >= 4:
        score += 0.20
        reasons.append("Artykul linkuje do kilku zewnetrznych zrodel.")
    elif source_link_count >= 1:
        score += 0.08
        reasons.append("Artykul zawiera przynajmniej jedno zewnetrzne zrodlo.")
    else:
        score -= 0.18
        reasons.append("Brak zewnetrznych linkow utrudnia cross-source verification.")
    if unique_source_domain_count >= 3:
        score += 0.12
        reasons.append("Linki prowadza do kilku roznych domen, co wzmacnia consensus.")
    elif source_link_count >= 3 and unique_source_domain_count <= 1:
        score -= 0.08
        reasons.append("Wiele linkow prowadzi do tej samej domeny, wiec consensus jest slaby.")
    if reputable_source_link_count >= 1:
        score += 0.08
        reasons.append("Wsrod linkow zrodlowych jest domena o wysokiej reputacji.")
    if features.source_word_count >= 3 and source_link_count >= 1:
        score += 0.10
    elif features.source_word_count >= 3:
        score += 0.03
    if source.source_link_count and source_link_count == 0:
        score = min(score, 0.34)
        reasons.append("Pominieto linki bez widocznego dopasowania do tresci, wiec consensus jest ograniczony.")

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


def _score_claims(claims: ClaimAnalysis, features: TextFeatures) -> tuple[float, list[str]]:
    score = 0.55
    reasons: list[str] = []

    if not claims.claims:
        score -= 0.12
        reasons.append("Nie wykryto wyraznych twierdzen faktograficznych do sprawdzenia.")
    if claims.numeric_claim_count >= 2:
        score += 0.10
        reasons.append("Wykryto konkretne twierdzenia liczbowe.")
    if claims.evidence_marker_count >= 2:
        score += 0.16
        reasons.append("Twierdzenia sa powiazane z markerami dowodow, np. raportem lub danymi.")
    if claims.unsupported_claim_count >= 2:
        score -= 0.24
        reasons.append("Czesc twierdzen brzmi faktograficznie, ale nie ma widocznego wsparcia.")
    if features.conspiracy_word_count and claims.evidence_marker_count == 0:
        score -= 0.12
        reasons.append("Narracja spiskowa pojawia sie bez markerow dowodowych.")

    return _clamp(score), reasons


def _score_profile(features: ProfileFeatures) -> tuple[float, list[str]]:
    score = 0.50
    reasons: list[str] = []

    if not features.has_profile_name and not features.has_profile_url:
        score -= 0.08
        reasons.append("Brak danych o profilu ogranicza ocene wiarygodnosci autora posta.")
    if features.has_profile_url:
        score += 0.08
        reasons.append("Podano URL profilu, co ulatwia weryfikacje autora.")
    if features.known_platform:
        score += 0.05
    if features.is_verified is True:
        score += 0.15
        reasons.append("Profil jest oznaczony jako zweryfikowany.")
    elif features.is_verified is False:
        score -= 0.04
        reasons.append("Profil nie jest oznaczony jako zweryfikowany.")
    if features.follower_count is not None:
        if features.follower_count >= 100_000:
            score += 0.08
            reasons.append("Profil ma duzy zasieg, co zwieksza odpowiedzialnosc i mozliwosc kontroli publicznej.")
        elif features.follower_count < 100:
            score -= 0.08
            reasons.append("Profil ma bardzo maly zasieg, co utrudnia ocene reputacji.")
    if features.account_age_days is not None:
        if features.account_age_days >= 365:
            score += 0.08
            reasons.append("Profil istnieje od co najmniej roku.")
        elif features.account_age_days < 30:
            score -= 0.12
            reasons.append("Profil jest bardzo nowy, co zwieksza ryzyko podszywania sie lub kampanii ad hoc.")
    if features.suspicious_handle_hint:
        score -= 0.10
        reasons.append("Nazwa profilu zawiera sygnaly typowe dla kont sensacyjnych lub podszywajacych sie.")
    if features.handle_from_text and not features.has_profile_url:
        score -= 0.08
        reasons.append("Wykryto uchwyt profilu w tekscie/OCR, ale brakuje URL do weryfikacji.")

    return _clamp(score), reasons


def _level(score: float) -> str:
    if score >= 0.75:
        return "wysoka wiarygodnosc / silnie uzrodlowiony material"
    if score >= 0.65:
        return "normalny artykul z autorem, data i weryfikowalnymi zrodlami"
    if score >= 0.50:
        return "opinia, blog lub czesciowy kontekst z pewna wartoscia weryfikacyjna"
    if score >= 0.35:
        return "jednostronny, slabo uzrodlowiony lub clickbaitowy tekst"
    return "wysokie ryzyko: scam, spisek, fabrykacja lub dezinformacja"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scores_to_dict(scores: ModuleScores) -> dict[str, float]:
    return {
        "source_score": scores.source_score,
        "linguistic_score": scores.linguistic_score,
        "fact_score": scores.fact_score,
        "consensus_score": scores.consensus_score,
        "consistency_score": scores.consistency_score,
        "claim_score": scores.claim_score,
        "profile_score": scores.profile_score,
        "ml_score": scores.ml_score,
        "text_ml_score": scores.text_ml_score,
    }


def _weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    return _clamp(sum(scores[name] * weight for name, weight in weights.items()) / total_weight)


def _disagreement_penalty(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    weighted_mean = _weighted_score(scores, weights)
    variance = sum(weights[name] * ((scores[name] - weighted_mean) ** 2) for name in weights) / total_weight
    weighted_std = variance**0.5
    penalty = max(0.0, weighted_std - SCORE_DISAGREEMENT_FREE_BAND) * SCORE_DISAGREEMENT_PENALTY_FACTOR
    return min(MAX_SCORE_DISAGREEMENT_PENALTY, penalty)


def _spread_weighted_score(score: float) -> float:
    return _clamp(0.50 + ((score - 0.50) * SCORE_SPREAD_FACTOR))


def _ml_source_inputs(article: ArticleInput) -> tuple[str | None, str | None, str | None, list[str] | None]:
    if article.input_type != "raw_text":
        return article.url, article.author, article.publish_date, article.source_links
    relevant_source_links = filter_relevant_source_links(article.source_links, article.content)
    if article.url:
        return article.url, article.author, article.publish_date, relevant_source_links
    return (
        None,
        None,
        None,
        relevant_source_links,
    )


def _trusted_source_link_count(features: SourceFeatures, input_type: InputType) -> int:
    if features.source_link_count:
        return features.relevant_source_link_count
    return features.source_link_count


def _trusted_unique_source_domain_count(features: SourceFeatures, input_type: InputType) -> int:
    if features.source_link_count:
        return features.relevant_unique_source_domain_count
    return features.unique_source_domain_count


def _trusted_reputable_source_link_count(features: SourceFeatures, input_type: InputType) -> int:
    if features.source_link_count:
        return features.relevant_reputable_source_link_count
    return features.reputable_source_link_count


def _diagnostic_scores(
    scores: dict[str, float],
    weights: dict[str, float],
    article: ArticleInput,
    profile_features: ProfileFeatures,
) -> dict[str, float]:
    diagnostic_names = [name for name in scores if name not in weights]
    if article.input_type == "url":
        diagnostic_names = [name for name in diagnostic_names if name in {"fact_score", "consistency_score"}]
    if article.input_type == "document":
        diagnostic_names = [name for name in diagnostic_names if name in {"fact_score", "consistency_score", "profile_score"}]
    if article.input_type == "raw_text":
        diagnostic_names = [name for name in diagnostic_names if name in {"fact_score", "consistency_score", "profile_score"}]
    if not (profile_features.has_profile_name or profile_features.has_profile_url):
        diagnostic_names = [name for name in diagnostic_names if name != "profile_score"]
    return {name: scores[name] for name in diagnostic_names}


def _content_quality_reasons(quality: ContentQuality) -> list[str]:
    reasons: list[str] = []
    if quality.quality_score < 0.20:
        reasons.append("Tekst nie zawiera wystarczajacej tresci semantycznej do wiarygodnej analizy.")
    elif quality.quality_score < 0.40:
        reasons.append("Jakosc tekstu jest niska, wiec finalny wynik zostal ograniczony.")
    if quality.high_risk_claim_count:
        reasons.append("Wykryto twierdzenia wysokiego ryzyka wymagajace mocnych dowodow zrodlowych.")
    return reasons


def _apply_calibration_caps(
    score: float,
    scores: dict[str, float],
    input_type: InputType,
    quality: ContentQuality,
    document: DocumentFeatures,
) -> float:
    calibrated = score

    if quality.quality_score < 0.20:
        return min(calibrated, 0.10)
    if quality.quality_score < 0.40:
        calibrated = min(calibrated, 0.25)
    if quality.high_risk_claim_count and scores["fact_score"] <= 0.45 and scores["consensus_score"] <= 0.45:
        calibrated = min(calibrated, 0.20)
    if scores["linguistic_score"] <= 0.35 and scores["claim_score"] <= 0.45:
        calibrated = min(calibrated, 0.38)
    if scores["source_score"] <= 0.30 and scores["consensus_score"] <= 0.35 and input_type in {"url", "document"}:
        calibrated = min(calibrated, 0.36)
    if scores["source_score"] <= 0.30 and scores["consensus_score"] <= 0.30 and input_type == "raw_text":
        calibrated = min(calibrated, 0.48)
    if scores["source_score"] <= 0.25 and input_type == "url":
        calibrated = min(calibrated, 0.32)
    if scores["profile_score"] <= 0.30 and input_type == "screenshot":
        calibrated = min(calibrated, 0.38)
    if scores["profile_score"] <= 0.35 and scores["text_ml_score"] <= 0.35 and input_type == "screenshot":
        calibrated = min(calibrated, 0.35)
    if scores["source_score"] >= 0.85 and scores["claim_score"] >= 0.70 and scores["ml_score"] >= 0.75:
        calibrated = max(calibrated, 0.78)
    if (
        input_type == "url"
        and scores["source_score"] >= 0.70
        and scores["claim_score"] >= 0.70
        and scores["fact_score"] >= 0.85
        and scores["consensus_score"] >= 0.80
    ):
        calibrated = max(calibrated, 0.75)
    if (
        input_type == "document"
        and scores["claim_score"] >= 0.70
        and scores["fact_score"] >= 0.85
        and scores["consensus_score"] >= 0.70
    ):
        calibrated = max(calibrated, 0.72)
    if input_type == "document" and document.scientific_document_score >= 0.70 and scores["fact_score"] >= 0.70:
        calibrated = max(calibrated, 0.76)
    if input_type == "document" and document.scientific_document_score >= 0.85 and scores["fact_score"] >= 0.80:
        calibrated = max(calibrated, 0.82)

    return _clamp(calibrated)
InputType = Literal["url", "screenshot", "document", "raw_text"]

SCORE_SPREAD_FACTOR = 1.30
SCORE_DISAGREEMENT_FREE_BAND = 0.14
SCORE_DISAGREEMENT_PENALTY_FACTOR = 0.45
MAX_SCORE_DISAGREEMENT_PENALTY = 0.12
RAW_TEXT_METADATA_WEIGHT = 0.25

SCORE_WEIGHTS: dict[InputType, dict[str, float]] = {
    "url": {
        "source_score": 0.20,
        "claim_score": 0.18,
        "ml_score": 0.27,
        "text_ml_score": 0.20,
        "consensus_score": 0.10,
        "linguistic_score": 0.05,
    },
    "screenshot": {
        "profile_score": 0.25,
        "claim_score": 0.20,
        "text_ml_score": 0.25,
        "linguistic_score": 0.10,
        "ml_score": 0.20,
    },
    "document": {
        "fact_score": 0.25,
        "claim_score": 0.18,
        "ml_score": 0.22,
        "text_ml_score": 0.18,
        "linguistic_score": 0.05,
        "source_score": 0.06,
        "consensus_score": 0.06,
    },
    "raw_text": {
        "claim_score": 0.25,
        "text_ml_score": 0.30,
        "linguistic_score": 0.10,
        "ml_score": 0.25,
        "source_score": 0.05,
        "consensus_score": 0.05,
    },
}
