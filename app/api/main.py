from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, HttpUrl

from app.services.article_fetcher import fetch_article
from app.services.file_extractor import extract_text_from_file
from ml.features.profile_features import ProfileInput
from ml.inference.pipeline import ArticleInput, InputType, analyze_article, result_to_dict


app = FastAPI(
    title="Credibility Detector API",
    description="Explainable MVP for article credibility scoring from text or URL.",
    version="0.1.0",
)


class TextAnalysisRequest(BaseModel):
    title: str | None = None
    content: str = Field(..., min_length=40)
    input_type: InputType = "raw_text"
    url: HttpUrl | None = None
    author: str | None = None
    publish_date: str | None = None
    source_links: list[HttpUrl] = Field(default_factory=list)
    profile_name: str | None = None
    profile_url: HttpUrl | None = None
    platform: str | None = None
    is_verified: bool | None = None
    follower_count: int | None = Field(default=None, ge=0)
    account_age_days: int | None = Field(default=None, ge=0)


class UrlAnalysisRequest(BaseModel):
    url: HttpUrl


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze/text")
def analyze_text(payload: TextAnalysisRequest) -> dict[str, object]:
    article = ArticleInput(
        title=payload.title,
        content=payload.content,
        input_type=payload.input_type,
        url=str(payload.url) if payload.url else None,
        author=payload.author,
        publish_date=payload.publish_date,
        source_links=[str(link) for link in payload.source_links],
        profile=ProfileInput(
            profile_name=payload.profile_name,
            profile_url=str(payload.profile_url) if payload.profile_url else None,
            platform=payload.platform,
            is_verified=payload.is_verified,
            follower_count=payload.follower_count,
            account_age_days=payload.account_age_days,
        ),
    )
    return result_to_dict(analyze_article(article))


@app.post("/analyze/url")
def analyze_url(payload: UrlAnalysisRequest) -> dict[str, object]:
    try:
        fetched = fetch_article(str(payload.url))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Nie udalo sie pobrac artykulu: {exc}") from exc

    article = ArticleInput(
        title=fetched.title,
        content=fetched.content,
        input_type="url",
        url=fetched.url,
        author=fetched.author,
        publish_date=fetched.publish_date,
        source_links=fetched.source_links,
    )
    return result_to_dict(analyze_article(article))


@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    profile_name: str | None = Form(default=None),
    profile_url: str | None = Form(default=None),
    platform: str | None = Form(default=None),
    is_verified: bool | None = Form(default=None),
    follower_count: int | None = Form(default=None),
    account_age_days: int | None = Form(default=None),
) -> dict[str, object]:
    try:
        data = await file.read()
        extracted = extract_text_from_file(file.filename or "uploaded-file", data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Nie udalo sie przetworzyc pliku: {exc}") from exc

    article = ArticleInput(
        title=extracted.filename,
        content=extracted.content,
        input_type="screenshot" if extracted.file_type in {"png", "jpg", "jpeg", "webp"} else "document",
        profile=ProfileInput(
            profile_name=profile_name,
            profile_url=profile_url,
            platform=platform,
            is_verified=is_verified,
            follower_count=follower_count,
            account_age_days=account_age_days,
        ),
    )
    result = result_to_dict(analyze_article(article))
    result["metadata"]["file"] = {
        "filename": extracted.filename,
        "file_type": extracted.file_type,
        "extraction_method": extracted.extraction_method,
        "characters": len(extracted.content),
    }
    return result
