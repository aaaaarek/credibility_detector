from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, HttpUrl

from app.services.article_fetcher import fetch_article
from app.services.file_extractor import extract_text_from_file
from ml.inference.pipeline import ArticleInput, analyze_article, result_to_dict


app = FastAPI(
    title="Credibility Detector API",
    description="Explainable MVP for article credibility scoring from text or URL.",
    version="0.1.0",
)


class TextAnalysisRequest(BaseModel):
    title: str | None = None
    content: str = Field(..., min_length=40)
    url: HttpUrl | None = None
    author: str | None = None
    publish_date: str | None = None
    source_links: list[HttpUrl] = Field(default_factory=list)


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
        url=str(payload.url) if payload.url else None,
        author=payload.author,
        publish_date=payload.publish_date,
        source_links=[str(link) for link in payload.source_links],
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
        url=fetched.url,
        author=fetched.author,
        publish_date=fetched.publish_date,
        source_links=fetched.source_links,
    )
    return result_to_dict(analyze_article(article))


@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)) -> dict[str, object]:
    try:
        data = await file.read()
        extracted = extract_text_from_file(file.filename or "uploaded-file", data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Nie udalo sie przetworzyc pliku: {exc}") from exc

    article = ArticleInput(
        title=extracted.filename,
        content=extracted.content,
    )
    result = result_to_dict(analyze_article(article))
    result["metadata"]["file"] = {
        "filename": extracted.filename,
        "file_type": extracted.file_type,
        "extraction_method": extracted.extraction_method,
        "characters": len(extracted.content),
    }
    return result
