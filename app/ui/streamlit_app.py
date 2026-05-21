from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.article_fetcher import fetch_article
from app.services.file_extractor import extract_text_from_file
from ml.inference.pipeline import ArticleInput, analyze_article, result_to_dict


def _render_result(result: dict[str, object]) -> None:
    score = float(result["credibility_score"])
    st.metric("Credibility score", f"{score:.3f}", result["credibility_level"])
    st.progress(score)

    col_1, col_2 = st.columns([1, 1])
    with col_1:
        st.subheader("Wyniki modulow")
        scores = result["module_scores"]
        st.dataframe(
            pd.DataFrame(
                [{"module": key, "score": value} for key, value in scores.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    with col_2:
        st.subheader("Uzasadnienie")
        for reason in result["reasons"]:
            st.write(f"- {reason}")

    with st.expander("Metadata i cechy"):
        st.json(result["metadata"])


st.set_page_config(page_title="Credibility Detector", layout="wide")

st.title("Credibility Detector")
st.caption("Analiza wklejonego tekstu, URL albo pliku.")

tab_text, tab_url, tab_file = st.tabs(["Tekst", "URL", "Plik"])

with tab_text:
    title = st.text_input("Tytul", placeholder="Opcjonalnie")
    content = st.text_area("Tresc artykulu", height=280, placeholder="Wklej tresc artykulu...")
    col_a, col_b = st.columns(2)
    author = col_a.text_input("Autor", placeholder="Opcjonalnie")
    publish_date = col_b.text_input("Data publikacji", placeholder="Opcjonalnie")
    links_raw = st.text_area("Linki zrodlowe", height=90, placeholder="Jeden link na linie, opcjonalnie")

    if st.button("Analizuj tekst", type="primary", use_container_width=True):
        if len(content.strip()) < 40:
            st.error("Wklej przynajmniej 40 znakow tekstu.")
        else:
            links = [line.strip() for line in links_raw.splitlines() if line.strip()]
            result = analyze_article(
                ArticleInput(
                    title=title or None,
                    content=content,
                    author=author or None,
                    publish_date=publish_date or None,
                    source_links=links,
                )
            )
            _render_result(result_to_dict(result))

with tab_url:
    url = st.text_input("URL artykulu", placeholder="https://...")
    if st.button("Pobierz i analizuj URL", type="primary", use_container_width=True):
        if not url.strip():
            st.error("Podaj URL artykulu.")
        else:
            with st.spinner("Pobieram i analizuje artykul..."):
                try:
                    fetched = fetch_article(url.strip())
                    result = analyze_article(
                        ArticleInput(
                            title=fetched.title,
                            content=fetched.content,
                            url=fetched.url,
                            author=fetched.author,
                            publish_date=fetched.publish_date,
                            source_links=fetched.source_links,
                        )
                    )
                except Exception as exc:
                    st.error(f"Nie udalo sie pobrac artykulu: {exc}")
                else:
                    st.subheader(fetched.title or "Pobrany artykul")
                    st.write(fetched.content[:1200] + ("..." if len(fetched.content) > 1200 else ""))
                    _render_result(result_to_dict(result))

with tab_file:
    uploaded_file = st.file_uploader(
        "Plik do analizy",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
    )
    if st.button("Wyciagnij tekst i analizuj plik", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("Wybierz plik PDF, DOCX, TXT albo obraz.")
        else:
            with st.spinner("Przetwarzam plik..."):
                try:
                    extracted = extract_text_from_file(uploaded_file.name, uploaded_file.getvalue())
                    result = analyze_article(
                        ArticleInput(
                            title=extracted.filename,
                            content=extracted.content,
                        )
                    )
                except Exception as exc:
                    st.error(f"Nie udalo sie przetworzyc pliku: {exc}")
                else:
                    st.subheader(extracted.filename)
                    st.caption(f"Metoda ekstrakcji: {extracted.extraction_method}")
                    st.write(extracted.content[:1200] + ("..." if len(extracted.content) > 1200 else ""))
                    rendered = result_to_dict(result)
                    rendered["metadata"]["file"] = {
                        "filename": extracted.filename,
                        "file_type": extracted.file_type,
                        "extraction_method": extracted.extraction_method,
                        "characters": len(extracted.content),
                    }
                    _render_result(rendered)
