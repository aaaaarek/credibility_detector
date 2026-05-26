from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.article_fetcher import fetch_article
from app.services.file_extractor import extract_text_from_file
from ml.features.profile_features import ProfileInput
from ml.inference.pipeline import ArticleInput, analyze_article, result_to_dict


def _profile_controls(section_key: str) -> ProfileInput:
    with st.expander("Profil autora / konto społecznościowe"):
        col_1, col_2 = st.columns(2)
        profile_name = col_1.text_input("Nazwa profilu", key=f"{section_key}_profile_name")
        profile_url = col_2.text_input("URL profilu", key=f"{section_key}_profile_url")
        col_3, col_4, col_5 = st.columns(3)
        platform = col_3.text_input("Platforma", placeholder="np. X, Facebook, TikTok", key=f"{section_key}_platform")
        verified_raw = col_4.selectbox(
            "Zweryfikowany",
            options=["Nie wiem", "Tak", "Nie"],
            key=f"{section_key}_verified",
        )
        follower_count = col_5.text_input("Liczba obserwujących", key=f"{section_key}_followers")
        account_age_days = st.text_input("Wiek konta w dniach", key=f"{section_key}_account_age")

    return ProfileInput(
        profile_name=profile_name or None,
        profile_url=profile_url or None,
        platform=platform or None,
        is_verified={"Tak": True, "Nie": False}.get(verified_raw),
        follower_count=_optional_int(follower_count),
        account_age_days=_optional_int(account_age_days),
    )


def _optional_int(value: str) -> int | None:
    value = value.replace(" ", "").strip()
    return int(value) if value.isdigit() else None


def _score_color(score: float) -> str:
    if score >= 0.80:
        return "#16803c"
    if score >= 0.70:
        return "#2f8f4e"
    if score >= 0.60:
        return "#b7791f"
    if score >= 0.50:
        return "#b45309"
    if score >= 0.40:
        return "#c2410c"
    return "#b91c1c"


def _render_result(result: dict[str, object]) -> None:
    score = float(result["credibility_score"])
    color = _score_color(score)
    st.metric("Credibility score", f"{score:.3f}")
    st.markdown(
        f"""
        <div style="
            color: {color};
            font-size: 2.25rem;
            line-height: 1.15;
            font-weight: 800;
            letter-spacing: 0;
            margin-top: -0.35rem;
            margin-bottom: 1rem;
            overflow-wrap: anywhere;
        ">
            {result["credibility_level"]}
        </div>
        """,
        unsafe_allow_html=True,
    )
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

    if result.get("diagnostic_scores"):
        with st.expander("Wyniki diagnostyczne"):
            st.dataframe(
                pd.DataFrame(
                    [{"module": key, "score": value} for key, value in result["diagnostic_scores"].items()]
                ),
                hide_index=True,
                use_container_width=True,
            )


st.set_page_config(page_title="Credibility Detector", layout="wide")

st.title("Credibility Detector")
st.caption("Analiza wklejonego tekstu, URL albo pliku.")

tab_text, tab_url, tab_file = st.tabs(["Tekst", "URL", "Plik"])

with tab_text:
    title = st.text_input("Tytul", placeholder="Opcjonalnie")
    text_mode = st.selectbox(
        "Rodzaj tekstu",
        options=["Wklejony tekst", "Tekst z posta/screenshotu"],
    )
    input_type = "screenshot" if text_mode == "Tekst z posta/screenshotu" else "raw_text"
    content = st.text_area("Tresc artykulu", height=280, placeholder="Wklej tresc artykulu...")
    source_url = st.text_input("URL zrodla", placeholder="Opcjonalnie, np. https://...")
    col_a, col_b = st.columns(2)
    author = col_a.text_input("Autor", placeholder="Opcjonalnie")
    publish_date = col_b.text_input("Data publikacji", placeholder="Opcjonalnie")
    links_raw = st.text_area("Linki zrodlowe", height=90, placeholder="Jeden link na linie, opcjonalnie")
    text_profile = _profile_controls("text") if input_type == "screenshot" else ProfileInput()

    if st.button("Analizuj tekst", type="primary", use_container_width=True):
        if len(content.strip()) < 40:
            st.error("Wklej przynajmniej 40 znakow tekstu.")
        else:
            links = [line.strip() for line in links_raw.splitlines() if line.strip()]
            source_url_clean = source_url.strip() or None
            analysis_input_type = input_type
            analysis_title = title or None
            analysis_author = author or None
            analysis_publish_date = publish_date or None
            analysis_links = links

            if source_url_clean and input_type == "raw_text":
                try:
                    fetched_context = fetch_article(source_url_clean)
                except Exception as exc:
                    st.warning(f"Nie udalo sie pobrac metadanych URL zrodla: {exc}")
                else:
                    analysis_input_type = "url"
                    analysis_title = analysis_title or fetched_context.title
                    analysis_author = analysis_author or fetched_context.author
                    analysis_publish_date = analysis_publish_date or fetched_context.publish_date
                    if not analysis_links:
                        analysis_links = fetched_context.source_links

            result = analyze_article(
                ArticleInput(
                    title=analysis_title,
                    content=content,
                    input_type=analysis_input_type,
                    url=source_url_clean,
                    author=analysis_author,
                    publish_date=analysis_publish_date,
                    source_links=analysis_links,
                    profile=text_profile,
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
                            input_type="url",
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
    file_profile = _profile_controls("file")
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
                            input_type="screenshot"
                            if extracted.file_type in {"png", "jpg", "jpeg", "webp"}
                            else "document",
                            profile=file_profile,
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
