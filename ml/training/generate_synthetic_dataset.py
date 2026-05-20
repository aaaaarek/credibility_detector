from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "datasets" / "synthetic_articles.csv"


@dataclass(frozen=True)
class Template:
    category: str
    source: str
    url_prefix: str
    author: str
    links: str
    label: float
    title_pl: str
    content_pl: str
    title_en: str
    content_en: str


TOPICS_PL = [
    ("inflacji", "cenach żywności", "3,1 procent"),
    ("bezrobociu", "rynku pracy", "4,8 procent"),
    ("jakości wody", "badaniach laboratoryjnych", "24 punkty pomiarowe"),
    ("transporcie publicznym", "opóźnieniach autobusów", "18 milionów godzin"),
    ("zdrowiu dzieci", "programie szczepień", "12 przychodni"),
    ("cenach mieszkań", "rynku nieruchomości", "2,4 procent"),
    ("powodzi", "ryzyku hydrologicznym", "38 miast"),
    ("edukacji", "braku nauczycieli", "600 szkół"),
    ("energii", "rachunkach za prąd", "1,2 miliona PLN"),
    ("sądzie", "wyroku w sprawie przetargu", "sygnatura akt"),
]

TOPICS_EN = [
    ("inflation", "food prices", "3.1 percent"),
    ("unemployment", "labour market", "4.8 percent"),
    ("water quality", "laboratory testing", "24 testing points"),
    ("public transport", "bus delays", "18 million hours"),
    ("child health", "vaccination schedule", "12 clinics"),
    ("housing prices", "property market", "2.4 percent"),
    ("flood risk", "hydrology model", "38 cities"),
    ("education", "teacher shortages", "600 schools"),
    ("energy", "electricity bills", "1.2 million PLN"),
    ("court ruling", "procurement case", "case number"),
]

TEMPLATES = [
    Template(
        category="high_news",
        source="News agency",
        url_prefix="https://pap.pl/report",
        author="Agency desk",
        links="https://gov.pl/report|https://stat.gov.pl/data",
        label=0.88,
        title_pl="Agencja publikuje raport o {topic}",
        content_pl=(
            "Agencja informacyjna opisała raport o {topic} i {subject}. Tekst podaje {metric}, datę publikacji "
            "oraz link do dokumentu źródłowego. Cytuje nazwanych ekspertów, oddziela komentarze od faktów i "
            "wyjaśnia ograniczenia danych."
        ),
        title_en="Agency report on {topic}",
        content_en=(
            "A news agency summarised a report on {topic} and {subject}. The article gives {metric}, publication "
            "date and a link to the source document. It quotes named experts, separates comments from facts and "
            "explains data limitations."
        ),
    ),
    Template(
        category="science",
        source="Science journal",
        url_prefix="https://science.org/article",
        author="Science desk",
        links="https://science.org/paper|https://data.example.org/dataset",
        label=0.86,
        title_pl="Badacze analizują dane o {topic}",
        content_pl=(
            "Zespół badaczy przeanalizował dane o {topic} i {subject}. Artykuł opisuje metodologię, próbę badawczą "
            "oraz {metric}. Linkuje do publikacji i zbioru danych, a niezależny ekspert komentuje ograniczenia badania."
        ),
        title_en="Researchers analyse {topic} data",
        content_en=(
            "A research team analysed data on {topic} and {subject}. The article describes methodology, sample size "
            "and {metric}. It links to the paper and dataset, while an independent expert comments on limitations."
        ),
    ),
    Template(
        category="local_news",
        source="Local news",
        url_prefix="https://localnews.example.com/story",
        author="Local desk",
        links="https://city.example.gov/notice",
        label=0.72,
        title_pl="Lokalny urząd informuje o {topic}",
        content_pl=(
            "Lokalny urząd opublikował komunikat o {topic} i {subject}. Tekst podaje daty, adresy, {metric} oraz "
            "link do oficjalnej strony. Brakuje szerszej analizy, ale podstawowe informacje są jawne i sprawdzalne."
        ),
        title_en="Local office announces {topic} update",
        content_en=(
            "A local office published a notice about {topic} and {subject}. The text gives dates, addresses, {metric} "
            "and a link to the official page. It lacks deeper analysis, but the basic facts are visible and checkable."
        ),
    ),
    Template(
        category="opinion",
        source="Opinion site",
        url_prefix="https://opinion.example.com/column",
        author="Opinion columnist",
        links="https://public.example.org/report",
        label=0.62,
        title_pl="Komentarz o {topic}",
        content_pl=(
            "Autor komentuje spór o {topic} i {subject}. Tekst jest oznaczony jako opinia, przywołuje publiczny "
            "raport i podaje {metric}. Zawiera kontrargumenty, ale nie jest neutralnym raportem informacyjnym."
        ),
        title_en="Opinion on {topic}",
        content_en=(
            "The author comments on the debate about {topic} and {subject}. The text is labelled as opinion, cites "
            "a public report and gives {metric}. It includes counterarguments, but it is not neutral reporting."
        ),
    ),
    Template(
        category="blog",
        source="Personal blog",
        url_prefix="https://blog.example.com/post",
        author="Blog author",
        links="",
        label=0.54,
        title_pl="Blogowy opis doświadczeń z {topic}",
        content_pl=(
            "Autor bloga opisuje własne doświadczenia z {topic} i {subject}. Podaje kilka liczb, w tym {metric}, "
            "ale zaznacza, że to przykład jednostkowy. Brakuje niezależnych danych i szerszej próby porównawczej."
        ),
        title_en="Blog experience with {topic}",
        content_en=(
            "The blog author describes personal experience with {topic} and {subject}. The text gives a few numbers, "
            "including {metric}, but says it is a single case. It lacks independent data and a broader comparison sample."
        ),
    ),
    Template(
        category="press_release",
        source="Industry portal",
        url_prefix="https://industry.example.com/press",
        author="Sponsored desk",
        links="https://company.example.com/press",
        label=0.42,
        title_pl="Firma ogłasza przełom w sprawie {topic}",
        content_pl=(
            "Firma twierdzi, że osiągnęła przełom w obszarze {topic} i {subject}. Artykuł powtarza komunikat prasowy "
            "i podaje {metric}, ale nie linkuje do niezależnego audytu ani recenzowanego badania."
        ),
        title_en="Company announces breakthrough in {topic}",
        content_en=(
            "A company claims a breakthrough in {topic} and {subject}. The article repeats a press release and gives "
            "{metric}, but it does not link to an independent audit or peer reviewed study."
        ),
    ),
    Template(
        category="rumor",
        source="Rumor site",
        url_prefix="http://rumor.example.com/story",
        author="",
        links="",
        label=0.25,
        title_pl="Anonimowe źródła mówią o {topic}",
        content_pl=(
            "Strona twierdzi, że anonimowe źródła ujawniły prawdę o {topic} i {subject}. Nie ma dokumentów, dat ani "
            "nazwanych ekspertów. Tekst używa sugestywnych pytań i wymaga potwierdzenia w innych źródłach."
        ),
        title_en="Anonymous sources claim {topic} story",
        content_en=(
            "The site claims anonymous sources revealed the truth about {topic} and {subject}. It gives no documents, "
            "dates or named experts. The text uses suggestive questions and requires confirmation elsewhere."
        ),
    ),
    Template(
        category="conspiracy",
        source="Conspiracy site",
        url_prefix="http://uncensoredtruth.example.com/post",
        author="",
        links="",
        label=0.08,
        title_pl="PILNE: ukrywają prawdę o {topic}",
        content_pl=(
            "PILNE! Strona twierdzi, że elity ukrywają prawdę o {topic} i {subject}. Nie pokazuje raportów, danych "
            "ani nazwisk ekspertów. Tekst straszy czytelnika i każe udostępniać zanim zostanie usunięty."
        ),
        title_en="URGENT: they hide the truth about {topic}",
        content_en=(
            "URGENT! The page claims elites hide the truth about {topic} and {subject}. It shows no reports, data or "
            "named experts. The text frightens readers and asks them to share before it is deleted."
        ),
    ),
    Template(
        category="scam",
        source="Viral sales page",
        url_prefix="http://viral-secret.example.com/offer",
        author="",
        links="http://shop.example.com/product",
        label=0.10,
        title_pl="Cudowny sposób na {topic}",
        content_pl=(
            "Strona obiecuje cudowny sposób na {topic} i {subject}. Mówi o gwarantowanym efekcie, podaje {metric} "
            "bez źródła i prowadzi do sklepu. Brakuje badań, autora i daty publikacji."
        ),
        title_en="Miracle trick for {topic}",
        content_en=(
            "The page promises a miracle trick for {topic} and {subject}. It claims a guaranteed effect, gives {metric} "
            "without a source and sends readers to a shop. It lacks studies, author and publication date."
        ),
    ),
    Template(
        category="social_thread",
        source="Social media thread",
        url_prefix="http://social-rumor.example.com/thread",
        author="",
        links="",
        label=0.18,
        title_pl="Wątek viralowy o {topic}",
        content_pl=(
            "Wątek viralowy twierdzi, że media milczą o {topic} i {subject}. Pokazuje screeny bez plików źródłowych "
            "i nie linkuje do danych. Autor prosi o natychmiastowe udostępnianie."
        ),
        title_en="Viral thread about {topic}",
        content_en=(
            "A viral thread claims the media are silent about {topic} and {subject}. It shows screenshots without "
            "source files and links to no data. The author asks followers to repost immediately."
        ),
    ),
]


def main() -> None:
    rows = []
    for idx in range(200):
        template = TEMPLATES[idx % len(TEMPLATES)]
        topic_set = TOPICS_PL if idx % 2 == 0 else TOPICS_EN
        topic, subject, metric = topic_set[(idx // len(TEMPLATES)) % len(topic_set)]
        is_polish = idx % 2 == 0
        title_template = template.title_pl if is_polish else template.title_en
        content_template = template.content_pl if is_polish else template.content_en
        rows.append(
            {
                "title": title_template.format(topic=topic, subject=subject, metric=metric),
                "content": content_template.format(topic=topic, subject=subject, metric=metric),
                "url": f"{template.url_prefix}-{idx + 1}",
                "source": template.source,
                "author": template.author,
                "publish_date": f"2026-{(idx % 5) + 1:02d}-{(idx % 27) + 1:02d}" if template.author else "",
                "source_links": template.links,
                "credibility_label": round(_jitter(template.label, idx // len(TEMPLATES)), 2),
            }
        )

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "content",
                "url",
                "source",
                "author",
                "publish_date",
                "source_links",
                "credibility_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} rows at {DATASET_PATH}")


def _jitter(label: float, idx: int) -> float:
    offsets = [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    return max(0.02, min(0.98, label + offsets[idx % len(offsets)]))


if __name__ == "__main__":
    main()
