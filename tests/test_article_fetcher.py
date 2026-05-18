import requests

from app.services.article_fetcher import _decode_html


def test_decode_html_keeps_polish_utf8_characters() -> None:
    response = requests.Response()
    response._content = (
        "<html><head><title>Policjant spod Częstochowy</title></head>"
        "<body>Wyjaśniamy, dlaczego to możliwe.</body></html>"
    ).encode("utf-8")
    response.encoding = "ISO-8859-1"

    html = _decode_html(response)

    assert "Częstochowy" in html
    assert "Wyjaśniamy" in html
    assert "możliwe" in html
