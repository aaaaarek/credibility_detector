from fastapi.testclient import TestClient

from app.api.main import app


def test_analyze_file_endpoint_accepts_txt_upload() -> None:
    client = TestClient(app)
    response = client.post(
        "/analyze/file",
        files={
            "file": (
                "article.txt",
                (
                    "The public agency published a report with data from 120 hospitals in 2026. "
                    "The article quotes experts and links to official research sources for verification."
                ),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "credibility_score" in payload
    assert payload["metadata"]["file"]["file_type"] == "txt"
