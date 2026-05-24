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
        data={
            "profile_name": "@cityoffice",
            "profile_url": "https://x.com/cityoffice",
            "is_verified": "true",
            "follower_count": "120000",
            "account_age_days": "1400",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "credibility_score" in payload
    assert payload["metadata"]["file"]["file_type"] == "txt"
    assert payload["metadata"]["profile_features"]["profile_name"] == "@cityoffice"
    assert payload["diagnostic_scores"]["profile_score"] > 0.5
