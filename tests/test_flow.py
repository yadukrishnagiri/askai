import io
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app


def client_for_demo(monkeypatch, tmp_path):
    monkeypatch.setenv("ASKAI_DEMO_MODE", "true")
    monkeypatch.setenv("ASKAI_DB_PATH", str(tmp_path / "test.sqlite3"))
    return TestClient(app)


def sample_cv():
    path = Path(__file__).resolve().parents[1] / "data" / "yadukrishnagiri (3).pdf"
    return io.BytesIO(path.read_bytes())


def test_recruiter_link_full_interview_and_access_separation(monkeypatch, tmp_path):
    with client_for_demo(monkeypatch, tmp_path) as client:
        response = client.post("/api/configurations", json={
            "role": "Product Designer",
            "skills": ["Research", "Collaboration"],
            "question_count": 3,
            "custom_questions": ["Describe a research decision you changed."],
        })
        assert response.status_code == 201
        candidate_url = response.json()["candidate_url"]
        dashboard_url = response.json()["dashboard_url"]
        candidate_token = candidate_url.rsplit("/", 1)[-1]
        dashboard_token = dashboard_url.rsplit("/", 1)[-1]
        assert client.get(f"/api/configurations/{candidate_token}").json()["settings"]["role"] == "Product Designer"

        response = client.post("/api/sessions", data={
            "candidate_name": "Alex Morgan", "role": "Ignored", "configuration_token": candidate_token,
        }, files={"cv": ("resume.pdf", sample_cv(), "application/pdf")})
        assert response.status_code == 201, response.text
        session = response.json()
        assert session["role"] == "Product Designer"
        assert session["current_question"]["kind"] == "custom"
        access_token = session["token"]

        for position in range(3):
            response = client.post(f"/api/sessions/{access_token}/answers", json={
                "answer": "I researched the problem with users, compared options, made a tradeoff, and measured the outcome with the team."
            })
            assert response.status_code == 200, response.text
            session = response.json()
            assert session["answered_count"] == position + 1
            if position < 2:
                assert "evaluation" not in session["turns"][-1]
                assert session["status"] == "active"
        assert session["status"] == "complete"
        assert session["report"]["next_steps"]
        assert session["score"] > 0
        assert all("evaluation" in turn for turn in session["turns"])
        assert client.get(f"/api/sessions/{access_token}").json()["status"] == "complete"
        assert client.post(f"/api/sessions/{access_token}/answers", json={"answer": "again"}).status_code == 409

        dashboard = client.get(f"/api/dashboard/{dashboard_token}")
        assert dashboard.status_code == 200
        assert dashboard.json()["sessions"][0]["score"] == session["score"]
        session_id = dashboard.json()["sessions"][0]["id"]
        recruiter = client.get(f"/api/dashboard/{dashboard_token}/sessions/{session_id}")
        assert recruiter.status_code == 200
        assert recruiter.json()["report"]["evidence"]
        assert client.get(f"/api/dashboard/{candidate_token}").status_code == 404
        assert client.get(f"/api/dashboard/bad-token/sessions/{session_id}").status_code == 404


def test_invalid_inputs_and_missing_gemini(monkeypatch, tmp_path):
    with client_for_demo(monkeypatch, tmp_path) as client:
        invalid_weights = client.post("/api/configurations", json={"role": "Designer", "technical_weight": 70})
        assert invalid_weights.status_code == 422
        invalid_pdf = client.post("/api/sessions", data={"candidate_name": "Alex Morgan", "role": "Designer"}, files={
            "cv": ("resume.pdf", b"not a pdf", "application/pdf")
        })
        assert invalid_pdf.status_code == 422
        assert client.get("/api/configurations/missing").status_code == 404

    monkeypatch.setenv("ASKAI_DEMO_MODE", "false")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.post("/api/sessions", data={"candidate_name": "Alex Morgan", "role": "Designer"}, files={
            "cv": ("resume.pdf", sample_cv(), "application/pdf")
        })
        assert response.status_code == 503
        assert "GEMINI_API_KEY" in response.json()["detail"]
