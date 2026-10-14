from __future__ import annotations

import io
import json
import secrets
import uuid
from contextlib import asynccontextmanager

import pdfplumber
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .ai import AIUnavailable, InterviewAI
from .db import connect, initialize, now, obj, unpack_session
from .models import AnswerRequest, CVAnalysis, Evaluation, InterviewConfig, Question


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize()
    app.state.ai = InterviewAI()
    yield


app = FastAPI(title="AskAI API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def model_or_503(call):
    try:
        return call()
    except AIUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def configuration_from_row(row) -> InterviewConfig:
    return InterviewConfig.model_validate_json(row["settings_json"])


def get_configuration(candidate_token: str):
    with connect() as db:
        row = db.execute("SELECT * FROM configurations WHERE candidate_token = ?", (candidate_token,)).fetchone()
    if row is None:
        raise HTTPException(404, "Interview link not found")
    return row


def get_dashboard_configuration(dashboard_token: str):
    with connect() as db:
        row = db.execute("SELECT * FROM configurations WHERE dashboard_token = ?", (dashboard_token,)).fetchone()
    if row is None:
        raise HTTPException(404, "Dashboard link not found")
    return row


def get_session(access_token: str):
    with connect() as db:
        row = db.execute("SELECT * FROM sessions WHERE access_token = ?", (access_token,)).fetchone()
        if row is None:
            raise HTTPException(404, "Interview session not found")
        turns = [dict(turn) for turn in db.execute("SELECT * FROM turns WHERE session_id = ? ORDER BY position", (row["id"],))]
    return unpack_session(row), decode_turns(turns)


def decode_turns(rows):
    return [
        {
            "position": row["position"],
            "question": json.loads(row["question_json"]),
            "answer": row["answer"],
            "evaluation": json.loads(row["evaluation_json"]),
        }
        for row in rows
    ]


def score_turns(turns: list[dict], settings: InterviewConfig) -> float:
    if not turns:
        return 0.0
    scores = []
    for turn in turns:
        evaluation = Evaluation.model_validate(turn["evaluation"])
        weighted = (
            evaluation.technical_accuracy * settings.technical_weight
            + evaluation.communication * settings.communication_weight
            + evaluation.evidence * settings.evidence_weight
        ) / 100
        scores.append(weighted)
    return round(sum(scores) / len(scores), 1)


def next_difficulty(current: str, adjustment: str) -> str:
    levels = ["basic", "intermediate", "advanced"]
    delta = 1 if adjustment == "harder" else -1 if adjustment == "easier" else 0
    return levels[max(0, min(2, levels.index(current) + delta))]


def pick_next_question(ai: InterviewAI, settings: InterviewConfig, analysis: CVAnalysis, turns: list[dict], difficulty: str) -> Question:
    used_custom = sum(turn["question"]["kind"] == "custom" for turn in turns)
    if used_custom < len(settings.custom_questions) and (not turns or len(turns) % 3 == 0):
        return ai.question(settings, analysis, turns, difficulty, "custom", settings.custom_questions[used_custom])
    if not turns:
        return ai.question(settings, analysis, turns, difficulty, "opening", settings.skills[0] if settings.skills else "project experience")
    previous = turns[-1]
    evaluation = previous["evaluation"]
    if evaluation.get("followup_worthwhile") and previous["question"]["kind"] != "followup" and len(turns) % 3 != 0:
        return ai.question(settings, analysis, turns, difficulty, "followup", evaluation.get("suggested_followup", ""))
    covered = {turn["question"]["topic"].lower() for turn in turns}
    target = next((skill for skill in settings.skills if skill.lower() not in covered), "project outcomes")
    return ai.question(settings, analysis, turns, difficulty, "coverage", target)


def candidate_session_view(session: dict, turns: list[dict]):
    settings = InterviewConfig.model_validate(session["settings"])
    complete = session["status"] == "complete"
    view = {
        "token": session["access_token"],
        "candidate_name": session["candidate_name"],
        "role": session["role"],
        "status": session["status"],
        "question_count": settings.question_count,
        "answered_count": len(turns),
        "difficulty": session["difficulty"],
        "cv_analysis": session["cv_analysis"],
        "current_question": session["current_question"],
        "turns": [
            {"position": turn["position"], "question": turn["question"], "answer": turn["answer"], **({"evaluation": turn["evaluation"]} if complete else {})}
            for turn in turns
        ],
    }
    if complete:
        view["score"] = score_turns(turns, settings)
        view["report"] = session["report"]["candidate"]
    return view


@app.get("/api/health")
def health():
    return {"status": "ok", "demo_mode": app.state.ai.demo, "gemini_ready": bool(app.state.ai.client)}


@app.post("/api/configurations", status_code=201)
def create_configuration(settings: InterviewConfig):
    configuration_id = str(uuid.uuid4())
    candidate_token = secrets.token_urlsafe(24)
    dashboard_token = secrets.token_urlsafe(32)
    with connect() as db:
        db.execute(
            "INSERT INTO configurations VALUES (?, ?, ?, ?, ?)",
            (configuration_id, candidate_token, dashboard_token, settings.model_dump_json(), now()),
        )
    return {
        "settings": settings,
        "candidate_url": f"/interview/{candidate_token}",
        "dashboard_url": f"/dashboard/{dashboard_token}",
    }


@app.get("/api/configurations/{candidate_token}")
def read_configuration(candidate_token: str):
    row = get_configuration(candidate_token)
    return {"settings": configuration_from_row(row)}


@app.post("/api/sessions", status_code=201)
async def start_session(
    candidate_name: str = Form(...),
    role: str = Form(""),
    configuration_token: str = Form(""),
    cv: UploadFile = File(...),
):
    candidate_name = candidate_name.strip()
    if not 2 <= len(candidate_name) <= 100:
        raise HTTPException(422, "Enter a name between 2 and 100 characters")
    configuration = get_configuration(configuration_token) if configuration_token else None
    if configuration:
        settings = configuration_from_row(configuration)
    else:
        try:
            settings = InterviewConfig(role=role.strip())
        except ValidationError as exc:
            raise HTTPException(422, "Enter a valid role") from exc
    if not cv.filename or not cv.filename.lower().endswith(".pdf"):
        raise HTTPException(422, "Upload a PDF CV")
    pdf_bytes = await cv.read(10 * 1024 * 1024 + 1)
    await cv.close()
    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "PDF must be 10 MB or smaller")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(422, "The uploaded file is not a valid PDF")
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            cv_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        raise HTTPException(422, "Could not read this PDF") from exc
    if len(cv_text.strip()) < 40:
        raise HTTPException(422, "This PDF has too little selectable text. Please upload a text-based CV")
    analysis = model_or_503(lambda: app.state.ai.analyze(cv_text, settings))
    question = model_or_503(lambda: pick_next_question(app.state.ai, settings, analysis, [], settings.starting_difficulty))
    session_id = str(uuid.uuid4())
    access_token = secrets.token_urlsafe(32)
    timestamp = now()
    with connect() as db:
        db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                access_token,
                configuration["id"] if configuration else None,
                candidate_name,
                settings.role,
                settings.model_dump_json(),
                analysis.model_dump_json(),
                question.model_dump_json(),
                settings.starting_difficulty,
                "active",
                None,
                timestamp,
                timestamp,
            ),
        )
    session, turns = get_session(access_token)
    return candidate_session_view(session, turns)


@app.get("/api/sessions/{access_token}")
def read_session(access_token: str):
    session, turns = get_session(access_token)
    return candidate_session_view(session, turns)


@app.post("/api/sessions/{access_token}/answers")
def submit_answer(access_token: str, body: AnswerRequest):
    session, turns = get_session(access_token)
    if session["status"] != "active" or not session["current_question"]:
        raise HTTPException(409, "This interview is already complete")
    answer = body.answer.strip()
    if not answer:
        raise HTTPException(422, "Please enter an answer")
    settings = InterviewConfig.model_validate(session["settings"])
    analysis = CVAnalysis.model_validate(session["cv_analysis"])
    question = Question.model_validate(session["current_question"])
    evaluation = model_or_503(lambda: app.state.ai.evaluate(settings, question, answer, analysis))
    next_turns = turns + [{"position": len(turns) + 1, "question": question.model_dump(), "answer": answer, "evaluation": evaluation.model_dump()}]
    difficulty = next_difficulty(session["difficulty"], evaluation.adjust_difficulty)
    complete = len(next_turns) >= settings.question_count
    if complete:
        report = model_or_503(lambda: app.state.ai.report(settings, analysis, next_turns))
        next_question = None
    else:
        next_question = model_or_503(lambda: pick_next_question(app.state.ai, settings, analysis, next_turns, difficulty))
        report = None
    with connect() as db:
        updated = db.execute(
            "UPDATE sessions SET current_question_json = ?, difficulty = ?, status = ?, report_json = ?, updated_at = ? WHERE id = ? AND current_question_json = ? AND updated_at = ? AND status = 'active'",
            (
                next_question.model_dump_json() if next_question else None,
                difficulty,
                "complete" if complete else "active",
                report.model_dump_json() if report else None,
                now(),
                session["id"],
                question.model_dump_json(),
                session["updated_at"],
            ),
        )
        if updated.rowcount != 1:
            raise HTTPException(409, "This question was already answered. Refresh to continue")
        db.execute(
            "INSERT INTO turns (session_id, position, question_json, answer, evaluation_json) VALUES (?, ?, ?, ?, ?)",
            (session["id"], len(next_turns), question.model_dump_json(), answer, evaluation.model_dump_json()),
        )
    refreshed, refreshed_turns = get_session(access_token)
    return candidate_session_view(refreshed, refreshed_turns)


@app.get("/api/dashboard/{dashboard_token}")
def dashboard(dashboard_token: str):
    configuration = get_dashboard_configuration(dashboard_token)
    with connect() as db:
        rows = db.execute(
            "SELECT id, candidate_name, role, status, updated_at, settings_json FROM sessions WHERE configuration_id = ? ORDER BY updated_at DESC",
            (configuration["id"],),
        ).fetchall()
        sessions = []
        for row in rows:
            turns = decode_turns([dict(turn) for turn in db.execute("SELECT * FROM turns WHERE session_id = ? ORDER BY position", (row["id"],))])
            settings = InterviewConfig.model_validate_json(row["settings_json"])
            sessions.append({
                "id": row["id"], "candidate_name": row["candidate_name"], "role": row["role"],
                "status": row["status"], "updated_at": row["updated_at"],
                "answered_count": len(turns), "question_count": settings.question_count,
                "score": score_turns(turns, settings) if row["status"] == "complete" else None,
            })
    return {"settings": configuration_from_row(configuration), "candidate_url": f"/interview/{configuration['candidate_token']}", "sessions": sessions}


@app.get("/api/dashboard/{dashboard_token}/sessions/{session_id}")
def recruiter_session(dashboard_token: str, session_id: str):
    configuration = get_dashboard_configuration(dashboard_token)
    with connect() as db:
        row = db.execute("SELECT * FROM sessions WHERE id = ? AND configuration_id = ?", (session_id, configuration["id"])).fetchone()
        if row is None:
            raise HTTPException(404, "Report not found")
        turns = decode_turns([dict(turn) for turn in db.execute("SELECT * FROM turns WHERE session_id = ? ORDER BY position", (session_id,))])
    session = unpack_session(row)
    settings = InterviewConfig.model_validate(session["settings"])
    return {
        "candidate_name": session["candidate_name"],
        "role": session["role"],
        "status": session["status"],
        "score": score_turns(turns, settings) if session["status"] == "complete" else None,
        "report": session["report"]["recruiter"] if session["report"] else None,
        "turns": turns,
        "cv_analysis": session["cv_analysis"],
    }
