# AskAI

AskAI is a local web application for CV-informed text interviews. Candidates can practice for a chosen role or use a recruiter-created interview link. Recruiters can configure a rubric and review completed assessments from a separate private dashboard link.

The application uses a React and TypeScript frontend, a FastAPI backend, SQLite, PDF text extraction with `pdfplumber`, and Gemini for structured CV analysis, question generation, answer evaluation, and final reports.

## Design preview

- [Editable Figma design](https://www.figma.com/design/4CUq6F9G0TNCzi9uazqTFC)
- [Desktop candidate prototype](https://www.figma.com/proto/4CUq6F9G0TNCzi9uazqTFC/AskAI?node-id=1-2)
- [Mobile candidate prototype](https://www.figma.com/proto/4CUq6F9G0TNCzi9uazqTFC/AskAI?node-id=2-2)
- [Desktop recruiter prototype](https://www.figma.com/proto/4CUq6F9G0TNCzi9uazqTFC/AskAI?node-id=1-38)
- [Mobile recruiter prototype](https://www.figma.com/proto/4CUq6F9G0TNCzi9uazqTFC/AskAI?node-id=2-69)
- [Desktop screen preview](docs/figma-desktop-preview.png) and [mobile screen preview](docs/figma-mobile-preview.png)

The Figma file contains candidate and recruiter screens at desktop and mobile sizes, linked prototype actions, and a small editable UI kit. The web app follows the same visual direction.

## Run locally

Requires Python 3.10+, Node.js 20+, and a Gemini API key. From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. Then start the backend and frontend in separate terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The frontend proxies `/api` requests to the backend. API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

For a local walkthrough without an API key, set `ASKAI_DEMO_MODE=true` in `.env`. The interface clearly labels this mode; generated analysis and scores are illustrative and should not be used for hiring decisions.

## Main flows

1. A candidate chooses a role or opens a recruiter link, enters a name, and uploads a text-based PDF CV (maximum 10 MB).
2. The backend extracts text in memory, deletes the uploaded file data after processing, and asks Gemini for a structured CV analysis.
3. The interview asks custom, role-coverage, and follow-up questions. It adjusts difficulty using previous answers and saves progress after every submission.
4. The candidate sees coaching and answer-level feedback at the end. A recruiter sees a separate assessment, evidence, and rubric scores through a private dashboard link.

Recruiter links work only while the local server is reachable. There are no accounts in this demo. Treat candidate session URLs and recruiter dashboard URLs as private access links. The database is stored at `backend/askai.sqlite3` by default and is excluded from Git.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm run build
```

The backend tests cover an entire recruiter-link interview, report access separation, score persistence, invalid PDF input, and missing Gemini configuration.

## Legacy script

`main.py` is the original command-line experiment and remains in the repository for reference. The web application runs through `backend.app` and `frontend/`.
