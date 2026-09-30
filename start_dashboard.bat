@echo off
echo ============================================================
echo  VeriTrust AI -- Hallucination Compliance Dashboard
echo ============================================================
echo.
echo Starting backend (FastAPI + uvicorn) on http://localhost:8000
echo Dashboard URL: http://localhost:8000/ui
echo API Docs:      http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server.
echo.
.venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
