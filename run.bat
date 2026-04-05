@echo off
echo =========================
echo Starting People Counter...
echo =========================

if not exist .venv\Scripts\python.exe (
    echo Environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate
python run_main.py
pause
