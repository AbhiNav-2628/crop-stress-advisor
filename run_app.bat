@echo off
REM Starts the Streamlit app. Run setup_windows.bat once before the first launch.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment is missing. Run setup_windows.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py %*
