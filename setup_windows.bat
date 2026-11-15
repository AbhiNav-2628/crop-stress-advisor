@echo off
REM One-time setup: creates a virtual environment and installs dependencies.
REM Usage: double-click, or run "setup_windows.bat" from a Command Prompt in this folder.
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set PYTHON=py -3) else (set PYTHON=python)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment. Is Python 3.10+ installed and on PATH?
        exit /b 1
    )
)

echo Installing dependencies (first time takes a few minutes) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: dependency installation failed. Check your internet connection and try again.
    exit /b 1
)

echo.
echo Setup complete. Start the app with:  run_app.bat
exit /b 0
