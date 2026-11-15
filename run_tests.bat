@echo off
REM Runs the automated test suite.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m pytest %*
