@echo off
setlocal
cd /d "%~dp0"
title »·¾³×Ô¼ì

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=py -3.11"
)
if not defined PY set "PY=python"

%PY% check_env.py
echo.
pause
