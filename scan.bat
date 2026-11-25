@echo off
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment .venv not found!
    exit /b 1
)
"%VENV_PY%" "%~dp0analyzer.py" %*
