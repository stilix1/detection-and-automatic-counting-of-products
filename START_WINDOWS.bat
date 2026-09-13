@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Shelf Product Counter
if not exist ".venv\Scripts\python.exe" (
  echo Сначала запусти 01_SETUP_WINDOWS.bat
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\windows_launcher.py
if errorlevel 1 pause
