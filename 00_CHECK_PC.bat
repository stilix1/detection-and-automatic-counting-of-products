@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === NVIDIA ===
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo nvidia-smi не найден.
) else (
  nvidia-smi
)
echo.
echo === PYTHON ===
py -3.11 --version 2>nul
if errorlevel 1 echo Python 3.11 через py launcher не найден.
echo.
pause
