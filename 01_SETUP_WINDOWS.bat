@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Shelf Product Counter - Setup

echo ===============================================================
echo  УСТАНОВКА ПОД WINDOWS + NVIDIA CUDA
echo ===============================================================
echo.

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo ОШИБКА: nvidia-smi не найден.
  pause
  exit /b 1
)

py -3.11 --version >nul 2>nul
if errorlevel 1 (
  echo ОШИБКА: не найден Python 3.11 x64.
  pause
  exit /b 1
)

echo [1/5] Создание .venv на Python 3.11...
if exist .venv rmdir /s /q .venv
py -3.11 -m venv .venv
if errorlevel 1 goto :fail

set "PY=%CD%\.venv\Scripts\python.exe"
echo [2/5] Обновление pip...
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo [3/5] Установка PyTorch 2.11 + CUDA 12.8...
"%PY%" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :fail

echo [4/5] Установка библиотек проекта...
"%PY%" -m pip install -r requirements-windows.txt
if errorlevel 1 goto :fail

echo [5/5] Проверка RTX и CUDA...
"%PY%" tools\check_environment.py
if errorlevel 1 goto :fail

echo.
echo ===============================================================
echo  ГОТОВО. Теперь START_WINDOWS.bat
 echo ===============================================================
pause
exit /b 0

:fail
echo.
echo УСТАНОВКА ОСТАНОВИЛАСЬ С ОШИБКОЙ.
pause
exit /b 1
