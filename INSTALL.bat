@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title LyricSync Setup (GPU Optimized)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install.ps1"
set "LS_RESULT=%ERRORLEVEL%"
echo.
if not "%LS_RESULT%"=="0" echo Setup encountered an issue. Please review the output above.
pause
exit /b %LS_RESULT%
