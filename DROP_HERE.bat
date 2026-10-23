@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PATH=%~dp0.venv\Scripts;%PATH%"
if not exist "%~dp0.venv\Scripts\python.exe" goto missing
if not exist "%~dp0tools\runtime_config.json" goto missing
"%~dp0.venv\Scripts\python.exe" -u "%~dp0src\launcher.py" %*
set "LS_RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %LS_RESULT%
:missing
echo Please extract the whole ZIP and run INSTALL.bat first.
echo If setup failed, send install.log back for help.
pause
exit /b 1
