@echo off
chcp 65001 >nul
rem Instala ou atualiza o Escritório de Recrutamento neste computador.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0instalar.ps1"
echo.
pause
