@echo off
setlocal
cd /d "%~dp0"

set STEPS=%1
if "%STEPS%"=="" set STEPS=1500

echo ==============================================
echo  Training Aster for %STEPS% steps...
echo ==============================================

aster.exe train --mode chat --steps %STEPS% --batch 8 --data data/demo_chat.jsonl --validation data/demo_valid.jsonl --vocab models/aster-small.vocab --out models/aster-small.bin --seed 1234

if %errorlevel% equ 0 (
    echo.
    echo Training complete! Checkpoint saved to models\aster-small.bin
    echo Run start.bat to launch the server.
)
