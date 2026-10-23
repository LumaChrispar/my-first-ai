@echo off
setlocal
cd /d "%~dp0"

set STEPS=%1
if "%STEPS%"=="" set STEPS=6000

echo =======================================================
echo  Training Aster on Large Corpus (1,563 Q&A Pairs)
echo  Target Steps: %STEPS%
echo =======================================================

aster.exe train --mode chat --steps %STEPS% --batch 8 --data data/claude_opus_train.jsonl --validation data/claude_opus_valid.jsonl --vocab models/aster-claude.vocab --out models/aster-small.bin --seed 1234

if %errorlevel% equ 0 (
    echo.
    echo =======================================================
    echo  Training complete!
    echo  Optimal checkpoint saved to models\aster-small.bin
    echo  Run start.bat to launch the chat server.
    echo =======================================================
)
