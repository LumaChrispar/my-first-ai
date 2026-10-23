@echo off
setlocal
cd /d "%~dp0"

set STEPS=%1
if "%STEPS%"=="" set STEPS=2000

echo =======================================================
echo  Training Aster on NVIDIA GPU (PyTorch + CUDA)
echo  Hardware: NVIDIA GeForce GTX 1050
echo  Target Steps: %STEPS%
echo =======================================================

python tools\train_gpu_aster.py --steps %STEPS% --batch 16 --data data\claude_opus_train.jsonl --vocab models\aster-claude.vocab --out models\aster-small.bin

if %errorlevel% equ 0 (
    echo.
    echo =======================================================
    echo  GPU Training complete!
    echo  Checkpoint exported to models\aster-small.bin
    echo  Run start.bat to launch Aster with the new model.
    echo =======================================================
)
