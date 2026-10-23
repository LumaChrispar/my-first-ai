@echo off
setlocal
cd /d "%~dp0"
if not exist aster.exe call build.bat
if not exist aster.exe exit /b 1
if not exist models\aster-small.bin goto missing_model
echo Open http://127.0.0.1:8080 in your browser. Press Ctrl+C to stop.
aster.exe serve --model models/aster-small.bin --port 8080 --max-new-tokens 60 --repetition-penalty 1.15
exit /b %errorlevel%
:missing_model
echo No trained checkpoint found. See README.md for the training command.
exit /b 1
