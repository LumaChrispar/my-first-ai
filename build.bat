@echo off
rem ---------------------------------------------------------------------------
rem  Aster - build script for Windows (MinGW-w64 GCC).
rem
rem  Run from the project folder:   build.bat
rem  Produces:                      aster.exe
rem
rem  Nothing is downloaded and nothing is vendored. You need MinGW-w64 GCC on
rem  your PATH first; see the message below if it is missing.
rem
rem  This script is written with goto labels rather than nested IF (...) blocks
rem  on purpose. Inside a nested parenthesised block, "exit /b" only leaves the
rem  block: cmd carries on running the rest of the file and the script still
rem  exits 0, so a caller chaining on the exit code would treat a failed build
rem  as a success. Flat labels cannot do that.
rem ---------------------------------------------------------------------------
setlocal

set "CC=gcc"
set "OUT=aster.exe"

echo.
echo  Aster build
echo  ===========
echo.

rem --- 1. Is the compiler present? -------------------------------------------
where %CC% >nul 2>&1
if errorlevel 1 goto no_compiler

rem --- 2. Which compiler is it? ---------------------------------------------
echo  Compiler:
%CC% --version | findstr /r /c:"gcc"
echo.

rem --- 3. Remove a stale executable -----------------------------------------
rem  A running aster.exe keeps its file locked and the linker then fails with
rem  "Permission denied". Stop the server (Ctrl+C in its terminal) and retry if
rem  deletion below fails.
if not exist %OUT% goto no_old_exe

echo  Removing the existing %OUT% ...
del /q %OUT%
if exist %OUT% goto cannot_delete
echo  Removed.
echo.
goto compile

:no_old_exe
echo  No existing %OUT% to remove.
echo.

rem --- 4. Compile -------------------------------------------------------------
:compile
echo  Compiling ...
echo.
%CC% -std=c11 -O2 -Wall -Wextra -Isrc -o %OUT% src/main.c src/model.c src/train.c src/server.c src/tokenizer.c src/util.c src/jsonstr.c -lws2_32 -lm
if errorlevel 1 goto compile_failed
if not exist %OUT% goto no_output

echo.
echo  BUILD SUCCEEDED - %OUT% was created.
echo  Any compiler warnings are listed above; this build is warning-free with
echo  -Wall -Wextra on MinGW-w64 GCC 15, so treat any warning as a regression.
echo.
echo  Check that the tokenizer, bounds handling and checkpoint IO all work:
echo.
echo      %OUT% selftest
echo.
echo  Then train a model and start the local server:
echo.
echo      %OUT% train --mode chat --steps 3000 --batch 8 --data data/demo_chat.jsonl --validation data/demo_valid.jsonl --out models/aster-small.bin --seed 1234
echo.
echo      %OUT% serve --model models/aster-small.bin --port 8080
echo.
endlocal
exit /b 0

rem --- Failure paths ----------------------------------------------------------
:no_compiler
echo  ERROR: gcc was not found on your PATH.
echo.
echo  Aster needs MinGW-w64 GCC. Install one of these, then reopen the
echo  terminal so the new PATH takes effect:
echo.
echo    * MSYS2  - https://www.msys2.org/     ^(recommended^)
echo                then run:  pacman -S mingw-w64-ucrt-x86_64-gcc
echo    * WinLibs - https://winlibs.com/
echo.
echo  Either installer gives you a gcc.exe you can call from this script.
echo.
endlocal
exit /b 1

:cannot_delete
echo.
echo  ERROR: could not delete %OUT%.
echo  It is most likely still running. Close it - press Ctrl+C in the
echo  terminal where "aster serve" is running - then run build.bat again.
echo.
endlocal
exit /b 1

:compile_failed
echo.
echo  BUILD FAILED. The compiler messages are above.
echo.
endlocal
exit /b 1

:no_output
echo.
echo  BUILD FAILED: the compiler reported success but %OUT% was not created.
echo.
endlocal
exit /b 1
