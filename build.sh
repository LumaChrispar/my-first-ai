#!/bin/sh
# -----------------------------------------------------------------------------
#  Aster - build script for POSIX shells.
#
#  Run from the project folder:   sh build.sh
#  Produces:                      aster.exe
#
#  Nothing is downloaded and nothing is vendored. You need MinGW-w64 GCC on
#  your PATH first; see the message below if it is missing.
# -----------------------------------------------------------------------------
set -eu

CC=${CC:-gcc}
OUT=aster.exe

printf '\n'
printf 'Aster build\n'
printf '===========\n'
printf '\n'

# --- 1. Is the compiler present? ---------------------------------------------
if ! command -v "$CC" >/dev/null 2>&1; then
    printf 'ERROR: gcc was not found on your PATH.\n\n'
    printf 'Aster needs MinGW-w64 GCC. Install one of these, then reopen the\n'
    printf 'terminal so the new PATH takes effect:\n\n'
    printf '  * MSYS2  - https://www.msys2.org/     (recommended)\n'
    printf '              then run:  pacman -S mingw-w64-ucrt-x86_64-gcc\n'
    printf '              and build from an "MSYS2 UCRT64" terminal\n'
    printf '  * WinLibs - https://winlibs.com/\n\n'
    printf 'Either installer gives you a gcc.exe you can call from this script.\n\n'
    exit 1
fi

# --- 2. Which compiler is it? ------------------------------------------------
printf 'Compiler:\n'
"$CC" --version 2>/dev/null | grep -i 'gcc' || "$CC" --version 2>/dev/null | head -n 1
printf '\n'

# --- 3. Remove a stale executable --------------------------------------------
#  A running aster.exe keeps its file locked and the linker then fails with
#  "Permission denied". Stop the server (Ctrl+C in its terminal) and retry if
#  deletion below fails.
if [ -e "$OUT" ]; then
    printf 'Removing the existing %s ...\n' "$OUT"
    if ! rm -f "$OUT" || [ -e "$OUT" ]; then
        printf '\n'
        printf 'ERROR: could not delete %s.\n' "$OUT"
        printf 'It is most likely still running. Close it - press Ctrl+C in the\n'
        printf 'terminal where "aster serve" is running - then run build.sh again.\n\n'
        exit 1
    fi
    printf 'Removed.\n\n'
else
    printf 'No existing %s to remove.\n\n' "$OUT"
fi

# --- 4. Compile ---------------------------------------------------------------
printf 'Compiling ...\n\n'
if ! "$CC" -std=c11 -O2 -Wall -Wextra -Isrc -o "$OUT" \
        src/main.c src/model.c src/train.c src/server.c src/tokenizer.c \
        src/util.c src/jsonstr.c -lws2_32 -lm; then
    printf '\n'
    printf 'BUILD FAILED. The compiler messages are above.\n\n'
    exit 1
fi

if [ ! -e "$OUT" ]; then
    printf '\n'
    printf 'BUILD FAILED: the compiler reported success but %s was not created.\n\n' "$OUT"
    exit 1
fi

printf '\n'
printf 'BUILD SUCCEEDED - %s was created.\n' "$OUT"
printf 'Any compiler warnings are listed above; this build is warning-free with\n'
printf -- '-Wall -Wextra on MinGW-w64 GCC 15, so treat any warning as a regression.\n\n'
printf 'Check that the tokenizer, bounds handling and checkpoint IO all work:\n\n'
printf '    ./%s selftest\n\n' "$OUT"
printf 'Then train a model and start the local server:\n\n'
printf '    ./%s train --mode chat --steps 600 \\\n' "$OUT"
printf '        --data data/demo_chat.jsonl --validation data/demo_valid.jsonl \\\n'
printf '        --out models/aster-small.bin --seed 1234\n\n'
printf '    ./%s serve --model models/aster-small.bin --port 8080\n\n' "$OUT"
exit 0