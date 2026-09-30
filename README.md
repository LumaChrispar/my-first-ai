# Aster AI — starter project

A small, local C chat prototype with a browser interface. The current response engine is intentionally a tiny rule-based placeholder; it is not a trained language model. This gives us a runnable foundation to replace with a tokenizer, transformer, and model weights as the project grows.

## Run on Windows

1. Install a C compiler such as MinGW-w64 or Visual Studio Build Tools.
2. From this folder, compile with MinGW:

   ```powershell
   gcc server.c -o aster.exe -lws2_32
   .\aster.exe
   ```

3. Open <http://127.0.0.1:8080> in your browser.

The server listens only on localhost. Press Ctrl+C to stop it.

## Project map

- `server.c` — local HTTP server and C response engine.
- `index.html` — responsive chat interface.

## Current limits

The response engine uses simple rules and does not learn, remember previous sessions, or connect to an online model. The interface labels it as a prototype. A useful next step is implementing a small tokenizer and a genuine inference backend.
