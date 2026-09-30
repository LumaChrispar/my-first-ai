# Aster

A tiny language model, written from scratch in C11, that trains on local text and
serves replies to a browser page from your own machine.

Everything happens locally. **No cloud API, no API key, no external model
download, no telemetry, no web search, no tools, and no remote data transfer.**
The server binds to `127.0.0.1` only and is not reachable from the network.

## Be clear about what this is

This is a **teaching project**. The model has **124 352 parameters** and was
trained on **332 short conversations**. It is not a capable assistant.

- It has **no reliable knowledge**. Most factual claims it makes will be wrong.
- It has **no web access**, **no tools**, and **no persistent memory**.
- It **cannot see earlier turns** in a conversation and forgets each reply as
  soon as it is sent.
- Its context is **128 bytes**, so keep messages short.
- It **drops and repeats characters**, because it predicts one byte at a time
  with no sub-word vocabulary.
- It is **not certified safe** for anything.

Do not use it for medical, legal, financial, emergency, hiring, or any other
high-impact decision. Full detail is in [`MODEL_CARD.md`](MODEL_CARD.md), which
is part of the project and not an afterthought.

It also invents text. When it does not know something it produces a confident,
fluent, wrong answer rather than admitting ignorance — which is the single most
important thing to understand about using a model this small.

## Requirements

- **Windows with MinGW-w64 GCC** (the primary target), or any C11 compiler with
  sockets and `libm` on another platform.
- Nothing else. No Python, no Node, no package manager, no downloads.

Check the compiler:

```powershell
gcc --version
```

If `gcc` is not found, install MinGW-w64 from
<https://www.mingw-w64.org/downloads/> (or MSYS2's `mingw-w64-ucrt-x86_64-gcc`
package) and make sure `C:\mingw64\bin` is on your `PATH`.

## Build

```powershell
.\build.bat
```

Or directly:

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Isrc -o aster.exe src/main.c src/model.c src/train.c src/server.c src/tokenizer.c src/util.c src/jsonstr.c -lws2_32 -lm
```

This produces `aster.exe` in this folder and should compile with **zero
warnings**.

## Verify it works

```powershell
.\aster.exe selftest
```

This checks the tokenizer and UTF-8 handling, the parameter count against a
hand-computed layout, that untrained loss is near `ln(261)`, the chat loss-mask
alignment, checkpoint save/load round trips, rejection of truncated and
non-checkpoint files, greedy generation determinism and step limits, that
generation really decodes every step it takes, the JSON helpers, and — most
importantly — that **the analytic gradients match a finite-difference
approximation** for one- and two-layer models.

A passing run ends with `self-test passed` and exit code 0.

## Train

```powershell
.\aster.exe train --mode chat --steps 3000 --batch 8 `
    --data data/demo_chat.jsonl --validation data/demo_valid.jsonl `
    --out models/aster-small.bin --seed 1234
```

About 7 minutes on an ordinary processor. Expected output, from a real run:

```
  train split: 332 window(s), 26578 tokens
  valid split: 68 window(s),  5762 tokens
  step      0  valid loss 5.5823  perplexity   265.69   (untrained)
  ...
  held-out loss was still rising at step 3000; reverting to the best
      held-out parameters from step 1075
  held-out loss 5.5823 -> 2.0938  (perplexity 8.12): improved
  wrote models/aster-small.bin
```

Training selects the checkpoint with the **best held-out loss**, not the last
one. A 124k-parameter model given 332 examples will drive its training loss
towards zero and keep going; that is memorisation, not learning, and stopping at
the best held-out step is what makes the reported number mean anything.

Read that figure narrowly — it measures a new question in a familiar register,
not general capability. `MODEL_CARD.md` explains why in detail.

## Generate

```powershell
.\aster.exe generate --model models/aster-small.bin --prompt "Who are you?" --max-new-tokens 80
```

Real output from the shipped checkpoint, unedited, from:

```powershell
.\aster.exe generate --model models/aster-small.bin --prompt "Who are you?"
```

```
--- generated text (raw model output) ---
I ave all ase seal a a a sonde anyor an a pllo.
--- end ---
```

That is the honest quality level. The intended reply starts "I am a very small
model…", and note what happened to it: characters dropped mid-word, and the
sentence is incoherent. Two other real examples:

| Prompt | Output |
|---|---|
| `Are you a doctor?` | `No. I am a a smandl and I ote a no a prre.` |
| `Can you remember me?` | `No. I am a sthe a a smor sthin pronact.` |

The refusals are often right — it says "No" to being a doctor and to having
memory, which is the behaviour the training data taught — but the wording is
broken. Dropping characters is expected here: the tokenizer is byte-level with
no merging, so most of the model's capacity goes into learning individual bytes.

## Serve

```powershell
.\aster.exe serve --model models/aster-small.bin --port 8080
```

Then open <http://127.0.0.1:8080>. Press Ctrl+C to stop.

The page renders every message as **plain text** — model output is untrusted
text and is never inserted as HTML. It shows the real model parameters and
context size reported by the server, labels replies as unverified machine
output, and tells you plainly that there is no web access, no tools, and no
memory.

**If no checkpoint is loaded**, the page says so and disables the input, and the
API returns `503 model_not_trained`:

```powershell
.\aster.exe serve --port 8080
curl.exe http://127.0.0.1:8080/api/status
```

```json
{"ready":false,"status":"model_not_trained","detail":"No checkpoint is loaded, so no
answer can be generated.","how_to_train":"aster train ...","network":"none","tools":"none"}
```

It never invents a pre-trained checkpoint, and it never returns random text as
if it were a useful answer.

## Evaluate a checkpoint

```powershell
.\aster.exe eval --model models/aster-small.bin --mode chat `
    --data data/demo_valid.jsonl
```

## Commands

| Command | Purpose |
|---|---|
| `aster train` | build a checkpoint from local files |
| `aster generate` | load a checkpoint and print one continuation |
| `aster serve` | run the loopback-only chat server |
| `aster eval` | report held-out loss for a checkpoint |
| `aster selftest` | run the checks |
| `aster help` | show usage |

Run `.\aster.exe help` for every flag.

## Model

| | |
|---|---|
| Architecture | decoder-only causal transformer, pre-LayerNorm |
| Layers | 2 |
| `d_model` | 64 |
| Heads | 4 (head dim 16), causal mask |
| `d_ff` | 256, GELU (tanh approximation) |
| Context | 128 tokens |
| Vocabulary | 261 (256 raw bytes + `SYSTEM`, `USER`, `ASSISTANT`, `END`, `PAD`) |
| Tokenizer | byte-level; one token is one byte |
| Weight tying | the token embedding is also the output head |
| **Parameters** | **124 352** |
| Optimiser | AdamW, decoupled weight decay, global-norm clipping, warmup + cosine decay |
| Checkpoint | `aster-checkpoint` format 1, architecture version 1, little-endian, with a CRC and a SHA-256 of the header |

Backpropagation is written by hand — there is no autodiff framework and no
third-party dependency of any kind.

## Project layout

```
src/
  main.c        CLI: train, generate, serve, eval, selftest, help
  model.c/.h    the transformer: forward, manual backward, checkpoint IO, generation
  train.c/.h    dataset building, training loop, evaluation
  server.c/.h   loopback-only HTTP server and JSON API
  tokenizer.c/.h byte tokenizer, UTF-8 validation and sanitisation
  util.c/.h     logging, checked allocation, atomic write, SHA-256, CRC-32, RNG
  jsonstr.c/.h  minimal JSON string extraction and escaping
data/           training and validation corpora, with provenance
models/         checkpoints (written by `train`)
index.html      the browser page
build.bat       Windows build script
build.sh        POSIX build script
MODEL_CARD.md   what this model can and cannot do
BUILD_PROGRESS.md  what was built, what was run, and what the results were
```

## Data

`data/` contains three files written by hand for this project — nothing was
scraped, downloaded, or copied. Provenance, licence, format, and SHA-256 hashes
are documented in [`data/README.md`](data/README.md), with a fill-in manifest
template in `data/manifest.example.json`.

If you add your own data, only use material you own or are licensed to use, and
exclude anything private: no credentials, no personal records, no confidential
work, and no private conversations.

## Privacy

- Prompts are **not logged**.
- The transcript lives in your browser page and is not sent anywhere.
- The server binds to `127.0.0.1`, so other machines cannot reach it.
- There is no account, no key, and no external service.
- This version has **no action tools and no tool endpoint**, by design. Nothing
  here executes a model-produced command, and nothing is added to your computer.

## Honest limitations

- **This is not a useful assistant.** It is a demonstration that the whole
  pipeline works and can be read and checked.
- **It has no knowledge.** 332 conversations do not contain the world's facts.
- **It has no tools, no web access, and no memory**, and this version is not
  going to grow them without being rewritten.
- **It is not certified safe** for anything.
- **Its output is untrusted text.** The UI renders it as text for that reason.
- Validation loss of 2.09 sounds better than the model is; see `MODEL_CARD.md`
  for the three reasons it overstates real ability.

## Licence

Project source: MIT. Bundled demo data: CC0-1.0. See `data/README.md`.
