# Aster

A tiny language model, written from scratch in C11, that trains on local text and
serves replies to a browser page from your own machine.

Everything happens locally. **No cloud API, no API key, no external model
download, no telemetry, no web search, no tools, and no remote data transfer.**
The server binds to `127.0.0.1` only and is not reachable from the network.

## Be clear about what this is

This is a **teaching project**. The model has **180 992 parameters** and was
trained on **332 short conversations**. It is not a capable assistant.

- It has **no reliable knowledge**. Most factual claims it makes will be wrong.
- It has **no web access**, **no tools**, and **no persistent memory**.
- It **cannot see earlier turns** in a conversation and forgets each reply as
  soon as it is sent.
- Its context is **128 tokens** — about 470 bytes of English on the demo corpus,
  so roughly 70 words. Keep messages short.
- Its **output is broken at the word level** much of the time. A sub-word
  vocabulary fixed the mid-word character dropping, not the coherence.
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
hand-computed layout, that untrained loss is near `ln(vocab)`, the chat loss-mask
alignment (under both byte fallback and learned merges), checkpoint save/load
round trips, rejection of truncated and non-checkpoint files, greedy generation
determinism and step limits, that generation really decodes every step it takes,
the JSON helpers, and — most importantly — that **the analytic gradients match a
finite-difference approximation** for one- and two-layer models.

The tokenizer-specific checks are the ones that matter most, because a tokenizer
bug produces a plausible-looking loss *and* a plausible-looking generation:

- `encode(decode(encode(s))) == s` over empty input, all 255 non-NUL byte
  values, lone UTF-8 continuation bytes, truncated sequences, a script the
  merges never saw, a 512-byte run of one repeated character, and 1024
  pseudo-random bytes
- an embedded NUL is **reported as an error**, not skipped — skipping it would
  quietly break the round trip for any string containing one
- unseen text falls back to byte tokens and still round-trips
- encoding is deterministic, and learning the same corpus twice gives an
  identical merge table
- a single flipped byte in a checkpoint is caught by the integrity block
- a byte-tokenizer checkpoint is refused with a message that says to retrain
- a fully masked batch produces **bitwise-zero** gradients, not merely small ones
- bits per byte is total nats over total bytes, not a byte-weighted average
- **nats → bits divides by ln 2** rather than exponentiating — the two are one
  measurement in two units, and `exp()` gets silently wrong numbers while
  still picking the same checkpoint

A passing run ends with `self-test passed` and exit code 0. There are **52
checks**.

## Vocabulary

The sub-word vocabulary is learned from the training split only:

```powershell
.\aster.exe vocab --mode chat --data data/demo_chat.jsonl `
    --merges 1024 --out models/aster-small.vocab
```

`--merges` is a ceiling, not a target: a small corpus runs out of frequent pairs
first and the tool says how many it actually got.

To see what was learned — which is a real check, not debug output, since it is
how a surprising merge gets spotted:

```powershell
.\aster.exe vocab --in models/aster-small.vocab
```

There is deliberately **no** flag to learn from the validation set. A merge
table fitted on held-out data leaks it, and the held-out loss then stops
meaning what it appears to mean. The learner is handed text through a callback
and never sees a file path, so it cannot be pointed at the wrong file.

`train` learns a vocabulary from `--data` when you do not pass `--vocab`, and
writes `<out>.vocab` beside the checkpoint so a second run can reproduce the
first. The checkpoint carries its own copy of the merge table, so a model never
depends on a separate file staying in sync.

## Train

```powershell
.\aster.exe train --mode chat --steps 3000 --batch 8 `
    --data data/demo_chat.jsonl --validation data/demo_valid.jsonl `
    --out models/aster-small.bin --seed 1234
```

About 9 minutes on an ordinary processor. Real output, from a real run on the
bundled demo corpus (seed 1234, `--vocab tmp/demo.vocab` — see
[BUILD_PROGRESS.md](BUILD_PROGRESS.md) for the exact commands):

```
  vocabulary: demo.vocab (885 merges, 1146 tokens, 3.68 bytes/token)
  parameters:  180992 (0.181 million), token embedding tied to the output head
  train split: 332 window(s), 8424 tokens, 45865 source bytes, 1 source(s)
  valid split: 68 window(s), 2100 tokens, 9737 source bytes, 1 source(s)
  step      0  valid 7.0541 nats/token  3.4042 bits/byte  (token ppl  1157.57)   (untrained)
    untrained check: 7.0541 nats/token x 2.99 bytes/token = 2.3596 nats/byte / ln2 =
    3.4042 bits/byte; log2(vocab=1146) / 2.99 = 3.3994  [consistent]
  ...
  finished 3000 step(s) in 532.6s
  held-out loss was still rising at step 3000; reverting to the best
      held-out parameters from step 450
  held-out 7.0541 -> 4.8693 nats/token   3.4042 -> 2.3498 bits/byte: improved
```

Note the `untrained check` line. An untrained model is uniform, so its
bits/byte is not a free parameter — it is `log2(vocab) / bytes-per-token`, and
with a byte vocabulary it is exactly 8.00. Printing it means the headline figure
can be checked by hand instead of taken on trust, and it caught a real bug (see
[BUILD_PROGRESS.md](BUILD_PROGRESS.md)).

**Bits per byte fell from 3.40 to 2.35.** The previous byte-tokenizer build
reached **3.02** bits/byte on this same held-out set, so the sub-word tokenizer
is a **22 % improvement on the one metric that survives a tokenizer change** —
fewer bits spent per byte of English.

The per-token figure went the *other* way, from 2.09 to 4.87 nats/token, and
**that comparison is meaningless**: a sub-word vocabulary makes every token
easier to predict whether or not the model improved at anything, so a higher
per-token loss here is expected and says nothing. This is exactly why bits per
byte is the figure the run reports and the one to quote.

Training selects the checkpoint with the **best held-out loss**, not the last
one, and selects on the same number it reports. A 181k-parameter model given 332
examples will drive its training loss towards zero and keep going — here it
reached 0.0028 by step 3000 while held-out loss was still climbing — and that
is memorisation, not learning.

Read the figure narrowly. It measures a new question in a familiar register,
not general capability. `MODEL_CARD.md` explains why in detail.

## Generate

```powershell
.\aster.exe generate --model models/aster-small.bin --prompt "Who are you?"
```

Real output from the run above, unedited:

```
--- generated text (raw model output) ---
I am a small model that runs on your own computer.
--- end ---
```

Three more, all unedited:

| Prompt | Output |
|---|---|
| `Are you a doctor?` | `No. I am a small model that runs on your own computer.` |
| `Can you remember me?` | `No. I am not a small model that runs on your own computer.` |
| `What is the capital of France?` | `I cannot. I am not qualified to your own computer.` |

**What this tokenizer change fixed, and what it did not.** Before it, the same
prompt produced `I ave all ase seal a a a sonde anyor an a pllo.` — characters
dropped mid-word. The output is now grammatical English that emits whole words.
What has *not* changed is the content: the model still does not know that Paris
is the capital of France, and it answers every question with the same sentence
about itself. It gives the right refusals — "No" to being a doctor, "No" to
having memory — for the wrong reason, because those are the patterns its 332
training conversations taught.

Left to run longer it degenerates into repetition, and long prompts make it
worse:

```
No. I have the the model model model model to a small model the the the the the
the the the model that qualified to a small model and I am not qualified to a
small models to a small model model model that qualified to the model model
that runs on your own computer.
```

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
| Vocabulary | 261 to 1285, depending on how many merges the corpus supports (see below) |
| Tokenizer | sub-word BPE **with byte fallback** |
| Weight tying | the token embedding is also the output head |
| **Parameters** | **124 352 at 261 tokens; 189 888 at the 1285 maximum** |
| Optimiser | AdamW, decoupled weight decay, global-norm clipping, warmup + cosine decay |
| Checkpoint | `ASTERMD2` format 2, architecture version 1, little-endian; the last 36 bytes are a CRC-32 and a SHA-256 of everything before them |

### The tokenizer, and why the parameter count moves

The first 261 ids are fixed: 256 raw bytes plus `SYSTEM`, `USER`, `ASSISTANT`,
`END`, `PAD`. Everything above that is a BPE merge learned from the training
split. The 256 byte ids are **always present**, which is what makes byte
fallback work — any byte sequence is representable, so there is no unknown
token and no input the encoder can reject. A vocabulary with zero merges is
therefore valid and behaves exactly like the old byte tokenizer, which makes it
the natural regression baseline.

Because the output head is tied to the token embedding, **every added token
costs exactly `d_model` parameters and changes nothing else** — the body of the
network is 107 648 parameters at any vocabulary size. Going from 261 to the
1285 maximum adds 65 536 parameters, 1.53×:

| Merges | Vocabulary | Parameters |
|---|---|---|
| 0 | 261 | 124 352 |
| 512 | 773 | 157 120 |
| 1024 (max) | 1285 | 189 888 |

A small corpus runs out of frequent pairs before it runs out of merge budget.
The bundled demo data (about 25 KB) supports 885 merges and 3.7 bytes per
token, so its model is 180 992 parameters; a 15 000-conversation corpus reaches
the full 1024. `aster vocab` reports which it got and warns when it fell short.

### Two loss numbers, and which one to compare

Loss is reported two ways because only one of them survives a tokenizer change:

- **bits per byte** — the headline. Total nats over total bytes, so it is
  comparable across tokenizers.
- **nats per token** — cheaper to compute, but a sub-word tokenizer makes every
  token easier whether or not the model improved. It is comparable only against
  another model using the *same* vocabulary, and is labelled as such wherever
  it appears.

Nats and bits are one measurement in two units, related by a **division** by
ln 2. Converting with `exp()` instead is monotonically increasing, so it changes
which checkpoint gets selected but makes every printed number wrong — and
badly: 1.63 nats/byte is 2.35 bits/byte, and the exponentiated form said 10.48.
The self-test now asserts the conversion directly, because nothing else can see
it.

Checkpoint selection uses bits per byte, the same number the run reports.
Selecting on one metric and quoting another would mean reporting the score of a
checkpoint chosen to minimise something else.

Backpropagation is written by hand — there is no autodiff framework and no
third-party dependency of any kind.

## Project layout

```
src/
  main.c        CLI: train, vocab, generate, serve, eval, selftest, help
  model.c/.h    the transformer: forward, manual backward, checkpoint IO, generation
  train.c/.h    dataset building, training loop, evaluation
  server.c/.h   loopback-only HTTP server and JSON API
  tokenizer.c/.h sub-word BPE with byte fallback, UTF-8 validation and sanitisation
  util.c/.h     logging, checked allocation, atomic write, SHA-256, CRC-32, RNG
  jsonstr.c/.h  minimal JSON string extraction and escaping
tools/
  oasst2jsonl.c data prep: OpenAssistant (oasst1) export -> chat JSONL
                NOT part of aster.exe, and it does not download anything
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

### Converting a public dataset

`tools/oasst2jsonl.c` turns an **OpenAssistant (oasst1)** export into Aster's
chat JSONL. It is a separate tool, not part of `aster.exe`, and **it downloads
nothing** — you fetch the dataset yourself, and this only reads a local file.

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Isrc -o oasst2jsonl.exe `
    tools/oasst2jsonl.c src/util.c src/jsonstr.c src/tokenizer.c

.\oasst2jsonl.exe --in oasst.jsonl --lang en `
    --out data/oasst_train.jsonl `
    --valid-out data/oasst_valid.jsonl --valid-every 20 `
    --max-conversations 15000
```

The split is made **by construction**: a whole conversation goes to one side or
the other, decided as it is emitted. Shuffling records after flattening the tree
would put a prompt and its own reply on opposite sides, and the held-out loss
would then measure memorisation rather than generalisation. The tool prints the
conversation count and the SHA-256 of each file, so a run is reproducible from
the log alone and a file that later changes is detectable.

oasst1 is a *tree* and Aster's format is linear, so the converter keeps one
reply per prompt and reports how many alternatives it discarded. Read the
counts before training on the result, and read the file: it is other people's
writing, and personal data should be assumed present.

Learn the vocabulary from the **training file only**:

```powershell
.\aster.exe vocab --mode chat --data data/oasst_train.jsonl `
    --merges 1024 --out models/aster-small.vocab
```

Evaluate on **both** files afterwards. The oasst1 holdout is in-distribution
and measures fit to that corpus; the small hand-written `data/demo_valid.jsonl`
is out-of-distribution and measures whether the model learned English or just
learned oasst1. If only the first improves, the second did not, and reporting
one without the other would be the dishonest choice.

Full details, including the `--probe` mode for diagnosing an unexpected file
layout, are in [`data/README.md`](data/README.md).

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
- **It has no knowledge.** 332 conversations do not contain the world's facts,
  and the sub-word tokenizer did not change that — it changed the *English*.
- **It has no tools, no web access, and no memory**, and this version is not
  going to grow them without being rewritten.
- **It repeats itself.** Given a long or unusual prompt it degenerates into
  `the the model model model`, which is a real output of the shipped
  checkpoint, not a worst case invented for the docs.
- **It is not certified safe** for anything.
- **Its output is untrusted text.** The UI renders it as text for that reason.
- A held-out loss of 2.35 bits/byte sounds better than the model is; see
  `MODEL_CARD.md` for the reasons it overstates real ability.

## Licence

Project source: MIT. Bundled demo data: CC0-1.0. See `data/README.md`.
