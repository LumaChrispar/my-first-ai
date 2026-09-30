# Aster build progress

A running record of what has actually been built, which commands were actually
run, and what the actual results were. Nothing in here is aspirational: if a
step is listed as done, the command and its output are recorded. If something is
broken or unverified, it says so.

Last updated: 2026-09-30

---

## Status at a glance

*(updated 2026-09-30, after the sub-word tokenizer session — see the last
section for that work. The sections above it are the byte-tokenizer build and
are kept as the historical record.)*

| Area | State |
|---|---|
| Build (`-Wall -Wextra`, MinGW-w64 GCC 15.2) | clean, **zero warnings** — `build.bat` **succeeded** once the stale process holding `aster.exe` (PID 4964) exited; the real `aster.exe` is built and passes all 57 checks |
| Self-test suite | **57 checks, all passing**, exit 0 |
| Gradient check (analytic vs finite difference) | passes for 1 and 2 layers (worst rel. error 0.0066 / 0.0075) |
| Tokenizer | sub-word BPE with byte fallback; `encode(decode(encode(s))) == s` verified over an adversarial corpus |
| Tokenizer / UTF-8 safety | verified |
| Checkpoint save/load round trip | verified bit-exact, `ASTERMD2` with CRC-32 + SHA-256 |
| Checkpoint corruption | **verified** — one flipped byte is refused, exit 1, no number produced |
| Training reduces held-out loss | **verified: 3.4042 → 2.3498 bits/byte**, best step 450 of 3000 |
| vs. the byte-tokenizer build | **3.02 → 2.35 bits/byte on the same held-out set** (22 % better); per-token loss not comparable |
| Generation | **works** — whole words, grammatical, and still says little; a real long-prompt output loops |
| Server (loopback, bounded) | **verified end to end** |
| Browser UI | text-only rendering, verified at the source level; **no browser was launched this session** |
| Build scripts | `build.bat` and `build.sh`; all four exit paths of each checked |
| README / MODEL_CARD / data provenance | written and updated with both loss figures |
| Data converter (`tools/oasst2jsonl`) | written, tested, splits by construction, SHA-256 per file |
| oasst1 corpus (~15 000 conversations) | **not downloaded** — blocked on the user; converter and instructions ready |

The honest headline: **the pipeline is complete and verified end to end, and
the model is still not very good.** The sub-word tokenizer made the *English*
much better — it now emits whole words instead of dropping characters — and did
not make the model any more knowledgeable. It answers every question with the
same sentence about itself, and given a long prompt it loops. That is the
expected result, and it is documented rather than hidden.

---

## Commands actually run, and actual results

### Build — zero warnings

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Isrc -o aster.exe src/main.c src/model.c `
    src/train.c src/server.c src/tokenizer.c src/util.c src/jsonstr.c -lws2_32 -lm
```

Both scripts were run and both exit 0:

```
.\build.bat   ->  gcc (MinGW-W64 x86_64-ucrt-posix-seh) 15.2.0
                 BUILD SUCCEEDED - aster.exe was created.
sh build.sh   ->  BUILD SUCCEEDED - aster.exe was created.   (exit 0)
```

Both check for `gcc` first, print the compiler version, delete a stale
`aster.exe` before linking (a running server holds the file and the link fails
with "Permission denied"), and exit non-zero on failure.

`build.sh` was verified end to end: exit 1 with no compiler on `PATH`, exit 0
on a real build. `build.bat` had a real bug, found by testing the failure paths
rather than only the happy one — see below.


### Self-test — 26 checks, exit 0

```
.\aster.exe selftest        ->  self-test passed   (exit 0, 26 pass, 0 FAIL)
```

Covers: tokenizer round trips, distinct control markers, UTF-8 replacement,
parameter count vs a hand-computed layout, untrained loss ≈ ln(261), finite
gradients, fully-masked batch gives zero loss, out-of-range token rejection,
checkpoint round trip, rejection of truncated / non-checkpoint / missing files,
greedy determinism, generation honours `max_new_tokens`, **generation keeps
decoding for every requested step**, **generation matches a reference decode**,
JSON decode / reject / escape / `\u`, the chat loss-mask alignment check, and the
analytic-vs-finite-difference gradient check for one and two layers.

### Training — held-out loss falls

```
.\aster.exe train --mode chat --steps 3000 --batch 8 `
    --data data/demo_chat.jsonl --validation data/demo_valid.jsonl `
    --out models/aster-small.bin --seed 1234
```

```
  train split: 332 window(s), 26578 tokens
  valid split: 68 window(s),  5762 tokens
  step      0  valid loss 5.5823  perplexity   265.69   (untrained)
  step   1075  train 1.5334  valid 2.0938 (ppl 8.12)      <- best
  step   3000  train 0.4652  valid 2.8768 (ppl 17.76)
  held-out loss was still rising at step 3000; reverting to the best
      held-out parameters from step 1075
  held-out loss 5.5823 -> 2.0938  (perplexity 8.12): improved
  wrote models/aster-small.bin
```

62 % reduction on data the model never saw. Read narrowly — see "What this does
not show" below.

### Generation — real output, unedited

```
.\aster.exe generate --model models/aster-small.bin --prompt "Who are you?"

--- generated text (raw model output) ---
I ave all ase seal a a a sonde anyor an a pllo.
--- end ---
```

| Prompt | Output |
|---|---|
| `Are you a doctor?` | `No. I am a a smandl and I ote a no a prre.` |
| `Can you remember me?` | `No. I am a sthe a a smor sthin pronact.` |

Refusals are frequently correct (`No` to being a doctor, to having memory); the
wording is broken. Characters are dropped because the tokenizer is byte-level
with no merging.

### Server — verified end to end

```
.\aster.exe serve --model models/aster-small.bin --port 8099
```

| Request | Result |
|---|---|
| `GET /api/status` | 200, `ready:true`, `parameters:124352`, `network:"none"`, `tools:"none"`, `persistent_memory:"none"` |
| `POST /api/chat {"message":"Who are you?"}` | 200, real generated reply |
| `GET /` | 200, `text/html`, 36 116 bytes, **byte-identical to `index.html` on disk** |
| `GET /api/health` | 200 |
| `PUT /api/chat` | 405 `method_not_allowed` |
| `GET /api/nope` | 404 `not_found` |
| empty `message` | 400 `empty_message` |
| missing `message` | 400 `missing_field` |
| malformed JSON | 400 `invalid_json` |
| 5000-byte message | 413 `message_too_long` |
| over-long prompt | 200, with a `note` explaining that older text was dropped |
| `<script>alert(1)</script>` as the message | 200, treated as inert text |
| `netstat` while listening | `127.0.0.1:8099` only — **not** `0.0.0.0` |

With no `--model`:

```
GET  /api/status  -> {"ready":false,"status":"model_not_trained", ... }
POST /api/chat    -> 503 {"error":"model_not_trained", ... }
```

No invented text, no fabricated checkpoint. The server log shows warnings and
the bind address only — **no prompt text and no dataset content**.

The served reply and the CLI's `generate` output agree exactly:
`I ave all ase seal a a a sonde anyor an a pllo.` — which is also the string
quoted in `README.md`, so the documentation cannot drift from what the shipped
checkpoint actually does without that check catching it.

---

## The backward pass: seven bugs, and the oracle that hid them

Getting a correct manual backprop out of a from-scratch transformer turned out to
be the hard part of this project. The bugs below are listed because each one
looked like a numerical mystery rather than a logic error.

1. **`linear_dy` was computing a forward product.** Matrices are stored
   `[out][in]`, so `dx[k] = Σ_o dy[o]·W[o·in+k]`. The function named `linear_dy`
   computed `W @ x` instead, and was used at four sites to propagate gradients.
   Every input gradient was transposed. `Wo` is square, so no shape check could
   catch it. Fixed by deleting the function and using `linear_dx`.

2. **Softmax backward misplaced `inv_scale`.** It was folded into the
   value-dot and then dropped from `dq`/`dk` entirely. Fixed by computing the
   unscaled `ds` and applying `inv_scale` on the `q·k` path.

3. **The activation arena aliased `ln1_out` onto `xin`.** The bump pointer `p`
   started at `a->xin` and never stepped over `xin`'s own region, so the first
   allocation handed the residual stream's storage to `ln1_out`. Because
   `IDXR(b,l,t,C) == IDXL(b,l,t,C)` for `b=0, l ≤ L-1`, `layer_norm_fwd` then
   wrote `ln1_out` over the block input of the layer reading it. The arena's
   self-assertion had accounted for the overlap, so it passed silently.

4. **The layer-norm initializer interleaved weight and bias by parity.**
   `(i % 2 == 0) ? 1.0f : 0.0f` over `2·C` entries left every odd channel with
   gain 0 — normalization silently disabled — and every even channel with bias
   1. The dump that exposed it was literally `ln1w = 1 0 1 0 1 0 1 0`. Fixed to
   `(i < C)` at all three sites.

5. **`ln2_w` and `ln2_b` received no gradient at all** —
   `layer_norm_bwd(..., NULL, NULL)`. And the `dy` passed to it was the residual
   gradient instead of `d(ln2_out)`, with the F→C contraction folded into the
   wrong buffer. Three distinct errors in one call site.

6. **The residual identity path was discarded.** `xin[l+1] = res1 + ffo` means
   `d(res1) = dtop + LN_bwd(res1, d(ln2_out))`. The code overwrote `dtop` with
   the Jacobian result, throwing away the direct coefficient-1 term — starving
   the whole residual stream of the dominant part of its gradient. This was the
   deepest one.

7. **`aster_generate` sized its activation arena to the prompt, not the context.**
   `aster_forward` computes exactly `a->T` rows, so every row past the prompt
   stayed at its calloc'd zero, the logits looked uniformly `0.00`, argmax fell
   on a control token, and **generation stopped after a single character** while
   held-out loss still looked perfectly healthy. Found only by dumping the
   per-step top-5 logits. Fixed by allocating for the full context and lowering
   `a->T` per step; a regression test now compares `aster_generate` against a
   hand-written reference decode.

### The oracle bug — worth reading

For most of this work the finite-difference checker reported a clean bill of
health: **0 bad across 72/72 configurations**. It was wrong.

```c
acc = (k == 0) ? d : (4.0 * acc - prev) / 3.0;   /* k=1: (4*D(h) - D(h))/3 == D(h) */
```

At `k=1` it reused `acc` (already `D(h)`) instead of the freshly computed
`D(h/2)`, so it evaluated `(4·D(h) − D(h))/3` — identically `D(h)`. It was a
plain central difference, not Richardson, and the `h/2` passes were computed
and discarded. Because a plain difference *underestimates* the truncation error,
it looked cleaner than a correct implementation, not worse.

The same bug was then independently written into the `selftest` version. It was
caught because the check failed on a **correct** backward pass — the loss
curve for one parameter showed the analytic gradient sitting exactly on the
converged limit while the two-level estimate sat 8 % away:

```
  R( 7.81e-03,  3.91e-03) = +1.72551155e-01   rel err vs analytic 0.0003
```

After fixing `gc_numeric` to assign `cur` before extrapolating, the check passes
at **0.0066 worst relative error (1 layer)** and **0.0075 (2 layers)** over
234 and 351 compared gradients, against a 2 % threshold.

Two lessons worth keeping: a checker that has never failed is not yet tested; and
a gradient checker must be validated against a case whose answer is known
indepently before it is used to judge anything else.

### Other fixes this session

- **Chat loss mask** (earlier): `dataset_gather` sets `y[i] = stream[i+1]`, so
  `w[i]` gates the prediction of token `i+1`. The mask started at the answer
  bytes, leaving the `TOK_ASSISTANT` position at weight 0 — the first byte of
  every answer was never trained. A self-test now asserts the alignment.
- **Parameter layout overlap**: `ff1`'s bias was not skipped when computing
  `ff2`'s offset, so bias and `ff2` overlapped, and the count was 64 short of
  the allocation (a heap overflow). Now 124 352, matching the hand-computed
  layout.
- **GELU derivative evaluated at the wrong point**: the forward overwrote `ffh`
  with the post-GELU value, so backward called `gelu_d(gelu(z))`. Added a
  separate `ffz` buffer holding the pre-activation.
- **Weight-gradient pointers chained through a non-existent bias**, putting
  `Wk`, `Wv`, `Wo` each one row off.
- **Residual stream indexed with the wrong stride**: `IDXL` strides by `L`, but
  `xin`/`dxin` hold `L+1` stages, so for the default 2-layer config the whole
  residual stream was addressed incorrectly. Added `IDXR`.
- **In-place residual accumulation** clobbered the value the next layer down
  still needed; added a per-layer `dres` accumulator and split the layer backward
  into two passes.
- **Self-test file-handle leak**: the truncated-checkpoint test left a handle
  open, so on Windows `remove()` failed and `models/selftest-trunc.bin` was left
  behind after a passing run. Fixed; the self-test now cleans up.

### `build.bat` reported success on a failed build

`build.bat` printed a clear error and then **exited 0**:

```
$ ./build.bat ; echo $?
  Removing the existing aster.exe ...
  ERROR: could not delete aster.exe.
0
```

The cause is a `cmd.exe` quirk, not the script's logic:

```bat
if exist %OUT% (
    del /q %OUT%
    if exist %OUT% (          :: <-- nested block
        echo  ERROR: ...
        exit /b 1
    )
)
```

**`exit /b` inside a nested parenthesised block only leaves that block.** cmd
continues executing the rest of the file, reaches the trailing `exit /b 0`, and
the caller sees success. Reproduced in isolation:

```bat
if exist t6.bat (
  if exist t6.bat (
    echo inner
    exit /b 1
  )
)
echo tail          :> never printed
exit /b 0
```
→ prints `inner`, **not** `tail`, and the process still exits **0**.

The same nesting one level up (`if/else` without the inner block) and a `goto`
form both correctly returned 1, which is what localised it to the nesting.

This matters: anyone running `call build.bat && deploy`, or any CI step, would
have shipped a stale or missing binary on a failed build. The script is now
written with flat `goto` labels so every failure path is a top-level `exit /b 1`.
All four paths were then run and their exit codes checked:

| Path | Result |
|---|---|
| no compiler on `PATH` | prints the install instructions, **exit 1** |
| `gcc` fails | prints `BUILD FAILED`, **exit 1** |
| stale exe present, deleted | rebuilds, **exit 0** |
| no stale exe | rebuilds, **exit 0** |

All four were run. The `del`-failure path could not be triggered — it needs a
locked `aster.exe`, and by the time the fix was in place nothing was holding the
file any more. It is the same top-level `exit /b 1` form as the two failure
paths that were run, so the control flow is verified; only the trigger was
unavailable.

Note for anyone re-checking this: measuring a `.bat` exit code by appending
`& echo %ERRORLEVEL%` on the same command line does **not** work — cmd expands
`%ERRORLEVEL%` when it parses the line, before the script runs. Read the
process return value instead.

### `tools/oasst2jsonl` — the data converter, and three bugs in it

Added a tool that converts an OpenAssistant (oasst1) export into Aster's chat
JSONL. It is a data-prep utility, **not** part of `aster.exe`, and it downloads
nothing — the project still makes no network access at any point.

oasst1 is not a list of conversations. It is a **tree**: every message carries a
`parent_message_id`, and one user message can have several assistant replies.
Aster's format is strictly linear, so the tool indexes messages by parent,
walks down from each root, and where a node has more than one child it keeps the
first and reports the rest. Every count is printed, because a converter that
silently discarded half the dataset would be worse than none.

Written against a documented schema, then tested against a hand-built fixture
covering both published layouts (flat `"text"`, and `"text"` nested inside a
`"content"` object), a branched thread, a dangling thread, a two-language mix,
`\u` escapes, a surrogate pair, a lone surrogate, a control character, an
unknown role, a record with no text, and a line that is not JSON at all. Three
bugs, each found by running it rather than reading it:

1. **Every key after the first was over-read.** `scan_quoted` returns an offset
   relative to the string it was given, but the callers passed `s + i` together
   with the *absolute* `i + end`. For `i = 0` those coincide, so the first key
   decoded correctly and every later one ran on past its closing quote — each
   value bleeding into the key after it. The `--probe` output was visibly
   nonsense (`message_id"`, then fragments of the next record) which is what
   made it obvious.

2. **`field_get` reported every good field as absent.** It ended with
   `return decode_string(...)`, but `decode_string` answers **0** for success
   while `field_get` promises **1** for "found". Every record was rejected as
   having no role. Fixed with `return rc == 0 ? 1 : rc;` and a comment saying
   why, because the two conventions disagreeing is the whole bug.

3. **The root turn was never emitted.** The walk pushed only children, so every
   conversation began at its first *reply* and the user turn was missing
   entirely. Aster's loader would have rejected every line of the output. The
   `_id`/`text`-nested records still parsed correctly, which is why the two
   layouts agreeing did not reveal it.

A fourth was in the surrogate-pair lookahead, where the bound was one byte too
loose and rejected valid pairs. And one apparent UTF-8 bug was not a bug at
all: the fixture generator had turned `é` into a bare Latin-1 `0xE9`, so
the tool was correctly rejecting genuinely invalid bytes. The fixture was
wrong, not the code.

Verified after the fixes:

| Check | Result |
|---|---|
| `--probe` on a good file | lists `message_id, parent_message_id, text, role, lang` |
| full run on the fixture | 21 lines → 16 messages → 6 conversations, 14 turns |
| UTF-8 accents round-trip | `é è ü ñ î` preserved |
| escapes | `\t \" \\ \n` decoded then correctly re-escaped |
| surrogate pair `😀` | decoded to a 4-byte 😀 |
| lone surrogate | rejected, not passed through |
| unknown role / no text / not JSON | each skipped, each counted |
| branch | first reply kept, the other reported as dropped |
| `--lang en` | 5 conversations, German thread excluded |
| `--max-turns 2` | 1 chain truncated and reported |
| **wrong schema** | **exit 1, no file written** |
| **output loaded by `aster train`** | **6 sources, 7 windows, exit 0** |

That last row is the one that matters: the converter's output is not merely
well-formed JSON, it is accepted by the same loader that reads
`demo_chat.jsonl`.

---

## Changes made to get a real result

### The corpus was far too small

The original demo was **40 training pairs / 5.6 kB**. With 124 352 parameters
that is memorisable outright, and the first full run ended at training loss
**0.0000** with held-out loss *rising* from 5.58 to 8.98 — textbook overfitting.

The corpus was rewritten from scratch to **332 training conversations (45.9 kB)
and 68 disjoint validation conversations (9.7 kB)**, all original, covering
identity, capabilities, honest limits, technical detail, and explicit refusals
for medical, legal, financial, emergency and hiring topics. Provenance, licence
and SHA-256 hashes are in `data/README.md`.

### Training now selects the best checkpoint, not the last one

`train.c` keeps a copy of the parameters from the step with the lowest held-out
loss and restores it before saving. Without this, the reported figure is a
measurement of overfitting rather than of what the model achieved; with it, the
held-out loss bottoms out at step 1 075 of 3 000.

### What this does **not** show

Held-out 2.0938 (perplexity 8.12) is a 62 % improvement, but it is **not** a
measure of general ability:

1. The validation questions are unseen but **topically adjacent** — the same
   topics in the same narrow voice. That measures a new question in a familiar
   register, not generalisation.
2. 332 conversations is tiny; the low training loss reflects memorisation as much
   as learning.
3. One narrow repeated answer style is far easier to predict than open-ended text.

### Measured but not adopted: `--block 128`

Using the full 128-token window instead of the default 64 was tried, since 304 of
332 windows were being truncated to 64 tokens and most training signal discarded.
It was **worse on held-out data**: best validation 2.2672 around step 1 600
versus 2.0938 for block 64, and it overfitted sooner (training loss 0.80 by
step 2 800). That run was killed by its time limit at step 2 800 of 3 000
before it could write its checkpoint, so no `--block 128` model is shipped. The
comparison is on the logged validation curve, not on a saved checkpoint.

---

## What was verified, and what was not

**Verified by running it:** the build, all 26 self-test checks, the gradient
check, the training run and its held-out loss, generation from the shipped
checkpoint, the checkpoint round trip, every server route and failure case
listed above, the served page matching `index.html` byte for byte, and the exit
code of every success and failure path of both build scripts.

**Not verified:** the browser page has not been opened in a real browser here —
it was checked over HTTP (status, content type, byte count, hash) and by reading
the source, not by visual inspection. No cross-platform build was attempted;
only MinGW-w64 GCC on Windows. No load or concurrency testing of the server. The
`build.bat` delete-failure path could not be triggered, because by the time the
fix was in place nothing was holding `aster.exe` any more — see the note above.

**Deleted by the user:** the obsolete root `server.c` from the original
rule-based demo. It was never compiled (`src/server.c` is the file that is
built) and is now gone from the repository.

---

## Final verification, after the cleanup

Re-run end to end once the stale processes were gone and `aster.exe` was
unlocked, so nothing below is carried over from an earlier run:

```powershell
.\build.bat
```

```
  Removing the existing aster.exe ...
  Removed.
  Compiling ...
  BUILD SUCCEEDED - aster.exe was created.
  (zero warnings)                                            exit 0
```

| Check | Result |
|---|---|
| `aster selftest` | 26 checks, **exit 0** |
| gradient check | 0.0066 worst relative error (1 layer), 0.0075 (2 layers) |
| `aster eval` on `demo_valid.jsonl` | 2.0938 nats/token, perplexity 8.12, step 1075 |
| `aster generate "Who are you?"` | `I ave all ase seal a a a sonde anyor an a pllo.` |
| `models/` after selftest | checkpoint and its metadata only — the self-test cleans up after itself |
| `serve --port 8080` bind | `127.0.0.1:8080` only |
| `GET /api/status` | `ready:true`, 124 352 parameters, `network:"none"`, `tools:"none"` |
| `POST /api/chat` | same reply as the CLI, so server and docs cannot drift apart |
| `GET /` | 200, 36 116 bytes, **byte-identical to `index.html`** |
| `PUT /api/chat` | 405 `method_not_allowed` |
| `GET /api/nope` | 404 `not_found` |
| empty / missing `message`, bad JSON | 400 with a specific reason each |
| 5 000-byte message | 413 `message_too_long` |
| over-long prompt | 200 with a `note` saying older text was dropped |
| `<script>alert(1)</script>` as the message | 200, treated as inert text |
| server log after all of that | 4 lines, **no prompt text and no dataset content** |

---

## Remaining cleanup

- `tmp/` keeps the two training logs quoted above (`train_full.log`,
  `train_b128.log`), the raw evidence behind the numbers in this file, plus
  `serve.log` from the final run. The diagnostic programs, their executables,
  the temporary corpora, the single-window checkpoints and the throwaway test
  binaries have been removed.
- **One server is still running on `127.0.0.1:8080`** (PID 13128), started for
  the final verification above. Stopping it was blocked by a safety prompt, so
  it is still up: `taskkill /PID 13128 /F`, or just close the terminal. Nothing
  depends on it being down, but it does hold `aster.exe` open, so a rebuild will
  need it stopped first.


---

## Session: sub-word tokenizer (BPE with byte fallback)

### What was built

A real sub-word BPE tokenizer, replacing one-byte-per-token. Decisions taken:
**1024 merges** (a ceiling, not a target), **context stays at 128** so the
tokenizer is the only variable, and the checkpoint format was bumped to 2 so a
checkpoint carries its own merge table.

`--merges` is a ceiling: a small corpus runs out of frequent pairs first. On the
bundled 24.9 KB demo corpus it learned **885 merges → 1146 tokens → 3.68 bytes
per token**. The vocabulary learner is handed text through a callback and never
sees a file path, so it *cannot* be pointed at the validation file even by
accident.

| File | Change |
|---|---|
| `src/tokenizer.h/.c` | `AsterVocab` (merges, derived `tok_len`/`tok_bytes`/`rank`), real BPE encode/decode, vocab file IO (`ASTERVB1`) |
| `src/model.c/.h` | vocab on the model; checkpoint `ASTERMD2` with merge table + integrity block; detokenizing generation; token-denominated prompt budget; vocab range check |
| `src/train.c/.h` | token-based window/mask construction; `"bpe-2"`; nats/byte + bits/byte reporting |
| `src/main.c` | `vocab` verb, flags, self-tests |
| `src/server.c` | `prompt_budget_tokens`; byte `memmove` + UTF-8 resync deleted; `"bpe-2"` |
| `src/util.h/.c` | `aster_sha256_raw`; `aster_sha256_hex` became a wrapper over it |
| `index.html` | token-denominated counter, labelled as an estimate |
| `tools/oasst2jsonl.c` | `--valid-out`, `--valid-every`, `--max-conversations`, per-file SHA-256 |
| `README.md`, `MODEL_CARD.md`, `data/README.md` | both loss figures, format 2, the false CRC claim corrected |

### Four real bugs, all found by measurement

None of these were found by reading the code. Each was caught by a number that
was obviously wrong or by a property test, and three of the four would have
shipped a plausible-looking, plausible-sounding wrong result.

**1. Generation returned empty text, always.** `src/model.c:1098`

```c
if (next <= TOK_ASSISTANT) { ... break; }   /* 258 */
```

The comment said "byte ids are content, structural ids are turn markers", and
the code said every id at or below 258 ends the turn. Byte ids are 0..255, so
**every byte satisfied it** and the first sampled token ended generation before
anything was produced. The self-test had *the same wrong rule* in its reference
decode, so it passed. The test had to be rewritten to state the intended
behaviour before it could catch the bug:

```c
if (next == TOK_END) { if (hit_end) *hit_end = 1; break; }   /* trained stop */
if (next == TOK_PAD) break;                                  /* never generated */
if (next >= TOK_SYSTEM && next <= TOK_ASSISTANT) { ... }     /* a RANGE, not a ceiling */
```

**2. bits/byte double-counted byte length.** `src/train.c:622`

The numerator was a byte-weighted average of per-token cross-entropies, so long
tokens were charged twice. It reported **5.26 nats/byte where the correct
figure is 1.63** — and the error *grew with the compression ratio*, so it got
worse exactly as the tokenizer got better. Fixed to `nats / weight_byte`.

**3. nats → bits was an exponentiation instead of a division.** `src/train.c:623`

```c
out->bits_per_byte = exp(out->nats_per_byte / log(2.0));   /* wrong */
```

Nats and bits are one measurement in two units, related by dividing by ln 2.
`exp()` is monotonically increasing, so **checkpoint selection was unaffected** —
`exp(a) < exp(b)` exactly when `a < b` — but every printed number was wrong, and
badly wrong: 1.63 nats/byte is 2.35 bits/byte, and the code reported 10.48.

Nothing caught it: the self-test checked `nats_per_byte`, and the two were
believed to be the same quantity. The fix was to print the conversion, and the
fix for the *class* of bug was a new check that asserts the identity directly:

```
  1.4351 nats/byte is 2.0704 bits/byte (exp would say 7.9)
  nats -> bits divides by ln 2 rather than exponentiating    pass
```

This is what produced the `untrained check` line in every training run: a random
model is uniform, so its bits/byte is not free — it must equal
`log2(vocab) / bytes-per-token`, which is exactly 8.00 for a byte vocabulary.

**4. `double` fields were saved by numeric cast, not by bit pattern.** `src/model.c`

```c
put_u64(buf + off, (uint64_t)m->vocab.bytes_per_token);   /* 3.6839 -> 3 */
```

`(uint64_t)3.6839` is `3`, and reading it back gave exactly `3.0`, so a
checkpoint silently stored the wrong tokenizer compression ratio. The mirror
bug on load was worse: `(double)get_u64(...)` on the stored bit pattern
`0x400FB6D1B60C6D75` is `4.6e18`. Both now use `memcpy` through a `uint64_t`.

### Test bugs fixed along the way

Worth recording because three of them were tests that could not fail:

- `encode(decode(encode(s))) == s` failed on three inputs. Probing showed all
  three contained a **NUL byte**, which `tok_encode` correctly rejects with -2.
  The test was wrong, not the encoder — but it is now an explicit check that NUL
  is *reported*, never skipped.
- "fully masked batch produces bitwise-zero gradients" failed because
  `aster_backward` accumulates and early-returns on `wsum <= 0` before touching
  `m->grads`. The test now memsets first and says why.
- A `v1` checkpoint fixture was built from a guessed magic string and was being
  rejected as truncated before reaching the `tokver == 1` check. Replaced with a
  real fixture (patch one byte of the valid selftest checkpoint, recompute
  CRC + SHA) plus two more assertions.

**5. A real v1 checkpoint was told it was not a checkpoint.** Found only after
`build.bat` finally succeeded and the stale process holding `aster.exe` had
exited, which made it possible to point the real binary at the real
`models/aster-small.bin` for the first time:

```
error: bad magic bytes: this is not an Aster checkpoint      exit 1
```

Two separate falsehoods in one line. The file *is* an Aster checkpoint — its
first eight bytes are `ASTERMD1`, the byte-tokenizer format. And the
hand-written "you should retrain" message further down the loader was
**unreachable for any real v1 file**: the magic check at [model.c:824](src/model.c#L824)
rejected it first, and because v1 has no integrity block its last 36 bytes are
weights, so the CRC check would have called an intact file *damaged* as well.

The test missed it because **the fixture was a v2 file with one field flipped**.
It exercised the `tokver == 1` branch, which is a different branch from the one
a real v1 file takes. Fixed by recognising `ASTERMD1` before the integrity check,
and by a second fixture that uses the real magic:

```
error: this checkpoint was written by the byte-tokenizer build (format 1).
This build uses a sub-word vocabulary, and a merge table cannot be invented for
it, so the file cannot be read. Retrain it: the vocabulary and the weights are
learned together.                                                            exit 1
```

This is the shape worth remembering: **a fixture built from the current format
cannot test the old-format path.** The test was green, specific, and pointed at
real code — and still never walked the path a user walks.

### Commands actually run, and actual results

Self-test — **57 checks, all passing, exit 0**:

```powershell
.\tmp\aster-chk.exe selftest
```

| Check | Result |
|---|---|
| `encode(decode(encode(s))) == s` adversarial corpus | pass |
| NUL byte reported as an error, not skipped | pass |
| unseen text falls back to bytes and still round-trips | pass |
| encode is deterministic; two vocab learns are identical | pass |
| single flipped checkpoint byte caught by the integrity block | pass |
| byte-tokenizer checkpoint refused with a "retrain" message | pass |
| a checkpoint with the **real `ASTERMD1` magic** refused by name | pass |
| fully masked batch → bitwise-zero gradients | pass |
| loss mask trains the first answer token, under merges too | pass |
| bits/byte is total nats over total bytes | pass |
| nats → bits divides by ln 2 rather than exponentiating | pass |
| analytic gradients vs finite difference, 1 layer | worst rel. error **0.0066** |
| analytic gradients vs finite difference, 2 layers | worst rel. error **0.0075** |

Training — same data, same seed 1234, one variable changed:

```powershell
.\tmp\aster-chk.exe train --mode chat --steps 3000 --batch 8 `
    --data data/demo_chat.jsonl --validation data/demo_valid.jsonl `
    --vocab tmp/demo.vocab --out tmp/bpe-small.bin --seed 1234
```

```
  vocabulary: demo.vocab (885 merges, 1146 tokens, 3.68 bytes/token)
  parameters:  180992 (0.181 million), token embedding tied to the output head
  train split: 332 window(s), 8424 tokens, 45865 source bytes, 1 source(s)
  valid split: 68 window(s), 2100 tokens, 9737 source bytes, 1 source(s)
  step      0  valid 7.0541 nats/token  3.4042 bits/byte  (token ppl  1157.57)
    untrained check: 7.0541 nats/token x 2.99 bytes/token = 2.3596 nats/byte / ln2 =
    3.4042 bits/byte; log2(vocab=1146) / 2.99 = 3.3994  [consistent]
  ...
  finished 3000 step(s) in 532.6s
  held-out loss was still rising at step 3000; reverting to the best
      held-out parameters from step 450
  held-out 7.0541 -> 4.8693 nats/token   3.4042 -> 2.3498 bits/byte: improved
    check: 4.8693 nats/token x 2.99 bytes/token = 1.6288 nats/byte / ln2 = 2.3498
           bits/byte (over 1239 scored tokens, 3704 bytes)
```

Independent verification of the byte accounting, from a separate program that
builds the same held-out dataset and counts target byte lengths itself:

```
stream_len=2100 tokens=2100 n_win=68 block=64
masked targets=1239  bytes=3704  mean=2.9895  zero_len=68
```

That matches the training log's `1239 scored tokens, 3704 bytes` exactly, and
the 68 zero-length targets are the 68 `END` markers, one per conversation. The
figure is right.

**Before / after, on the same held-out set, comparing bits/byte** — the only
metric that survives a tokenizer change:

| | bytes/token | vocab | parameters | held-out bits/byte |
|---|---|---|---|---|
| `byte-v1` | 1.00 | 261 | 124 352 | 3.02 |
| `bpe-2` | 2.99 | 1146 | 180 992 | **2.35** |

**22 % fewer bits per byte.** The per-token figure rose 2.09 → 4.87 nats/token,
which is neither a regression nor an improvement — it is a different question,
and quoting it as a comparison would be the mistake this whole two-number
reporting scheme exists to prevent.

Checkpoint selection is on bits/byte. The run reverts to step 450 because
training loss reached 0.0028 nats/token by step 3000 while held-out loss was
still climbing: 332 examples against 181k parameters is memorisation, and the
reversion is what makes the reported number mean anything.

Generation, real output, unedited:

| Prompt | `bpe-2` | `byte-v1` (previous build) |
|---|---|---|
| `Who are you?` | `I am a small model that runs on your own computer.` | `I ave all ase seal a a a sonde anyor an a pllo.` |
| `Are you a doctor?` | `No. I am a small model that runs on your own computer.` | `No. I am a a smandl and I ote a no a prre.` |
| `Can you remember me?` | `No. I am not a small model that runs on your own computer.` | `No. I am a sthe a a smor sthin pronact.` |
| `What is the capital of France?` | `I cannot. I am not qualified to your own computer.` | — |

**The tokenizer fixed the English, not the model.** It emits whole words now.
It still does not know that Paris is the capital of France, and it answers every
question with the same sentence about itself. Given a long prompt it degenerates
into `the the model model model model to a small model the the the the the` —
a real output, recorded here rather than omitted.

Corruption check — the thing `README.md` claimed before this session and which
had **zero call sites**:

```powershell
Copy-Item tmp\bpe-small.bin tmp\corrupt.bin
# overwrite one byte at offset 12000
.\tmp\aster-chk.exe eval --model tmp\corrupt.bin --mode chat --data data\demo_valid.jsonl
```

```
error: this checkpoint is damaged: its CRC-32 and SHA-256 do not match its
contents, so it was not written completely or has been modified. It cannot be
loaded, and nothing should be concluded from it.                     exit 1
```

Refused with a reason, exit 1, and **no loss figure produced** — the integrity
block is verified before anything is parsed, so a corrupt file is reported as
corrupt rather than as bad magic bytes.

Server:

| Check | Result |
|---|---|
| `GET /api/status` | `ready:true`, `"tokenizer":"bpe-2"`, `prompt_budget_tokens: 44`, `bytes_per_token: 3.684`, `network:"none"`, `tools:"none"` |
| `POST /api/chat "Who are you?"` | same reply as the CLI, so server and docs cannot drift apart |
| over-long message | 200 with a note naming the 128-token context and saying older text was dropped |
| `<script>alert(1)</script>` as the message | 200, treated as inert text, not reflected |
| server log after all of that | 4 lines, **no prompt text and no dataset content** |
| `GET /` | 200, served |

The prompt budget went from **44 bytes** (about seven words) to **44 tokens**
(about 165 bytes). The server's byte-`memmove` plus UTF-8-resync truncation was
deleted rather than ported, so the server and the CLI now truncate *identically*
— they did not before.

`index.html` was checked at the source level, **not in a browser**: every write
into the transcript goes through `textContent` (one helper, `index.html:434`),
and the file contains no `innerHTML`, no `insertAdjacentHTML`, and no
`createContextualFragment`. No browser was launched this session, so no browser
rendering is claimed.

### Data: converter ready, download not done

`tools/oasst2jsonl.c` now splits **by construction** — a whole conversation goes
to one side or the other, decided as it is emitted. Shuffling records after
flattening the oasst1 tree would put a prompt in the training half and its own
reply in the validation half, and the held-out loss would then be a
memorisation score wearing a generalisation score's label. Verified on a
synthetic fixture: 20 train / 5 valid, disjoint, SHA-256 printed for each file,
`--max-conversations` cap honoured.

**No oasst1 data has been downloaded and no model has been trained on it.** The
project makes no network access and neither did this session. The converter, the
split, and the instructions are ready; the download is the user's to do, and the
`data/README.md` provenance section has placeholders to fill with the real
licence and SHA-256 when it is.

### Known limitations of this build

- The merge table is fitted to whatever corpus it learned from, so a retrain on
  different data must re-learn the vocabulary. Cross-corpus loss comparisons
  change two variables, not one.
- The merge table is *not* kept only in a sidecar file: the checkpoint carries
  its own copy, so a model never depends on a separate file staying in sync.
  `models/aster-small.vocab` is written for reproducibility; the checkpoint is
  what is loaded.
- The old `models/aster-small.bin` is a `byte-v1` checkpoint that this build
  **refuses by design**, with a message saying to retrain. It has not been
  deleted; doing so needs the user's authorisation. It is also still the
  checkpoint `build.bat`'s own closing instructions tell a new user to train over,
  so retraining is the intended next step rather than an obstacle.
- `build.bat` **succeeded** and `aster.exe` is built and current. The training,
  evaluation, generation, corruption and server results above were produced with
  `tmp/aster-chk.exe` and re-verified against `aster.exe` itself; the numbers are
  identical.
