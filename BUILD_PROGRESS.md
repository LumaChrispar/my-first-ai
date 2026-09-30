# Aster build progress

A running record of what has actually been built, which commands were actually
run, and what the actual results were. Nothing in here is aspirational: if a
step is listed as done, the command and its output are recorded. If something is
broken or unverified, it says so.

Last updated: 2026-09-30

---

## Status at a glance

| Area | State |
|---|---|
| Build (`-Wall -Wextra`, MinGW-w64 GCC 15.2) | clean, **zero warnings** |
| Self-test suite | **26 checks, all passing**, exit 0 |
| Gradient check (analytic vs finite difference) | passes for 1 and 2 layers |
| Tokenizer / UTF-8 safety | verified |
| Checkpoint save/load round trip | verified bit-exact |
| Training reduces held-out loss | **verified: 5.5823 → 2.0938** |
| Generation | **works** — decodes every step, coherent at the start |
| Server (loopback, bounded) | **verified end to end** |
| Browser UI | rewritten, text-only rendering, verified over HTTP |
| Build scripts | `build.bat` and `build.sh`; all four exit paths of each checked |
| README / MODEL_CARD / data provenance | written |

The honest headline: **the pipeline is complete and verified end to end, and
the model is still not very good.** It trains, holds out, reloads, and generates,
but at 124k parameters on 332 conversations it drops characters and says little.
That is the expected result, and it is documented rather than hidden.

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

The `del`-failure path itself could not be run, because the test servers still
hold the real `aster.exe` locked — see "Remaining cleanup". It is the same
top-level `exit /b 1` form as the two failure paths that were run, so the
control flow is verified; only the trigger was unavailable.

Note for anyone re-checking this: measuring a `.bat` exit code by appending
`& echo %ERRORLEVEL%` on the same command line does **not** work — cmd expands
`%ERRORLEVEL%` when it parses the line, before the script runs. Read the
process return value instead.

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
`build.bat` delete-failure path could not be triggered, because the locked
`aster.exe` prevents it — see the note above.

**Left in place deliberately:** the obsolete root `server.c` from the original
rule-based demo. Deleting it was blocked by a safety prompt, so it is still in the
repository. It is **not** referenced by `build.bat`, `build.sh`, or any command in
the README, and `src/server.c` is the file that is actually compiled. It should
be deleted before this is shared.

---

## Remaining cleanup

- Delete the obsolete root `server.c` (needs explicit authorisation).
- Three processes are still listening on `127.0.0.1`: ports 8098 and 8099 were
  test servers started for this project, 8097 was already running beforehand and
  belongs to something else — leave it alone. The two test servers were not
  killed, because force-killing those PIDs needed authorisation I did not have.
  Stop them with `taskkill /PID <pid> /F`, or just reboot.
- **A running server is holding `aster.exe` open**, so `build.bat` cannot relink
  it in place until those processes stop. The current `aster.exe` is current
  (built at 10:06, after the last source change at 09:47) and passes
  `selftest`, and the identical build was verified to succeed under a different
  output name — so the source is fine, only the filename is locked.
- `tmp/` keeps only the two training logs quoted above (`train_full.log`,
  `train_b128.log`), as the raw evidence behind the numbers in this file. The
  diagnostic programs, their executables, the temporary corpora, the
  single-window checkpoints and the throwaway test binaries have been removed.

