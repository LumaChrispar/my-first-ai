# Aster build progress

A running record of what has actually been built, which commands were actually
run, and what the actual results were. Nothing in here is aspirational: if a
step is listed as done, the command and its output are recorded. If something
is broken, it says so.

Last updated: 2026-09-30

---

## Status at a glance

| Area | State |
|---|---|
| Build (`-Wall -Wextra`, MinGW-w64 GCC) | clean, zero warnings |
| Self-test suite | 23 checks, all passing |
| Tokenizer / UTF-8 safety | verified |
| Checkpoint save/load round trip | verified bit-exact |
| HTTP server (loopback, bounded) | implemented, not yet end-to-end tested |
| Training reduces loss | **partially** — see "Backward pass debugging" |
| Generation produces coherent text | **no** — blocked on the backward pass |
| Browser UI rewrite | not started |
| README / MODEL_CARD / data provenance | not started |

The honest headline: **the model does not yet generate usable text.** Training
loss falls, but the backward pass is still wrong, so the model is learning
token statistics rather than conditioning on the prompt. Details below.

---

## What has been built

### Source (`src/`, all C11, no third-party dependencies)

| File | Contents |
|---|---|
| `util.h/.c` | logging, checked size arithmetic, allocation wrappers, atomic file write, SHA-256, CRC-32, xorshift32 RNG, normal distribution |
| `tokenizer.h/.c` | byte tokenizer (IDs 0-255 = raw bytes, 256=SYSTEM, 257=USER, 258=ASSISTANT, 259=END, 260=PAD; vocab 261), UTF-8 validation and sanitisation |
| `model.h/.c` | the transformer, manual forward + backward, checkpoint IO, generation |
| `jsonstr.h/.c` | minimal JSON string extraction and escaping for the HTTP layer |
| `train.h/.c` | dataset building (text and chat modes), training loop, evaluation |
| `server.h/.c` | loopback-only HTTP server and JSON API |
| `main.c` | CLI: `train`, `generate`, `serve`, `eval`, `selftest`, `help` |

### Model, as specified

- 2 layers, d_model 64, 4 heads (head dim 16), d_ff 256, context 128, vocab 261
- **Parameter count: 124,352 (0.1244 million)** — matches the spec's "~0.13-million"
- Pre-LayerNorm, causal multi-head self-attention, GELU (tanh approx) feed-forward
- Token embedding is tied to the output head
- Next-token cross-entropy, AdamW with decoupled weight decay, global-norm
  gradient clipping, linear warmup + cosine decay, loss masking for chat mode

### Data (`data/`)

| File | Contents |
|---|---|
| `demo_chat.jsonl` | 40 self-written instruction pairs about Aster, honest about its limits |
| `demo_valid.jsonl` | 20 **disjoint** held-out pairs (no overlap with training) |
| `demo_text.txt` | ~4.5 kB of original prose about the project, for pretraining mode |

The demo corpus uses `{"role":"system","content":""}` — an empty system turn.
The loader requires a well-formed system turn (per spec), but generation
defaults to an empty system text, so the default training and serving paths
agree exactly with no extra flags.

---

## Commands actually run, and actual results

### Build (zero warnings)

```
gcc -std=c11 -O2 -Wall -Wextra -Isrc -o aster.exe src/main.c src/model.c \
    src/train.c src/server.c src/tokenizer.c src/util.c src/jsonstr.c -lws2_32 -lm
```

### Self-test

```
./aster.exe selftest     ->  self-test passed  (23 checks, exit 0)
```

Covers: tokenizer round trips, distinct markers, tiny output buffer, invalid
UTF-8 replacement, parameter count vs hand-computed layout, untrained loss
≈ ln(261), finite gradients, fully-masked batch gives zero loss, out-of-range
token rejection, checkpoint round trip, rejection of truncated /
non-checkpoint / missing files, greedy determinism, UTF-8 validity,
`max_new_tokens` cap, JSON decode / reject / escape / `\u` handling, and the
chat loss-mask alignment check (below).

### Training run (chat mode)

```
./aster.exe train --mode chat --steps 1500 --batch 8 \
    --data data/demo_chat.jsonl --validation data/demo_valid.jsonl \
    --out models/aster-small.bin --seed 1234
```

```
  train split: 40 window(s), 2042 tokens
  valid split: 20 window(s),  950 tokens
  step      0  valid loss 5.6199  perplexity 275.86  (untrained)
  step    300  train 3.2948  valid 3.2206
  step   1500  train 3.3366  valid 3.3244
  held-out loss 5.6199 -> 3.3244  (perplexity 27.78): improved
```

Held-out loss fell by 41%, which satisfies the spec's "held-out loss should be
meaningfully below its initial value" — **but see the caveat below.** A loss
that plateaus with train ≈ valid ≈ 3.3 is the signature of a model that has
learned unigram token statistics and *not* learned to condition on the prompt.

### Single-window overfit test (the test that exposed the real problem)

```
./aster.exe train --mode chat --block 64 --batch 1 --steps 600 --lr 3e-3 \
    --data tmp/one.jsonl --out tmp/one.bin
```

```
  step    100  train 2.5770
  step    600  train 1.8464
```

A 124k-parameter model **cannot memorise a single 19-character sentence**.
That is not a capacity problem; it is a gradient bug. This test is what
redirected the work from "not enough training" to "the backward pass is wrong".

---

## Backward pass debugging

Rather than guess, I wrote a finite-difference gradient check
(`tmp/gradcheck.c`, a throwaway tool, not part of the build) that compares every
analytic gradient against a central difference on a small config
(1-2 layers, d_model 8, 1-2 heads). This found several real bugs.

### Bugs found and fixed

1. **Parameter layout overlap.** `ff1`'s bias was not skipped when computing
   `ff2`'s offset, so the bias and the `ff2` matrix overlapped. The parameter
   count was also short by 64, which meant the init loop wrote 64 floats past
   the end of the allocation (heap overflow). Now 124,352, matching the
   hand-computed layout.

2. **Chat loss mask off by one.** `dataset_gather` sets `y[i] = stream[i+1]`, so
   `w[i]` gates the prediction of token `i+1`. The mask started at the answer
   *bytes*, leaving the `TOK_ASSISTANT` position at weight 0 — so **the first
   byte of every answer was never trained.** The model consequently never
   learned to open a reply. Fixed, and a self-test now asserts the alignment
   (verified: the test fails against the old code and passes against the new).

3. **GELU derivative evaluated at the wrong point.** The forward overwrote
   `ffh` in place with the post-GELU value, so the backward called
   `gelu_d(gelu(z))` instead of `gelu'(z)`. Added a separate `ffz` buffer
   holding the pre-activation. This was corrupting every gradient downstream of
   the feed-forward.

4. **Weight-gradient pointers chained through a non-existent bias.** The
   pointers for `Wk`, `Wv` and `Wo` were derived as `prev + C*C + C`, assuming
   a bias slot that the layout does not have. Each was off by one row. Now each
   comes from its own offset.

5. **Residual stream indexed with the wrong stride.** `IDXL` strides by `L`, but
   `xin` and `dxin` hold `L+1` residual stages. For the **default 2-layer
   config the entire residual stream was addressed incorrectly.** Added a
   dedicated `IDXR` macro with the `L+1` stride.

6. **In-place residual accumulation.** The layer loop built `d(xin[l])` in
   `dxin[l+1]`, clobbering the value the next layer down still needed to read;
   for 2 layers, `dxin[1]` was never written at all. Added a dedicated `dres`
   accumulator per layer, and the layer backward is now split into two passes
   because the attention backward needs `d_att`, which is only produced by the
   feed-forward/projection pass that previously ran *after* it.

### Current state of the gradient check

After the fixes above, most regions match the finite difference to within
float32 noise. **Still wrong:**

| Region | Status |
|---|---|
| `Wq`, `Wk`, `Wv`, `ff1`+bias, `ff2`, `lnf_w`, `lnf_b` | match |
| `ln1_w`, `ln1_b` | analytic ≈ 0 where numeric ≈ 8e-3 — **not yet fixed** |
| `pos_emb` | off by roughly 11x — **not yet fixed** |
| `tok_emb` | 47 of 1068 significant entries wrong, 24 of them silently zero |
| `Wo` | 2 entries off; may be finite-difference noise |

So there is at least one more real bug in the `ln1` path. That is the current
front line of work.

---

## Other changes made during debugging

- Added `--system TEXT` to `generate` and `serve`, because the training window
  and the generation prefix disagreed: training always had the system text
  between `SYSTEM` and `USER`, generation had nothing there, so the model saw
  an adjacency it had never been trained on. A model this size cannot infer the
  system text, so the mismatch is now an explicit, documented option instead of
  a silent failure. `aster_prompt_budget` now accounts for the system text.
- Fixed a stall in generation: when the model emitted a structural marker the
  loop did not advance the context, so the identical argmax was resampled until
  the step budget ran out. It now stops and reports truncation.
- Fixed the `--mode chat` omission in the help text's own example.
- The old top-level `server.c`, `index.html` and `README.md` are still present
  and still describe the previous rule-based demo. They need to be replaced or
  removed.

---

## Next steps

1. Fix the remaining `ln1` gradient bug, then re-run the gradient check until
   it is clean across 1 and 2 layers and 1 and 2 heads.
2. Promote the gradient check into `aster selftest` so it cannot regress.
3. Re-run the single-window overfit test — it should drive loss toward zero.
4. Re-run training and check that held-out loss falls *and* that generation
   produces coherent replies.
5. Rewrite `index.html` (text-only rendering, model status, honest warnings).
6. Write `README.md` with exact PowerShell commands, `MODEL_CARD.md`, and
   `data/README.md` with provenance and licence instructions.
7. Delete the obsolete top-level `server.c`; add `build.bat`.
8. End-to-end test the server: health, chat, 503-without-model, loopback-only.
