# Model card — Aster (`aster-small`)

An experimental, deliberately tiny text model. This card records what it is,
what it was trained on, what it can do, and — in more detail — what it must not
be used for.

## Summary

| | |
|---|---|
| Name | `aster-small` |
| Type | decoder-only causal transformer, trained from scratch |
| Parameters | **124 352** (0.124 M) |
| Layers / width | 2 layers, `d_model` 64 |
| Attention | 4 heads, head dim 16, causal mask |
| Feed-forward | `d_ff` 256, GELU (tanh approximation) |
| Context | 128 tokens |
| Tokenizer | `byte-v1` — 261 tokens: 256 raw byte values plus `SYSTEM`, `USER`, `ASSISTANT`, `END`, `PAD` |
| Weight tying | the token embedding **is** the output head |
| Checkpoint format | `aster-checkpoint` v1, architecture version 1 |

One token is one byte, so the 128-token context is about 128 bytes of text.

## Intended use

This model exists to demonstrate, honestly and end to end, that a small language
model can be built from first principles in plain C11: read a corpus, train a
transformer by manual backpropagation, write a versioned checkpoint, reload it,
generate text, and serve it from a loopback-only local HTTP server.

Its intended use is **learning, reading the source, and evaluating the pipeline**.
That is the whole of it.

## Out of scope — do not use it for

- **Medical, legal, financial, or emergency decisions.** It has no reliable
  knowledge in any of these areas and no way to check itself. For medical
  questions it should refer you to a doctor; for legal ones to a lawyer; for
  money to a financial adviser; for anything urgent to local emergency
  services.
- **Hiring, firing, credit, housing, education, or any other decision that
  affects a person's rights or opportunities.** A 124k-parameter model with no
  validation and no fairness testing has no business influencing these.
- **Anything where being wrong is expensive or dangerous.** Including
  production systems, safety-critical work, medical devices, and legal filings.
- **Generating facts, citations, numbers, or code to rely on.** It invents
  plausible-looking text, including confident and wrong text, and it gives no
  signal when it is doing so.
- **Summarising, translating, or rewriting** text you care about.
- **Live information.** It has no clock, no news, no weather, no prices, and no
  way to look anything up. It was trained on a fixed small corpus at a fixed
  time and knows nothing about the world since.

## Training data

332 short conversations written by hand for this project, plus 68 **disjoint**
held-out conversations for validation. Full provenance, licences, and SHA-256
hashes are in [`data/README.md`](data/README.md) and
[`data/manifest.example.json`](data/manifest.example.json).

- Nothing was scraped, downloaded, or copied from a copyrighted source.
- No personal data, credentials, API keys, or private conversations.
- Licence: CC0-1.0.

This corpus is **far too small** to teach a model general knowledge, and this
card does not claim it does.

## Training recipe

| | |
|---|---|
| Steps | 3 000 |
| Batch | 8 windows × 64 tokens |
| Optimiser | AdamW, decoupled weight decay 0.01 |
| Learning rate | 1e-3, 100 warmup steps, cosine decay to 10 % |
| Gradient clipping | global norm 1.0 |
| Seed | 1234 |
| Checkpoint selection | best held-out loss during training (step 1 075) |
| Weight tying | enabled |

Selection on held-out data is deliberate. A 124k-parameter model given 332
examples drives its training loss towards zero and then keeps going —
memorisation, not learning. Stopping at the best held-out step and reporting
that number is what makes the figure below mean anything.

## Results

Held-out loss on 68 conversations the model never saw:

| | Loss | Perplexity |
|---|---|---|
| Before training (random init) | 5.5823 | 265.69 |
| Best (step 1 075) | **2.0938** | **8.12** |

That is a 62 % reduction. **Read it narrowly.** It is not a measure of general
capability, for three reasons:

1. **The validation questions are topically adjacent to the training ones.** They
   are unseen but ask about the same topics in the same narrow voice, so this
   measures "can it answer a new question in a familiar register", not
   "can it generalise".
2. **332 training conversations is tiny.** The model can memorise a large
   fraction of them; the low training loss reflects that as much as learning.
3. **A single narrow answer style inflates the score.** One assistant voice that
   repeats itself is much easier to predict than open-ended text.

Perplexity 8.12 means roughly "about 8 plausible continuations at each
character", which for open-ended use is poor.

A single-window overfit test reaches training loss 0.0000 and reproduces its one
training answer verbatim — which is the evidence that the forward pass, the
backward pass, the loss mask, and generation are mutually consistent.

## Intended users' out-of-scope behaviour to expect

- **It drops characters.** At this size, generated text often loses or repeats
  characters mid-word. This is the normal failure mode, not an edge case.
- **It gives short answers and stops early.** An answer may be a single
  truncated sentence.
- **It answers questions it has no answer to**, in the same confident register.
  There is no uncertainty signal.
- **It is unreliable on repetition.** Left to run, it loops.
- **It has no memory.** Every message is handled independently; it cannot see
  earlier turns, and it forgets as soon as the response is sent.

## Safety

This model has **not** been evaluated, tested, or certified for safety by any
standard, and no claim of safety certification should be inferred from its
existence or from the checks in `selftest`.

The training corpus contains explicit refusals for medical, legal, financial,
emergency, and hiring topics, and the corpus is deliberately narrow so that the
model mostly stays inside them. **That is a mitigation by construction, not a
guarantee.** A model this small will still produce confident, wrong, and
occasionally harmful text on inputs it was never trained on.

Treat all model output as **untrusted text**. The browser UI renders it as
plain text and never as HTML.

## Network and data handling

- The server binds to `127.0.0.1` only. It is not reachable from the network.
- There is **no cloud API, no API key, no external model download, no
  telemetry, no web search, no tools, and no remote data transfer.**
- Prompts are not logged. The transcript stays in the browser page.
- This version has **no action tools and no tool endpoint**, by design. Nothing
  in the codebase executes a model-produced command.
- If no checkpoint is loaded, the server reports `model_not_trained` rather than
  returning random text as if it were a useful answer.

## Bias

A model trained on 332 hand-written conversations reflects those conversations:
one author, one culture, one language (English), and a deliberately narrow
register. It has no representation, no testing, and no fairness measurement
behind it. Do not use it in any decision that affects people.

## Known issues

- Byte-level tokenisation with no merging: the model spends most of its capacity
  learning individual characters, which is why it drops characters in longer
  words.
- No dropout or other regularisation beyond decoupled weight decay and
  best-checkpoint selection; overfitting begins early (validation loss bottoms
  out around step 1 075 of 3 000).
- Greedy decoding is the default. Sampling with a non-zero temperature makes the
  output worse, not better, at this scale.

## Intended lifecycle

This is a teaching artifact. It is not intended for release, deployment, or
wider distribution. If you extend it, re-run `aster selftest`, retrain, and
update this card with what you actually measured.
