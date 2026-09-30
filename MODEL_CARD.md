# Model card — Aster (`aster-small`)

An experimental, deliberately tiny text model. This card records what it is,
what it was trained on, what it can do, and — in more detail — what it must not
be used for.

## Summary

| | |
|---|---|
| Name | `aster-small` |
| Type | decoder-only causal transformer, trained from scratch |
| Parameters | **189 888** at the full 1285-token vocabulary (0.190 M); 180 992 on the bundled demo corpus, which supports 885 merges |
| Layers / width | 2 layers, `d_model` 64 |
| Attention | 4 heads, head dim 16, causal mask |
| Feed-forward | `d_ff` 256, GELU (tanh approximation) |
| Context | 128 tokens |
| Tokenizer | `bpe-2` — sub-word BPE **with byte fallback**. 256 raw byte values plus `SYSTEM`, `USER`, `ASSISTANT`, `END`, `PAD`, plus up to 1024 learned merges |
| Weight tying | the token embedding **is** the output head |
| Checkpoint format | `ASTERMD2` v2, architecture version 1; the last 36 bytes are a CRC-32 and a SHA-256 covering everything before them |

The 256 byte ids are always present, so any byte sequence is encodable, there is
no unknown token, and no input can be rejected by the encoder. On the bundled
demo corpus the learned merges give **3.68 bytes per token**, so the 128-token
context holds roughly 470 bytes of text instead of 128.

**A previous `byte-v1` checkpoint is refused by this build, by design.** One
byte per token is a different model, not a slower one. Retrain rather than
expecting a conversion.

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
  affects a person's rights or opportunities.** A 190k-parameter model with no
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
| Steps | 3 000 (best checkpoint at step 450) |
| Batch | 8 windows × 64 tokens |
| Optimiser | AdamW, decoupled weight decay 0.01 |
| Learning rate | 1e-3, 100 warmup steps, cosine decay to 10 % |
| Gradient clipping | global norm 1.0 |
| Seed | 1234 |
| Vocabulary | 885 merges learned from the **training split only**, 1146 tokens, 3.68 bytes/token |
| Checkpoint selection | best held-out **bits per byte** during training (step 450) |
| Weight tying | enabled |
| Wall clock | 532.6 s on an ordinary processor |

Selection on held-out data is deliberate. A 181k-parameter model given 332
examples drives its training loss towards zero and then keeps going — here it
reached **0.0028** nats/token by step 3000 while held-out loss was still
climbing. That is memorisation, not learning. Stopping at the best held-out step
and reporting that number is what makes the figure below mean anything.

## Results

Held-out loss on 68 conversations the model never saw, 1 239 scored tokens over
3 704 bytes:

| | bits/byte | nats/byte | nats/token |
|---|---|---|---|
| Before training (random init) | 3.4042 | 2.3596 | 7.0541 |
| Best (step 450) | **2.3498** | **1.6288** | 4.8693 |

**Bits per byte is the figure to compare.** Nats per byte is the same
measurement in different units, divided by ln 2. Nats per token is *not*
comparable against a byte-tokenizer model, because a sub-word vocabulary makes
every token easier to predict whether or not the model improved.

The untrained row is not a free parameter — a random model is uniform, so its
bits/byte must equal `log2(vocab) / bytes-per-token` = 10.164 / 2.99 = 3.3994,
against the 3.4042 measured. The training run prints that identity every time
so the headline can be checked by hand.

### Against the previous byte-tokenizer build

Same data, same seed, same held-out set, one variable changed:

| | bytes/token | vocab | parameters | held-out bits/byte |
|---|---|---|---|---|
| `byte-v1` | 1.00 | 261 | 124 352 | 3.02 |
| `bpe-2` (this card) | 2.99 (scored targets) | 1146 | 180 992 | **2.35** |

A **22 % reduction in bits per byte** — fewer bits spent per byte of English.
The per-token loss rose from 2.09 to 4.87 nats/token, which is not a regression
and not an improvement; it is a different question.

**This is not a 22 % better model.** It is a better *tokenizer*. What improved
is the English: the same prompt that used to produce

```
I ave all ase seal a a a sonde anyor an a pllo.
```

now produces

```
I am a small model that runs on your own computer.
```

and it answers `What is the capital of France?` with `I cannot. I am not
qualified to your own computer.` — grammatical, correctly-shaped, and wrong.
The model has no more knowledge than it did before.

**Read the figure narrowly.** It is not a measure of general capability, for
three reasons:

1. **The validation questions are topically adjacent to the training ones.** They
   are unseen but ask about the same topics in the same narrow voice, so this
   measures "can it answer a new question in a familiar register", not
   "can it generalise".
2. **332 training conversations is tiny.** The model memorises most of them; the
   training loss of 0.0028 reflects that as much as learning.
3. **A single narrow answer style inflates the score.** One assistant voice that
   repeats itself is much easier to predict than open-ended text — which is
   exactly what generation shows it doing.

A single-window overfit test reaches training loss 0.0000 and reproduces its one
training answer verbatim — which is the evidence that the forward pass, the
backward pass, the loss mask, and generation are mutually consistent.

## Intended users' out-of-scope behaviour to expect

- **It repeats itself.** This is the dominant failure mode. The model learned
  one answer voice and reuses it; a long or unusual prompt produces
  `the the model model model model to a small model the the the the the`.
  Greedy decoding makes this worse, not better, at this scale.
- **It gives short answers and stops early.** Most replies are one sentence,
  terminated by the trained `END` marker.
- **It answers questions it has no answer to**, in the same confident register.
  There is no uncertainty signal. Its refusals ("No. I am not a doctor") are
  right, and it gives them because they are in the training data, not because it
  understood the question.
- **It has no memory.** Every message is handled independently; it cannot see
  earlier turns, and it forgets as soon as the response is sent.
- **It used to drop characters mid-word**, which is why the tokenizer is
  sub-word. That specific symptom is fixed. Nothing replaced it except a
  different way of being wrong.

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

- **The vocabulary is learned, so it is a second thing being fitted.** A merge
  table fitted on one corpus does not transfer to another. If you retrain on
  different data, re-learn the vocabulary and re-measure; a before/after
  comparison across corpora changes two variables, not one.
- **The tokenizer is a confound for loss numbers.** Loss is reported both as
  bits per byte and as nats per token. Only bits per byte is comparable across
  tokenizers: a sub-word vocabulary makes every token easier to predict whether
  or not the model improved, so a fall in nats per token on its own means
  nothing.
- The 128-token context is still short. Sub-word tokenization bought roughly 3.7×
  more text per context, which is a large gain and still not much.
- No dropout or other regularisation beyond decoupled weight decay and
  best-checkpoint selection; overfitting begins early.
- Greedy decoding is the default. Sampling with a non-zero temperature makes the
  output worse, not better, at this scale.
- The integrity block (CRC-32 and SHA-256) detects corruption. It is **not** a
  signature and proves nothing about provenance — anyone can write a valid
  block over a forged file.

## Intended lifecycle

This is a teaching artifact. It is not intended for release, deployment, or
wider distribution. If you extend it, re-run `aster selftest`, retrain, and
update this card with what you actually measured.
