# Data

Everything Aster trains on lives in this folder. There is no download step, no
scraping, and no external corpus: the three files here were written by hand for
this project, and the model is trained from random numbers on them.

## Files

| File | Bytes | Conversations | Purpose |
|---|---|---|---|
| `demo_chat.jsonl` | 45 865 | 332 | training set (`--mode chat`) |
| `demo_valid.jsonl` | 9 737 | 68 | held-out set (`--validation`) |
| `demo_text.txt` | 4 577 | — | plain prose, for `--mode text` |

`demo_valid.jsonl` is **disjoint from** `demo_chat.jsonl`: the questions differ,
not just their wording. It exists so the loss reported at the end of training is
measured on examples the model never saw. A validation set that repeats training
questions would make the number meaningless.

## SHA-256

Verify these before trusting a checkpoint:

```
76785434f78e29c02820bb38efabed70fadb1a7aa655c4f33f3b7adbcb320417  demo_chat.jsonl
dae7cb730c367ff4199b60585ed6d7f1c5a295c890e128e48bd1348928c199b9  demo_valid.jsonl
3c90cbb5e8dc163a9251dfe4662a7a98255c9f86120c8fb0cb93b6d543c01f9c  demo_text.txt
```

In PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 data\demo_chat.jsonl, data\demo_valid.jsonl, data\demo_text.txt
```

The trainer also writes an FNV-1a hash of each split into the run log and into
`<checkpoint>.meta.json`, so a log can be tied to an exact input without
re-hashing the file by hand.

## Provenance and licence

All three files are **original work written for this project**. Nothing was
copied, scraped, transcribed, or downloaded. They contain:

- no personal data, and no real person's name, contact detail, or identifier
- no private conversations, credentials, API keys, or confidential material
- no text from any copyrighted work, dataset, or language-model training set

The licence is **CC0 1.0** (public domain dedication) for `demo_chat.jsonl` and
`demo_valid.jsonl`, and the same for `demo_text.txt`. Put your own data in the
folder with your own licence — see below.

## Format

`--mode chat` reads **JSON Lines**: one JSON object per line, each with a
`role` and a `content` string.

```json
{"role":"system","content":""}
{"role":"user","content":"Who are you?"}
{"role":"assistant","content":"I am Aster, a very small model."}
```

Rules the loader enforces:

- `role` must be `system`, `user`, or `assistant`.
- A conversation starts with exactly one `system` turn, then any number of
  `user`/`assistant` pairs.
- `content` must be valid UTF-8. Invalid bytes are rejected, not silently
  replaced, so a mis-encoded file fails loudly.
- Every assistant turn must be non-empty.

The demo uses an **empty** `system` turn. The loader still requires the turn to
be present and well-formed, but generation defaults to an empty system text too,
so the training and serving paths agree exactly without extra flags.

`--mode text` reads plain UTF-8 text instead, treating the whole file as one
continuous stream and making next-token predictions with no loss mask.

## Adding your own data

Before you do, check that you are allowed to:

1. **Exclude anything you do not own.** Do not add private conversations,
   credentials, API keys, personal records, or confidential work. None of that
   belongs in a training set, and no amount of care makes it safe to publish in
   a repository.
2. **Only use material whose licence or terms permit it.** Check the licence
   explicitly and record it.
3. **Strip identifiers.** Remove names, addresses, phone numbers, email
   addresses, and anything else that identifies a person.
4. **Record what you did.** Copy `manifest.example.json` to
   `manifest.json`, fill it in, and commit it next to the data.

A one-line record of provenance is the difference between a dataset you can
publish and one you cannot.

## Using a public dataset: `tools/oasst2jsonl`

`tools/oasst2jsonl.c` converts an **OpenAssistant (oasst1)** export into
Aster's chat JSONL. It is a data-prep tool and is deliberately **not** part of
`aster.exe`.

**It does not download anything.** The project makes no network access at any
point, and neither does this tool. You fetch the dataset yourself with whatever
tooling you trust; this only reads a file already on disk.

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Isrc -o oasst2jsonl.exe `
    tools/oasst2jsonl.c src/util.c src/jsonstr.c src/tokenizer.c

.\oasst2jsonl.exe --in oasst.jsonl --lang en `
    --out data/oasst_train.jsonl `
    --valid-out data/oasst_valid.jsonl --valid-every 20 `
    --max-conversations 15000
```

If the file is not in the layout the tool understands, run `--probe` to see the
keys it actually has. A real run against an unrecognised file **exits non-zero
and writes nothing** rather than producing a wrong training set.

| Option | Meaning |
|---|---|
| `--in FILE` | the oasst1 export to read (JSON Lines) |
| `--out FILE` | where to write Aster-format JSONL |
| `--valid-out FILE` | also write a held-out split here |
| `--valid-every N` | every Nth conversation goes to `--valid-out`. Default: 20; off when `--valid-out` is absent |
| `--max-conversations N` | stop after N conversations in total. Default: 0 (all) |
| `--lang CODE` | keep only this language, e.g. `en`. Default: keep all |
| `--min-chars N` | drop turns shorter than N bytes. Default: 8 |
| `--max-turns N` | keep at most N turns per conversation. Default: 6 |
| `--probe` | print the record layout and exit; writes nothing |

### The split is made by construction

`--valid-every 20` sends every 20th conversation to `--valid-out` **as it is
emitted**, so a conversation's turns can never land on both sides. This is not
just tidiness — the alternative is wrong:

> If you flattened the tree first and *then* shuffled records, siblings would
> land on both sides. A prompt in the training half with its own reply in the
> validation half is not a held-out example at all, and the loss you report is a
> memorisation score wearing a generalisation score's label.

The tool prints the conversation count and the SHA-256 of each file it wrote, so
a run is reproducible from its own log and a file that later changes is
detectable.

### What it does to the data, and what that costs you

oasst1 is a **tree**, not a list of conversations: every message points at its
parent, and one user message can have several assistant replies. Aster's format
is strictly linear. So the tool walks down from each root and, where a node has
more than one child, **keeps the first and drops the rest**. That is lossy, and
the run reports exactly how lossy:

```
warning: 1 alternative replies were dropped: oasst1 is a tree and Aster's
         format is linear, so one reply per prompt is kept
```

Read those counts before you trust the file. A converter that quietly threw away
half the dataset and said nothing would be worse than no converter at all.

### Before you train on the result

1. **Read it.** It is other people's writing. Look for names, contact details,
   and anything personal. oasst1 was written by volunteers, so assume personal
   data is present — do not assume the dataset was scrubbed for you.
2. **Record the licence and the SHA-256** in this file, and add an entry to
   `manifest.json`.
3. **Learn the vocabulary from the training file only.**

   ```powershell
   .\aster.exe vocab --mode chat --data data/oasst_train.jsonl `
       --merges 1024 --out models/aster-small.vocab
   ```

   There is deliberately no `--validation` flag on `vocab`, because a merge
   table fitted on held-out data leaks it: the tokenizer would have already
   seen the words whose frequency it is about to be scored on, and the
   held-out loss would stop meaning what it appears to mean. The learner is
   handed text through a callback and never sees a file path, so it cannot be
   pointed at the wrong file even by accident.

   **One confound to be aware of when comparing across corpora:** the merge
   table is fitted to whichever corpus it learned from, so a model trained on
   oasst1 with 1024 merges and a model trained on the demo set with 885 merges
   are not the same tokenizer. If you change the training corpus, re-learn the
   vocabulary, and treat any cross-corpus loss comparison as two variables
   changed rather than one.

4. **Evaluate on two sets, not one.**

   | Set | Meaning |
   |---|---|
   | `data/oasst_valid.jsonl` | **in-distribution.** Measures fit to the oasst1 corpus. |
   | `data/demo_valid.jsonl` | **out-of-distribution.** Measures whether the model learned English, or merely learned oasst1. |

   If the first improves and the second does not, the model learned the
   corpus and not the language, and reporting only the first would be the
   dishonest choice.

## Why the demo corpus is small

332 conversations is tiny — far too small for a model this size to learn much,
and the training loss will still drop to near zero because a model this size can
memorise this much text. That is expected here and is not presented as
success. See `../MODEL_CARD.md` for what the model can and cannot do.
