# Data

The three bundled demo files here were written by hand for this project, and
the demo model is trained from random numbers on them. Optional external
experiments have separate provenance: see [aviator/README.md](aviator/README.md)
for the user-requested Kaggle numeric sequence model.

## Files

| File | Bytes | Conversations | Purpose |
|---|---|---|---|
| `demo_chat.jsonl` | 45 865 | 332 | training set (`--mode chat`) |
| `demo_valid.jsonl` | 9 737 | 68 | held-out set (`--validation`) |
| `demo_text.txt` | 4 577 | — | plain prose, for `--mode text` |

There are also `oasst_train.jsonl` and `oasst_valid.jsonl` once you convert an
oasst1 export — those are **not** in the repository, are **not** hand-written,
and have their own provenance below. Nothing in this project downloads them.

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

### Which file to download

oasst1 is published as several `.jsonl.gz` exports. **The `.trees.` file is the
one to download**, and this corrects an earlier version of this document, which
said the opposite.

The reason is not a preference. The flat `.messages.` file has **no
`parent_message_id` in any of its 88 838 records** — it carries `message_tree_id`
and `created_date` instead, and parentage was discarded without being recorded.
Every message in it therefore looks like a root, and the converter correctly
refuses it:

```
roots found 15728, conversations written 0
error: no conversation survived the filters, so nothing was written
```

The `.trees.` file holds the **same 88 838 messages** nested under `children`,
with the parent/child relationship implied by the structure. It is the only
published file from which a conversation tree can be recovered. The nesting has
to be flattened first, which is what `tools/oassttrees2flat.awk` does.

| File | Compressed | Use it? |
|---|---|---|
| `2023-04-12_oasst_ready.trees.jsonl.gz` | 34 145 252 B | **yes — this one** |
| `2023-04-12_oasst_ready.messages.jsonl.gz` | 34 196 309 B (32.6 MiB) | no — no parent ids, see above |
| `2023-04-12_oasst_all.messages.jsonl.gz` | 53 622 827 B (51.1 MiB) | no — same problem, plus unreviewed messages |
| `2023-04-12_oasst_prompts.messages.jsonl.gz` | 12 783 960 B | no — prompts only, no replies |
| `2023-04-12_oasst_spam.messages.jsonl.gz` | 1 513 962 B | no — removed spam |

Download and decompress it **outside this repository**, in an ordinary
terminal — the project has no network access and this is not something Aster
does for you:

```bash
curl -L -o oasst_ready_trees.jsonl.gz \
  "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz"
gunzip oasst_ready_trees.jsonl.gz
```

Budget **170 MB** of disk once decompressed — measured, not estimated: the
trees file is 166 284 779 B and the messages file 172 132 559 B.

**Scale, so the cap is not a surprise:** the `ready` export holds **88 838
messages across 10 364 conversation trees in 35 languages**. There is no
15 000-conversation oasst1, so `--max-conversations 15000` never binds. After
`--lang en` and the length filter, the measured yield is **2 276 English
conversations** — 4 662 869 bytes, against the 332-conversation demo corpus.

### Flattening the trees first

```bash
awk -f tools/oassttrees2flat.awk oasst_ready_trees.jsonl > oasst_flat.jsonl
```

```
# trees 10364  messages 88838
```

That count is the check worth trusting: it equals the record count of the
separately published `.messages.` file, so two independent files agree that
every message was recovered. The flattener's own invariants, also measured:

| Check | Result |
|---|---|
| distinct `message_id` values | 88 838 (no duplicates) |
| roots (empty parent) | 10 364 — one per tree |
| parent ids referenced but absent | 0 |
| messages whose parent is not in the file | 0 |
| roots whose role is not `prompter` | 0 |
| nodes whose role equals their parent's role | 0 |

The last row is the meaningful one: it holds across the whole file, so the
recovered parentage really is the conversation structure and not an artefact of
the nesting.

The script is deliberately dependency-free and prints no message text. Its
comments record the two bugs that shaped it — that brace depth is not tree
depth, and that a message object is not a leaf, so its own closing brace is not
the first `}` after its `message_id`.

### Provenance

| | |
|---|---|
| Dataset | OpenAssistant conversations (oasst1) |
| Publisher | LAION-AI / OpenAssistant |
| Release | 2023-04-12 |
| Licence | **Apache-2.0** |
| Source | <https://huggingface.co/datasets/OpenAssistant/oasst1> |
| File used | `2023-04-12_oasst_ready.trees.jsonl.gz` |
| SHA-256 of the **downloaded** trees file | `2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7` |
| SHA-256 of the downloaded messages file (for cross-checking) | `286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c` |
| SHA-256 of `oasst_train.jsonl` | `bfe039eeac7ed7f180b4ab5f57e29aaafcb2d28fce735d945004b5f49cba2ec6` |
| SHA-256 of `oasst_valid.jsonl` | `0e44610bb0f9f726b968bbe19e1e22c695f5a0700fc13bfc4c1c49a85aa0ef1c` |

Those last two are the hashes of the **unredacted** converter output, printed by
the run. Redaction changes them — see below.

Apache-2.0 permits this use, which is why the converter exists at all. It does
**not** make the contents private: oasst1 was written by volunteers, so assume
personal data is present and read the output before training on it.

### Running it

The full pipeline, as actually run. The three awk passes are data prep and are
not part of `aster.exe`.

```bash
# 1. flatten the nested trees into the flat form the converter reads
awk -f tools/oassttrees2flat.awk oasst_ready_trees.jsonl > oasst_flat.jsonl

# 2. build the converter
gcc -std=c11 -O2 -Wall -Wextra -Isrc -o oasst2jsonl.exe \
    tools/oasst2jsonl.c src/util.c src/jsonstr.c src/tokenizer.c

# 3. check the layout before committing to a run
./oasst2jsonl.exe --in oasst_flat.jsonl --probe

# 4. convert
./oasst2jsonl.exe --in oasst_flat.jsonl --lang en \
    --out oasst_train.jsonl \
    --valid-out oasst_valid.jsonl --valid-every 20 \
    --max-conversations 15000

# 5. redact, BEFORE training on it
awk -f tools/oasst-redact.awk oasst_train.jsonl   > red_train.jsonl
awk -f tools/oasst-redact.awk oasst_valid.jsonl  > red_valid.jsonl
```

Measured output of steps 4 and 5, unedited:

```
  read oasst_flat.jsonl
    lines 88838, parsed as messages 40822
    skipped: language filter (47533)
    skipped: shorter than --min-chars (483)
    roots found 3652, conversations written 2276
    held-out split: 119 conversations (every 20th), 238586 bytes
    turns written 7694, training total 4662869 bytes
warning: 10417 alternative replies were dropped: oasst1 is a tree and Aster's
         format is linear, so one reply per prompt is kept
    dropped, last turn had no reply: 1257
  wrote oasst_train.jsonl
    2276 conversation(s), 4662869 bytes, sha256 bfe039ea...
  wrote oasst_valid.jsonl
    119 conversation(s), 238586 bytes, sha256 0e44610b...

# redacted: 15 email, 8 ip, 73 number runs
# redacted:  1 email, 0 ip, 20 number runs
```

Note **40 822 of 88 838 messages are not even English** and are dropped by
`--lang en`, and 3 652 English roots yield only 2 276 conversations. The
`--max-conversations 15000` cap never binds: oasst1 in English is smaller than
the 15 000 this project originally aimed for.

### Redacting, and what redaction does not do

`tools/oasst-redact.awk` removes email addresses, IPv4 addresses, and runs of
digits with separators. A scan of the converted corpus found **11 email
addresses, 27 long number runs and 6 IPv4 addresses**; the redacting pass, which
loops until no match remains, replaced **16 emails, 8 IPs and 93 number runs**,
and a re-scan of the result finds **zero** of each.

It redacts rather than deletes. Dropping a whole conversation because it holds
one address would lose the other two thousand characters of it, and a corpus
missing exactly the conversations where someone described their own situation
is a biased corpus.

**It does not catch real names, employers, towns, health details, or anything
else a person wrote about themselves.** Those cannot be found by pattern. If
this corpus is to be committed or published, a human has to read it; this step
narrows the problem and does not close it.

If the file is not in the layout the tool understands, run `--probe` to see the
keys it actually has. A real run against an unrecognised file **exits non-zero
and writes nothing** rather than producing a wrong training set.

| Option | Meaning |
|---|---|
| `--in FILE` | the flat oasst1 export to read (JSON Lines) |
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
warning: 10417 alternative replies were dropped: oasst1 is a tree and Aster's
         format is linear, so one reply per prompt is kept
    dropped, last turn had no reply: 1257
```

Read those counts before you trust the file. A converter that quietly threw away
half the dataset and said nothing would be worse than no converter at all.

The 10 417 is a real loss, not a rounding error: oasst1 branched heavily, and
keeping the first child of a branch discards the alternatives **together with
their descendants**, which is why 88 838 messages become 7 694 turns. The
converter does not sort by rank when choosing which child to keep, so "first"
means first in the file. A rank-ordered choice would be a different corpus and a
different loss number, and would need measuring before being claimed as better.

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
