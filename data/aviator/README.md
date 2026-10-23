# Aviator numeric sequence experiment

Downloaded on 2026-10-11 from
[kaggleaccount7537/aviator-game-dataset](https://www.kaggle.com/datasets/kaggleaccount7537/aviator-game-dataset),
version 2. Kaggle reports its license as **Unknown**. The downloaded archive is
kept locally; this document does not assert redistribution rights.

This is a separate numeric text model trained from scratch, not a fine-tune of
the conversational checkpoint. `models/aster-small.bin` is unchanged. No
KaggleHub installation was needed: the same public dataset was downloaded
through Kaggle's dataset API.

The archive contains `multipliers.csv` and `aviator_dataset_clean.csv`. We use
only the raw `Multiplier` column. The publisher explicitly defines `color`
from the current target and `mean` using the current and preceding targets;
other cleaned features inherit that leakage. They cannot be treated as known
inputs when forecasting the current target.

There are 732,367 finite positive values. Their original numeric spellings are
preserved, with one value per line and no header. Split boundaries use source
file order: 585,893 training rows, 73,237 validation rows, and 73,237 test rows.
There are no timestamps, so actual chronological order is unverified. Repeated
numeric values may occur in different splits; source rows do not overlap.

`manifest.json` records version, transformation, source hashes, split hashes,
and row boundaries. `kaggle-metadata.json` retains the publisher's description.

Reproduce preparation and training from the project root:

```powershell
python tools/prepare_aviator.py
.\aster.exe train --mode text --data data/aviator/train.txt `
  --validation data/aviator/validation.txt --out models/aster-aviator.bin `
  --layers 1 --width 32 --heads 4 --ff 128 --context 128 `
  --block 64 --stride 64 --merges 64 --steps 1000 --batch 8 `
  --log-every 200 --seed 1234
.\aster.exe eval --model models/aster-aviator.bin --mode text `
  --data data/aviator/test.txt --block 64 --stride 64
```

The model has 27,104 parameters and a 325-token vocabulary. Training sampled
1,000 batches of eight 64-token windows (512,000 input token positions), not a
full pass over the training set. The tokenizer was learned on training only.
Windows start 64 tokens apart; within each window the trainer scores 63 next
tokens, so boundary transitions and the final short tail are not scored.

Measured results from this run:

| Measurement | Bits per byte |
|---|---:|
| Untrained validation loss | 3.661802 |
| Best validation loss, step 1,000 | 1.942678 |
| Untouched test loss | 1.9432 |

Test loss was 3.0668 nats per token over 157,437 scored tokens covering 358,472
bytes. The test set was evaluated only after checkpoint selection. Lower loss
demonstrates learning the corpus's numeric text patterns, including formatting
and number frequencies. It does **not** measure profitable betting, next-round
prediction accuracy, causal structure, or reliable future outcomes. No
next-round prediction accuracy was evaluated.

Numeric text continuation is available explicitly through the CLI:

```powershell
.\aster.exe generate --model models/aster-aviator.bin --mode text `
  --prompt "2.8" --max-new-tokens 24 --temp 0 --repetition-penalty 1
```

Generated text is synthetic and can repeat or become incoherent. Do not load
this checkpoint into the chat UI: it was not trained with chat role markers.
The app's regular `start.bat` continues using the conversational checkpoint.
