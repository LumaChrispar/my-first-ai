# Below-2x classifier and BetPawa source inspection

The requested site is BetPawa Cameroon, and the Aviator entry is:
https://www.betpawa.cm/casino/game/3255?redirectBack=%2Fcasino

Inspected on 2026-10-11. The game entry required Login or Join Now in the
available browser session. No completed-round history or authenticated game
feed was available. **Live scraping and automatic synchronization are not
connected.** No account was created, no credentials were entered, and no bets
were placed.

BetPawa's [Aviator page](https://www.betpawa.cm/aviator-crash-game) states that
rounds are independent and past results do not predict future ones.
[TypeSafe's Jev](https://typesafe.ai/) returns typed decisions and confidence
estimates. That output style does not establish knowledge of hidden future
game outcomes. This local experiment is not Jev and does not call its API.

## The local experiment

`tools/aviator_signal.py` fits a logistic classifier using only the 64 completed
multipliers preceding each target. Features include the last 12 log multipliers,
12 below-2x indicators, and trailing summaries. Targets are `multiplier < 2.0`;
exactly 2.0 is not included. There are no target-derived cleaned features.

Source-order splits reuse the original prepared corpus. Features and scaling
are fitted on training only, and training uses every fifth eligible row.
The validation block is divided into 36,618 calibration rows and 36,619
qualification rows. The threshold is fixed at 95%, with at least 100
qualification alerts and a 95% Wilson precision lower bound of at least 90%
required. Brier score must also beat a constant training-rate baseline. This
is a screening rule, not proof of certainty or independent live calibration.
The test block is examined after fitting and qualification. Actual chronology
and correspondence with BetPawa Cameroon's live feed remain unverified.

Measured results, over 73,237 held-out test targets:

| Measurement | Result |
|---|---:|
| ROC AUC (0.5 is chance ranking) | 0.497633 |
| Brier score (lower is better) | 0.249800 |
| Constant baseline Brier score | 0.249701 |
| Largest estimated probability below 2x | 55.25% |
| Predictions meeting 95% threshold | 0 |
| Qualification gate | Failed |
| Cached local inference p95 | 0.452445 ms |

The classifier did not demonstrate useful predictive discrimination. Its
`decision` is **NO_SIGNAL**. The latency measures local computation only;
collection and network latency are not included. File-based replay never emits
a live alert, even if a future offline model meets the threshold.

The model and metrics are saved in `models/aviator-classifier.json`. Reproduce:

```powershell
python tools/aviator_signal.py train
python tools/aviator_signal.py predict --history data/aviator/test.txt
python tools/test_aviator_signal.py
```

Python and NumPy are required for this optional classifier; Aster's C runtime
still has no Python dependency. Unit checks cover past-only alignment,
malformed histories, tied-probability AUC, and blocking unsupported alerts.

## Live tracking requirements

An accessible completed-round source with documented round IDs and timestamps
is still required. A collector should deduplicate IDs, preserve source order,
detect gaps and reconnects, and expose lag or disconnected state. Each prediction
would have to be logged before its target round begins, then checked against
the eventual result. New live observations must not automatically unlock alerts:
they first need separate forward evaluation and calibration. No implementation
claims a live connection or a validated signal until these requirements hold.
