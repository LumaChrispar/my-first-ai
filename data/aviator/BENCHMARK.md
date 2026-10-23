# Aviator walk-forward benchmark

Three fixed expanding windows compare models using past-only features. Each window fits scaling/model parameters on its fit block and calibrates probabilities on the following block before evaluation.

**Source order only:** timestamps are absent, so chronology is unverified. The earlier test file was not opened, but it was evaluated in previous work. There is no fresh final test or connected live source.

The nonlinear candidate adds 16 fixed tanh projections to the existing 31 features and trains a logistic output head. It is a small local nonlinear classifier, not a recreation of Jev.

| Model | Evaluation rows | AUC | Brier ↓ | Brier gain vs baseline ↑ | Better folds | 95% signals |
|---|---:|---:|---:|---:|---:|---:|
| constant | 197720 | 0.499213 | 0.249722 | 0.00000000 | 0/3 | 0 |
| logistic | 197720 | 0.500533 | 0.249793 | -0.00007144 | 0/3 | 0 |
| nonlinear | 197720 | 0.500409 | 0.249802 | -0.00008012 | 0/3 | 0 |

AUC 0.5 is chance ranking. Positive Brier gain favors the candidate. The JSON includes each fold, frozen parameters, reliability bins, precision/coverage at fixed thresholds, and approximate intervals from contiguous block means. These intervals are descriptive, not proof of independent live performance.

No model or threshold is automatically promoted. Newly collected, timestamped rounds with predictions recorded before outcomes are needed for a prospective final evaluation.

Reproduce: `python tools/benchmark_aviator.py` (Python + NumPy).
