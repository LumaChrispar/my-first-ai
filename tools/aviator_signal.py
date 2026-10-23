"""Local, past-only binary classification experiment with an abstention gate.

This is not TypeSafe's Jev and has no live betting-site connection.
Requires NumPy; training and calibration use separate source-order blocks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HISTORY = 64
THRESHOLD = 0.95
MIN_ALERTS = 100
MIN_PRECISION_LOWER_BOUND = 0.90


def features(history):
    x = np.asarray(history, dtype=np.float64)
    if x.shape[-1] != HISTORY or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError('Exactly 64 finite positive completed multipliers are required')
    logs = np.log(np.minimum(x, 1000.0))
    below = x < 2.0
    blocks = [logs[..., -12:], below[..., -12:].astype(np.float64)]
    for size in (8, 32, 64):
        blocks.extend([logs[..., -size:].mean(axis=-1, keepdims=True),
                       below[..., -size:].mean(axis=-1, keepdims=True)])
    blocks.append(logs.std(axis=-1, keepdims=True))
    return np.concatenate(blocks, axis=-1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit(x, y, epochs=160):
    weights = np.zeros(x.shape[1])
    intercept = float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
    for _ in range(epochs):
        residual = sigmoid(x @ weights + intercept) - y
        weights -= 0.2 * (x.T @ residual / len(y) + 0.001 * weights)
        intercept -= 0.2 * residual.mean()
    return weights, intercept


def auc(y, p):
    order = np.argsort(p)
    sorted_p = p[order]
    _, starts, counts = np.unique(sorted_p, return_index=True, return_counts=True)
    ranks = np.repeat(starts + (counts + 1) / 2, counts)
    positives = int(y.sum())
    negatives = len(y) - positives
    return float((ranks[y[order] == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def metrics(y, p):
    selected = p >= THRESHOLD
    count = int(selected.sum())
    correct = int(y[selected].sum())
    precision = correct / count if count else None
    lower = None
    if count:
        z = 1.96
        lower = ((precision + z*z/(2*count)) - z*np.sqrt(
            precision*(1-precision)/count + z*z/(4*count*count))) / (1+z*z/count)
    clipped = np.clip(p, 1e-8, 1-1e-8)
    return {'rows': len(y), 'observed_below_2x_rate': float(y.mean()),
            'brier_score': float(np.mean((p-y)**2)), 'auc': auc(y, p),
            'log_loss': float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
            'probability_min': float(p.min()), 'probability_max': float(p.max()),
            'alerts_at_95_percent': count, 'alert_precision': precision,
            'precision_wilson_95_percent_lower_bound': lower}


def probability(model, history):
    x = (features(history) - np.asarray(model['mean'])) / np.asarray(model['scale'])
    raw = float(x @ np.asarray(model['weights']) + model['intercept'])
    return float(sigmoid(model['calibration_slope'] * raw + model['calibration_intercept']))


def decide(model, history):
    p = probability(model, history)
    # File-based replay is never a live alert, even if the offline threshold passes.
    return {'decision': 'NO_SIGNAL',
            'experimental_probability_below_2x': p,
            'validated_for_live_use': False,
            'offline_threshold_met': bool(model['validation_gate_passed'] and p >= model['threshold']),
            'reason': 'validation_gate_failed' if not model['validation_gate_passed']
                      else 'live_validation_pending' if p >= model['threshold']
                      else 'probability_below_threshold',
            'live_source_connected': False}


def train(root, output):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8-sig'))
    values = []
    for name in ('train', 'validation', 'test'):
        raw = (root / (name + '.txt')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['splits'][name]['sha256']:
            raise ValueError('Split hash differs from the prepared corpus: ' + name)
        values.append(np.fromstring(raw.decode('utf-8'), sep='\n'))
    n_train, n_valid, n_test = map(len, values)
    all_values = np.concatenate(values)
    # At target position t, the window ends at t-1. It never includes the target.
    windows = np.lib.stride_tricks.sliding_window_view(all_values, HISTORY)[:-1]
    x = features(windows)
    y = (all_values[HISTORY:] < 2.0).astype(float)
    t = n_train - HISTORY
    v = t + n_valid
    mean, scale = x[:t].mean(axis=0), x[:t].std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    # Deterministic downsampling limits local training cost; all held-out targets remain.
    weights, intercept = fit(standardized[:t:5], y[:t:5])
    raw_valid = standardized[t:v] @ weights + intercept
    calibration_size = n_valid // 2
    raw_cal = raw_valid[:calibration_size]
    y_cal = y[t:t+calibration_size]
    slope, offset = 1.0, 0.0
    for _ in range(160):
        error = sigmoid(slope*raw_cal+offset)-y_cal
        slope -= 0.2 * float(np.mean(error*raw_cal))
        offset -= 0.2 * float(error.mean())
    p_valid = sigmoid(slope*raw_valid[calibration_size:]+offset)
    valid_y = y[t+calibration_size:v]
    validation = metrics(valid_y, p_valid)
    baseline = float(y[:t].mean())
    validation['constant_baseline_brier'] = float(np.mean((baseline-valid_y)**2))
    lower = validation['precision_wilson_95_percent_lower_bound']
    passed = (validation['alerts_at_95_percent'] >= MIN_ALERTS and lower is not None
              and lower >= MIN_PRECISION_LOWER_BOUND
              and validation['brier_score'] < validation['constant_baseline_brier'])
    # Only now examine the final test block, with model and threshold fixed.
    p_test = sigmoid(slope*(standardized[v:] @ weights+intercept)+offset)
    test = metrics(y[v:], p_test)
    test['constant_baseline_brier'] = float(np.mean((baseline-y[v:])**2))
    model = {'kind': 'local_past_only_logistic_classifier', 'history_length': HISTORY,
             'threshold': THRESHOLD, 'validation_gate_passed': bool(passed),
             'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': weights.tolist(),
             'intercept': intercept, 'calibration_slope': slope, 'calibration_intercept': offset,
             'training_targets_available': t, 'training_targets_used': len(y[:t:5]),
             'calibration_rows': calibration_size, 'validation': validation, 'test': test,
             'source_order_chronology_verified': False, 'live_source_connected': False,
             'minimum_validation_alerts': MIN_ALERTS,
             'minimum_precision_lower_bound': MIN_PRECISION_LOWER_BOUND}
    times = []
    for _ in range(1000):
        started = time.perf_counter_ns()
        probability(model, all_values[-HISTORY:])
        times.append((time.perf_counter_ns()-started)/1e6)
    model['local_inference_p95_ms'] = float(np.quantile(times, 0.95))
    model['latency_scope'] = 'Cached local inference only; excludes collection, network and file reads'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: model[k] for k in ('validation_gate_passed', 'validation', 'test',
                                          'local_inference_p95_ms')}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--data', type=Path, default=Path('data/aviator'))
    parser.add_argument('--model', type=Path, default=Path('models/aviator-classifier.json'))
    parser.add_argument('--history', type=Path)
    args = parser.parse_args()
    if args.command == 'train':
        train(args.data, args.model)
    else:
        if not args.history:
            parser.error('predict requires --history with one completed multiplier per line')
        model = json.loads(args.model.read_text(encoding='utf-8-sig'))
        history = np.fromstring(args.history.read_text(), sep='\n')
        if len(history) < HISTORY:
            raise ValueError('At least 64 completed multipliers are required')
        print(json.dumps(decide(model, history[-HISTORY:]), indent=2))


if __name__ == '__main__':
    main()
