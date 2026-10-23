"""Fixed walk-forward comparison in source order, using NumPy only.

The previously evaluated test.txt is deliberately never opened. This benchmark
cannot establish real chronology or prospective/live performance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aviator_signal import HISTORY, features, fit, metrics, sigmoid


def load_development(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8-sig'))
    parts = []
    for name in ('train', 'validation'):
        raw = (root / (name + '.txt')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['splits'][name]['sha256']:
            raise ValueError('Corpus hash mismatch: ' + name)
        values = np.fromstring(raw.decode('utf-8'), sep='\n')
        if len(values) != manifest['splits'][name]['rows']:
            raise ValueError('Corpus row count mismatch: ' + name)
        parts.append(values)
    values = np.concatenate(parts)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Multipliers must be finite and positive')
    return values, manifest


def folds(n):
    # Indexes address eligible targets (source row = index + HISTORY).
    # Disjoint evaluation blocks; expanding fit blocks may use earlier results.
    return [(int(n*f), int(n*(f+0.05)), int(n*(f+0.15)))
            for f in (0.40, 0.60, 0.80)]


def calibrate(raw, y):
    slope, offset = 1.0, 0.0
    for _ in range(160):
        error = sigmoid(slope*raw+offset)-y
        slope -= 0.2 * float(np.mean(error*raw))
        offset -= 0.2 * float(error.mean())
    return slope, offset


def reliability(y, p):
    bins, ece = [], 0.0
    for i in range(10):
        mask = (p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
        count = int(mask.sum())
        if count:
            predicted, observed = float(p[mask].mean()), float(y[mask].mean())
            ece += count/len(y)*abs(predicted-observed)
            bins.append({'lower': i/10, 'upper': (i+1)/10, 'rows': count,
                         'mean_probability': predicted, 'observed_rate': observed})
    return {'ece_10_bins': ece, 'bins': bins}


def selection_curve(y, p):
    result = []
    for threshold in (0.55, 0.60, 0.70, 0.80, 0.90, 0.95):
        selected = p >= threshold
        count = int(selected.sum())
        precision = float(y[selected].mean()) if count else None
        lower = None
        if count:
            z = 1.96
            lower = ((precision+z*z/(2*count))-z*np.sqrt(
                precision*(1-precision)/count+z*z/(4*count*count)))/(1+z*z/count)
        result.append({'threshold': threshold, 'count': count,
                       'coverage': count/len(y), 'precision': precision,
                       'wilson_95_lower': lower})
    return result


def improvement(y, p, baseline):
    delta = (baseline-y)**2 - (p-y)**2  # Positive favors the candidate.
    # Contiguous block means account approximately for short-range dependence.
    blocks = np.array([chunk.mean() for chunk in np.array_split(
        delta, max(2, len(delta)//1000))])
    margin = 1.96*blocks.std(ddof=1)/np.sqrt(len(blocks))
    return {'brier_improvement': float(delta.mean()),
            'block_mean_approx_95_interval': [float(delta.mean()-margin),
                                            float(delta.mean()+margin)],
            'blocks': len(blocks)}


def run(root, output):
    values, manifest = load_development(root)
    windows = np.lib.stride_tricks.sliding_window_view(values, HISTORY)[:-1]
    x = features(windows)
    y = (values[HISTORY:] < 2.0).astype(float)
    # Fixed random nonlinear basis: no hidden weight search or tuning on evaluation.
    rng = np.random.default_rng(20261011)
    projection = rng.normal(size=(x.shape[1], 16))/np.sqrt(x.shape[1])
    bias = rng.normal(scale=0.2, size=16)
    report = {'protocol': 'Three fixed expanding source-order folds; fit/calibrate/evaluate',
              'chronology_verified': False, 'live_source_connected': False,
              'fresh_final_test_available': False, 'previous_test_opened': False,
              'data_hashes': {k: manifest['splits'][k]['sha256']
                              for k in ('train', 'validation')},
              'recipe': {'history': HISTORY, 'fit_stride': 5, 'epochs': 160,
                         'learning_rate': 0.2, 'l2': 0.001, 'seed': 20261011,
                         'nonlinear': '31 original features + 16 fixed tanh projections; logistic head',
                         'selection': 'No hyperparameter or threshold search; no production promotion'},
              'folds': []}
    predictions = {k: [] for k in ('constant', 'logistic', 'nonlinear')}
    targets = []
    for fold_id, (a, b, c) in enumerate(folds(len(y)), 1):
        print(f'Fold {fold_id}: fit {a}, calibrate {b-a}, evaluate {c-b}', flush=True)
        mean, scale = x[:a].mean(axis=0), x[:a].std(axis=0)
        scale[scale < 1e-8] = 1.0
        z = (x[:c]-mean)/scale
        baseline = np.full(c-b, y[:a].mean())
        fold = {'fold': fold_id,
                'target_source_rows_0based_half_open': {
                    'fit': [HISTORY, a+HISTORY],
                    'calibration': [a+HISTORY, b+HISTORY],
                    'evaluation': [b+HISTORY, c+HISTORY]}, 'models': {}}
        for name in predictions:
            model = None
            if name == 'constant':
                p = baseline
            else:
                design = z if name == 'logistic' else np.concatenate(
                    [z, np.tanh(z @ projection+bias)], axis=1)
                weights, intercept = fit(design[:a:5], y[:a:5])
                slope, offset = calibrate(design[a:b] @ weights+intercept, y[a:b])
                p = sigmoid(slope*(design[b:c] @ weights+intercept)+offset)
                model = {'mean': mean.tolist(), 'scale': scale.tolist(),
                         'weights': weights.tolist(), 'intercept': intercept,
                         'calibration_slope': slope, 'calibration_intercept': offset}
                if name == 'nonlinear':
                    model.update(projection=projection.tolist(), bias=bias.tolist())
            measured = metrics(y[b:c], p)
            measured.update(reliability(y[b:c], p))
            measured.update(improvement(y[b:c], p, baseline))
            measured['selection_curve'] = selection_curve(y[b:c], p)
            fold['models'][name] = {'metrics': measured, 'parameters': model}
            predictions[name].append(p)
            print(f"  {name}: AUC {measured['auc']:.6f}, Brier {measured['brier_score']:.6f}", flush=True)
        targets.append(y[b:c])
        report['folds'].append(fold)
    combined_y = np.concatenate(targets)
    combined_baseline = np.concatenate(predictions['constant'])
    report['aggregate'] = {}
    for name, parts in predictions.items():
        p = np.concatenate(parts)
        measured = metrics(combined_y, p)
        measured.update(reliability(combined_y, p))
        measured.update(improvement(combined_y, p, combined_baseline))
        measured['selection_curve'] = selection_curve(combined_y, p)
        measured['folds_with_lower_brier_than_baseline'] = sum(
            f['models'][name]['metrics']['brier_improvement'] > 0 for f in report['folds'])
        report['aggregate'][name] = measured
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    write_summary(report, output.with_suffix('.md'))
    print(json.dumps(report['aggregate'], indent=2), flush=True)


def write_summary(report, path):
    lines = ['# Aviator walk-forward benchmark', '',
             'Three fixed expanding windows compare models using past-only features. '
             'Each window fits scaling/model parameters on its fit block and calibrates '
             'probabilities on the following block before evaluation.', '',
             '**Source order only:** timestamps are absent, so chronology is unverified. '
             'The earlier test file was not opened, but it was evaluated in previous work. '
             'There is no fresh final test or connected live source.', '',
             'The nonlinear candidate adds 16 fixed tanh projections to the existing '
             '31 features and trains a logistic output head. It is a small local '
             'nonlinear classifier, not a recreation of Jev.', '',
             '| Model | Evaluation rows | AUC | Brier ↓ | Brier gain vs baseline ↑ | Better folds | 95% signals |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in report['aggregate'].items():
        lines.append(f"| {name} | {m['rows']} | {m['auc']:.6f} | {m['brier_score']:.6f} | "
                     f"{m['brier_improvement']:.8f} | {m['folds_with_lower_brier_than_baseline']}/3 | {m['alerts_at_95_percent']} |")
    lines += ['', 'AUC 0.5 is chance ranking. Positive Brier gain favors the candidate. '
              'The JSON includes each fold, frozen parameters, reliability bins, '
              'precision/coverage at fixed thresholds, and approximate intervals '
              'from contiguous block means. These intervals are descriptive, not '
              'proof of independent live performance.', '',
              'No model or threshold is automatically promoted. Newly collected, '
              'timestamped rounds with predictions recorded before outcomes are '
              'needed for a prospective final evaluation.', '',
              'Reproduce: `python tools/benchmark_aviator.py` (Python + NumPy).', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/aviator'))
    parser.add_argument('--output', type=Path, default=Path('data/aviator/BENCHMARK.json'))
    args = parser.parse_args()
    run(args.data, args.output)
