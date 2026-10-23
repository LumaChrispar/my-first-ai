"""Prepare the requested numeric corpus, without target-derived features."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path('data/aviator/source.zip'))
    parser.add_argument('--out', type=Path, default=Path('data/aviator'))
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        raw = archive.read('multipliers.csv')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if reader.fieldnames != ['Multiplier']:
        raise ValueError(f'Unexpected columns: {reader.fieldnames}')
    values = []
    minimum, maximum = math.inf, -math.inf
    for line, row in enumerate(reader, 2):
        text = row['Multiplier'].strip()
        value = float(text)
        if not math.isfinite(value) or value <= 0 or '\n' in text or '\r' in text:
            raise ValueError(f'Invalid multiplier at source line {line}')
        values.append(text)
        minimum, maximum = min(minimum, value), max(maximum, value)
    if len(values) < 100:
        raise ValueError('Insufficient records')
    first, second = len(values) * 8 // 10, len(values) * 9 // 10
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {}
    training_counts = None
    for name, start, stop in [('train', 0, first), ('validation', first, second), ('test', second, len(values))]:
        encoded = ('\n'.join(values[start:stop]) + '\n').encode('utf-8')
        destination = args.out / (name + '.txt')
        destination.write_bytes(encoded)
        counts = Counter(encoded)
        if name == 'train':
            training_counts = counts
            training_bytes = len(encoded)
        # Add-one smoothed byte unigram: no sequence history, fitted on train only.
        baseline = sum(n * -math.log2((training_counts.get(b, 0) + 1) / (training_bytes + 256))
                       for b, n in counts.items()) / len(encoded)
        splits[name] = {'file': str(destination), 'rows': stop - start,
                        'source_row_start_inclusive_0based': start,
                        'source_row_stop_exclusive_0based': stop,
                        'bytes': len(encoded), 'sha256': sha(encoded),
                        'byte_unigram_bits_per_byte': baseline}
    metadata_file = args.out / 'kaggle-metadata.json'
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    manifest = {
        'dataset': 'kaggleaccount7537/aviator-game-dataset',
        'url': 'https://www.kaggle.com/datasets/kaggleaccount7537/aviator-game-dataset',
        'version': metadata.get('currentVersionNumber'),
        'license_reported_by_kaggle': metadata.get('licenseName', 'Unknown'),
        'archive_sha256': sha(args.archive.read_bytes()),
        'source_csv_sha256': sha(raw), 'source_csv': 'multipliers.csv',
        'rows': len(values), 'min_multiplier': minimum, 'max_multiplier': maximum,
        'split': '80/10/10 contiguous in source file order; timestamps absent, chronology unverified',
        'transform': 'Header removed; original numeric spelling preserved; LF newline per record',
        'excluded': 'All cleaned feature columns: some are explicitly derived from the current target',
        'task': 'Experimental numeric text sequence modeling, not demonstrated future-round prediction',
        'splits': splits,
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
