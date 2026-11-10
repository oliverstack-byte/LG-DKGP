"""Verify separately distributed weights and query inputs without importing Torch."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='Root containing checkpoints/ and fixtures/ (default: checkout)')
    args = parser.parse_args()
    expected = {}
    manifest = json.loads((ROOT / 'assets/checkpoints/manifest.json').read_text())
    for env in manifest['environments'].values():
        for item in env.values():
            expected[item['file']] = item['sha256']
    for path in sorted((ROOT / 'assets/manifests').glob('*/evaluation.json')):
        for query in json.loads(path.read_text()):
            if 'fixture' in query:
                expected[query['fixture']] = query['fixture_sha256']
    for path, digest in expected.items():
        if sha(args.root / path) != digest:
            raise ValueError('Artifact SHA mismatch: ' + path)
    print(f'Verified {len(expected)} released checkpoints/configs and query fixtures.')


if __name__ == '__main__':
    main()
