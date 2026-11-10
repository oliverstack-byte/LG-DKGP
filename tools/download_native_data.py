"""Download exact native source file names; validate training archive parts."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=['libero_caddy', 'libero_scene2', 'robotwin_a2b'], required=True)
    p.add_argument('--root', type=Path, default=ROOT / 'data/sage/source')
    p.add_argument('--revision', default='main')
    a = p.parse_args()
    snapshot_download(repo_id='CLTRAY/SAGE-data', repo_type='dataset', revision=a.revision,
                      allow_patterns=[f'training/{a.env}_*'], local_dir=a.root)
    from lgdkgp.common.io import file_hash
    expected = json.loads((ROOT / f'assets/provenance/archives/{a.env}_train.tar.parts.json').read_text())
    for part in expected['parts']:
        path = a.root / part['path']
        if path.stat().st_size != part['bytes'] or file_hash(path) != part['sha256']:
            raise ValueError('Training part mismatch: ' + str(path))
    print('Training parts verified. Archives are read in place; do not unpack/merge them.')


if __name__ == '__main__':
    main()
