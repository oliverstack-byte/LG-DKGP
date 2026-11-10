"""Verify and extract an already downloaded asset archive in a new runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=ROOT / 'third_party/robotwin')
    a = p.parse_args()
    spec = json.loads((ROOT / 'third_party/setup/robotwin/asset_archive.json').read_text())['robotwin_a2b_assets.tar.gz']
    if sha(a.archive) != spec['sha256']:
        raise ValueError('Asset archive SHA mismatch')
    runtime = a.runtime.resolve()
    if (runtime / 'assets').exists():
        raise FileExistsError('Runtime assets already exist')
    runtime.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.archive) as tar:
        for member in tar.getmembers():
            if not (runtime / member.name).resolve().is_relative_to(runtime) or member.issym() or member.islnk():
                raise ValueError('Unsafe archive member')
        tar.extractall(runtime)
    for key, entry in json.loads((ROOT / 'third_party/setup/robotwin/bootstrap_assets.json').read_text()).items():
        name = Path(entry['bundled_file']).name
        target = runtime / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'third_party/setup/robotwin/configs' / name, target)
    # These two kinematics configs use portable asset-relative paths.
    for side in ['left', 'right']:
        source = ROOT / f'third_party/setup/robotwin/configs/curobo_{side}.yml'
        if source.exists():
            shutil.copy2(source, runtime / f'assets/embodiments/aloha-agilex/curobo_{side}.yml')


if __name__ == '__main__':
    main()
