"""Portable configuration, artifact identity, and single-device ownership."""
import hashlib
import json
import os
from pathlib import Path

_checkout = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('LGDKGP_ROOT',
            str(_checkout if (_checkout / 'pyproject.toml').is_file() else Path.cwd()))).resolve()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path, overrides=()):
    cfg = json.loads(resolve(path).read_text())
    for item in overrides:
        key, sep, raw = item.partition('=')
        if not sep:
            raise ValueError('Overrides must be key=value')
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        dest = cfg
        fields = key.split('.')
        for field in fields[:-1]:
            dest = dest[field]
        if fields[-1] not in dest:
            raise KeyError(key)
        dest[fields[-1]] = value
    return cfg


def single_device(device):
    # One visible CUDA device, one process, one serial environment. CPU is for
    # artifact inspection and unit tests; it is not a second worker.
    import torch
    dev = torch.device(device)
    if dev.type not in ('cpu', 'cuda') or dev.index not in (None, 0):
        raise ValueError('Select one physical GPU with CUDA_VISIBLE_DEVICES; use cuda:0')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Distributed execution is unsupported')
    if dev.type == 'cuda':
        if torch.cuda.device_count() != 1:
            raise ValueError('Expose exactly one GPU, e.g. CUDA_VISIBLE_DEVICES=0')
        torch.cuda.set_device(0)
    return str(dev)


def load_world(path, device):
    from lgdkgp.models.compat import install_legacy_model_aliases
    install_legacy_model_aliases()
    from stable_worldmodel.wm.utils import load_pretrained
    return load_pretrained(str(resolve(path)), cache_dir=str(ROOT / 'outputs/model_cache')).to(device).eval().requires_grad_(False)
