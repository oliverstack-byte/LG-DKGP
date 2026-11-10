"""Build episode-disjoint latent/action windows from the released raw splits."""
import io
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2
from lgdkgp.common.io import resolve, file_hash, digest, write_json, atomic_torch_save, load_world


def image_transform(size=224):
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize([.485, .456, .406], [.229, .224, .225]),
                       v2.Resize((size, size))])


def decode(frame):
    frame = np.asarray(frame)
    if frame.ndim < 3:
        with Image.open(io.BytesIO(frame.tobytes())) as image:
            return np.asarray(image.convert('RGB')).copy()
    return frame.copy()


def episode_sources(cfg):
    """Yield split, episode metadata, HDF5 handle, offset, length, camera map."""
    env = cfg['environment']
    root = resolve(cfg['data_root'])
    if env in ('pusht', 'cube'):
        with h5py.File(root / cfg['dataset'], 'r') as h:
            offsets = np.asarray(h['ep_offset'])
            for split, key in [('train', 'train_split'), ('val', 'validation_split')]:
                selection = json.loads(resolve(cfg[key]).read_text())
                for entry in selection['episodes']:
                    yield split, entry, h, int(offsets[entry['episode_id']]), entry['length'], {'pixels': 'pixels'}
    else:
        from lgdkgp.data.windows.native_world_windows import NativeWorldWindows, VIEWS
        for split, key in [('train', 'train_split'), ('val', 'validation_split')]:
            selection = json.loads(resolve(cfg[key]).read_text())
            selection['data_root'] = str(root / cfg['dataset'])
            # A relocatable transient manifest is written inside the new run only.
            dest = resolve(cfg['output']) / 'inputs' / (split + '.json')
            write_json(dest, selection)
            reader = NativeWorldWindows(dest, cfg['action_mean'], cfg['action_std'])
            try:
                for ordinal, entry in enumerate(selection['prior_episodes']):
                    if entry['success'] is not True or not entry['prior_eligible']:
                        raise ValueError('Prior training requires successful expert episodes')
                    item = dict(entry, episode_id=ordinal)
                    yield split, item, reader._handle(entry), entry['raw_offset'], entry['length'], VIEWS[env]
            finally:
                reader.close()


@torch.no_grad()
def prepare(cfg, device):
    output = resolve(cfg['output']) / 'cache'
    output.mkdir(parents=True, exist_ok=True)
    binding = {'world': file_hash(resolve(cfg['world_model'])),
               'train': file_hash(resolve(cfg['train_split'])),
               'validation': file_hash(resolve(cfg['validation_split'])),
               'mean': cfg['action_mean'], 'std': cfg['action_std'],
               'image_size': cfg['image_size']}
    if (output / 'binding.json').exists() and json.loads((output / 'binding.json').read_text()) != binding:
        raise ValueError('Cache binding differs; select a new output directory')
    write_json(output / 'binding.json', binding)
    model = load_world(cfg['world_model'], device)
    transform = image_transform(cfg['image_size'])
    mean, std = np.asarray(cfg['action_mean']), np.asarray(cfg['action_std'])
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid action normalization')
    entries, q_sum, pairs = [], np.zeros(len(mean)), 0
    for split, entry, source, offset, length, views in episode_sources(cfg):
        if cfg['environment'] == 'robotwin_a2b':
            raw = np.asarray(source['lewm/action_blocks'][:length, 0])
            visual = np.asarray(source['lewm/visual_raw_indices'][:length])
            if not np.array_equal(visual, np.arange(length)):
                raise ValueError('Noncausal RoboTwin visual/action alignment')
        else:
            raw = np.asarray(source['action'][offset:offset + length])
        finite = np.isfinite(raw).all(1)
        usable = int(np.flatnonzero(~finite)[0]) if not finite.all() else len(raw)
        if finite[usable:].any() or usable < 26:
            raise ValueError('Invalid interior command or too-short episode')
        raw = raw[:usable]
        if 'actual_action_sha256' in entry:
            import hashlib
            if hashlib.sha256(raw.tobytes()).hexdigest() != entry['actual_action_sha256']:
                raise ValueError('Native executed-command source changed')
        if cfg['environment'] in ('pusht', 'cube'):
            action = torch.from_numpy(((raw.astype(np.float64) - mean) / std).astype(np.float32))
        else:
            # The original native feature exporter used float32 Torch arithmetic.
            action = (torch.from_numpy(raw.copy()).float() - torch.tensor(cfg['action_mean'])) / torch.tensor(cfg['action_std'])
        if split == 'train':
            q_sum += np.diff((raw.astype(np.float64) - mean) / std, axis=0).__pow__(2).sum(0)
            pairs += len(raw) - 1
        name = f'{split}_{entry["episode_id"]:05d}.pt'
        dest = output / 'features' / name
        meta = dest.with_suffix('.json')
        if dest.exists():
            record = json.loads(meta.read_text())
            if file_hash(dest) != record['sha256'] or record['binding'] != digest(binding):
                raise ValueError('Cached episode changed')
        else:
            latents = []
            for start in range(0, usable, cfg['batch_size']):
                stop = min(usable, start + cfg['batch_size'])
                batch = {key: torch.stack([transform(decode(source[column][offset + i]))
                                           for i in range(start, stop)]).to(device)[:, None]
                         for key, column in views.items()}
                latents.append(model.encode(batch)['emb'][:, 0].float().cpu())
            atomic_torch_save({'latent': torch.cat(latents), 'action': action}, dest)
            write_json(meta, {'sha256': file_hash(dest), 'binding': digest(binding)})
        entries.append({'file': name, 'sha256': file_hash(dest), 'split': split,
                        'length': usable, 'episode_id': entry['episode_id']})
        print(json.dumps({'cache_split': split, 'episode': entry['episode_id']}), flush=True)
    if pairs == 0 or not entries:
        raise ValueError('Empty training dataset')
    first = torch.load(output / 'features' / entries[0]['file'], weights_only=True)
    shared = {'environment': cfg['environment'], 'parent_sha256': binding['world'],
              'identity': {'allocation_sha256': digest(binding)},
              'feature_directory': str(output / 'features'), 'episodes': entries,
              'action_dim': len(mean), 'latent_dim': first['latent'].shape[1], 'akf_parameters': {}}
    if cfg['environment'] not in ('pusht', 'cube'):
        shared['goal_policy'] = 'terminal'
    write_json(output / 'dataset.json', shared)
    from lgdkgp.data.windows.variable_history import prepare_variable_dataset
    prepare_variable_dataset(output, output / 'lg')
    # Existing checkpoint loading uses embedded latent statistics; new training
    # computes them from training episodes only through the original reader.
    calibration = {'environment': cfg['environment'], 'train_only': True,
                   'adjacent_pairs': pairs, 'expert_delta_second_moment': (q_sum / pairs).tolist(),
                   'sigma': float(np.sqrt((q_sum / pairs).mean() / 2)), 'binding': binding}
    write_json(resolve(cfg['output']) / 'calibration.json', calibration)
    return output / 'lg'
