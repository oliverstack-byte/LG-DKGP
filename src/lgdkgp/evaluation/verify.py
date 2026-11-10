"""Read-only checkpoint compatibility checks; no simulator or fitting calls."""
import json
import torch
from lgdkgp.common.io import resolve, file_hash, write_json
from lgdkgp.training.prior import construct_checkpoint


def check(cfg):
    manifest = json.loads(resolve('assets/checkpoints/manifest.json').read_text())
    reference = manifest['environments'][cfg['environment']]
    for item in reference.values():
        if file_hash(resolve(item['file'])) != item['sha256']:
            raise ValueError('Released checkpoint SHA differs: ' + item['file'])
    saved = torch.load(resolve(cfg['prior']), map_location='cpu', weights_only=False)
    model = construct_checkpoint(saved, 'cpu').eval()
    if saved['metadata']['parent_sha256'] != file_hash(resolve(cfg['world_model'])):
        raise ValueError('Prior/world-model binding mismatch')
    conf = model.config
    context = {'history': torch.zeros(1, conf['history_max'], conf['action_dim']),
               'mask': torch.zeros(1, conf['history_max'], dtype=torch.bool),
               'z_current': torch.zeros(1, conf['latent_dim']), 'z_goal': torch.zeros(1, conf['latent_dim'])}
    with torch.no_grad():
        result = model.predict_distribution(context, conf['horizon'])
    if result.mean.shape != (1, conf['horizon'], conf['action_dim']) or not torch.isfinite(result.mean).all():
        raise ValueError('Checkpoint forward compatibility failed')
    # Check the LeWM config/state loader as well, without evaluating a policy.
    from lgdkgp.common.io import load_world
    world = load_world(cfg['world_model'], 'cpu')
    result = {'environment': cfg['environment'], 'checkpoint_sha_verified': True,
              'prior_strict_state_load': True, 'prior_output_shape': list(result.mean.shape),
              'gp_layout': conf.get('gp_layout', 'independent'), 'world_model_class': type(world).__name__,
              'training_runs': 0, 'simulator_episodes': 0}
    print(json.dumps(result))
    return result
