"""Single-device training of the frozen LeWM backbone used by LG-DKGP."""
import json
import os
import subprocess
import sys
from functools import partial
from lgdkgp.common.io import ROOT, resolve, load_config, write_json, digest


def train_world(cfg, device):
    import torch
    env = cfg['environment']
    out = resolve(cfg['output']) / 'world'
    if (out / 'complete.json').exists():
        raise FileExistsError('World-model training already completed in this output')
    out.mkdir(parents=True, exist_ok=True)
    settings = load_config(cfg['world_settings'])
    settings['trainer']['devices'] = 1
    settings['trainer']['accelerator'] = 'cpu' if device == 'cpu' else 'gpu'
    settings['subdir'] = env
    settings['output_model_name'] = env + '/lewm'
    os.environ['STABLEWM_HOME'] = str(out / 'cache')
    settings['wandb'] = {'enabled': False}
    if env in ('pusht', 'cube'):
        # Use the original single-view data split and normalization implementation.
        import yaml
        settings['data']['dataset']['name'] = str(resolve(cfg['data_root']) / cfg['dataset'])
        settings.pop('run_identity', None)
        settings.pop('resume_from', None)
        # Hydra's config search path is based on the entry-point directory.
        path = out / 'recipe.yaml'
        path.write_text(yaml.safe_dump(settings))
        subprocess.run([sys.executable, '-m', 'lgdkgp.training.backbone', '--config-path', str(out),
                        '--config-name', 'recipe', 'hydra.run.dir=' + str(out / 'hydra')], cwd=ROOT, check=True)
    else:
        import lightning as pl
        import stable_pretraining as spt
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from lightning.pytorch.callbacks import ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger
        from lgdkgp.data.windows.native_world_windows import NativeWorldWindows
        from lgdkgp.training.backbone import lejepa_forward
        from lgdkgp.models.world.blocks import SIGReg
        from lgdkgp.training.utils import SaveCkptCallback
        manifests = []
        for split, key in [('train', 'train_split'), ('validation', 'validation_split')]:
            m = json.loads(resolve(cfg[key]).read_text())
            m['data_root'] = str(resolve(cfg['data_root']) / cfg['dataset'])
            path = out / (split + '.json')
            write_json(path, m)
            manifests.append(path)
        c = OmegaConf.create(settings)
        pl.seed_everything(c.seed, workers=True)
        datasets = [NativeWorldWindows(p, cfg['action_mean'], cfg['action_std'],
                                      image_size=c.img_size, context=c.history_size) for p in manifests]
        generator = torch.Generator().manual_seed(c.seed)
        options = dict(c.loader)
        if options['num_workers'] == 0:
            options.pop('prefetch_factor', None)
            options['persistent_workers'] = False
        loaders = [torch.utils.data.DataLoader(d, **options, shuffle=i == 0, drop_last=i == 0,
                                               generator=generator if i == 0 else None)
                   for i, d in enumerate(datasets)]
        model = instantiate(c.model)
        module = spt.Module(model=model, sigreg=SIGReg(**c.loss.sigreg.kwargs),
                            forward=partial(lejepa_forward, cfg=c),
                            optim={'model_opt': {'modules': 'model', 'optimizer': dict(c.optimizer),
                                                'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'},
                                                'interval': 'epoch'}})
        callbacks = [SaveCkptCallback(c.output_model_name, c.model, epoch_interval=1),
                     ModelCheckpoint(dirpath=str(out / 'resume'), save_last=True, save_top_k=0)]
        trainer = pl.Trainer(**dict(c.trainer), callbacks=callbacks, num_sanity_val_steps=1,
                              logger=CSVLogger(str(out), name='csv'), enable_checkpointing=True)
        try:
            spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=loaders[0], val=loaders[1]),
                        weights_only=False)()
        finally:
            for d in datasets:
                d.close()
    weights = out / 'cache/checkpoints' / env / 'lewm' / f'weights_epoch_{settings["trainer"]["max_epochs"]}.pt'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    from lgdkgp.common.io import file_hash
    write_json(out / 'complete.json', {'weights': str(weights), 'sha256': file_hash(weights),
                                      'settings_digest': digest(settings)})
    return weights
