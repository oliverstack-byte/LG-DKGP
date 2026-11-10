import argparse
import os
import json
from pathlib import Path
from lgdkgp.common.io import resolve, load_config, single_device

ENVS = ['pusht', 'cube', 'libero_caddy', 'libero_scene2', 'robotwin_a2b']


def main():
    os.environ.setdefault('MUJOCO_GL', 'egl')
    parser = argparse.ArgumentParser(description='Single-GPU LG-DKGP training and evaluation')
    sub = parser.add_subparsers(dest='command', required=True,
                                metavar='{prepare,train,train-world,eval,check}')
    for name in ['prepare', 'train', 'train-world', 'eval', 'check']:
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
        p.add_argument('--device', default='cuda:0')
        if name == 'train':
            p.add_argument('--stage', choices=['mean', 'gp', 'both'], default='both')
            p.add_argument('--resume', action='store_true')
            p.add_argument('--mean-checkpoint', help='Completed mean-stage checkpoint for --stage gp; defaults to <output>/priors/mean.pt')
            p.add_argument('--settings')
            p.add_argument('--prior-set', action='append', default=[])
    p = sub.add_parser('_case')
    p.add_argument('--job', required=True)
    p.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = single_device(args.device)
    if args.command == '_case':
        from lgdkgp.evaluation.native import run_case
        job = json.loads(Path(args.job).read_text())
        run_case(job['config'], job['query'], device, Path(job['output']))
        return
    cfg = load_config(args.config, args.set)
    if cfg['environment'] not in ENVS:
        raise ValueError('Unsupported environment')
    if args.command == 'check':
        from lgdkgp.evaluation.verify import check
        check(cfg)
    elif args.command == 'prepare':
        from lgdkgp.data.prepare import prepare
        prepare(cfg, device)
    elif args.command == 'train':
        from lgdkgp.training.prior import train
        if args.mean_checkpoint and args.stage != 'gp':
            parser.error('--mean-checkpoint requires --stage gp')
        settings = load_config(args.settings or cfg['prior_settings'], args.prior_set)
        folder = resolve(cfg['output']) / 'priors'
        for stage in ['mean', 'gp'] if args.stage == 'both' else [args.stage]:
            train(resolve(cfg['output']) / 'cache/lg', folder / (stage + '.pt'), stage, 'direct', device,
                  settings, mean_checkpoint=(resolve(args.mean_checkpoint) if args.mean_checkpoint else folder / 'mean.pt') if stage == 'gp' else None, resume=args.resume)
    elif args.command == 'train-world':
        from lgdkgp.training.world import train_world
        train_world(cfg, device)
    elif args.command == 'eval':
        from lgdkgp.evaluation.runner import evaluate
        evaluate(cfg, device)


if __name__ == '__main__':
    main()
