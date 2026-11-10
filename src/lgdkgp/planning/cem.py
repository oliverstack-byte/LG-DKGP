"""Full conditional prior with expert-scaled diagonal CEM. No alternate arms."""
import torch
from lgdkgp.common.io import resolve
from lgdkgp.training.prior import construct_checkpoint


@torch.no_grad()
def cem(center, sigma, cost, *, num_samples=300, iterations=30, elites=30, generator=None):
    if not 2 <= elites <= num_samples or iterations < 1 or sigma <= 0:
        raise ValueError('Invalid CEM configuration')
    mean, std = torch.zeros_like(center), torch.ones_like(center)
    scale = torch.full_like(center, float(sigma) ** 2).sqrt()
    for _ in range(iterations):
        noise = mean + std * torch.randn((num_samples, *center.shape), device=center.device,
                                        dtype=center.dtype, generator=generator)
        noise[0] = mean
        values = cost(center + scale * noise)
        if values.shape != (num_samples,) or not torch.isfinite(values).all():
            raise ValueError('Invalid world-model costs')
        chosen = noise[values.topk(elites, largest=False).indices]
        mean, std = chosen.mean(0), chosen.std(0).clamp_min(1e-6)
    return center + scale * mean


class Planner:
    def __init__(self, world, checkpoint, cfg, device='cuda:0', seed=101):
        saved = torch.load(resolve(checkpoint), map_location='cpu', weights_only=False)
        if saved.get('kind') != 'lg_dkgp' or saved.get('stage') != 'gp' or not saved.get('fit_complete'):
            raise ValueError('A completed full LG-DKGP checkpoint is required')
        if saved['model_config'].get('history_only') or saved['metadata']['variant'] != 'direct':
            raise ValueError('Only the full conditional method is released')
        self.prior = construct_checkpoint(saved, device).eval().requires_grad_(False)
        self.world, self.cfg, self.device = world, cfg, device
        self.history = []
        self.generator = torch.Generator(device=device).manual_seed(seed)
        h, block = cfg['horizon'], cfg['action_block']
        if not 1 <= h <= saved['model_config']['horizon'] or h % block:
            raise ValueError('Horizon must fit the checkpoint and action block')
        if not 1 <= cfg['execution'] <= h or cfg['budget'] < 1:
            raise ValueError('Invalid execution/budget')
        if not 0 <= cfg['history_limit'] <= saved['model_config']['history_max']:
            raise ValueError('History limit exceeds trained context')
        self.action_dim = saved['model_config']['action_dim']
        self.history_max = saved['model_config']['history_max']
        self.last = None

    def reset(self, seed=None):
        self.history = []
        if seed is not None:
            self.generator.manual_seed(seed)

    def observe(self, standardized_executed_action):
        a = torch.as_tensor(standardized_executed_action, device=self.device, dtype=torch.float32).flatten()
        if a.shape != (self.action_dim,) or not torch.isfinite(a).all():
            raise ValueError('Invalid executed action')
        self.history.append(a.clone())
        self.history = self.history[-self.history_max:]

    @torch.no_grad()
    def solve(self, info, sigma):
        h, block = self.cfg['horizon'], self.cfg['action_block']
        history = torch.zeros(1, self.history_max, self.action_dim, device=self.device)
        mask = torch.zeros(1, self.history_max, device=self.device, dtype=torch.bool)
        length = min(len(self.history), self.cfg['history_limit'])
        if length:
            history[0, -length:] = torch.stack(self.history[-length:])
            mask[0, -length:] = True
        context = {'history': history, 'mask': mask}
        views = getattr(self.world, 'view_keys', ('pixels',))
        for target, goal in [('z_current', False), ('z_goal', True)]:
            encoded = {v: info[('goal' if v == 'pixels' else 'goal_' + v) if goal else v].to(self.device)
                       for v in views}
            context[target] = self.world.encode(encoded)['emb'][:, -1]
        center = self.prior.predict_distribution(context, h).mean[0]
        expanded = {k: v.to(self.device).unsqueeze(1).expand(1, self.cfg['num_samples'], *v.shape[1:])
                    for k, v in info.items() if torch.is_tensor(v)}

        def cost(actions):
            packed = actions.reshape(len(actions), h // block, block * self.action_dim)[None]
            return self.world.get_cost(dict(expanded), packed)[0]

        actions = cem(center, sigma, cost, generator=self.generator,
                      **{k: self.cfg[k] for k in ('num_samples', 'iterations', 'elites')})
        # Preserve the final full-retention arithmetic of the recorded evaluator.
        actions = center + (actions - center)
        self.last = {'history_length': length, 'prior_mean': center.cpu().tolist(),
                     'planned_actions_standardized': actions.cpu().tolist(),
                     'candidate_scores': self.cfg['num_samples'] * self.cfg['iterations']}
        return actions.cpu()
