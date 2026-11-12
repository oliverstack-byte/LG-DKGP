"""Expert-scaled isotropic or GP-covariance CEM for the full prior."""
import math
import torch
from lgdkgp.common.io import resolve
from lgdkgp.training.prior import construct_checkpoint


def expert_scaled_covariance(function_covariance, sigma, *, covariance_scale=2., floor_std=.1):
    """Match mean marginal variance to sigma**2, retaining GP covariance shape.

    The input is residual-function covariance in time-major (H, d) order,
    without Gaussian likelihood noise. Calibration precedes normalization.
    """
    if (not math.isfinite(sigma) or sigma <= 0 or
            not math.isfinite(covariance_scale) or covariance_scale <= 0 or
            not math.isfinite(floor_std) or floor_std <= 0):
        raise ValueError('Expert sigma, GP multiplier, and covariance floor must be finite and positive')
    if (function_covariance.ndim != 2 or function_covariance.shape[0] == 0 or
            function_covariance.shape[0] != function_covariance.shape[1] or
            not function_covariance.is_floating_point() or
            not torch.isfinite(function_covariance).all()):
        raise ValueError('Expected a finite square GP function covariance')
    covariance = covariance_scale * (function_covariance + function_covariance.mT) * .5
    covariance = covariance + floor_std ** 2 * torch.eye(
        len(covariance), device=covariance.device, dtype=covariance.dtype)
    average_variance = covariance.diagonal().mean()
    if not torch.isfinite(average_variance) or average_variance <= 0:
        raise ValueError('GP covariance must have positive mean marginal variance')
    return covariance * (float(sigma) ** 2 / average_variance)


def covariance_factor(covariance):
    """Factor once per call, using the manuscript's numerical-jitter schedule."""
    covariance = (covariance + covariance.mT) * .5
    eye = torch.eye(len(covariance), device=covariance.device, dtype=covariance.dtype)
    for jitter in (0., 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        factor, info = torch.linalg.cholesky_ex(covariance + jitter * eye)
        if info.item() == 0:
            return factor, jitter
    raise ValueError('Search covariance is not positive definite after allowed numerical jitter')


@torch.no_grad()
def cem(center, sigma, cost, *, num_samples=300, iterations=30, elites=30,
        generator=None, covariance=None, diagnostics=None):
    """Refit diagonal noise in fixed whitened coordinates, then map to actions.

    ``covariance`` is an already scaled (H*d, H*d) initial covariance. When
    omitted, the original scalar-sigma arithmetic is preserved exactly.
    """
    if not 2 <= elites <= num_samples or iterations < 1 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('Invalid CEM configuration')
    if center.ndim != 2 or not torch.isfinite(center).all():
        raise ValueError('Expected a finite (horizon, action dimension) center')
    mean, std = torch.zeros_like(center), torch.ones_like(center)
    factor, jitter = None, 0.
    if covariance is None:
        scale = torch.full_like(center, float(sigma) ** 2).sqrt()
        average_variance = float(sigma) ** 2
    else:
        covariance = covariance.to(device=center.device, dtype=center.dtype)
        if covariance.shape != (center.numel(), center.numel()) or not torch.isfinite(covariance).all():
            raise ValueError('Search covariance must be finite and match the flattened action shape')
        factor, jitter = covariance_factor(covariance)
        average_variance = covariance.diagonal().mean().item()

    def actions_from_noise(noise):
        if factor is None:
            return center + scale * noise
        return center + (noise.reshape(-1, center.numel()) @ factor.mT).reshape_as(noise)

    if diagnostics is not None:
        diagnostics.update(initial_mean_variance=average_variance,
                           cholesky_jitter=jitter,
                           effective_initial_mean_variance=average_variance + jitter)
    for _ in range(iterations):
        noise = mean + std * torch.randn((num_samples, *center.shape), device=center.device,
                                        dtype=center.dtype, generator=generator)
        noise[0] = mean
        values = cost(actions_from_noise(noise))
        if values.shape != (num_samples,) or not torch.isfinite(values).all():
            raise ValueError('Invalid world-model costs')
        chosen = noise[values.topk(elites, largest=False).indices]
        mean, std = chosen.mean(0), chosen.std(0).clamp_min(1e-6)
    return actions_from_noise(mean)


class Planner:
    def __init__(self, world, checkpoint, cfg, device='cuda:0', seed=101):
        saved = torch.load(resolve(checkpoint), map_location='cpu', weights_only=False)
        if saved.get('kind') != 'lg_dkgp' or saved.get('stage') != 'gp' or not saved.get('fit_complete'):
            raise ValueError('A completed full LG-DKGP checkpoint is required')
        if saved['model_config'].get('history_only') or saved['metadata']['variant'] != 'direct':
            raise ValueError('Only the full conditional method is released')
        self.prior = construct_checkpoint(saved, device).eval().requires_grad_(False)
        self.world, self.cfg, self.device = world, cfg, device
        self.covariance_mode = cfg.get('covariance_mode', 'isotropic')
        if self.covariance_mode not in ('isotropic', 'gp_covariance'):
            raise ValueError('covariance_mode must be isotropic or gp_covariance')
        self.gp_covariance_scale = float(cfg.get('gp_covariance_scale', 2.))
        self.gp_covariance_floor_std = float(cfg.get('gp_covariance_floor_std', .1))
        if self.covariance_mode == 'gp_covariance' and any(
                not math.isfinite(value) or value <= 0
                for value in (self.gp_covariance_scale, self.gp_covariance_floor_std)):
            raise ValueError('GP covariance multiplier and floor must be finite and positive')
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
        distribution = self.prior.predict_distribution(context, h)
        center = distribution.mean[0]
        covariance = None
        if self.covariance_mode == 'gp_covariance':
            covariance = expert_scaled_covariance(
                distribution.function_covariance[0], sigma,
                covariance_scale=self.gp_covariance_scale,
                floor_std=self.gp_covariance_floor_std)
        expanded = {k: v.to(self.device).unsqueeze(1).expand(1, self.cfg['num_samples'], *v.shape[1:])
                    for k, v in info.items() if torch.is_tensor(v)}

        def cost(actions):
            packed = actions.reshape(len(actions), h // block, block * self.action_dim)[None]
            return self.world.get_cost(dict(expanded), packed)[0]

        search_diagnostics = {}
        actions = cem(center, sigma, cost, generator=self.generator,
                      covariance=covariance, diagnostics=search_diagnostics,
                      **{k: self.cfg[k] for k in ('num_samples', 'iterations', 'elites')})
        # Preserve the final full-retention arithmetic of the recorded evaluator.
        actions = center + (actions - center)
        self.last = {'history_length': length, 'prior_mean': center.cpu().tolist(),
                     'planned_actions_standardized': actions.cpu().tolist(),
                     'candidate_scores': self.cfg['num_samples'] * self.cfg['iterations'],
                     'covariance_mode': self.covariance_mode, 'expert_sigma': float(sigma),
                     'search': search_diagnostics}
        if covariance is not None:
            self.last['search'].update(gp_covariance_scale=self.gp_covariance_scale,
                                       gp_covariance_floor_std=self.gp_covariance_floor_std)
        return actions.cpu()
