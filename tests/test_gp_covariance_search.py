"""Search geometry, expert-scale matching, and checkpoint adapter contracts."""
from types import SimpleNamespace

import pytest
import torch

from lgdkgp.planning.cem import (
    Planner, cem, covariance_factor, expert_scaled_covariance,
)


def temporal_covariance():
    # Time-major H=3, d=2: independent action coordinates, correlated times.
    covariance = torch.zeros(6, 6, dtype=torch.float64)
    temporal = torch.tensor([[1., .7, .4], [.7, 2., .8], [.4, .8, 1.5]],
                            dtype=torch.float64)
    covariance[0::2, 0::2] = temporal
    covariance[1::2, 1::2] = temporal * .3
    return covariance


@pytest.mark.parametrize('sigma', [.362953, .349368, .572329, .653212, .026753])
def test_expert_variance_and_covariance_shape(sigma):
    function = temporal_covariance()
    original = function.clone()
    covariance = expert_scaled_covariance(function, sigma)
    calibrated = 2 * function + .1 ** 2 * torch.eye(6, dtype=function.dtype)
    assert covariance.diagonal().mean().item() == pytest.approx(sigma ** 2)
    def correlation(matrix):
        std = matrix.diagonal().sqrt()
        return matrix / (std[:, None] * std[None, :])
    torch.testing.assert_close(correlation(covariance), correlation(calibrated))
    assert covariance[0, 2] > 0
    assert covariance[0, 1] == 0
    torch.testing.assert_close(function, original)


def test_zero_function_covariance_reduces_to_isotropic():
    covariance = expert_scaled_covariance(torch.zeros(6, 6), .2)
    torch.testing.assert_close(covariance, .2 ** 2 * torch.eye(6))
    center = torch.arange(6).reshape(3, 2).float() / 13
    def cost(actions):
        return (actions - .37).square().sum((1, 2))
    def run(covariance):
        return cem(center, .2, cost, covariance=covariance, num_samples=40,
                   iterations=4, elites=7, generator=torch.Generator().manual_seed(101))
    torch.testing.assert_close(run(covariance), run(None), atol=1e-6, rtol=1e-5)


def test_gp_geometry_is_retained_across_cem_refits():
    sigma = .3
    covariance = expert_scaled_covariance(temporal_covariance(), sigma)
    center = torch.zeros(3, 2, dtype=torch.float64)
    batches = []
    def cost(actions):
        batches.append(actions.clone())
        return (actions - .25).square().sum((1, 2))
    result = cem(center, sigma, cost, covariance=covariance,
                 num_samples=16000, iterations=3, elites=800,
                 generator=torch.Generator().manual_seed(23))
    factor = torch.linalg.cholesky(covariance)
    expected_mean = torch.zeros(6, dtype=center.dtype)
    expected_variance = torch.ones(6, dtype=center.dtype)
    for actions in batches:
        flat = (actions - center).flatten(1)
        white = torch.linalg.solve_triangular(factor, flat.T, upper=False).T
        torch.testing.assert_close(white[0], expected_mean, atol=1e-10, rtol=1e-10)
        empirical = torch.cov(flat[1:].T)
        expected = factor @ torch.diag(expected_variance) @ factor.T
        torch.testing.assert_close(empirical, expected, rtol=.08, atol=.001)
        assert empirical[0, 2] > 0  # Temporal correlation persists after refitting.
        selected = white[(actions - .25).square().sum((1, 2)).topk(800, largest=False).indices]
        expected_mean = selected.mean(0)
        expected_variance = selected.var(0)
    torch.testing.assert_close(result.flatten(), factor @ expected_mean)


def test_cholesky_jitter_and_invalid_covariances():
    factor, jitter = covariance_factor(torch.ones(2, 2))
    assert jitter > 0
    torch.testing.assert_close(factor @ factor.T, torch.ones(2, 2) + jitter * torch.eye(2))
    with pytest.raises(ValueError, match='positive definite'):
        covariance_factor(torch.tensor([[1., 2.], [2., 1.]]))
    with pytest.raises(ValueError, match='flattened action shape'):
        cem(torch.zeros(3, 2), .2, lambda a: a.square().sum((1, 2)), covariance=torch.eye(2))
    for sigma in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite and positive'):
            expert_scaled_covariance(torch.eye(6), sigma)


@pytest.mark.parametrize('mode', ['isotropic', 'gp_covariance'])
def test_planner_uses_function_covariance_once_and_keeps_checkpoint(tmp_path, monkeypatch, mode):
    checkpoint = tmp_path / 'prior.pt'
    torch.save({'kind': 'lg_dkgp', 'stage': 'gp', 'fit_complete': True,
                'metadata': {'variant': 'direct'},
                'model_config': {'history_only': False, 'history_max': 32,
                                 'horizon': 25, 'action_dim': 2}}, checkpoint)
    original = checkpoint.read_bytes()

    class Prior:
        calls = 0
        def eval(self): return self
        def requires_grad_(self, enabled): return self
        def predict_distribution(self, context, horizon):
            self.calls += 1
            return SimpleNamespace(mean=torch.zeros(1, 3, 2),
                                   function_covariance=temporal_covariance().float()[None],
                                   observation_covariance=torch.full((1, 6, 6), float('nan')),
                                   search_covariance=torch.full((1, 6, 6), float('nan')))

    prior = Prior()
    import importlib
    module = importlib.import_module('lgdkgp.planning.cem')
    monkeypatch.setattr(module, 'construct_checkpoint', lambda saved, device: prior)

    class World:
        def encode(self, observations): return {'emb': observations['pixels']}
        def get_cost(self, observations, actions): return actions.square().sum((-1, -2))

    cfg = {'horizon': 3, 'action_block': 1, 'execution': 3, 'budget': 75,
           'history_limit': 32, 'num_samples': 16, 'iterations': 3, 'elites': 4,
           'covariance_mode': mode, 'gp_covariance_scale': 2., 'gp_covariance_floor_std': .1}
    planner = Planner(World(), checkpoint, cfg, device='cpu')
    result = planner.solve({'pixels': torch.zeros(1, 1, 8), 'goal': torch.ones(1, 1, 8)}, .026753)
    assert result.shape == (3, 2) and torch.isfinite(result).all()
    assert prior.calls == 1
    assert planner.last['candidate_scores'] == 48
    assert planner.last['covariance_mode'] == mode
    assert planner.last['search']['initial_mean_variance'] == pytest.approx(.026753 ** 2)
    assert checkpoint.read_bytes() == original
