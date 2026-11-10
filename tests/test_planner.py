import os
import torch
import pytest
from lgdkgp.common.io import single_device
from lgdkgp.planning.cem import cem


def reference(center, sigma, cost, seed):
    # The reported solver refits whitened diagonal perturbations, with the
    # current distribution mean as the first candidate in every iteration.
    generator = torch.Generator().manual_seed(seed)
    mean, std = torch.zeros_like(center), torch.ones_like(center)
    scale = torch.full_like(center, sigma ** 2).sqrt()
    for _ in range(4):
        xi = mean + std * torch.randn((12, *center.shape), generator=generator)
        xi[0] = mean
        actions = center + scale * xi
        chosen = xi[cost(actions).topk(3, largest=False).indices]
        mean, std = chosen.mean(0), chosen.std(0).clamp_min(1e-6)
    return center + scale * mean


@pytest.mark.parametrize('dim', [2, 5, 7, 14])
def test_cem_matches_frozen_scalar_scale_update(dim):
    center = torch.arange(5 * dim).reshape(5, dim).float() / 30
    cost = lambda actions: (actions - .2).square().sum((1, 2))
    expected = reference(center, .36, cost, 101)
    got = cem(center, .36, cost, num_samples=12, iterations=4, elites=3,
              generator=torch.Generator().manual_seed(101))
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_cem_budget_and_invalid_score():
    counts = []
    def cost(a):
        counts.append(len(a))
        return a.square().sum((1, 2))
    cem(torch.zeros(5, 2), .2, cost, num_samples=12, iterations=4, elites=3)
    assert sum(counts) == 48
    with pytest.raises(ValueError, match='cost'):
        cem(torch.zeros(5, 2), .2, lambda a: torch.full((len(a),), float('nan')),
            num_samples=12, iterations=1, elites=3)


def test_reject_distributed_or_second_device(monkeypatch):
    with pytest.raises(ValueError):
        single_device('cuda:1')
    monkeypatch.setenv('WORLD_SIZE', '2')
    with pytest.raises(ValueError):
        single_device('cpu')
