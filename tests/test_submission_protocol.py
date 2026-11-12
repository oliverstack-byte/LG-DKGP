"""Protocol contracts shared by isotropic and GP-covariance search."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lgdkgp.common.io import load_config
from lgdkgp.data.windows.variable_history import VariableHistoryWindows


@pytest.mark.parametrize('task', ['pusht', 'cube', 'libero_caddy', 'libero_scene2', 'robotwin_a2b'])
def test_task_search_modes_and_shared_planning_protocol(task):
    cfg = load_config(f'configs/tasks/{task}.json', ['planning.covariance_mode=gp_covariance'])
    assert cfg['planning']['covariance_mode'] == 'gp_covariance'
    assert cfg['planning']['history_limit'] == 32
    assert cfg['planning']['budget'] == 75
    assert cfg['planning']['horizon'] == cfg['planning']['execution'] == 25
    assert cfg['planning']['gp_covariance_scale'] == 2.
    assert cfg['planning']['gp_covariance_floor_std'] == .1


def test_native_windows_use_future_25_and_reject_terminal_cache(tmp_path):
    stats = tmp_path / 'normalizer.pt'
    torch.save({'mean': torch.zeros(2), 'std': torch.ones(2)}, stats)
    index = {'identity': {'schema': 'lg_windows_v1'},
             'normalizer_sha256': hashlib.sha256(stats.read_bytes()).hexdigest(),
             'episodes': [], 'action_dim': 2, 'latent_dim': 2,
             'horizon': 25, 'goal_policy': 'future_25'}
    path = tmp_path / 'dataset.json'
    for task in ['libero_caddy', 'libero_scene2', 'robotwin_a2b']:
        index['environment'] = task
        path.write_text(json.dumps(index))
        reader = VariableHistoryWindows(tmp_path, 'train')
        payload = {'action': torch.arange(120).reshape(60, 2).float(),
                   'latent': torch.arange(120).reshape(60, 2).float()}
        batch = reader._assemble(payload, {'episode_id': 1},
                                 torch.tensor([0, 7]), torch.tensor([0, 7]), 1)
        torch.testing.assert_close(batch['z_goal'], payload['latent'][[25, 32]])
    index['goal_policy'] = 'terminal'
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match=r't\+25'):
        VariableHistoryWindows(tmp_path, 'train')


@pytest.mark.parametrize('reach_at, expected_success, expected_steps', [(None, False, 3), (2, True, 2)])
def test_native_terminal_flag_does_not_award_or_end_goal_evaluation(
        tmp_path, monkeypatch, reach_at, expected_success, expected_steps):
    from lgdkgp.evaluation import native
    from lgdkgp.envs import native_robotwin, native_tail, native_local_goal

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    class Adapter:
        def __init__(self):
            self.env = SimpleNamespace(eval_success=True)
            self.executed_actions = []
            self.closed = False
        def reset(self, *args): pass
        def observe(self):
            return {'pixels': frame, 'left_wrist_pixels': frame,
                    'right_wrist_pixels': frame, 'success': True}
        def step(self, action): self.executed_actions.append(action)
        def close(self): self.closed = True
    adapter = Adapter()
    monkeypatch.setattr(native_robotwin, 'NativeRoboTwin', lambda *a, **kw: adapter)
    monkeypatch.setattr(native_tail, 'replay_prefix', lambda *a: [])
    monkeypatch.setattr(native_tail, 'handoff_alignment', lambda *a: ({}, True))
    components = ['object_A', 'left_eef', 'right_eef']
    def geometry(distance):
        return {'position_success_components': components,
                'components': {k: {'position': [distance, 0., 0.],
                                   'quaternion': [1., 0., 0., 0.]} for k in components}}
    monkeypatch.setattr(native_local_goal, 'geometry', lambda *a: geometry(
        0. if reach_at is not None and len(adapter.executed_actions) >= reach_at else 1.))
    monkeypatch.setattr(native, 'load_world', lambda *a: object())
    monkeypatch.setattr(native, 'image_transform', lambda size: lambda frame: torch.zeros(3, 2, 2))
    class FakePlanner:
        history_max = 32
        last = {}
        def __init__(self, *a): pass
        def observe(self, action): pass
        def solve(self, info, sigma): return torch.zeros(3, 14)
    monkeypatch.setattr(native, 'Planner', FakePlanner)
    fixture = tmp_path / 'fixture.npz'
    np.savez(fixture, model_ids_json='{}', episode_index=0, seed=101,
             settle_steps=0, task='a2b', agent=np.stack([frame, frame]),
             left_wrist=np.stack([frame, frame]), right_wrist=np.stack([frame, frame]))
    query = {'fixture': str(fixture), 'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
             'trajectory': str(tmp_path / 'trajectory'), 'planning_seed': 101,
             'goal_geometry': geometry(0.), 'id': 'case', 'seed': 101}
    cfg = {'environment': 'robotwin_a2b', 'runtime': {'robotwin_root': str(tmp_path)},
           'world_model': 'unused', 'prior': 'unused', 'action_mean': [0.] * 14,
           'action_std': [1.] * 14, 'image_size': 2, 'sigma': .026753,
           'frozen_first_observation': False,
           'planning': {'budget': 3, 'execution': 3, 'action_block': 1}}
    result = native.run_case(cfg, query, 'cpu', tmp_path / 'result.json')
    assert result['success'] is expected_success
    assert result['executed_raw_steps'] == expected_steps
    assert result['native_terminal_diagnostic'] is True
    assert adapter.closed
