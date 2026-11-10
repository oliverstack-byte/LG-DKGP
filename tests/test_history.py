import torch
from lgdkgp.models.prior.locality_gated_encoder import history_features


def test_empty_and_single_history_have_no_invented_difference():
    history = torch.arange(12).reshape(1, 4, 3).float()
    for n in [0, 1]:
        mask = torch.zeros(1, 4, dtype=torch.bool)
        if n:
            mask[:, -n:] = True
        clean, delta, valid, _ = history_features(history, mask)
        assert not delta.any()
        assert not valid.any()
        assert (clean[~mask] == 0).all()


def test_window_history_precedes_target_and_empty_mask_is_zero(tmp_path):
    import json
    import hashlib
    from lgdkgp.data.windows.variable_history import VariableHistoryWindows
    path = tmp_path / 'normalizer.pt'
    torch.save({'mean': torch.zeros(2), 'std': torch.ones(2)}, path)
    index = {'identity': {'schema': 'lg_windows_v1'}, 'normalizer_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'episodes': [], 'action_dim': 2, 'latent_dim': 2, 'horizon': 25}
    (tmp_path / 'dataset.json').write_text(json.dumps(index))
    reader = VariableHistoryWindows(tmp_path, 'train', history_max=5, horizon=5)
    actions = torch.arange(80).reshape(40, 2).float()
    payload = {'action': actions, 'latent': torch.arange(80).reshape(40, 2).float()}
    batch = reader._assemble(payload, {'episode_id': 7}, torch.tensor([0, 4]), torch.tensor([0, 1]), 1)
    assert not batch['mask'][0].any()
    assert not batch['history'][0].any()
    torch.testing.assert_close(batch['history'][1, -1], actions[3])
    torch.testing.assert_close(batch['target'][1, 0], actions[4])
    torch.testing.assert_close(batch['z_goal'][1], payload['latent'][29])
