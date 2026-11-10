"""Integration checks for portable recipes and original checkpoint targets."""
import json

from hydra.utils import get_class
from lgdkgp.common.io import ROOT, load_config, resolve
from lgdkgp.models.compat import install_legacy_model_aliases


def test_five_task_recipes_resolve_outside_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for task in ['pusht', 'cube', 'libero_caddy', 'libero_scene2', 'robotwin_a2b']:
        cfg = load_config(f'configs/tasks/{task}.json')
        assert cfg['environment'] == task
        for key in ['prior_settings', 'world_settings', 'train_split', 'validation_split', 'queries']:
            assert resolve(cfg[key]).is_file(), (task, key)
        assert resolve(cfg['output']).is_relative_to(ROOT)


def test_original_checkpoint_targets_resolve_to_the_reorganized_models():
    install_legacy_model_aliases()
    old_targets = set()

    def visit(value):
        if isinstance(value, dict):
            target = value.get('_target_', '')
            if target.startswith(('jepa.', 'module.', 'planning.native_world_model.')):
                old_targets.add(target)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in (ROOT / 'assets/checkpoints').glob('*/config.json'):
        visit(json.loads(path.read_text()))
    assert 'jepa.JEPA' in old_targets
    assert 'planning.native_world_model.MultiViewJEPA' in old_targets
    for target in old_targets:
        assert get_class(target).__module__.startswith('lgdkgp.models.world.')
