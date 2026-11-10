"""Resolve original LeWM Hydra targets without modifying released checkpoints."""
import importlib
import sys
import types


def install_legacy_model_aliases():
    aliases = {
        'jepa': 'lgdkgp.models.world.jepa',
        'module': 'lgdkgp.models.world.blocks',
        'planning.native_world_model': 'lgdkgp.models.world.multiview',
    }
    legacy_planning = sys.modules.get('planning')
    if legacy_planning is None:
        legacy_planning = types.ModuleType('planning')
        legacy_planning.__path__ = []
        sys.modules['planning'] = legacy_planning
    for old, canonical in aliases.items():
        module = importlib.import_module(canonical)
        sys.modules[old] = module
        if old.startswith('planning.'):
            legacy_planning.native_world_model = module
