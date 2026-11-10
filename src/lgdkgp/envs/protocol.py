"""Preserve manifest seeds through nested dataset-evaluation resets.

PushT reset(None) samples fresh entropy during World's second dataset reset;
CubeEnv drops its seed argument and OGBench uses global numpy randomness.
Installed dependencies are not modified.
Call once per serial environment; then set the manifest reset_seed per episode.
"""
import random,types
import numpy as np
import torch


class NativeStepTrace:
    """Observe the real vector-env step without changing its mask/reset API."""

    def __init__(self, step):
        self.step = step
        self.reset()

    def reset(self):
        self.success = False
        self.steps = 0
        self.actions = []

    def __call__(self, action, *args, **kwargs):
        result = self.step(action, *args, **kwargs)
        self.actions.append(np.asarray(action).copy())
        self.success = self.success or bool(np.asarray(result[2]).any())
        self.steps += 1
        return result

def install_seeded_reset(env):
    original=env.reset
    env.manifest_reset_seed=0
    def reset(this,seed=None,**kwargs):
        actual=int(this.manifest_reset_seed if seed is None else seed)
        random.seed(actual);np.random.seed(actual);torch.manual_seed(actual)
        this._np_random=np.random.default_rng(actual)
        this._np_random_seed=actual
        return original(seed=actual,**kwargs)
    env.reset=types.MethodType(reset,env)
