"""PushT/Cube evaluation with the original World reset and success interfaces."""
import json
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.policy import WorldModelPolicy
import stable_worldmodel as swm
from lgdkgp.data.prepare import image_transform
from lgdkgp.common.io import resolve, load_world, write_json
from lgdkgp.planning.cem import Planner
from lgdkgp.envs.protocol import install_seeded_reset, NativeStepTrace


class SerialSolver(CEMSolver):
    def __init__(self, model, planner, sigma, **kwargs):
        super().__init__(model, **kwargs)
        self.planner, self.sigma = planner, sigma
        self.timings = []

    def reset(self):
        self.planner.reset()
        self.timings = []

    def observe_executed(self, action):
        self.planner.observe(action)

    @torch.no_grad()
    def solve(self, info_dict, init_action=None):
        # World.evaluate reseeds the official solver generator per query.
        self.planner.generator = self.torch_gen
        actions = self.planner.solve(info_dict, self.sigma)
        self.timings.append(self.planner.last)
        block = self.planner.cfg['action_block']
        return {'actions': actions.reshape(1, len(actions) // block, -1), 'costs': []}


class ExecutedActionPolicy(WorldModelPolicy):
    def __init__(self, *args, clip_executed=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous = None
        self.clip_executed = clip_executed

    def reset(self):
        self.previous = None
        self.solver.reset()
        if self._action_buffer:
            for b in self._action_buffer:
                b.clear()
        self._next_init = None

    def get_action(self, info_dict, **kwargs):
        if self.previous is not None:
            actual = np.clip(self.previous, self.env.single_action_space.low,
                             self.env.single_action_space.high) if self.clip_executed else self.previous.copy()
            self.solver.observe_executed(self.process['action'].transform(actual)[0])
        action = super().get_action(info_dict, **kwargs)
        self.previous = action.copy()
        return action


def run(cfg, queries, device, output):
    env = cfg['environment']
    spec = yaml.safe_load(resolve('configs/evaluation/' + env + '.yaml').read_text())
    dataset = swm.data.load_dataset(str(resolve(cfg['data_root']) / cfg['dataset']),
                                    keys_to_cache=spec['dataset']['keys_to_cache'])
    process = {}
    for key in spec['dataset']['keys_to_cache']:
        x = dataset.get_col_data(key)
        x = x[np.isfinite(x).all(1)]
        process[key] = StandardScaler().fit(x)
        if key != 'action':
            process['goal_' + key] = process[key]
    # Keep the released action coordinate system exact, including population SD.
    np.testing.assert_allclose(process['action'].mean_, cfg['action_mean'], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(process['action'].scale_, cfg['action_std'], rtol=1e-6, atol=1e-8)
    options = dict(spec['world'], num_envs=1, max_episode_steps=max(100, cfg['planning']['budget']))
    world = swm.World(**options, image_shape=(cfg['image_size'], cfg['image_size']))
    try:
        model = load_world(cfg['world_model'], device)
        planner = Planner(model, cfg['prior'], cfg['planning'], device)
        plan = cfg['planning']
        pc = swm.PlanConfig(warm_start=False, horizon=plan['horizon'] // plan['action_block'],
                            receding_horizon=plan['execution'] // plan['action_block'], action_block=plan['action_block'])
        if plan['execution'] % plan['action_block']:
            raise ValueError('PushT/Cube execution must be divisible by action block')
        solver = SerialSolver(model, planner, cfg['sigma'], num_samples=plan['num_samples'],
                              n_steps=plan['iterations'], topk=plan['elites'], batch_size=1, device=device, seed=101)
        transform = image_transform(cfg['image_size'])
        policy = ExecutedActionPolicy(solver, pc, process, {'pixels': transform, 'goal': transform},
                                      clip_executed=env != 'pusht')
        world.set_policy(policy)
        install_seeded_reset(world.envs.envs[0].unwrapped)
        trace = NativeStepTrace(world.envs.step)
        world.envs.step = trace
        results = []
        for query in queries:
            policy.reset()
            trace.reset()
            solver.torch_gen.manual_seed(query['reset_seed'])
            world.envs.envs[0].unwrapped.manifest_reset_seed = query['reset_seed']
            world.reset(seed=query['reset_seed'])
            metrics = world.evaluate(dataset=dataset, start_steps=[query['start']],
                                     goal_offset=25, eval_budget=plan['budget'], episodes_idx=[query['episode']],
                                     callables=spec['eval']['callables'], video=None)
            success = bool(metrics['episode_successes'][0])
            if success != trace.success or trace.steps > plan['budget']:
                raise ValueError('Native success/budget accounting mismatch')
            record = {'id': query['id'], 'seed': query['seed'], 'environment': env,
                      'state': 'complete', 'success': success, 'executed_raw_steps': trace.steps,
                      'plans': solver.timings}
            results.append(record)
            write_json(output / (str(len(results) - 1).zfill(4) + '.json'), record)
        return results
    finally:
        world.envs.close()
