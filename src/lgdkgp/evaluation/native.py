"""Serial simulator evaluation from frozen scene/goal query fixtures."""
import json
import os
import numpy as np
import torch
from lgdkgp.common.io import resolve, file_hash, write_json, load_world
from lgdkgp.data.prepare import image_transform
from lgdkgp.planning.cem import Planner


def run_case(cfg, query, device, output):
    from lgdkgp.envs.native_tail import replay_prefix, handoff_alignment
    from lgdkgp.envs.native_local_goal import geometry, errors
    env = cfg['environment']
    fixture_path = resolve(query['fixture'])
    if file_hash(fixture_path) != query['fixture_sha256']:
        raise ValueError('Query fixture SHA mismatch')
    with np.load(fixture_path) as data:
        fixture = {k: data[k] for k in data.files}
    if env == 'robotwin_a2b':
        from lgdkgp.envs.native_robotwin import NativeRoboTwin
        fixture['trajectory_path'] = str(resolve(query['trajectory']))
        adapter = NativeRoboTwin(resolve(cfg['runtime']['robotwin_root']),
                                json.loads(str(fixture['model_ids_json'])), int(fixture['episode_index']))
    else:
        from lgdkgp.envs.native_libero import NativeLibero
        selected = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        render_gpu = int(selected) if selected.isdigit() else 0
        os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', str(render_gpu))
        adapter = NativeLibero(env, str(fixture['task']), resolve(cfg['runtime']['libero_root']),
                               resolve('configs/environments/libero.json'), output.parent / 'libero_config',
                               gpu=render_gpu, datasets=fixture_path.parent)
    try:
        adapter.reset(int(fixture['seed']), int(fixture['settle_steps']))
        prefix = replay_prefix(adapter, fixture, env)
        alignment, accepted = handoff_alignment(adapter.observe(), fixture, env)
        if not accepted:
            raise ValueError('Scene alignment failed: ' + str(alignment))
        world = load_world(cfg['world_model'], device)
        torch.manual_seed(query['planning_seed'])
        planner = Planner(world, cfg['prior'], cfg['planning'], device, query['planning_seed'])
        mean, std = np.asarray(cfg['action_mean']), np.asarray(cfg['action_std'])
        for a in prefix[-planner.history_max:]:
            planner.observe((a - mean) / std)
        transform = image_transform(cfg['image_size'])
        views = [('pixels', 'agent'), ('wrist_pixels', 'wrist')] if env != 'robotwin_a2b' else [
            ('pixels', 'agent'), ('left_wrist_pixels', 'left_wrist'), ('right_wrist_pixels', 'right_wrist')]
        goal = {('goal' if k == 'pixels' else 'goal_' + k): transform(fixture[col][-1].copy())[None, None].to(device)
                for k, col in views}
        lower, upper = np.full(len(mean), -1.), np.ones(len(mean))
        if env == 'robotwin_a2b':
            lower[:], upper[:] = -10, 10
            lower[[6, 13]], upper[[6, 13]] = 0, 1
        plans, local, terminal, steps = [], False, False, 0
        task = str(fixture['task'])
        goal_geometry = query['goal_geometry']
        if errors(geometry(adapter, env, task), goal_geometry)['local_goal_success']:
            raise ValueError('Frozen query is already at its goal')
        while steps < cfg['planning']['budget']:
            obs = adapter.observe()
            if not plans and cfg.get('frozen_first_observation', True):
                obs = dict(obs, **{key: fixture[col][1].copy() for key, col in views})
            info = {k: transform(obs[k].copy())[None, None].to(device) for k, _ in views}
            info.update(goal)
            info['action'] = torch.zeros(1, 1, len(mean) * cfg['planning']['action_block'], device=device)
            actions = planner.solve(info, cfg['sigma']).numpy() * std + mean
            plans.append(planner.last)
            count = min(cfg['planning']['execution'], cfg['planning']['budget'] - steps)
            for a in actions[:count]:
                adapter.step(np.clip(a, lower, upper))
                actual = adapter.executed_actions[-1]
                planner.observe((actual - mean) / std)
                steps += 1
                local = errors(geometry(adapter, env, task), goal_geometry)['local_goal_success']
                terminal = env == 'robotwin_a2b' and bool(getattr(adapter.env, 'eval_success', False))
                if terminal and not bool(adapter.observe()['success']):
                    raise ValueError('Native terminal lacks success evidence')
                if local or terminal:
                    break
            if local or terminal:
                break
        result = {'state': 'complete', 'id': query['id'], 'seed': query['seed'],
                  'environment': env, 'success': bool(local or terminal), 'local_success': bool(local),
                  'verified_native_terminal': bool(terminal), 'executed_raw_steps': steps,
                  'fixture_sha256': query['fixture_sha256'], 'alignment': alignment, 'plans': plans}
        write_json(output, result)
        return result
    finally:
        adapter.close()
