"""RoboTwin source-identity reconstruction and native joint-target control."""
from contextlib import contextmanager
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np


@contextmanager
def source_asset_ids(model_ids):
    """Choose declared source files while preserving every original RNG draw."""
    original = glob.glob
    resolved = {}
    def ordered(pattern, *args, **kwargs):
        paths = original(pattern, *args, **kwargs)
        name = Path(pattern).parent.name
        if Path(pattern).name != 'model_data*.json' or name not in model_ids:
            return paths
        ids = [int(Path(p).stem.replace('model_data', '')) for p in paths]
        desired = int(model_ids[name])
        if desired not in ids:
            raise ValueError('Source model ID is not installed: ' + name)
        state = np.random.get_state()
        try:
            index = int(np.random.choice(len(paths)))
        finally:
            np.random.set_state(state)
        target = ids.index(desired)
        paths[index], paths[target] = paths[target], paths[index]
        resolved[name] = [Path(p).name for p in paths]
        return paths
    glob.glob = ordered
    try:
        yield resolved
    finally:
        glob.glob = original


@contextmanager
def render_device(alias=None):
    """Bind native Vulkan rendering to the same visible CUDA device as the worker."""
    if not alias:
        yield
        return
    import sapien
    original=sapien.Engine.create_scene
    def create_scene(engine,config=None):
        if config is not None:
            sapien.physx.set_scene_config(config)
        return sapien.Scene([sapien.physx.PhysxCpuSystem(),sapien.render.RenderSystem(alias)])
    sapien.Engine.create_scene=create_scene
    try:
        yield
    finally:
        sapien.Engine.create_scene=original


def capture_camera_frames_serially(cameras):
    """Finish each native camera frame before submitting another render.

    SAPIEN 3.0.0b1 concurrent RT camera submissions can lose their GPU channel
    on the local RTX 5090 runtime. Reading Color waits for completion without
    adding a render or changing the image returned later by get_rgba.
    """
    pending = ([cameras.left_camera, cameras.right_camera]
               if cameras.collect_wrist_camera else []) + list(cameras.static_camera_list)
    for camera in pending:
        camera.take_picture()
        camera.get_picture("Color")


def install_serial_camera_capture():
    """Patch only this worker process; shared simulator files stay immutable."""
    from envs.camera.camera import Camera
    Camera.update_picture = capture_camera_frames_serially


class NativeRoboTwin:
    def __init__(self, runtime, model_ids, episode_index):
        self.runtime = Path(runtime).resolve()
        self.model_ids = dict(model_ids)
        self.episode_index = int(episode_index)
        self.env = None
        self.executed_actions = []

    def reset(self, seed, settle_steps=0):
        if settle_steps:
            raise ValueError('RoboTwin setup already performs its native stabilization')
        import yaml
        os.chdir(self.runtime)
        sys.path.insert(0, str(self.runtime))
        from envs.place_a2b_left import place_a2b_left
        install_serial_camera_capture()
        config = self.runtime / 'env_cfg/task_config/lewm_place_a2b_left_v1.yml'
        cfg = yaml.safe_load(config.read_text())
        choices = yaml.safe_load((config.parent / '_embodiment_config.yml').read_text())
        robot = (self.runtime / choices[cfg['embodiment'][0]]['file_path']).resolve()
        embodiment = yaml.safe_load((robot / 'config.yml').read_text())
        cfg.update(task_name='place_a2b_left', seed=int(seed), now_ep_num=self.episode_index,
            need_plan=False, save_data=False, save_video=False, render_freq=0, eval_mode=False,
            dual_arm_embodied=True, left_robot_file=str(robot), right_robot_file=str(robot),
            left_embodiment_config=embodiment, right_embodiment_config=embodiment)
        self.replay_config = dict(cfg)
        self.env = place_a2b_left()
        with source_asset_ids(self.model_ids) as order, render_device(os.environ.get('LEWM_ROBOTWIN_RENDER_DEVICE')):
            self.env.setup_demo(**cfg)
        self.asset_order = order
        self.scene_identity = {}
        for role in ('A', 'B'):
            name = str(getattr(self.env, 'selected_modelname_' + role))
            model_id = int(getattr(self.env, 'selected_model_id_' + role))
            if self.model_ids.get(name) != model_id:
                raise ValueError('Native scene differs from decoded source identity')
            self.scene_identity[role] = dict(model_name=name, model_id=model_id)
        self.env.step_lim = 300
        self.executed_actions = []
        return self.env.get_obs()

    def success(self):
        return bool(self.env.eval_success or self.env.check_success())

    def observe(self):
        from lgdkgp.data.windows.native_image_codec import robotwin_live_training_pixels
        obs = self.env.get_obs()
        result = dict(proprio=np.asarray(obs['joint_action']['vector'], np.float32),
            success=self.success(), endpose=np.concatenate([obs['endpose']['left_endpose'],
                                                           obs['endpose']['right_endpose']]))
        result['camera_intrinsics']=np.stack([obs['observation'][camera]['intrinsic_cv'] for camera in ('head_camera','left_camera','right_camera')])
        result['camera_extrinsics_gl']=np.stack([obs['observation'][camera]['cam2world_gl'] for camera in ('head_camera','left_camera','right_camera')])
        for key, camera in [('pixels', 'head_camera'), ('left_wrist_pixels', 'left_camera'),
                            ('right_wrist_pixels', 'right_camera')]:
            result[key] = robotwin_live_training_pixels(obs['observation'][camera]['rgb'])
        return result

    def step(self, action):
        command = np.asarray(action, np.float64)
        if command.shape != (14,) or not np.isfinite(command).all():
            raise ValueError('Expected finite14D joint targets')
        before = self.env.take_action_cnt
        self.env.take_action(command, action_type='qpos')
        if self.env.take_action_cnt != before + 1:
            raise ValueError('Native task dropped a control command')
        actual = np.asarray(self.env.robot.get_left_arm_jointState() +
                            self.env.robot.get_right_arm_jointState(), np.float64)
        self.executed_actions.append(actual.copy())

    def close(self):
        if self.env is not None:
            self.env.close_env()
            self.env = None
