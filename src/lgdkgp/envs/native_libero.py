"""Isolated native LIBERO control adapter; no learned policy or SAGE controller."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import yaml


class NativeLibero:
    def __init__(self,suite,task,libero_root,environment_config,config_dir,render=True,gpu=0,factory=None,datasets=None):
        self.suite,self.task=suite,task
        config=json.loads(Path(environment_config).read_text());spec=config['suites'][suite]
        root=Path(libero_root).resolve()/'libero/libero';out=Path(config_dir).resolve();out.mkdir(parents=True,exist_ok=True)
        settings=dict(benchmark_root=str(root),bddl_files=str(root/'bddl_files'),init_states=str(root/'init_files'),
                      assets=str(root/'assets'),datasets=str(Path(datasets).resolve()) if datasets is not None else str(out))
        destination=out/'config.yaml'
        if destination.exists() and yaml.safe_load(destination.read_text())!=settings:raise ValueError('Existing native runtime config differs')
        destination.write_text(yaml.safe_dump(settings));os.environ['LIBERO_CONFIG_PATH']=str(out)
        sys.path.insert(0,str(Path(libero_root).resolve()))
        if factory is None:
            from libero.libero.envs import TASK_MAPPING
            factory=TASK_MAPPING[spec['problem_name']]
        bddl=next((root/'bddl_files').rglob(spec['tasks'][task]))
        kwargs=dict(config['env_kwargs']);kwargs.update(bddl_file_name=str(bddl),has_offscreen_renderer=render,
                   use_camera_obs=render,render_gpu_device_id=gpu)
        self.env=factory(**kwargs);self.last_observation=None;self.executed_actions=[]

    def reset(self,seed,settle_steps=0):
        self.executed_actions=[];self.env.seed(int(seed));self.last_observation=self.env.reset()
        for _ in range(settle_steps):self.last_observation,_,_,_=self.env.step(np.zeros(7,np.float32))
        # Settling belongs to reset, not to planner-visible executed action history.
        return self.last_observation

    def step(self,action):
        action=np.asarray(action,dtype=np.float32)
        if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Expected finite seven-dimensional native action')
        lower,upper=self.env.action_spec;actual=np.clip(action,lower,upper)
        result=self.env.step(actual);self.last_observation=result[0];self.executed_actions.append(actual.copy())
        return result

    def observe(self):
        if self.last_observation is None:raise ValueError('Reset native environment before observing')
        obs=self.last_observation
        result=dict(proprio=np.concatenate([obs['robot0_joint_pos'],obs['robot0_gripper_qpos']]).astype(np.float32),
                    success=bool(self.env._check_success()))
        for target,key in [('pixels','agentview_image'),('wrist_pixels','robot0_eye_in_hand_image')]:
            if key in obs:result[target]=np.asarray(obs[key])
        return result

    def close(self):self.env.close()
