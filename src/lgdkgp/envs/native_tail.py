"""Source-locked short terminal windows, with causal prefix and explicit eligibility."""
import numpy as np


def handoff_alignment(observation, fixture, environment):
    views=[('pixels','agent'),('wrist_pixels','wrist')] if environment!='robotwin_a2b' else [
        ('pixels','agent'),('left_wrist_pixels','left_wrist'),('right_wrist_pixels','right_wrist')]
    pixels={}
    for key,col in views:
        delta=np.abs(observation[key].astype(np.int16)-fixture[col][1].astype(np.int16))
        pixels[key]=dict(max_abs=int(delta.max()),mae=float(delta.mean()))
    state=float(np.max(np.abs(observation['proprio']-fixture['proprio'][1])))
    result=dict(pixels=pixels,proprio_max_error=state)
    accepted=state<=2e-5
    if environment=='robotwin_a2b':
        actual=observation['endpose'];ref=fixture['endpose'][1]
        position=max(float(np.max(np.abs(actual[i:i+3]-ref[i:i+3])))for i in (0,7))
        quat=max(float(min(np.linalg.norm(actual[i:i+4]-ref[i:i+4]),np.linalg.norm(actual[i:i+4]+ref[i:i+4])))for i in (3,10))
        intrinsic=float(np.max(np.abs(observation['camera_intrinsics']-fixture['camera_intrinsics'][1])))
        extrinsic=float(np.max(np.abs(observation['camera_extrinsics_gl']-fixture['camera_extrinsics_gl'][1])))
        result.update(endpose_position_max=position,quaternion_error=quat,intrinsic_max=intrinsic,extrinsic_max=extrinsic,
                      pixel_contract='JPEG/render diagnostics; physical and camera calibration admission')
        accepted=accepted and position<=2e-5 and quat<=2e-5 and intrinsic<=1e-4 and extrinsic<=2e-5
    else:
        accepted=accepted and all(v['max_abs']<=1 and v['mae']<=.01 for v in pixels.values())
    return result,bool(accepted)


def replay_prefix(adapter, fixture, environment):
    prefix=np.asarray(fixture['prefix_actions'])
    if environment!='robotwin_a2b':
        for action in prefix:adapter.step(action)
        actual=np.asarray(adapter.executed_actions).reshape(prefix.shape).copy()
        if not np.allclose(actual,prefix,atol=1e-7,rtol=0):raise ValueError('Source prefix commands changed')
    else:
        import pickle
        from pathlib import Path
        import hashlib
        path=Path(str(fixture['trajectory_path']))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=str(fixture['trajectory_sha256']):raise ValueError('Source expert path changed')
        with path.open('rb')as stream:trajectory=pickle.load(stream)
        cfg=dict(adapter.replay_config,left_joint_path=trajectory['left_joint_path'],right_joint_path=trajectory['right_joint_path'])
        adapter.env.set_path_lst(cfg)
        capture=0;reached=False
        class HandoffReached(Exception):pass
        original=adapter.env._take_picture
        def tap():
            nonlocal capture,reached
            if capture==len(prefix):
                reached=True
                raise HandoffReached()
            capture+=1
        adapter.env._take_picture=tap
        try:
            try:adapter.env.play_once()
            except HandoffReached:pass
        finally:adapter.env._take_picture=original
        if not reached:raise ValueError('Recorded expert ended before tail handoff')
        actual=prefix.copy()
        # Only the post-prefix control counter is reset, never physical state.
        adapter.env.take_action_cnt=0
        adapter.env.step_lim=50
    adapter.executed_actions=[]
    return actual
