"""Environment-specific simulator geometry for local-goal evaluation only."""
import numpy as np


def geometry(adapter,environment,task):
    if environment=='robotwin_a2b':
        components={}
        for name,attribute in [('object_A','object'),('target_B','target_object')]:
            pose=getattr(adapter.env,attribute).get_pose();components[name]=dict(position=np.asarray(pose.p).tolist(),quaternion=np.asarray(pose.q).tolist())
        for arm in ['left','right']:
            # get_obs also renders/copies all camera images. Geometry-only success
            # checks must read the SAME native arm pose without that side effect.
            pose=np.asarray(adapter.env.get_arm_pose(arm));components[arm+'_eef']=dict(position=pose[:3].tolist(),quaternion=pose[3:].tolist())
        return dict(components=components,position_success_components=['object_A','left_eef','right_eef'])
    env=adapter.env;keys=list(env.obj_body_id)
    word='book'if environment=='libero_caddy'else{'alphabet':'alphabet_soup','butter':'butter','milk':'milk','orange':'orange_juice','tomato':'tomato_sauce'}[task]
    names=[k for k in keys if word in k.lower()]
    if len(names)!=1:raise ValueError('Ambiguous manipulated object: '+str((word,keys)))
    name=names[0];body=env.obj_body_id[name];obs=adapter.last_observation
    components=dict(object=dict(position=np.asarray(env.sim.data.body_xpos[body]).tolist(),quaternion=np.asarray(env.sim.data.body_xquat[body]).tolist()),
                    eef=dict(position=np.asarray(obs['robot0_eef_pos']).tolist(),quaternion=np.asarray(obs['robot0_eef_quat']).tolist()))
    return dict(components=components,position_success_components=['object','eef'],object_name=name)


def errors(actual,goal):
    if actual['position_success_components']!=goal['position_success_components']:raise ValueError('Goal component identity differs')
    result={};success=True
    for key,ref in goal['components'].items():
        cur=actual['components'][key];distance=float(np.linalg.norm(np.asarray(cur['position'])-ref['position']))
        q=np.asarray(cur['quaternion']);r=np.asarray(ref['quaternion']);dot=abs(float(q@r/(np.linalg.norm(q)*np.linalg.norm(r))))
        angle=float(2*np.arccos(np.clip(dot,0,1)))
        if not np.isfinite(distance+angle):raise ValueError('Nonfinite evaluation geometry')
        result[key]=dict(position_m=distance,orientation_rad=angle)
        if key in goal['position_success_components']:success=success and distance<=.03
    return dict(components=result,local_goal_success=bool(success))
