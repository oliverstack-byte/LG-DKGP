"""Multiview adapter of the original JEPA; inherited rollout and score unchanged."""
import torch
from einops import rearrange
from lgdkgp.models.world.jepa import JEPA


class MultiViewJEPA(JEPA):
    def __init__(self,*args,view_keys,fusion=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.fusion=fusion if fusion is not None else torch.nn.Identity()
        self.view_keys=tuple(view_keys)
        if len(self.view_keys)<2 or len(set(self.view_keys))!=len(self.view_keys):raise ValueError('Distinct native camera slots required')

    def encode(self,info):
        if any(key not in info for key in self.view_keys):raise ValueError('Missing native camera')
        views=[info[key] for key in self.view_keys]
        if any(x.shape!=views[0].shape for x in views):raise ValueError('Native cameras must share time indices/shapes')
        pixels=torch.stack(views,dim=2).float();b,t,v=pixels.shape[:3]
        encoded=self.encoder(rearrange(pixels,'b t v ... -> (b t v) ...'),interpolate_pos_encoding=True)
        projected=self.projector(encoded.last_hidden_state[:,0])
        slots=rearrange(projected,'(b t v) d -> b t v d',b=b,t=t,v=v)
        info['view_emb']=slots
        concatenated=rearrange(slots,'b t v d -> (b t) (v d)')
        info['emb']=rearrange(self.fusion(concatenated),'(b t) d -> b t d',b=b,t=t)
        if 'action' in info:info['act_emb']=self.action_encoder(info['action'])
        return info
