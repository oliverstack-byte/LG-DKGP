"""Masked, causal action-history features and soft local/global cross attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def position_encoding(positions, width):
    frequency = torch.exp(torch.arange(0, width, 2, device=positions.device,
                                      dtype=positions.dtype) * (-math.log(10000.)/width))
    angle = positions[..., None]*frequency
    return torch.stack((angle.sin(), angle.cos()), -1).flatten(-2)[..., :width]


def history_features(history, mask):
    """Mask before arithmetic: even NaN padding is irrelevant; no fictional predecessor."""
    if history.ndim != 3 or mask.shape != history.shape[:2]:
        raise ValueError('Expected history[B,L,d] and mask[B,L]')
    mask = mask.bool()
    if ((mask[:, :-1]) & (~mask[:, 1:])).any():
        raise ValueError('History must be right-aligned, in temporal order')
    clean = torch.where(mask[..., None], history, torch.zeros_like(history))
    if not torch.isfinite(clean).all():
        raise ValueError('Nonfinite valid history')
    previous = F.pad(clean[:, :-1], (0,0,1,0))
    delta_valid = mask & F.pad(mask[:, :-1], (1,0), value=False)
    delta = torch.where(delta_valid[..., None], clean-previous, torch.zeros_like(clean))
    age = torch.arange(history.shape[1], 0, -1, device=history.device, dtype=history.dtype)
    return clean, delta, delta_valid, age.expand(history.shape[0], -1)


class GatedHistoryLayer(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        if hidden % heads:
            raise ValueError('Hidden width must be divisible by heads')
        self.heads, self.width = heads, hidden//heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.locality = nn.Linear(hidden, heads)
        self.gate = nn.Sequential(nn.Linear(3*hidden, hidden), nn.GELU(), nn.Linear(hidden,1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(.25/.75))
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2*hidden), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2*hidden,hidden))
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, query, tokens, mask, age, forced_gate=None):
        batch, length, hidden = tokens.shape
        q = self.query(query).reshape(batch,self.heads,self.width)
        k = self.key(tokens).reshape(batch,length,self.heads,self.width).transpose(1,2)
        v = self.value(tokens).reshape(batch,length,self.heads,self.width).transpose(1,2)
        score = torch.einsum('bhd,bhld->bhl',q,k)/math.sqrt(self.width)
        locality = F.softplus(self.locality(query))
        # Empty rows use a safe dummy softmax location, then all weights become zero.
        safe = mask.clone()
        safe[~mask.any(1), -1] = True
        global_weights = score.masked_fill(~safe[:,None], -torch.inf).softmax(-1)*mask[:,None]
        local_score = score-locality[...,None]*age[:,None]
        local_weights = local_score.masked_fill(~safe[:,None], -torch.inf).softmax(-1)*mask[:,None]
        local = torch.einsum('bhl,bhld->bhd',local_weights,v).reshape(batch,hidden)
        global_ = torch.einsum('bhl,bhld->bhd',global_weights,v).reshape(batch,hidden)
        gamma = self.gate(torch.cat((query,local,global_),-1)).sigmoid()
        if forced_gate is not None:
            if not 0 <= forced_gate <= 1:
                raise ValueError('Gate override outside [0,1]')
            gamma = torch.full_like(gamma, forced_gate)
        fused = (1-gamma)*local+gamma*global_
        updated = self.norm(query+fused)
        output = self.output_norm(updated+self.ff(updated))
        return output, {'local':local_weights,'global':global_weights,
                        'gamma':gamma,'lambda':locality,'fused':fused,
                        'local_summary':local,'global_summary':global_}


class LocalityGatedEncoder(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden=128, heads=4, layers=2,
                 dropout=.1, history_only=False):
        super().__init__()
        self.latent_dim, self.action_dim = latent_dim, action_dim
        self.hidden, self.history_only = hidden, history_only
        self.action = nn.Linear(action_dim,hidden)
        self.delta = nn.Linear(action_dim,hidden,bias=False)
        self.delta_presence = nn.Linear(1,hidden,bias=False)
        query_dim = 2*action_dim+1 if history_only else 3*latent_dim
        self.query = nn.Linear(query_dim,hidden)
        self.empty = nn.Parameter(torch.zeros(hidden))
        self.layers = nn.ModuleList([GatedHistoryLayer(hidden,heads,dropout) for _ in range(layers)])

    def forward(self, z_current, z_goal, history, mask, forced_gate=None):
        if history.shape[-1] != self.action_dim or history.shape[1] < 1:
            raise ValueError('Invalid history shape')
        clean, delta, delta_valid, age = history_features(history,mask)
        tokens = self.action(clean)+self.delta(delta)+position_encoding(age,self.hidden)
        tokens = tokens+self.delta_presence(delta_valid[...,None].to(clean))
        tokens = tokens*mask[...,None]
        count = mask.sum(1)
        if self.history_only:
            source = torch.cat((clean[:,-1],clean.sum(1)/count[:,None].clamp_min(1),
                                count[:,None].to(clean)/history.shape[1]),-1)
        else:
            if z_current is None or z_goal is None or z_current.shape != (len(history),self.latent_dim) or z_goal.shape != z_current.shape:
                raise ValueError('Current and goal latent shapes do not match encoder')
            source = torch.cat((z_current,z_goal,z_goal-z_current),-1)
        query = self.query(source)
        diagnostics=[]
        for layer in self.layers:
            query, diag = layer(query,tokens,mask.bool(),age,forced_gate)
            diagnostics.append(diag)
        output = torch.where(count[:,None]>0,query,self.empty[None])
        return output, diagnostics
