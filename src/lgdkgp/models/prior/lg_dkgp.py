"""Locality-gated neural mean and deep-kernel residual distribution, frozen LeWM external."""
from dataclasses import dataclass
import time as clock
import torch
from torch import nn
import gpytorch
from lgdkgp.models.prior.locality_gated_encoder import LocalityGatedEncoder,position_encoding
from lgdkgp.models.prior.locality_gated_encoder import history_features
from lgdkgp.models.prior.deep_kernel_gp import DeepSparseGP


@dataclass
class ActionDistribution:
    mean: torch.Tensor
    function_covariance: torch.Tensor
    observation_covariance: torch.Tensor
    search_covariance: torch.Tensor
    diagnostics: list
    context: torch.Tensor = None


class LGDKGP(nn.Module):
    def __init__(self,latent_dim,action_dim,history_max=32,horizon=25,hidden=128,
                 heads=4,layers=2,feature_dim=32,dropout=.1,history_only=False,
                 temporal='matern32',covariance_scale=1.,exploration_std=.05,
                 gate_mode='learned',feature_mode='learned',context_mode='conditional',gp_layout='independent'):
        super().__init__()
        if gp_layout not in ('independent','batched'):raise ValueError('Unknown GP layout')
        self.gp_layout=gp_layout
        if gate_mode not in ('learned','local','global','fixed'):raise ValueError('Unknown gate ablation')
        if feature_mode not in ('learned','raw'):raise ValueError('Unknown feature map ablation')
        if context_mode not in ('conditional','time_only'):raise ValueError('Unknown context ablation')
        if context_mode=='time_only' and not history_only:raise ValueError('Time-only ablation must disable visual inputs')
        self.config=dict(latent_dim=latent_dim,action_dim=action_dim,history_max=history_max,
            horizon=horizon,hidden=hidden,heads=heads,layers=layers,feature_dim=feature_dim,
            dropout=dropout,history_only=history_only,temporal=temporal,
            covariance_scale=covariance_scale,exploration_std=exploration_std)
        for key,value,default in [('gate_mode',gate_mode,'learned'),('feature_mode',feature_mode,'learned'),
                                  ('context_mode',context_mode,'conditional')]:
            if value!=default:self.config[key]=value
        self.gate_mode,self.feature_mode,self.context_mode=gate_mode,feature_mode,context_mode
        if gp_layout!='independent':self.config['gp_layout']=gp_layout
        raw_width=(0 if history_only else 2*latent_dim)+2*history_max*action_dim+2*history_max+1
        self.kernel_feature_dim=1 if context_mode=='time_only' else (raw_width if feature_mode=='raw' else feature_dim)
        self.encoder=LocalityGatedEncoder(latent_dim,action_dim,hidden,heads,layers,dropout,history_only)
        self.context_dim=hidden if history_only else 2*latent_dim+hidden
        self.time_width=32
        query_dim=self.context_dim+self.time_width
        def mlp(input_dim,output_dim):
            return nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Linear(hidden,output_dim))
        self.mean_head=mlp(query_dim,action_dim)
        self.phi=mlp(query_dim,feature_dim)
        self.psi=mlp(self.context_dim,feature_dim)
        self.register_buffer('latent_mean',torch.zeros(latent_dim))
        self.register_buffer('latent_std',torch.ones(latent_dim))
        self.gps=nn.ModuleList()
        self.likelihoods=nn.ModuleList()

    def neural_query(self,batch,horizon=None):
        horizon=horizon or self.config['horizon']
        if not 1<=horizon<=self.config['horizon']:raise ValueError('Requested horizon outside model support')
        history,mask=batch['history'],batch['mask']
        if history.shape[1]!=self.config['history_max']:raise ValueError('History schema mismatch')
        if self.context_mode=='time_only':
            clean,_,_,_=history_features(history,mask)
            positions=torch.arange(history.shape[1],device=history.device)[None].expand(len(history),-1)
            last=torch.where(mask,positions,-1).max(-1).values
            hold=clean[torch.arange(len(history),device=history.device),last.clamp_min(0)]
            hold=torch.where((last>=0)[:,None],hold,torch.zeros_like(hold))
            context=history.new_zeros(len(history),self.context_dim)
            time=torch.arange(1,horizon+1,device=history.device,dtype=history.dtype)
            query=torch.cat((context[:,None].expand(-1,horizon,-1),
                position_encoding(time,self.time_width)[None].expand(len(history),-1,-1)),-1)
            # Explicit hold base plus a learned global time trend; no conditional kernel features.
            self.last_attention_seconds=0.
            return self.mean_head(query)+hold[:,None],query,context,time,[]
        if self.config['history_only']:z=goal=None
        else:
            z=(batch['z_current']-self.latent_mean)/self.latent_std
            goal=(batch['z_goal']-self.latent_mean)/self.latent_std
        profile=batch.get('profile',False)
        if profile and history.device.type=='cuda':torch.cuda.synchronize(history.device)
        attention_started=clock.perf_counter()
        fixed={'local':0.,'global':1.,'fixed':.25}.get(self.gate_mode)
        h,diagnostics=self.encoder(z,goal,history,mask,batch.get('forced_gate',fixed))
        if profile and history.device.type=='cuda':torch.cuda.synchronize(history.device)
        self.last_attention_seconds=clock.perf_counter()-attention_started
        context=h if self.config['history_only'] else torch.cat((z,goal,h),-1)
        time=torch.arange(1,horizon+1,device=h.device,dtype=h.dtype)
        pe=position_encoding(time,self.time_width)
        query=torch.cat((context[:,None].expand(-1,horizon,-1),pe[None].expand(len(h),-1,-1)),-1)
        return self.mean_head(query),query,context,time,diagnostics

    def mean_prediction(self,batch,horizon=None):
        return self.neural_query(batch,horizon)[0]

    def lifted_query(self,batch,horizon=None):
        mean,query,context,time,diagnostics=self.neural_query(batch,horizon)
        x=self.kernel_query(batch,query,context,time)
        return mean,x,diagnostics

    def kernel_query(self,batch,query,context,time):
        if self.context_mode=='time_only':
            u=query.new_zeros(len(query),len(time),1);s=u
        elif self.feature_mode=='raw':
            clean,delta,valid,_=history_features(batch['history'],batch['mask'])
            parts=[clean.flatten(1),delta.flatten(1),batch['mask'].to(clean),valid.to(clean)]
            if not self.config['history_only']:
                parts=[(batch['z_current']-self.latent_mean)/self.latent_std,
                       (batch['z_goal']-self.latent_mean)/self.latent_std]+parts
            raw=torch.cat(parts,-1)
            repeated=raw[:,None].expand(-1,len(time),-1)
            u=torch.cat((repeated,(time/self.config['horizon'])[None,:,None].expand(len(query),-1,-1)),-1)
            s=torch.cat((repeated,repeated.new_zeros(len(query),len(time),1)),-1)
        else:
            u=self.phi(query);s=self.psi(context)[:,None].expand(-1,len(time),-1)
        return torch.cat((u,s,time[None,:,None].expand(len(query),-1,-1)),-1)

    def initialize_gp(self,inducing):
        if self.gps:raise ValueError('GP already initialized')
        c=self.config
        if self.gp_layout=='batched':
            if inducing.ndim==2:inducing=inducing.unsqueeze(0).expand(c['action_dim'],-1,-1).clone()
            if inducing.ndim!=3 or inducing.shape[0]!=c['action_dim']:raise ValueError('Batched inducing action dimension mismatch')
            self.gps=nn.ModuleList([DeepSparseGP(inducing,c['horizon'],self.kernel_feature_dim,c['temporal']).to(inducing)])
            self.likelihoods=nn.ModuleList([gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([c['action_dim']])).to(inducing)])
        else:
            self.gps=nn.ModuleList([DeepSparseGP(inducing,c['horizon'],self.kernel_feature_dim,c['temporal']).to(inducing) for _ in range(c['action_dim'])])
            self.likelihoods=nn.ModuleList([gpytorch.likelihoods.GaussianLikelihood().to(inducing) for _ in self.gps])
        # Kernel features must be deterministic even while optimizing parameters.
        for module in self.modules():
            if isinstance(module,nn.Dropout):module.p=0.

    def elbo_loss(self,batch,num_data):
        if not self.gps:raise ValueError('GP stage has not been initialized')
        mean,x,_=self.lifted_query(batch)
        residual=(batch['target']-mean).reshape(-1,self.config['action_dim'])
        flat=x.reshape(-1,x.shape[-1])
        if self.gp_layout=='batched':
            # A batch of independent outputs, not a correlated multitask GP.
            gp,like=self.gps[0],self.likelihoods[0]
            return -gpytorch.mlls.VariationalELBO(like,gp,num_data=num_data)(gp(flat),residual.mT).sum()
        return sum(-gpytorch.mlls.VariationalELBO(like,gp,num_data=num_data)(gp(flat),residual[:,j])
                   for j,(gp,like) in enumerate(zip(self.gps,self.likelihoods)))

    @torch.no_grad()
    def predict_distribution(self,batch,horizon=None):
        self.eval()
        if not self.gps:raise ValueError('A mean-only checkpoint is not a fitted LG-DKGP')
        profile=batch.get('profile',False)
        def tick():
            if profile and batch['history'].device.type=='cuda':torch.cuda.synchronize(batch['history'].device)
            return clock.perf_counter()
        started=tick()
        mean,query,context,time,diagnostics=self.neural_query(batch,horizon)
        encoded=tick()
        x=self.kernel_query(batch,query,context,time)
        b,h,d=mean.shape
        function=mean.new_zeros(b,h*d,h*d)
        noise=mean.new_zeros(h*d)
        if self.gp_layout=='batched':
            posterior=self.gps[0](x.unsqueeze(1))  # [B,D,H], covariance [B,D,H,H]
            mean=mean+posterior.mean.transpose(-1,-2)
            # Preserve time-major action ordering and zero cross-action covariance.
            covariance=posterior.covariance_matrix
            for j in range(d):function[:,j::d,j::d]=covariance[:,j]
            noise=self.likelihoods[0].noise.squeeze(-1).repeat(h)
        else:
            for j,(gp,like) in enumerate(zip(self.gps,self.likelihoods)):
                posterior=gp(x)
                mean[:,:,j]+=posterior.mean
                function[:,j::d,j::d]=posterior.covariance_matrix
                noise[j::d]=like.noise.reshape(())
        function=(function+function.mT)*.5
        observation=function+torch.diag(noise)[None]
        search=self.config['covariance_scale']*function+self.config['exploration_std']**2*torch.eye(h*d,device=mean.device,dtype=mean.dtype)[None]
        self.last_prediction_timing={'encoder_mean_seconds':encoded-started,
            'attention_seconds':self.last_attention_seconds,'features_gp_seconds':tick()-encoded}
        return ActionDistribution(mean,function,observation,search,diagnostics,context)
