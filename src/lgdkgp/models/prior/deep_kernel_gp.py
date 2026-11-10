"""A sparse variational GP on deterministic lifted [phi, psi, time] inputs."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
import gpytorch


class BoundedInducingTime(nn.Module):
    def __init__(self,horizon):
        super().__init__()
        self.horizon=horizon

    def forward(self,raw):
        return torch.cat((raw[...,:-1],1+(self.horizon-1)*raw[...,-1:].sigmoid()),-1)

    def right_inverse(self,value):
        if self.horizon==1:
            return torch.cat((value[...,:-1],torch.zeros_like(value[...,-1:])), -1)
        unit=((value[...,-1:]-1)/(self.horizon-1)).clamp(1e-5,1-1e-5)
        return torch.cat((value[...,:-1],torch.logit(unit)),-1)


class DeepTemporalKernel(gpytorch.kernels.Kernel):
    def __init__(self,feature_dim=32,temporal='matern32',batch_shape=torch.Size()):
        super().__init__(batch_shape=batch_shape)
        self.deep=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
            ard_num_dims=feature_dim,active_dims=tuple(range(feature_dim)),batch_shape=batch_shape),batch_shape=batch_shape)
        self.context=gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim,
            active_dims=tuple(range(feature_dim,2*feature_dim)),batch_shape=batch_shape)
        kw={'active_dims':(2*feature_dim,),'batch_shape':batch_shape}
        if temporal=='rbf':self.time=gpytorch.kernels.RBFKernel(**kw)
        elif temporal in ('matern32','matern52'):
            self.time=gpytorch.kernels.MaternKernel(nu=1.5 if temporal=='matern32' else 2.5,**kw)
        else:raise ValueError('Unknown temporal kernel')
        self.time.lengthscale=4.
        self.raw_eta=nn.Parameter(torch.zeros(batch_shape))

    def forward(self,x1,x2,diag=False,**params):
        eta=F.softplus(self.raw_eta)
        if self.batch_shape:eta=eta.reshape(*self.batch_shape,*([1] if diag else [1,1]))
        return self.deep(x1,x2,diag=diag,**params)+eta*self.context(x1,x2,diag=diag,**params)*self.time(x1,x2,diag=diag,**params)


class DeepSparseGP(gpytorch.models.ApproximateGP):
    def __init__(self,inducing,horizon=25,feature_dim=32,temporal='matern32'):
        if inducing.ndim not in (2,3) or inducing.shape[-1]!=2*feature_dim+1:
            raise ValueError('Inducing inputs must use the declared lifted feature domain')
        batch_shape=torch.Size(inducing.shape[:-2])
        distribution=gpytorch.variational.CholeskyVariationalDistribution(inducing.shape[-2],batch_shape=batch_shape)
        strategy=gpytorch.variational.VariationalStrategy(self,inducing.clone(),distribution,
            learn_inducing_locations=True,jitter_val=1e-5)
        super().__init__(strategy)
        parametrize.register_parametrization(self.variational_strategy,'inducing_points',BoundedInducingTime(horizon))
        self.mean_module=gpytorch.means.ZeroMean(batch_shape=batch_shape)
        self.covar_module=DeepTemporalKernel(feature_dim,temporal,batch_shape)

    def forward(self,x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x),self.covar_module(x))
