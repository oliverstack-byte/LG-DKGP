"""Replayable neural-mean and DKGP stages on frozen episode shards, no LeWM training."""
import json
import math
import time
from pathlib import Path

import torch
from lgdkgp.models.prior.lg_dkgp import LGDKGP
from lgdkgp.data.windows.variable_history import VariableHistoryWindows
from lgdkgp.common.io import digest,file_hash,write_json,resolve
from lgdkgp.common.io import atomic_torch_save


def to_device(batch,device):
    # Episode/start are provenance only. Keep them on CPU rather than enqueue
    # tiny transfers; length is still needed by validation masks on the device.
    return {key:value.to(device,non_blocking=True) if torch.is_tensor(value) and key not in ('episode','start')
            else value for key,value in batch.items()}


def construct_checkpoint(checkpoint,device='cpu'):
    if checkpoint.get('kind') not in ('lg_dkgp_mean','lg_dkgp'):
        raise ValueError('Expected a new LG-DKGP checkpoint; legacy GP cannot be loaded')
    model=LGDKGP(**checkpoint['model_config']).to(device)
    if checkpoint['stage']=='gp':model.initialize_gp(checkpoint['inducing'].to(device))
    model.load_state_dict(checkpoint['model'])
    return model


@torch.no_grad()
def validate(model,reader,stage,batch_size,device,fixed_length=None):
    model.eval();keys=('mse','joint_nll_per_scalar','marginal_nll','coverage95');count=0
    sums=torch.zeros(4,device=device,dtype=torch.float64)
    lengths={}
    for batch in reader.batches(batch_size,shuffle=False,fixed_length=fixed_length):
        # Inspect CPU metadata before transfer; GPU unique()/int() synchronize
        # once per history bucket. Keep reductions on device and fetch together.
        history_values,history_sizes=batch['length'].unique(return_counts=True) if stage=='gp' else ([],[])
        history_lengths=history_values.tolist() if stage=='gp' else []
        history_counts=dict(zip(history_lengths,history_sizes.tolist())) if stage=='gp' else {}
        batch=to_device(batch,device)
        if stage=='mean':
            mean=model.mean_prediction(batch);error=(mean-batch['target']).square()
            sums[0].add_(error.sum().double());count+=error.numel()
            continue
        distribution=model.predict_distribution(batch)
        diff=(batch['target']-distribution.mean).double()
        K=distribution.observation_covariance.double()
        L=torch.linalg.cholesky(K+1e-5*torch.eye(K.shape[-1],device=device,dtype=K.dtype))
        flat=diff.flatten(1)
        white=torch.linalg.solve_triangular(L,flat[...,None],upper=False).squeeze(-1)
        joint=.5*(white.square().sum(-1)+2*L.diagonal(dim1=-2,dim2=-1).log().sum(-1)+flat.shape[-1]*math.log(2*math.pi))
        variance=K.diagonal(dim1=-2,dim2=-1).reshape_as(diff)
        reductions=[diff.square().sum(),joint.sum(),
                    (.5*(diff.square()/variance+variance.log()+math.log(2*math.pi))).sum(),
                    (diff.abs()<=1.96*variance.sqrt()).sum().double()]
        count+=diff.numel()
        for length in history_lengths:
            selected=batch['length']==length
            # Fixed-shape masking avoids CUDA boolean-index output-size sync.
            key=str(length)
            if key not in lengths:
                lengths[key]={'squared_error':torch.zeros((),device=device,dtype=torch.float64),'scalars':0}
            item=lengths[key]
            item['squared_error'].add_((diff.square()*selected[:,None,None]).sum())
            item['scalars']+=history_counts[length]*diff.shape[1]*diff.shape[2]
        sums.add_(torch.stack(reductions))
    if not count:raise ValueError('Independent validation windows are required')
    # One device-to-host fetch for the entire validation pass, including buckets.
    values=torch.cat((sums,torch.stack([item['squared_error'] for item in lengths.values()]) if lengths else sums[:0])).cpu().tolist()
    result={key:value/count for key,value in zip(keys,values[:4]) if stage=='gp' or key=='mse'}
    for item,value in zip(lengths.values(),values[4:]):item['squared_error']=value
    result['scalars']=count;result['history_lengths']=lengths
    return result


def train(data,output,stage='mean',variant='direct',device='cpu',settings=None,
          mean_checkpoint=None,resume=False,stop_after_batches=None):
    settings=settings or json.loads(resolve('configs/priors/default.json').read_text())
    output=Path(output);data=Path(data)
    training=settings['training'];seed=settings.get('seed',2027)
    history_only=False;residual=False
    if variant != 'direct':raise ValueError('Unknown LG target variant')
    config=dict(settings['model'])
    ablation=settings.get('ablation')
    fixed_length=settings.get('history_fixed_length')
    if fixed_length is not None and (not ablation or not 0<=fixed_length<=config['history_max']):
        raise ValueError('Fixed history requires a separate valid ablation identity')
    if not ablation and any(config.get(key,default)!=default for key,default in
                           [('gate_mode','learned'),('feature_mode','learned'),('context_mode','conditional')]):
        raise ValueError('Structural interventions require a separate ablation identity')
    reader=VariableHistoryWindows(data,'train',config['history_max'],config['horizon'],history_only,residual,seed=seed)
    val=VariableHistoryWindows(data,'val',config['history_max'],config['horizon'],history_only,residual,seed=seed)
    if not reader.window_count or not val.window_count:raise ValueError('Training and independent validation episodes required')
    config.update(latent_dim=reader.latent_dim,action_dim=reader.d,history_only=history_only)
    metadata={'environment':reader.index['environment'],'parent_sha256':reader.index['parent_sha256'],
              'split_sha256':reader.index['identity']['split_sha256'],'dataset_sha256':file_hash(data/'dataset.json'),
              'normalizer_sha256':reader.index['normalizer_sha256'],'variant':variant,
              'akf_parameters':reader.index['akf_parameters'] if residual else None,'schema':'lg_dkgp_v1'}
    mean_kind,gp_kind,schema='lg_dkgp_mean','lg_dkgp','lg_dkgp_v1'
    if ablation:raise ValueError('This release trains the full conditional prior only')
    if stage not in ('mean','gp'):raise ValueError('Unknown training stage')
    if stage=='gp' and not mean_checkpoint:raise ValueError('GP stage requires a completed neural mean stage')
    parent_sha=file_hash(mean_checkpoint) if mean_checkpoint else None
    identity=digest({'metadata':metadata,'model':config,'training':training,'stage':stage,
                     'mean_stage_sha256':parent_sha,'seed':seed})
    last=Path(str(output)+'.last')
    if output.exists() and not resume:raise FileExistsError('Use explicit compatible resume; do not overwrite prior')
    if resume and not last.exists():raise FileNotFoundError(last)
    torch.manual_seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    checkpoint=torch.load(last,map_location='cpu',weights_only=False) if resume else None
    if checkpoint and checkpoint['identity_sha256']!=identity:raise ValueError('LG resume identity differs')
    if checkpoint and math.isfinite(checkpoint['best']) and not output.exists() and not checkpoint.get('stage_complete'):
        raise ValueError('Resume requires the historical best checkpoint too; migrate both output and output.last')
    if checkpoint:model=construct_checkpoint(checkpoint,device)
    else:
        model=LGDKGP(**config).to(device)
        stats=torch.load(data/'normalizer.pt',weights_only=True)
        model.latent_mean.copy_(stats['mean']);model.latent_std.copy_(stats['std'])
        if stage=='gp':
            parent=torch.load(mean_checkpoint,map_location='cpu',weights_only=False)
            # Layout is storage/execution only; mean-stage tensors are identical.
            mean_config={k:v for k,v in parent['model_config'].items() if k!='gp_layout'}
            expected_mean_config={k:v for k,v in config.items() if k!='gp_layout'}
            if (parent.get('kind')!=mean_kind or not parent.get('fit_complete')
                    or parent['metadata']!=metadata or mean_config!=expected_mean_config):
                raise ValueError('Completed matching mean-stage checkpoint required')
            model.load_state_dict(parent['model'])
            model.eval();points=[]
            with torch.no_grad():
                for batch,indices in reader.inducing_batches(training['inducing_points']):
                    batch=to_device(batch,device);_,x,_=model.lifted_query(batch)
                    points.append(x[torch.arange(len(indices),device=device),indices.to(device)])
            model.initialize_gp(torch.cat(points))
    gp_parameters=list(model.gps.parameters())+list(model.likelihoods.parameters())
    gp_ids={id(p) for p in gp_parameters}
    neural=[p for p in model.parameters() if id(p) not in gp_ids]
    groups=[{'params':neural,'lr':training['mean_lr']}]
    if gp_parameters:groups.append({'params':gp_parameters,'lr':training['gp_lr']})
    optimizer=torch.optim.AdamW(groups)
    epoch,cursor,best,stale=0,0,float('inf'),0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch,cursor,best,stale=checkpoint['epoch'],checkpoint['cursor'],checkpoint['best'],checkpoint['stale']
        torch.set_rng_state(checkpoint['rng'])
        if device.startswith('cuda') and checkpoint.get('cuda_rng'):torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    output.parent.mkdir(parents=True,exist_ok=True)
    status_path=Path(str(output)+'.status.json')
    started=time.time();processed=0
    epoch_loss=torch.tensor(checkpoint.get('epoch_loss',0.) if checkpoint else 0.,device=device,dtype=torch.float64)
    epoch_scalars=checkpoint.get('epoch_scalars',0) if checkpoint else 0
    def save(path,complete=False,metrics=None):
        payload={'kind':mean_kind if stage=='mean' else gp_kind,'schema':schema,
            'stage':stage,'stage_complete':complete,'identity_sha256':identity,'metadata':metadata,
            'model_config':config,'training_config':training,'mean_stage_sha256':parent_sha,
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':epoch,'cursor':cursor,'best':best,'stale':stale,
            'epoch_loss':float(epoch_loss),'epoch_scalars':epoch_scalars,'num_data':reader.point_count,
            'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else None,
            'metrics':metrics,'resume_semantics':'same epoch batch cursor with deterministic private history generator'}
        if model.gps:payload['inducing']=model.gps[0].variational_strategy.inducing_points.detach()
        atomic_torch_save(payload,path)
    if checkpoint and checkpoint.get('stage_complete'):
        return checkpoint.get('metrics')
    while epoch<training['epochs'] and stale<training['patience']:
        model.train()
        for batch in reader.batches(training['batch_size'],epoch,skip_batches=cursor,fixed_length=fixed_length):
            batch=to_device(batch,device)
            optimizer.zero_grad()
            loss=(model.mean_prediction(batch)-batch['target']).square().mean() if stage=='mean' else model.elbo_loss(batch,reader.point_count)
            if not torch.isfinite(loss):raise ValueError('Nonfinite LG training objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),training['gradient_clip']);optimizer.step()
            cursor=batch['batch_index'];processed+=1
            epoch_loss.add_(loss.detach().double()*len(batch['history']));epoch_scalars+=len(batch['history'])
            if cursor%training['checkpoint_every_batches']==0:
                save(last)
                write_json(status_path,{'state':'running','stage':stage,'variant':variant,'epoch':epoch,
                    'batch_cursor':cursor,'window_count':reader.window_count,'best':best if math.isfinite(best) else None,
                    'recent_loss':float(loss.detach()),'updated_unix':time.time(),
                    'elapsed_seconds':time.time()-started})
                print(json.dumps({'stage':stage,'epoch':epoch,'batch_cursor':cursor,'loss':float(loss.detach())}),flush=True)
            if stop_after_batches is not None and processed>=stop_after_batches:
                save(last)
                return {'interrupted_after_batches':processed,'cursor':cursor}
        metrics=validate(model,val,stage,training['batch_size'],device,fixed_length=fixed_length)
        score=metrics['mse' if stage=='mean' else 'joint_nll_per_scalar']
        improved=score<best;best=min(best,score);stale=0 if improved else stale+1
        metrics.update(train_loss=float(epoch_loss)/max(epoch_scalars,1),epoch=epoch,elapsed_seconds=time.time()-started)
        if improved:save(output,metrics=metrics)
        with Path(str(output)+'.jsonl').open('a') as log:log.write(json.dumps(metrics)+'\n')
        print(json.dumps({'stage':stage,**metrics}),flush=True)
        epoch+=1;cursor=0;epoch_loss.zero_();epoch_scalars=0
        save(last)
    # The best model is complete only after this stage's stopping policy has run.
    best_state=torch.load(output,map_location='cpu',weights_only=False)
    best_state['stage_complete']=True;best_state['fit_complete']=True
    atomic_torch_save(best_state,output)
    write_json(status_path,{'state':'complete','stage':stage,'variant':variant,'epochs_completed':epoch,
        'best':best,'early_stopped':stale>=training['patience'],'output_sha256':file_hash(output)})
    return best_state['metrics']
