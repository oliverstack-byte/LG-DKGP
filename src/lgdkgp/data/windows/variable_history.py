"""Episode-local, replayable LG windows; frozen feature files are never rewritten."""
import json
from pathlib import Path
import torch

from lgdkgp.common.io import file_hash,write_json,digest

VAL_LENGTHS=(0,1,5,16,25,32)


def prepare_variable_dataset(shared,output):
    shared,output=Path(shared).resolve(),Path(output).resolve()
    original=json.loads((shared/'dataset.json').read_text())
    identity={'schema':'lg_windows_v1','shared_manifest_sha256':file_hash(shared/'dataset.json'),
              'parent_sha256':original['parent_sha256'],'split_sha256':original['identity']['allocation_sha256']}
    target=output/'dataset.json'
    if target.exists():
        record=json.loads(target.read_text())
        if record['identity']!=identity or file_hash(output/'normalizer.pt')!=record['normalizer_sha256']:
            raise ValueError('LG dataset identity or normalizer changed')
        return record
    sums=squares=None;count=0
    for entry in original['episodes']:
        if entry['split']!='train':continue
        feature=Path(original['feature_directory'])/entry['file']
        if file_hash(feature)!=entry['sha256']:raise ValueError('Frozen shard changed')
        z=torch.load(feature,map_location='cpu',weights_only=True)['latent'].double()
        sums=z.sum(0) if sums is None else sums+z.sum(0)
        squares=z.square().sum(0) if squares is None else squares+z.square().sum(0)
        count+=len(z)
    if count<2:raise ValueError('Train-only latent statistics require source episodes')
    mean=sums/count;std=((squares-count*mean.square())/(count-1)).clamp_min(1e-12).sqrt()
    output.mkdir(parents=True,exist_ok=True)
    temporary=output/'normalizer.tmp';torch.save({'mean':mean.float(),'std':std.float(),'count':count},temporary)
    temporary.replace(output/'normalizer.pt')
    record={**original,'identity':identity,'shared_directory':str(shared),
            'normalizer_sha256':file_hash(output/'normalizer.pt'),
            'history_max':32,'horizon':25,'validation_lengths':list(VAL_LENGTHS),
            'validation_schedule':'lengths[(source_episode_index+start_step)%6], clamped to actual history',
            'normalizer_source':'all finite latent frames in prior train episodes only',
            'window_rule':'0 <= t < usable_length-25; goal at t+25',
            'episodes':[{**entry,'windows':max(0,entry['length']-25)} for entry in original['episodes']]}
    write_json(target,record)
    return record


class VariableHistoryWindows:
    def __init__(self,directory,split,history_max=32,horizon=25,history_only=False,residual=False,seed=2027):
        self.directory=Path(directory)
        self.seed=seed
        self.index=json.loads((self.directory/'dataset.json').read_text())
        if self.index['identity'].get('schema')!='lg_windows_v1':raise ValueError('Expected LG data schema')
        if file_hash(self.directory/'normalizer.pt')!=self.index['normalizer_sha256']:
            raise ValueError('Changed latent normalizer')
        if history_max<1 or not 1<=horizon<=self.index['horizon']:raise ValueError('Unsupported LG history/horizon')
        self.history_max,self.horizon=history_max,horizon
        self.history_only,self.residual=history_only,residual
        if history_only or residual:raise ValueError('Full conditional prior only')
        self.episodes=[entry for entry in self.index['episodes'] if entry['split']==split]
        self.d=self.index['action_dim'];self.latent_dim=self.index['latent_dim']
        self.projection=None

    @property
    def window_count(self):return sum(entry['windows'] for entry in self.episodes)

    @property
    def point_count(self):return self.window_count*self.horizon

    def _load(self,entry):
        feature=Path(self.index['feature_directory'])
        if not feature.is_absolute():feature=self.directory/feature
        feature=feature/entry['file']
        if file_hash(feature)!=entry['sha256']:raise ValueError('Frozen shard changed')
        return torch.load(feature,map_location='cpu',weights_only=True,mmap=True)

    def _assemble(self,payload,entry,times,lengths,batch_index):
        actions=payload['action']
        lag=torch.arange(-self.history_max,0)
        mask=lag[None]>=-lengths[:,None]
        rows=(times[:,None]+lag[None]).clamp_min(0)
        history=torch.where(mask[...,None],actions[rows],torch.zeros(len(times),self.history_max,self.d))
        target=actions[times[:,None]+torch.arange(self.horizon)].clone()
        if self.residual:
            state_path=Path(self.index['shared_directory'])/entry['filter_file']
            if file_hash(state_path)!=entry['filter_sha256']:raise ValueError('Causal AKF state changed')
            states=torch.load(state_path,map_location='cpu',weights_only=True,mmap=True)['states']
            target-=(states[times]@self.projection.T).reshape(len(times),self.horizon,self.d).to(target)
        result={'history':history,'mask':mask,'target':target,'start':times,
                'episode':torch.full_like(times,entry['episode_id']),
                'length':lengths,'batch_index':batch_index}
        if not self.history_only:
            goal_times=(torch.full_like(times,len(payload['latent'])-1)
                        if self.index.get('goal_policy')=='terminal' else times+25)
            result.update(z_current=payload['latent'][times],z_goal=payload['latent'][goal_times])
        return result

    def inducing_batches(self,count=128):
        """Uniform random training point initialization, without scanning every window."""
        count=min(count,self.point_count)
        generator=torch.Generator().manual_seed(self.seed)
        selected=torch.empty(0,dtype=torch.long)
        while len(selected)<count:
            selected=torch.cat((selected,torch.randint(self.point_count,(2*count,),generator=generator))).unique()
        selected=selected[torch.randperm(len(selected),generator=generator)[:count]].sort().values
        base=0
        for entry in self.episodes:
            width=entry['windows']*self.horizon
            points=selected[(selected>=base)&(selected<base+width)]-base
            base+=width
            if not len(points):continue
            times=points//self.horizon
            available=times.clamp_max(self.history_max)
            lengths=torch.tensor([int(torch.randint(int(cap)+1,(1,),generator=generator)) for cap in available])
            yield self._assemble(self._load(entry),entry,times,lengths,0),points%self.horizon

    def batches(self,batch_size=128,epoch=0,shuffle=True,skip_batches=0,fixed_length=None):
        generator=torch.Generator().manual_seed(self.seed+epoch)
        order=torch.randperm(len(self.episodes),generator=generator).tolist() if shuffle else range(len(self.episodes))
        batch_index=0
        for ordinal in order:
            entry=self.episodes[ordinal]
            payload=self._load(entry)
            actions=payload['action'];starts=torch.arange(entry['windows'])
            if shuffle:starts=starts[torch.randperm(len(starts),generator=generator)]
            for offset in range(0,len(starts),batch_size):
                times=starts[offset:offset+batch_size]
                available=times.clamp_max(self.history_max)
                if fixed_length is not None:
                    lengths=available.clamp_max(fixed_length)
                elif shuffle:
                    lengths=torch.empty_like(available)
                    for cap in available.unique():
                        selected=available==cap
                        lengths[selected]=torch.randint(int(cap)+1,(int(selected.sum()),),generator=generator)
                else:
                    requested=torch.tensor(VAL_LENGTHS)[(times+ordinal)%len(VAL_LENGTHS)]
                    lengths=torch.minimum(available,requested)
                batch_index+=1
                if batch_index<=skip_batches:continue
                yield self._assemble(payload,entry,times,lengths,batch_index)
