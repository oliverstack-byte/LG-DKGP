"""Episode-local multiview WM windows directly from verified split tar views."""
import bisect
from collections import OrderedDict
import io
import json
import os
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from lgdkgp.data.windows.archive_source import PartsReader,MemberReader,archive_parts

VIEWS={'libero_scene2':{'pixels':'pixels','wrist_pixels':'wrist_pixels'},
       'libero_caddy':{'pixels':'pixels','wrist_pixels':'wrist_pixels'},
       'robotwin_a2b':{'pixels':'vision/cam_head/colors','left_wrist_pixels':'vision/cam_left_wrist/colors','right_wrist_pixels':'vision/cam_right_wrist/colors'}}


class NativeWorldWindows(Dataset):
    def __init__(self,manifest,action_mean,action_std,image_size=224,context=3,frameskip=5,image_decoder='pil'):
        if frameskip!=5 or context<1:raise ValueError('Native WM requires frameskip5 and positive context')
        self.manifest_path=Path(manifest);self.manifest=json.loads(self.manifest_path.read_text())
        self.environment=self.manifest['environment'];self.entries=self.manifest['world_model_episodes']
        if image_decoder not in ('pil','torchcodec_cpu'):raise ValueError('Unsupported image decoder')
        self.image_decoder=image_decoder
        self._decode_jpeg=None
        if self.manifest['frameskip']!=5 or self.manifest['action_block']!=5:raise ValueError('Native manifest frameskip mismatch')
        self.view_keys=tuple(VIEWS[self.environment]);self.context=context;self.span=5*context
        self.mean=torch.as_tensor(action_mean,dtype=torch.float32);self.std=torch.as_tensor(action_std,dtype=torch.float32)
        self.d=len(self.mean)
        if self.std.shape!=self.mean.shape or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or not (self.std>0).all():raise ValueError('Invalid action normalization')
        if any(x['action_dim']!=self.d for x in self.entries):raise ValueError('Source action dimension mismatch')
        self._owner=None;self._source=None;self._files=OrderedDict()
        self.ends=[];total=0
        for entry in self.entries:
            states=entry['length']
            if self.environment=='robotwin_a2b':
                handle=self._handle(entry)
                counts=[len(handle[column]) for column in VIEWS[self.environment].values()]
                if len(set(counts))!=1 or counts[0] not in (entry['length'],entry['length']+1):
                    raise ValueError('RoboTwin cameras have missing or inconsistent raw frames')
                states=counts[0]
                # visual_raw_indices may include a trailing action-boundary sentinel;
                # it is NOT a camera frame. Never invent the last observation.
            total+=max(0,min(entry['length']-self.span+1,states-self.span))
            self.ends.append(total)
        self.transform=transforms.Compose([transforms.ToImage(),transforms.ToDtype(torch.float32,scale=True),
             transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]),transforms.Resize((image_size,image_size))])
        self.close()

    def __len__(self):return self.ends[-1] if self.ends else 0

    def close(self):
        for handle in self._files.values():handle.close()
        self._files.clear()
        if self._source:self._source.close()
        self._source=None;self._owner=None

    def _handle(self,entry):
        if self._owner!=os.getpid():
            self.close();self._source=PartsReader(archive_parts(self.manifest['data_root'],self.environment,self.manifest['split']));self._owner=os.getpid()
        key=(entry['member_offset'],entry['member_bytes'])
        if key not in self._files:
            if len(self._files)>=4:self._files.popitem(last=False)[1].close()
            self._files[key]=h5py.File(MemberReader(self._source,*key),'r')
        self._files.move_to_end(key);return self._files[key]

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        episode=bisect.bisect_right(self.ends,index);entry=self.entries[episode]
        start=index-(self.ends[episode-1] if episode else 0);offset=entry['raw_offset']
        handle=self._handle(entry);times=np.arange(start,start+self.span+1,5,dtype=np.int64)
        if self.environment=='robotwin_a2b':
            indices=np.asarray(handle['lewm/visual_raw_indices'][times])
            if not np.array_equal(indices,times):raise ValueError('RoboTwin images are not raw-step aligned')
            action=np.asarray(handle['lewm/action_blocks'][start:start+self.span])
            if action.shape!=(self.span,1,self.d):raise ValueError('RoboTwin source action block is not raw stride1')
            action=action[:,0]
        else:action=np.asarray(handle['action'][offset+start:offset+start+self.span])
        if action.shape!=(self.span,self.d) or not np.isfinite(action).all():raise ValueError('Incomplete/nonfinite actual action sequence')
        normalized=(torch.from_numpy(action.copy()).float()-self.mean)/self.std
        result={'action':normalized.reshape(self.context,5*self.d),'raw_state_indices':torch.from_numpy(times),
                'source_episode_index':episode,'raw_start':start}
        for view,column in VIEWS[self.environment].items():
            frames=[]
            for t in times:
                raw=np.asarray(handle[column][offset+int(t)])
                if raw.ndim==1 or (raw.ndim==0 and raw.dtype.kind in 'SVO'):
                    if self.image_decoder=='torchcodec_cpu':
                        if self._decode_jpeg is None:
                            from torchcodec.decoders import decode_jpeg
                            self._decode_jpeg=decode_jpeg
                        raw=self._decode_jpeg(raw.tobytes(),mode='RGB',device='cpu')
                    else:
                        with Image.open(io.BytesIO(raw.tobytes())) as image:raw=np.asarray(image.convert('RGB')).copy()
                if raw.ndim!=3 or (raw.shape[0]!=3 if torch.is_tensor(raw) else raw.shape[-1]!=3):raise ValueError('Expected an RGB observation')
                frames.append(self.transform(raw))
            result[view]=torch.stack(frames)
        return result
