
from inspect import stack
import stat
from mmengine import fileio
import io
import h5py
import numpy as np
import cv2
import json
import random
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.interpolate import interp1d, BSpline, make_interp_spline

def random_shifts_aug(x: torch.Tensor, 
                      pad: int = 30):
    x = x.float()
    c, h, w = x.size()
    assert h == w
    padding = tuple([pad] * 4)
    x = F.pad(x, padding, "replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)

    shift = torch.randint(0, 2 * pad + 1, size=(1, 1, 2), device=x.device, dtype=x.dtype)
    shift *= 2.0 / (h + 2 * pad)

    grid = base_grid + shift
    return F.grid_sample(x.unsqueeze(0), grid.unsqueeze(0), padding_mode="zeros", align_corners=False).squeeze(0)



class MapStyleReader(Dataset):
    def __init__(self, 
                 metas_path:str,
                 DOMAIN_NAME_TO_INFO: dict = {},
                 
                 proprio_normalization = "mean-std",
                 action_normalization = "min-max",
                #  standard_freq:int = 4,
                 max_freq:int=30,
                 repeat_padding: bool = False,
                 action_sequence_length: int = 0,
                 blind_mode: bool = False,
                 codebook_metas_path: str = '',
                 clip_ending: bool = False,
                #  dim_action:int= 7
                 ):
        
        #### read meta files, please put all json file in a one directory（metas_path）
        self.metas = {}
        self.datalist = []
        # reading setting
        self.max_freq = max_freq
        # self.standard_freq = standard_freq
        self.DOMAIN_NAME_TO_INFO = DOMAIN_NAME_TO_INFO
        self.action_normalization = action_normalization
        self.proprio_normalization = proprio_normalization
        self.action_sequence_length = action_sequence_length
        self.blind_mode = blind_mode
        self.repeat_padding = repeat_padding
        self.use_codebook_norm = True if codebook_metas_path else False
        self.codebook_meta = None
        self.clip_ending = clip_ending
        # self.dim_action = dim_action
        if self.clip_ending:
            print(f"=============Clip Ending for last seq================")
        elif self.repeat_padding:
            print(f"=============Repeat Padding for last seq================")

        self.num_views = 0
        for file in fileio.list_dir_or_file(metas_path, suffix='.json', recursive=True, list_dir=False):
            with io.BytesIO(fileio.get(fileio.join_path(metas_path, file))) as f:
                meta = json.load(f)
                print(f"================detect dataset {meta['dataset_name']} with traj {len(meta['datalist'])}==================")
                self.datalist.extend([(path, meta['dataset_name'], idx) 
                     for path, traj_len in meta['datalist'] 
                     for idx in range(0, traj_len if not self.clip_ending else traj_len - self.action_sequence_length + 1)
                     ])
                print(len(self.datalist))
                del meta['datalist']
                self.metas[meta['dataset_name']] = meta
                self.num_views = max(self.num_views, len(meta["observation_key"]))

        print(f"=================Max Num Views : {self.num_views}====================")
        
        # load codebook metas if provided
        if codebook_metas_path:
            for file in fileio.list_dir_or_file(codebook_metas_path, suffix='.json', recursive=True, list_dir=False):
                with io.BytesIO(fileio.get(fileio.join_path(codebook_metas_path, file))) as f:
                    meta = json.load(f)
                    print(f"================detect codebook metas path==================")
                    del meta['datalist']
                    self.codebook_meta = meta

        # image augmentations
        self.image_aug = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])

        
    
    def __len__(self):
        return len(self.datalist)
    
    def decode_and_augment_image(self, img):
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        pil_img = Image.fromarray(img)
        return random_shifts_aug(self.image_aug(pil_img))
    
    def action_interpolation(self, action, time_stamp, idx, hz, type):
        original_time = time_stamp[()]
        
        query_duration = 1 # 2s
        start_time = time_stamp[()][idx]
        end_time =  min(start_time + query_duration, time_stamp[()].max())
        chunk_interval = end_time - start_time
        query_time = np.linspace(original_time[idx], end_time, int(chunk_interval * hz))
        
        if type == 'b':
            actions_interp = np.zeros((len(query_time), action.shape[1]))
            for i in range(action.shape[1]):
                spline = make_interp_spline(original_time, action[:, i], k=5)
                actions_interp[:, i] = spline(query_time)
        elif type == 'linear':
            interp_func = interp1d(original_time, action, axis=0, kind='linear')
            actions_interp = interp_func(query_time)
        else:
            raise NotImplementedError
        return actions_interp
    
    def read_obs(self, datas, idx):
        x =  torch.stack([self.decode_and_augment_image(data[idx]).unsqueeze(0) for data in datas])
        V_exist = x.size(0)
        if V_exist < self.num_views:
            x = torch.cat([x, x.new_zeros(self.num_views - V_exist, *x.shape[1:])], dim=0) 
        mask = torch.zeros(self.num_views, x.size(1), dtype=torch.bool, device=x.device)
        mask[:V_exist] = True 

        return {'image_input': x, 'image_mask': mask}

    def read_proprio(self, data, idx, statics):
        idx_proprio = max(idx - 1, 0)
        extracted_data = data[idx_proprio]

        if self.proprio_normalization == 'min-max':
            extracted_data = (extracted_data - np.array(statics['proprio_stactics']['min'])) / \
                (np.array(statics['proprio_stactics']['max']) -
                np.array(statics['proprio_stactics']['min']) + 1e-6)
            extracted_data = extracted_data * 2 - 1
        elif self.proprio_normalization == 'mean-std':
            extracted_data = (extracted_data - np.array(statics['proprio_stactics']['mean'])) / \
                (np.array(statics['proprio_stactics']['std']) + 1e-6)
        else:
            raise NotImplementedError

        # 加噪声（可选）
        noise = np.random.randn(extracted_data.shape[0])
        extracted_data += noise * 0.3

        return {'proprio': extracted_data.astype(np.float32)}
            
    
    def read_actions(self, data, times, idx, statics):
        ## read actions
        T, D = data.shape
        # action_left = data[:, :int(D/2)]
        # action_right = data[:, int(D/2):]
        
        
        # extracted_data = self.action_interpolation(data, times, idx, 10, 'linear')
        # extracted_right = self.action_interpolation(action_right, right_time, times, idx, 30, 'linear')
        # extracted_data = np.concatenate([extracted_left, extracted_right], axis=-1)
        
        # end_idx = idx + self.frequency
        # extracted_data = data[idx:min(self.traj_len, end_idx)]
        
        
        end_idx = idx + 10
        extracted_data = data[idx:min(T, end_idx)]
        
        if self.action_normalization == 'min-max':
            extracted_data = (extracted_data - np.array(statics['action_statics']['min'])[None,]) /\
            (np.array(statics['action_statics']['max'])[None,] - np.array(statics['action_statics']['min'])[None,] + 1e-6)
            extracted_data = extracted_data * 2 - 1
        elif self.action_normalization == 'mean-std':
            extracted_data = (extracted_data - np.array(statics['action_statics']['mean'])[None,]) /\
            (np.array(statics['action_statics']['std'])[None,] + 1e-6)            
        else: raise NotImplementedError

        ## action padding
        mask = np.ones_like(extracted_data)

        ## horizon padding
        assert extracted_data.shape[0] <= self.frequency
        if extracted_data.shape[0] < self.frequency:
            # mask = np.concatenate([mask, np.zeros((self.max_freq - extracted_data.shape[0], self.dim_action))])
            # extracted_data = np.concatenate([extracted_data, np.zeros((self.max_freq - extracted_data.shape[0], self.dim_action))])
            mask = np.concatenate([mask, np.zeros((self.frequency - extracted_data.shape[0], extracted_data.shape[1]))])
            if not self.repeat_padding:
                extracted_data = np.concatenate([extracted_data, np.zeros((self.frequency - extracted_data.shape[0], extracted_data.shape[1]))])
            else:
                # extracted_data shape is [frequency, dim_action]
                extracted_data = np.concatenate([extracted_data, np.tile(extracted_data[-1:], (self.frequency - extracted_data.shape[0], 1))])
            
        return {'action': extracted_data.astype(np.float32), 'action_mask': mask.astype(np.bool_)}
    
    def read_next_obs(self, datas, idx):
        end_idx = idx + self.frequency
        query_idx = min(end_idx, self.traj_len-1)
        
        x =  torch.stack([self.decode_and_augment_image(data[query_idx]).unsqueeze(0) for data in datas])
        V_exist = x.size(0)
        if V_exist < self.num_views:
            x = torch.cat([x, x.new_zeros(self.num_views - V_exist, *x.shape[1:])], dim=0) 
        mask = torch.zeros(self.num_views, x.size(1), dtype=torch.bool, device=x.device)
        mask[:V_exist] = True 

        return {'next_image_input': x, 'next_image_mask': mask}
    
    def read_next_proprio(self, data, idx, statics):
        end_idx = idx + self.frequency
        query_idx = min(end_idx, self.traj_len-1)
        
        extracted_data = data[query_idx]
        if self.proprio_normalization == 'min-max':
            extracted_data = (extracted_data - np.array(statics['proprio_stactics']['min'])) /\
            (np.array(statics['proprio_stactics']['max']) - np.array(statics['proprio_stactics']['min']) + 1e-6)
            extracted_data = extracted_data * 2 - 1
        elif self.proprio_normalization == 'mean-std':
            extracted_data = (extracted_data - np.array(statics['proprio_stactics']['mean'])) /\
            (np.array(statics['proprio_stactics']['std']) + 1e-6)
        else:
            raise NotImplementedError
    
        return {'next_proprio': extracted_data.astype(np.float32)}  
          

    def read_reward_done(self, data, reward_key, done_key, idx):
        if not self.clip_ending:
            end_idx = idx + self.frequency
        else:
            end_idx = idx + 1
        
        if reward_key in data and done_key in data:
            reward = data[reward_key][idx:min(self.traj_len, end_idx)]
            done = data[done_key][idx]
            mask = np.ones_like(reward)
            if reward.shape[0] < self.frequency:
                mask = np.concatenate([mask, np.zeros((self.frequency - mask.shape[0],))])
                reward = np.concatenate([reward, np.ones((self.frequency - reward.shape[0],))])
        else:
            # reward auto generate
            if not self.clip_ending:
            
                # reward = np.asarray([1 if i > self.traj_len - int(self.frequency/3) else 0 for i in range(idx, end_idx)])
                # done = 1 if idx >= self.traj_len - self.frequency else 0
                # mask = np.asarray([1 if i <= self.traj_len - 1 else 0 for i in range(idx, end_idx)])
                
                reward = np.asarray([1 if i >= self.traj_len - 1 else 0 for i in range(idx, end_idx)])
                done = 1 if idx >= self.traj_len - self.frequency else 0
                mask = np.asarray([1 if i <= self.traj_len - 1 else 0 for i in range(idx, end_idx)])
            
            else:
                reward = np.asarray([1 if i >= (self.traj_len - self.action_sequence_length * 1.3) else 0 for i in range(idx, end_idx)])
                done = 1 if idx >= (self.traj_len - self.action_sequence_length * 1.3) else 0
                mask = np.asarray([1 if i <= self.traj_len - self.action_sequence_length else 0 for i in range(idx, end_idx)])
            
        return {'reward': reward.astype(np.float32) - 1, 'done': done, 'r_mask': mask.astype(np.float32)} # reward relabel, minus 1


    def add_traj(self, traj):
        path = traj['path']
        dataset_name = traj['dataset_name']
        traj_len = traj['length']
        self.datalist.extend([(path, dataset_name, idx) for idx in range(0, traj_len)])
    
    def clear_buffer(self):
        self.datalist = []
        
    def drop_old_buffer(self, keep_num):
        self.datalist = self.datalist[-keep_num:]

    def __getitem__(self, index):
        path, dataset_name, idx = self.datalist[index]
        meta = self.metas[dataset_name]
        f = io.BytesIO(fileio.get(fileio.join_path(meta["top_path"], path)))
        data = h5py.File(f,'r')
            
        # ins = data[meta["language_instruction_key"]][()] if len(data[meta["language_instruction_key"]].shape) == 0 else \
            # data[meta["language_instruction_key"]][idx]
        # ins = ins.decode()
        
        self.traj_len = data[meta["action_key"]].shape[0]
        self.frequency = meta['frequency'] if self.action_sequence_length == 0 else self.action_sequence_length
                
        # next done
        if idx + self.frequency >= self.traj_len:
            next_done = 1
        else:
            next_done = 0

        item =  {
            # 'hetero_info': torch.tensor(self.DOMAIN_NAME_TO_INFO[dataset_name]),
            # 'language_instruction': ins,
            **self.read_actions(data[meta["action_key"]], data['/time_stamp'], idx, meta),
            **self.read_obs([data[key] for key in meta['observation_key']], idx),
            **self.read_proprio(data[meta['proprio_key']], idx, meta),
            'next_done': next_done
        } 
            
        return item

def create_dataloader(
                 batch_size: int,
                 metas_path:str,
                 rank:int,
                 world_size:int,
                 DOMAIN_NAME_TO_INFO,
                 dim_action: int= 14,
                 max_freq:int = 30,
                #  standard_freq: int = 5,
                 action_normalization = "min-max",
                 repeat_padding: bool = False,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 action_sequence_length: int = 0,
                 blind_mode: bool = False,
                 codebook_metas_path: str = '',
                 clip_ending: bool = False,
                 **kwargs
                 ):
    dataset = MapStyleReader(
                 metas_path = metas_path,
                 DOMAIN_NAME_TO_INFO = DOMAIN_NAME_TO_INFO,
                 action_normalization = action_normalization,
                 max_freq = max_freq,
                 repeat_padding = repeat_padding,
                 action_sequence_length= action_sequence_length,
                 blind_mode= blind_mode,
                 codebook_metas_path = codebook_metas_path,
                 clip_ending = clip_ending,
                #  standard_freq = standard_freq,
                #  dim_action = dim_action,
                 )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)
    return DataLoader(dataset,
        sampler=sampler,
        batch_size=batch_size, 
        num_workers=4, 
        pin_memory=True
    )