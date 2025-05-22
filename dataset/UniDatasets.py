
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
                #  max_freq:int=50,
                 dim_action:int= 7
                 ):
        
        #### read meta files, please put all json file in a one directory（metas_path）
        self.metas = {}
        self.datalist = []
        # reading setting
        # self.max_freq = max_freq
        # self.standard_freq = standard_freq
        self.DOMAIN_NAME_TO_INFO = DOMAIN_NAME_TO_INFO
        self.action_normalization = action_normalization
        self.proprio_normalization = proprio_normalization
        self.dim_action = dim_action

        self.num_views = 0
        for file in fileio.list_dir_or_file(metas_path, suffix='.json', recursive=True, list_dir=False):
            with io.BytesIO(fileio.get(fileio.join_path(metas_path, file))) as f:
                meta = json.load(f)
                print(f"================detect dataset {meta['dataset_name']} with traj {len(meta['datalist'])}==================")
                self.datalist.extend([(path, meta['dataset_name'], idx) 
                     for path, traj_len in meta['datalist'] 
                     for idx in range(0, traj_len - meta['frequency'])
                     ])
                print(len(self.datalist))
                del meta['datalist']
                self.metas[meta['dataset_name']] = meta
                self.num_views = max(self.num_views, len(meta["observation_key"]))

        print(f"=================Max Num Views : {self.num_views}====================")

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
    
    def read_obs(self, datas, idx):
        x =  torch.stack([self.decode_and_augment_image(data[idx]).unsqueeze(0) for data in datas])
        V_exist = x.size(0)
        if V_exist < self.num_views:
            x = torch.cat([x, x.new_zeros(self.num_views - V_exist, *x.shape[1:])], dim=0) 
        mask = torch.zeros(self.num_views, x.size(1), dtype=torch.bool, device=x.device)
        mask[:V_exist] = True 

        return {'image_input': x, 'image_mask': mask}

    def read_proprio(self, data, idx, statics):
        extracted_data = data[idx]
        if self.proprio_normalization == 'min-max':
            extracted_data = (extracted_data - np.array(statics['proprio_stactics']['min'])) /\
            (np.array(statics['proprio_stactics']['max']) - np.array(statics['proprio_stactics']['min']) + 1e-6)
            extracted_data = extracted_data * 2 - 1
        elif self.proprio_normalization == 'mean-std':
            extracted_data = (extracted_data - np.array(statics['proprio_stactics']['mean'])) /\
            (np.array(statics['proprio_stactics']['std']) + 1e-6)
        else:
            raise NotImplementedError
    
        return {'proprio': extracted_data.astype(np.float32)}
            
    
    def read_actions(self, data, idx, statics):
        traj_len = data.shape[0]
        ## read actions
        end_idx = idx + statics['frequency']
        extracted_data = data[idx:min(traj_len, end_idx)]
        
        if self.action_normalization == 'min-max':
            extracted_data = (extracted_data - np.array(statics['action_statics']['min'])[None,]) /\
            (np.array(statics['action_statics']['max'])[None,] - np.array(statics['action_statics']['min'])[None,] + 1e-6)
            extracted_data = extracted_data * 2 - 1
        else: raise NotImplementedError

        ## action padding
        mask = np.ones_like(extracted_data)
        ## action length padding
        assert extracted_data.shape[1] <= self.dim_action
        if extracted_data.shape[1] < self.dim_action:
            mask = np.concatenate([mask, np.zeros((mask.shape[0], self.dim_action - extracted_data.shape[1]))], axis=1)
            extracted_data = np.concatenate([extracted_data, np.zeros((mask.shape[0], self.dim_action - extracted_data.shape[1]))], axis=1)

        ## horizon padding
        # assert extracted_data.shape[0] <= self.max_freq
        # if extracted_data.shape[0] < self.max_freq:
            # mask = np.concatenate([mask, np.zeros((self.max_freq - extracted_data.shape[0], self.dim_action))])
            # extracted_data = np.concatenate([extracted_data, np.zeros((self.max_freq - extracted_data.shape[0], self.dim_action))])

        return {'action': extracted_data.astype(np.float32), 'action_mask': mask.astype(np.bool_)}


    def __getitem__(self, index):
        path, dataset_name, idx = self.datalist[index]
        meta = self.metas[dataset_name]
        f = io.BytesIO(fileio.get(fileio.join_path(meta["top_path"], path)))
        data = h5py.File(f,'r')
            
        ins = data[meta["language_instruction_key"]][()] if len(data[meta["language_instruction_key"]].shape) == 0 else \
            data[meta["language_instruction_key"]][idx]
        ins = ins.decode()
                
        item =  {
            'hetero_info': torch.tensor(self.DOMAIN_NAME_TO_INFO[dataset_name]),
            'language_instruction': ins,
            **self.read_actions(data[meta["action_key"]], idx, meta),
            **self.read_obs([data[key] for key in meta['observation_key']], idx),
            **self.read_proprio(data[meta['proprio_key']], idx, meta)
        }
        return item

def create_dataloader(
                 batch_size: int,
                 metas_path:str,
                 rank:int,
                 world_size:int,
                 DOMAIN_NAME_TO_INFO,
                 dim_action: int= 14,
                #  max_freq:int = 50,
                #  standard_freq: int = 5,
                 action_normalization = "min-max",
                 **kwargs
                 ):
    dataset = MapStyleReader(
                 metas_path = metas_path,
                 DOMAIN_NAME_TO_INFO = DOMAIN_NAME_TO_INFO,
                 action_normalization = action_normalization,
                #  max_freq = max_freq,
                #  standard_freq = standard_freq,
                 dim_action = dim_action,
                 )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset,
        sampler=sampler,
        batch_size=batch_size, 
        num_workers=4, 
        pin_memory=True
    )