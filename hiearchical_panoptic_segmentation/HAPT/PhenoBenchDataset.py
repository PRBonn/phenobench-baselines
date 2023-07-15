import torch
import yaml
import torchvision
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import os
import os.path as path
from PIL import Image, ImageFile
import utils.utils as utils
import numpy as np
import torch.nn.functional as F
import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True

class StatDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        # from cfg i can access to all my shit
        # as data path, data size and so on 
        self.cfg = cfg
        self.len = -1
        self.setup()
        self.loader = [ self.train_dataloader(), self.val_dataloader() ]

    def prepare_data(self):
        # Augmentations are applied using self.transform 
        # no data to download, for now everything is local 
        pass

    def setup(self, stage=None):
        self.data_train = PhenoRobPlantsBase(os.path.join(self.cfg['data']['ft-path'], 'train'), overfit=self.cfg['train']['overfit'])
        self.data_val = PhenoRobPlantsBase(os.path.join(self.cfg['data']['ft-path'], 'val'), overfit=self.cfg['train']['overfit'])
        return

    def train_dataloader(self):
        loader = DataLoader(self.data_train, 
                            batch_size = self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers = self.cfg['train']['workers'],
                            pin_memory=True,
                            shuffle=True)
        self.len = self.data_train.__len__()
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data_val, 
                            batch_size = self.cfg['train']['batch_size'] // self.cfg['train']['n_gpus'],
                            num_workers = self.cfg['train']['workers'],
                            pin_memory=True,
                            shuffle=False)
        self.len = self.data_val.__len__()
        return loader
            
    def test_dataloader(self):
        pass

#################################################
################## Data loader ##################
#################################################

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os


class PhenoRobPlantsBase(Dataset):
    def __init__(self, data_path, overfit=False):
        super().__init__()

        self.data_path = data_path
        self.overfit = overfit

        self.image_list = [
            x for x in os.listdir(os.path.join(self.data_path, "images")) if ".png" in x
        ]
        self.image_list.sort()

        self.len = len(self.image_list)

        # preload the data to memory
        self.field_list = os.listdir(self.data_path)

    def get_centers(self, mask):
        if mask.sum() == 0:
            return torch.zeros(mask.shape, device=mask.device, dtype=torch.float)
        
        masks = F.one_hot(mask.long())
        masks = masks.permute(2,0,1)[1:,:,:]
        num, H, W = masks.shape
        center_mask = torch.zeros( (H, W) , device=masks.device, dtype=torch.float)

        for submask in masks:
            if submask.sum() == 0:
                continue
            x, y = torch.where(submask != 0)
            xy = torch.cat([x.unsqueeze(0),y.unsqueeze(0)], dim=0)
            mu, _ = torch.median(xy,dim=1, keepdim=True)
            center_idx = torch.argmin(torch.sum(torch.abs(xy - mu), dim=0))
            center = xy[:,center_idx]
            center_mask[center[0], center[1]] = 1.
    
        return center_mask

    def get_offsets(self, mask, centers):
        if mask.sum() == 0:
            return torch.zeros((0, 4), device=mask.device, dtype=torch.float)

        masks = F.one_hot(mask.long())
        masks = masks.permute(2,0,1)[1:,:,:]
        num, H, W = masks.shape
        
        total_mask = torch.zeros((H,W,2), device = masks.device, dtype=torch.float)

        for submask in masks:
            coords = torch.ones((H,W,2))
            tmp = torch.ones((H,W,2))
            tmp[:,:,1] = torch.cumsum(coords[:,:,0],0) - 1
            tmp[:,:,0] = torch.cumsum(coords[:,:,1],1) - 1

            current_center = torch.where(submask * centers)
            offset_mask = (tmp - torch.tensor([current_center[1], current_center[0]])) * submask.unsqueeze(2)
            total_mask += offset_mask
        return total_mask


    @staticmethod
    def load_one_data(data_path, image_list, field_list, idx):
        data_frame = {}
        for field in field_list:
            # data_frame[field] = []
            image = image_list[idx]
            image = cv2.imread(
                os.path.join(os.path.join(data_path, field), image),
                cv2.IMREAD_UNCHANGED,
            )
            if len(image.shape) > 2:
                sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                sample = torch.tensor(sample).permute(2, 0, 1)
            else:
                sample = torch.tensor(image.astype("int16"))

            data_frame[field] = sample
        return data_frame


    def get_sample(self, index):
        sample = {}

        sample = self.load_one_data(self.data_path, self.image_list, self.field_list, index)
        partial_crops = sample["semantics"] == 3
        partial_weeds = sample["semantics"] == 4

        # 1 where there's stuff to be ignored by instance segmentation, 0 elsewhere
        sample["ignore_mask"] = torch.logical_or(partial_crops, partial_weeds).bool()

        # remove partial plants
        sample["semantics"][partial_crops] = 1
        sample["semantics"][partial_weeds] = 2

        # remove instances that aren't crops or weeds
        sample["plant_instances"][sample["semantics"] == 0] = 0
        sample["leaf_instances"][sample["semantics"] == 0] = 0

        # make ids successive
        sample["plant_instances"] = torch.unique(
            sample["plant_instances"] + sample["semantics"] * 1e6, return_inverse=True
        )[1]
        sample["leaf_instances"] = torch.unique(
            sample["leaf_instances"] + sample["plant_instances"] * 1e6, return_inverse=True
        )[1]
        sample['leaf_instances'][sample['semantics'] == 2] = 0

        return sample

    def __getitem__(self, index):
        sample = self.get_sample(index)
        plants = sample['plant_instances'] * (~sample['ignore_mask'])
        leaves = sample['leaf_instances'] * (~sample['ignore_mask'])
        sample['plant_instances'] = torch.unique(plants, return_inverse=True)[1]
        sample['leaf_instances'] = torch.unique(leaves, return_inverse=True)[1]

        p_centers = self.get_centers(sample['plant_instances'])
        l_centers = self.get_centers(sample['leaf_instances'])
        p_offsets = self.get_offsets(sample['plant_instances'], p_centers).permute(2,0,1)
        l_offset = self.get_offsets(sample['leaf_instances'], l_centers).permute(2,0,1)
        
        p_centers = torchvision.transforms.GaussianBlur(11, 5.0)(p_centers.unsqueeze(0).float())
        p_centers = ((torch.max(p_centers, dim=0)[0] / torch.max(p_centers))) * plants.bool().long()
        
        exp_centers = torch.zeros(l_centers.shape[0] + 10, l_centers.shape[1] + 10)
        exp_centers[5:-5, 5:-5] = l_centers
        exp_centers = torchvision.transforms.GaussianBlur(7, 3.0)(exp_centers.unsqueeze(0).float())
        l_centers = exp_centers[0, 5:-5, 5:-5]
        l_centers = torchvision.transforms.GaussianBlur(7, 3.0)(exp_centers.unsqueeze(0).float()).squeeze()
        l_centers = l_centers[5:-5, 5:-5].unsqueeze(0)
        l_centers = ((torch.max(l_centers, dim=0)[0] / torch.max(l_centers))) * leaves.bool().long()

        sample['global_centers'] = p_centers
        sample['parts_centers'] = l_centers
        sample['global_offsets'] = p_offsets
        sample['parts_offsets'] = l_offset
        sample["image_name"] = self.image_list[index]
        return sample

    def __len__(self):
        return self.len

