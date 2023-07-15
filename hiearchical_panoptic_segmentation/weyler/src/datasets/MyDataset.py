"""
Dataset parser
"""
import pdb
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple

def image_rgb_split(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """ Extract each channel of a rgb image seperately.

  Args:
      image (torch.Tensor): RGB image of shape [3 x H x W]

  Returns:
      Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Each channel of shape [H x W]
  """
  assert image.dim() == 3

  red = image[0]
  green = image[1]
  blue = image[2]

  return red, green, blue

def channel_stats(channel: torch.Tensor) -> Tuple[float, float]:
  """ Compute mean and std of an image channel 

  Args:
      channel (torch.Tensor): image channel of shape [H x W]

  Returns:
      Tuple[float, float]: mean and std
  """
  assert channel.dim() == 2

  mean = float(channel.mean())
  std = float(channel.std())

  return mean, std

def image_rgb_stack(red: torch.Tensor, green: torch.Tensor, blue: torch.Tensor) -> torch.Tensor:
  assert red.dim() == 2
  assert green.dim() == 2
  assert blue.dim() == 2

  rgb_image = torch.stack([red, green, blue], dim=0)  # [3 x H x W]

  return rgb_image


class MyDataset(Dataset):

    def __init__(self, root_dir='./', type_="train", size=None, stems=False, transform=None):
        self.type = type_
        self.root_dir = root_dir
        self.stems = stems
        # get images 
        image_list = glob.glob(os.path.join(self.root_dir, '{}/images'.format(self.type), '*.png'))
        image_list.sort()
        self.image_list = image_list
        print("# image files: ", len(image_list))

        if self.type == 'train':
          # get global and part annotation
          global_instance_list = glob.glob(os.path.join(self.root_dir, '{}/plant_instances'.format(self.type), '*.png'))
          parts_instance_list = glob.glob(os.path.join(self.root_dir, '{}/leaf_instances'.format(self.type), '*.png'))
          semantics_list = glob.glob(os.path.join(self.root_dir, '{}/semantics'.format(self.type), '*.png'))

          global_instance_list.sort()
          parts_instance_list.sort()
          semantics_list.sort()

          self.global_instance_list = global_instance_list
          self.parts_instance_list = parts_instance_list
          self.semantics_list = semantics_list
          print("# global instance files: ", len(self.global_instance_list))
          print("# part instance files: ", len(self.parts_instance_list))
          print("# semantic files: ", len(self.semantics_list))
   
          # check if there are additional annotations for the stem location
          self.stem_list = []
          path_to_stem_anno = os.path.join(self.root_dir, 'annos', self.type, 'stems')
          if stems:
              stem_list = glob.glob(os.path.join(path_to_stem_anno, '*.npy'))
              stem_list.sort()
              self.stem_list = stem_list
              print("# stem anno files: ", len(self.stem_list))


        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

        print('MyDataset created - [{} file(s)]'.format(self.real_size))

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size-1)
        sample = {}

        # load image
        image = Image.open(self.image_list[index])
        width, height = image.size
        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        if self.type == 'train':
          # load labels and instances
          global_annos = np.array(Image.open(self.global_instance_list[index]))
          parts_annos = np.array(Image.open(self.parts_instance_list[index]))
          semantic_annos = np.array(Image.open(self.semantics_list[index]))
   
          global_labels = (semantic_annos == 1) | (semantic_annos == 3) # crops only
          global_instances = global_annos
          global_instances[global_labels != 1] = 0
          
          # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
          global_instance_ids = np.unique(global_instances)[1:] # no background
          global_instances_successive =  np.zeros_like(global_instances)
          for idx, id_ in enumerate(global_instance_ids):
              instance_mask = global_instances == id_
              global_instances_successive[instance_mask] = idx + 1
          global_instances = global_instances_successive
   
          assert np.max(global_instances) <= 255, 'Currently we do not suppot more than 255 instances in an image'
   
          parts_labels = parts_annos > 0
          parts_instances = parts_annos
          # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
          parts_instance_ids = np.unique(parts_instances)[1:] # no background
          parts_instances_successive =  np.zeros_like(parts_instances)
          for idx, id_ in enumerate(parts_instance_ids):
              instance_mask = parts_instances == id_
              parts_instances_successive[instance_mask] = idx + 1
          parts_instances = parts_instances_successive
   
          assert np.max(parts_instances) <= 255, 'Currently we do not suppot more than 255 instances in an image'

          global_labels = Image.fromarray(np.uint8(global_labels))
          # TODO there might be more than 255 instances
          global_instances = Image.fromarray(np.uint8(global_instances))
          
          parts_labels = Image.fromarray(np.uint8(parts_labels))
          # TODO there might be more than 255 instances
          parts_instances = Image.fromarray(np.uint8(parts_instances))
   
          sample['global_instances'] = global_instances
          sample['global_labels'] = global_labels
          
          sample['parts_instances'] = parts_instances
          sample['parts_labels'] = parts_labels
   
          # load stems if provided
          if self.stems:
              stem_anno = np.fromfile(self.stem_list[index], dtype=np.uint8)
              stem_anno = stem_anno.reshape(height, width)
              stem_anno = Image.fromarray(np.uint8(stem_anno))
              
              sample['stem_anno'] = stem_anno

        # transform
        if(self.transform is not None):
            sample = self.transform(sample)
            
        return sample
