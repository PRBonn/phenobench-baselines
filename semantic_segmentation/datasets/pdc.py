import os
import pdb
import random
from typing import Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import datasets.common as common
from datasets.augmentations_color import get_color_augmentations
from datasets.augmentations_geometry import (GeometricDataAugmentation,
                                             get_geometric_augmentations)
from datasets.image_normalizer import ImageNormalizer, get_image_normalizer


class PDC(Dataset):
  """ Represents the PDC dataset.

  The directory structure is as following:
  ├── test
  │   ├── images
  │   ├── leaf_instances
  │   ├── leaf_visibility
  │   ├── plant_instances
  │   ├── plant_visibility
  │   └── semantics
  ├── train
  │   ├── images
  │   ├── leaf_instances
  │   ├── leaf_visibility
  │   ├── plant_instances
  │   ├── plant_visibility
  │   └── semantics
  └── val
      ├── images
      ├── leaf_instances
      ├── leaf_visibility
      ├── plant_instances
      ├── plant_visibility
      ├── semantics
      └── visualize
  """

  def __init__(self, path_to_dataset: str, 
                     mode: str, 
                     img_normalizer: ImageNormalizer,
                     augmentations_geometric: List[GeometricDataAugmentation], 
                     augmentations_color: List[Callable]):
    """ Get the path to all images and its corresponding annotations.

    Args:
        path_to_dataset (str): Path to dir that contains the images and annotations
        mode(str): train, val, or test
        img_normalizer (ImageNormalizer): Specifies how to normalize the input images
        augmentations_geometric (List[GeometricDataAugmentation]): Geometric data augmentations applied to the image and its annotations
        augmentations_color (List[Callable]): Color data augmentations applied to the image
    """

    assert os.path.exists(path_to_dataset), f"The path to the dataset does not exist: {path_to_dataset}."

    super().__init__()

    assert mode in ['train', 'val', 'test']
    self.mode = mode

    self.img_normalizer = img_normalizer
    self.augmentations_geometric = augmentations_geometric
    self.augmentations_color = augmentations_color

    # ------------- Prepare Training -------------
    self.path_to_train_images = os.path.join(path_to_dataset, "train", "images")
    self.path_to_train_annos = os.path.join(path_to_dataset, "train", "semantics")
    self.filenames_train = common.get_img_fnames_in_dir(self.path_to_train_images)

    # ------------- Prepare Training -------------
    self.path_to_val_images = os.path.join(path_to_dataset, "val", "images")
    self.path_to_val_annos = os.path.join(path_to_dataset, "val", "semantics")
    self.filenames_val = common.get_img_fnames_in_dir(self.path_to_val_images)

    # ------------- Prepare Testing -------------
    self.path_to_test_images = os.path.join(path_to_dataset, "test", "images")
    self.path_to_test_annos = os.path.join(path_to_dataset, "test", "semantics")
    self.filenames_test = common.get_img_fnames_in_dir(self.path_to_test_images)

    # specify image transformations
    self.img_to_tensor = transforms.ToTensor()

  def get_train_item(self, idx: int) -> Dict:
    path_to_current_img = os.path.join(self.path_to_train_images, self.filenames_train[idx])
    img_pil = Image.open(path_to_current_img)
    img = self.img_to_tensor(img_pil)  # [C x H x W] with values in [0, 1]
    
    if random.random() > 0.25:
      for augmentor_color_fn in self.augmentations_color:
        img = augmentor_color_fn(img)
    
    path_to_current_anno = os.path.join(self.path_to_train_annos, self.filenames_train[idx])
    anno = np.array(Image.open(path_to_current_anno))  # dtype: int32
    if len(anno.shape) > 2:
        anno = anno[:, :, 0]
    anno = anno.astype(np.int64)
    anno = torch.Tensor(anno).type(torch.int64)  # [H x W]
    anno = anno.unsqueeze(0)  # [1 x H x W]

    for augmentor_geometric in self.augmentations_geometric:
      img, anno = augmentor_geometric(img, anno)
    anno = anno.squeeze(0)  # [H x W]

    mask_3 = anno == 3
    anno[mask_3] = 1 

    mask_4 = anno == 4
    anno[mask_4] = 2

    img_before_norm = img.clone()
    img = self.img_normalizer.normalize(img)
    
    return {'input_image_before_norm': img_before_norm, 
            'input_image': img, 
            'anno': anno, 
            'fname': self.filenames_train[idx]}

  def get_val_item(self, idx: int) -> Dict:
    path_to_current_img = os.path.join(self.path_to_val_images, self.filenames_val[idx])
    img_pil = Image.open(path_to_current_img)
    img = self.img_to_tensor(img_pil)  # [C x H x W] with values in [0, 1]
    
    path_to_current_anno = os.path.join(self.path_to_val_annos, self.filenames_val[idx])
    anno = np.array(Image.open(path_to_current_anno))  # dtype: int32
    if len(anno.shape) > 2:
        anno = anno[:, :, 0]
    anno = anno.astype(np.int64)
    anno = torch.Tensor(anno).type(torch.int64)  # [H x W]

    img_before_norm = img.clone()
    img = self.img_normalizer.normalize(img)

    mask_3 = anno == 3
    anno[mask_3] = 1 

    mask_4 = anno == 4
    anno[mask_4] = 2
    
    return {'input_image_before_norm': img_before_norm, 
            'input_image': img, 
            'anno': anno, 
            'fname': self.filenames_val[idx]}

  def get_test_item(self, idx: int) -> Dict:
    path_to_current_img = os.path.join(self.path_to_test_images, self.filenames_test[idx])
    img_pil = Image.open(path_to_current_img)
    img = self.img_to_tensor(img_pil)  # [C x H x W] with values in [0, 1]

    path_to_current_anno = os.path.join(self.path_to_test_annos, self.filenames_test[idx])
    anno = np.array(Image.open(path_to_current_anno))  # dtype: int32
    if len(anno.shape) > 2:
        anno = anno[:, :, 0]
    anno = anno.astype(np.int64)
    anno = torch.Tensor(anno).type(torch.int64)  # [H x W]

    img_before_norm = img.clone()
    img = self.img_normalizer.normalize(img)

    mask_3 = anno == 3
    anno[mask_3] = 1 

    mask_4 = anno == 4
    anno[mask_4] = 2
    
    return {'input_image_before_norm': img_before_norm, 
            'input_image': img, 
            'anno': anno, 
            'fname': self.filenames_test[idx]}

  def __getitem__(self, idx: int):
    if self.mode == 'train':
      items = self.get_train_item(idx)
      return items

    if self.mode == 'val':
      items = self.get_val_item(idx)
      return items

    if self.mode == 'test':
      items = self.get_test_item(idx)
      return items
    
  def __len__(self):
    if self.mode == 'train':
      return len(self.filenames_train)

    if self.mode == 'val':
      return len(self.filenames_val)

    if self.mode == 'test':
      return len(self.filenames_test)


class PDCModule(pl.LightningDataModule):
  """ Encapsulates all the steps needed to process data from the PDC Challenge.
  """

  def __init__(self, cfg: Dict):
    super().__init__()

    self.cfg = cfg

  def setup(self, stage: Optional[str] = None):
    """ Data operations we perform on every GPU.

    Here we define the how to split the dataset.

    Args:
        stage (Optional[str], optional): _description_. Defaults to None.
    """
    path_to_dataset = self.cfg['data']['path_to_dataset']
    image_normalizer = get_image_normalizer(self.cfg)

    if (stage == 'fit') or (stage == 'validate') or (stage is None):
      # ----------- TRAIN -----------
      train_augmentations_geometric = get_geometric_augmentations(self.cfg, 'train')
      train_augmentations_color = get_color_augmentations(self.cfg, 'train')
      self.train_ds = PDC(
          path_to_dataset,
          mode='train',
          img_normalizer=image_normalizer,
          augmentations_geometric=train_augmentations_geometric,
          augmentations_color=train_augmentations_color)
  
      # ----------- VAL -----------
      val_augmentations_geometric = get_geometric_augmentations(self.cfg, 'val')
      self.val_ds = PDC(
          path_to_dataset,
          mode='val',
          img_normalizer=image_normalizer,
          augmentations_geometric=val_augmentations_geometric,
          augmentations_color=[])
  
    if stage == 'test' or stage is None:
      # ----------- TEST -----------
      test_augmentations_geometric = get_geometric_augmentations(self.cfg, 'test')
      self.test_ds = PDC(
          path_to_dataset,
          mode='test',
          img_normalizer=image_normalizer,
          augmentations_geometric=test_augmentations_geometric,
          augmentations_color=[])
  

  def train_dataloader(self) -> DataLoader:
    # Return DataLoader for Training Data here
    shuffle: bool = self.cfg['train']['shuffle']
    batch_size: int = self.cfg['train']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, drop_last=True, pin_memory=True)

    return loader

  def val_dataloader(self) -> DataLoader:
    batch_size: int = self.cfg['val']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self.val_ds, batch_size=batch_size, num_workers=n_workers, shuffle=False, drop_last=True, pin_memory=True)

    return loader

  def test_dataloader(self) -> DataLoader:
    batch_size: int = self.cfg['test']['batch_size']
    n_workers: int = self.cfg['data']['num_workers']

    loader = DataLoader(self.test_ds, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True)

    return loader
