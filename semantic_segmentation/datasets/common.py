""" Auxiliary functions shared across different datasets.
"""
import torch
import os
from typing import List, Tuple

import yaml


def image_rgb_split(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  assert image.shape[0] == 3
  
  # get channels
  red_chan = image[0]
  green_chan = image[1]
  blue_chan = image[2]

  return red_chan, green_chan, blue_chan


def image_stats(image: torch.Tensor) -> Tuple[float, float, float, float, float, float]:

  # compute the mean and standard deviation of each channel
  r, g, b = image_rgb_split(image)

  r_mean = float(r.mean())
  r_std = float(r.std())
  g_mean = float(g.mean())
  g_std = float(g.std())
  b_mean = float(b.mean())
  b_std = float(b.std())

  # return the color statistics
  return r_mean, r_std, g_mean, g_std, b_mean, b_std


def image_rgb_stack(r_chan: torch.Tensor, g_chan: torch.Tensor, b_chan: torch.Tensor) -> torch.Tensor:
  rgb_image = torch.stack([r_chan, g_chan, b_chan], dim=0)  # [3 x H x W]

  return rgb_image


def is_image(filename: str) -> bool:
  """ Check whether or not a given file is an image based on its format.

  Args:
      filename (str): filename

  Returns:
      bool: whether or not a given file is an image
  """
  return any(filename.endswith(ext) for ext in ['.jpg', '.png'])


def get_images_in_dir(path_to_dir: str, sort: bool = True) -> List[str]:
  """ Get the paths to all images in directory.

  Args:
      path_to_dir (str): path to directory

  Returns:
      List[str]: contain the full path to all images in given directory
      sort (bool, optional): whether or not to sort the output. Defaults to True.
  """
  fpath_images = []

  files_in_dir = os.listdir(path_to_dir)
  for fname in files_in_dir:
    if is_image(fname):
      fpath_img = os.path.join(path_to_dir, fname)
      fpath_images.append(fpath_img)

  if sort:
    fpath_images.sort()

  return fpath_images

def get_img_fnames_in_dir(path_to_dir: str, sort: bool = True) -> List[str]:
  """ Get the filenames of all images in directory.

  Args:
      path_to_dir (str): path to directory

  Returns:
      List[str]: filenames of all images in given directory
      sort (bool, optional): whether or not to sort the output. Defaults to True.
  """
  fnames_images = []

  files_in_dir = os.listdir(path_to_dir)
  for fname in files_in_dir:
    if is_image(fname):
      fnames_images.append(fname)

  if sort:
    fnames_images.sort()

  return fnames_images


def check_split_file(path_to_split_file: str) -> bool:
  """ Check if the split file is valid.

  The train, val, and test splits of the UAV dataset
  are determined by a 'split.yaml' file. This function
  checks if there is any overlap between the sets.

  Args:
      path_to_split_file (str): path to split.yaml file

  Returns:
      bool: checks if there is any overlap between the train, val, and test split.
  """
  assert path_to_split_file.endswith("yaml"), "The split file should be a yaml file."

  with open(path_to_split_file, 'r') as istream:
    split_info = yaml.safe_load(istream)

    train_fnames = set(split_info['train'])
    test_fnames = set(split_info['test'])
    valid_fnames = set(split_info['valid'])

    # check if there are any identical files between two sets
    train_test_intersection = train_fnames.intersection(test_fnames)
    train_val_intersection = train_fnames.intersection(valid_fnames)
    val_test_intersection = valid_fnames.intersection(test_fnames)
    no_intersection = (len(train_test_intersection) == 0) and (len(train_val_intersection) == 0) and (
        len(val_test_intersection) == 0)

    return no_intersection
