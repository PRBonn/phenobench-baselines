#!/usr/bin/env python3
""" Compute mean and std of train images.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import oyaml as yaml
import skimage.io

sys.path.append(os.path.abspath("./agri_semantics"))


def parse_args() -> Dict[str, str]:
  """ Parse command line arguments.

  Returns:
      Dict[str, str]: see code below
  """

  parser = argparse.ArgumentParser()
  parser.add_argument("--image_dir", required=True, type=str, help="Path to directory to get the images from.")
  parser.add_argument("--split", required=True, type=str, help="Path to split (.yaml) file.")
  args = vars(parser.parse_args())

  assert args['split'].endswith('.yaml')

  return args

def get_image_fnames_from_split(fpath_split: str) -> List[str]:
  with open(fpath_split, 'r') as istream:
    split_info = yaml.safe_load(istream)

  image_fnames = split_info['train']

  return image_fnames

def compute_mean_std(path_to_dir: str, fpath_split: str) -> Tuple[List[float], List[float]]:
  """ Compute channelwise mean and std of all images in given directory.

  Args:
      path_to_dir (str): Path to Directory to get the images from
      fpath_split (str): Path to split file (.yaml)
  Returns:
      Tuple[List[float], List[float]]: channelwise means and stds.
  """
  filenames = get_image_fnames_from_split(fpath_split)

  # examine individually pixel values
  counter = 0.0
  pix_val = np.zeros(3, dtype=float)
  for filename in filenames:
    # open as rgb
    fpath_img = os.path.join(path_to_dir, filename)
    img = skimage.io.imread(fpath_img)

    # normalize to 1
    img = img.astype(float) / 255.0

    # count pixels and add them to counter
    h, w, _ = img.shape
    counter += h * w

    # sum to moving pix value counter in each channel
    pix_val += np.sum(img, (0, 1))

  # calculate means
  means = (pix_val / counter)

  # pass again and calculate variance
  pix_var = np.zeros(3, dtype=float)
  for filename in filenames:
    # open as rgb
    fpath_img = os.path.join(path_to_dir, filename)
    img = skimage.io.imread(fpath_img)

    # normalize to 1
    img = img.astype(float) / 255.0

    # sum to moving pix value counter in each channel
    pix_var += np.sum(np.square(img - means), (0, 1))

  # calculate the standard deviations
  stds = np.sqrt(pix_var / counter)

  means = [float(x) for x in means]
  stds = [float(x) for x in stds]

  return means, stds


if __name__ == '__main__':
  args = parse_args()
  means, stds = compute_mean_std(args['image_dir'], args['split'])

  print("*" * 80)
  print("means(rgb): ", means)
  print("stds(rgb): ", stds)
  print("*" * 80)

  # # Dump means and stds to config file in case its provided
  # if "cfg" in args.keys():
  #   # load config
  #   with open(args['cfg'], 'r') as istream:
  #     cfg = yaml.safe_load(istream)
  #     cfg['data']['img_means'] = means
  #     cfg['data']['img_stds'] = stds

  #   # dump config
  #   with open(args['cfg'], 'w') as ostream:
  #     yaml.dump(cfg, ostream)
