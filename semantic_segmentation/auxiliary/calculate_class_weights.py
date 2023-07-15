#!/usr/bin/env python3
""" Compute class weights for semantic segmentation based on ground truth annotations
"""
import argparse
import math
import os
import pdb
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import oyaml as yaml
from PIL import Image


def parse_args() -> Dict[str, str]:
  """ Parse command line arguments.

  Returns:
      Dict[str, str]: see code below
  """

  parser = argparse.ArgumentParser()
  parser.add_argument("--anno_dir", required=True, type=str, help="Path to Directory to get the annotations from.")
  args = vars(parser.parse_args())

  return args

def get_class_frequencies(anno: np.ndarray) -> Dict[int, int]:
  """ Compute the (total) class frequencies for a single annotation.

  Args:
      anno (np.ndarray): ground truth annotation of single image.

  Returns:
      Dict[int, int]: maps each class id to its total count
  """
  class_ids = np.unique(anno)
  class_ids = class_ids[class_ids != 255]

  class_frequencies = {}
  for class_id in class_ids:
    class_mask = (anno == class_id)
    class_frequency = int(np.sum(class_mask))
    assert class_frequency <= np.prod(anno.shape)

    class_frequencies[int(class_id)] = class_frequency

  return class_frequencies

def convert_frequency_to_weight(relative_frequency: int) -> float:
  """ Convert a (relative) class frequency into a weights.

  We follow the definiton of http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf.

  Args:
      relative_frequency (int): relativ class frequency

  Returns:
      float: class weights
  """
  assert relative_frequency >= 0.0
  assert relative_frequency <= 1.0

  CONSTANT = 1.10

  weight = 1 / math.log(CONSTANT + relative_frequency)

  return weight

if __name__ == '__main__':
  args = parse_args()

  # fpaths to all annotations
  filenames = [fname for fname in os.listdir(args["anno_dir"]) if fname.endswith('png')]

  # compute total class frequencies of entire dataset
  total_class_frequencies = defaultdict(int)
  for fname in filenames:
    # compute the total class frequencies for a single annotation.
    fpath_anno = os.path.join(args['anno_dir'], fname)
    anno = np.array(Image.open(fpath_anno))
    
    mask_3 = anno == 3
    anno[mask_3] = 1 
    
    mask_4 = anno == 4
    anno[mask_4] = 2

    class_frequencies = get_class_frequencies(anno)

    # update
    for class_id, class_frequency in class_frequencies.items():
      total_class_frequencies[class_id] += class_frequency

  # convert total to relative frequencies
  total = np.sum([total_class_frequencies[class_id] for class_id in total_class_frequencies.keys()])

  relativ_class_frequencies = {}
  for class_id, class_frequency in total_class_frequencies.items():
      relativ_class_frequencies[class_id] = total_class_frequencies[class_id] / total

  # compute and print final class weights
  print("*" * 80)
  for class_id, class_relativ_frequency in relativ_class_frequencies.items():
    class_weight = convert_frequency_to_weight(class_relativ_frequency)
    assert class_weight >= 0.0
    
    print(f"class weights [{class_id}]: {class_weight:4.2f}")
  print("*" * 80)
