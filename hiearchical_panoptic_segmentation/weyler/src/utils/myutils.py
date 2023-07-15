"""
Peform automated postprocessing step to cluster individual crop leaf and plant instances.
"""
import collections
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch
import torch.nn as nn
from matplotlib import patches
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def bounding_box_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
  """ Compute bounding box from binary mask.

  Args:
      mask (np.ndarray): binary mask of shape (h x w)

  Returns:
      Tuple[int, int, int, int]: x_top_left, y_top_left, width, height
  """
  mask = mask.squeeze()
  mask_height, mask_width = mask.shape
  
  s = np.where(mask > 0)
  
  py = s[0]
  px = s[1]
  
  y_top_left = np.min(py)
  x_top_left = np.min(px)
  assert y_top_left >= 0,'y_top_left of bounding box must be >= 0 but is {:d} px'.format(y_top_left)
  assert x_top_left >= 0,'x_top_left of bounding box must be >= 0 but is {:d} px'.format(x_top_left)
  
  y_bottom_right = np.max(py)
  x_bottom_right = np.max(px)
  assert y_bottom_right < mask_height, 'y_bottom_right of bounding box must be < {:d} px but is {:d} px'.format(mask_height, y_bottom_right)
  assert x_bottom_right < mask_width, 'x_bottom_right of bounding box must be < {:d} px but is {:d} px'.format(mask_width, y_top_left)
  
  box_width = x_bottom_right - x_top_left
  box_height = y_bottom_right - y_top_left
  assert box_width > 0, 'Width of bounding box must be > 0 but is {:d} px'.format(box_width)
  assert box_height > 0, 'Height of bounding box must be > 0 but is {:d} px'.format(box_height)

  return x_top_left, y_top_left, box_width, box_height

def get_contours(mask: np.ndarray) -> List:
  """ Compute contours based on a binary mask.

  Args:
      mask (np.ndarray): binary mask of shape (h x w)

  Returns:
      List: contours (see openCV doc)
  """
  mask = mask.squeeze().astype(np.uint8)
  if cv2.__version__ == '3.4.2':
    _, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
  else:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  return cnts

class Prediction():
  """ Container of all predictions (paths to files) available for a single image
  """
  def __init__(self, path: str, pred: str, channels: int, width: int, height: int, dtype: str):
    self.path = path

    # network prediction (filename)
    self.pred = pred

    # shape of prediction C x H x W
    self.channels = channels
    self.width = width
    self.height = height

    dtype_converter = {"float32" : np.float32, "float64" : np.float64}
    self.dtype = dtype_converter[dtype]

  def load(self) -> np.ndarray:
    """ Load network prediction from disk.

    Returns:
        np.ndarray: network prediction
    """
    path_to_pred = os.path.join(self.path, self.pred)
    pred = np.fromfile(path_to_pred, dtype=self.dtype)
    pred = np.reshape(pred, (self.channels, self.height, self.width))

    return pred

  @staticmethod
  def is_pred(filename: str) -> bool:
    """ Check if filename is a prediction file.

    Args:
        filename (str): filename

    Returns:
        bool: whether or not the filename is a prediction file
    """
    extensions = ['.pred']

    return any([filename.endswith(ext) for ext in extensions])

  @staticmethod
  def is_meta(filename: str) -> bool:
    """ Check if filename is a meta file.

    Args:
        filename (str): filename

    Returns:
        bool: whether or not the filename is a meta file
    """
    extensions = ['.meta']

    return any([filename.endswith(ext) for ext in extensions])

def get_all_predictions(path: str) -> List[Prediction]:
  """ Get all predicition saved on disk.

  This function should be used in the report.py script.

  Args:
      path (str): path to directory which contains all predictions

      This dir needs to have the following structure:
      ├── train
      │   ├── 0002
      │      ├── 0056.meta
      │      ├── 0056.pred
      │   ├── 0004
      │      ├── 0056.meta
      │      ├── 0056.pred

  Returns:
      List[Prediction]: list of all predicitions
  """
  if not os.path.isdir(path):
    return []

  predictions = []

  directories = [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]
  directories.sort(reverse=True)

  for directory in directories:
    path_to_pred_dir = os.path.join(path, directory)
    network_preds = [filename for filename in os.listdir(path_to_pred_dir) if Prediction.is_pred(filename)]

    for network_pred in network_preds:
      filename = network_pred.split(".")[0]
      meta_data = filename + ".meta"
      with open(os.path.join(path_to_pred_dir, meta_data), "r") as istream:
        lines = istream.readlines()
        channels = int(lines[0].strip())
        height = int(lines[1].strip())
        width = int(lines[2].strip())
        dtype = str(lines[3].strip()) 

      predictions.append(Prediction(path_to_pred_dir, network_pred, channels, width, height, dtype))

  return predictions

def sigmoid(x: np.ndarray) -> np.ndarray:
  """ Compute sigmoid activation function

  This function should be applied to numpy ndarrys.

  Args:
      x (np.ndarray): input array

  Returns:
      np.ndarray: output array
  """
  return 1 / (1 + np.exp(-x))

class PrecisionMatrix():
  """ Contains parameters of precision matrix and computes probability margins.
  """
  def __init__(self, mode: str, values: Union[np.ndarray, torch.Tensor], sigma_scale: float, alpha_scale: float, height: float):
    self.mode = mode # {'np', 'torch'}
    self.alpha_scale = alpha_scale
    self.sigma_scale = sigma_scale
    self.height = height
    self.alpha = 0 # rotation

    if self.mode == 'np':
      l_11 = np.exp(values[0] * self.sigma_scale)
  
      try:
        l_22 = np.exp(values[1] * self.sigma_scale)
      except IndexError:
        l_22 = l_11
  
      try:
        self.alpha = (values[2] * self.alpha_scale) * (math.pi / 2)
      except IndexError:
        self.alpha = np.zeros_like(l_11)
  
      self.p_xx = np.power(np.cos(self.alpha), 2) *  l_11 + np.power(np.sin(self.alpha), 2) * l_22
      self.p_yy = np.power(np.sin(self.alpha), 2) *  l_11 + np.power(np.cos(self.alpha), 2) * l_22
      self.p_yx = -np.cos(self.alpha) * np.sin(self.alpha) * l_11 + np.sin(self.alpha)*np.cos(self.alpha) * l_22
      self.p_xy = self.p_yx
        
      # precision matrix
      self.precision_mat = np.array([[self.p_xx, self.p_xy], [self.p_yx, self.p_yy]])
  
      # positive defintie check
      eigvals, _ = np.linalg.eig(self.precision_mat)
      assert np.all((eigvals + 1e-8) >= 0)

    elif self.mode == 'torch':
      l_11 = torch.exp(values[0] * self.sigma_scale)

      try:
        l_22 = torch.exp(values[1] * self.sigma_scale)
      except IndexError:
        l_22 = l_11

      try:
        self.alpha = torch.tanh(values[2] * self.alpha_scale) * (math.pi / 2)
      except IndexError:
        self.alpha = torch.zeros_like(l_11)

      _p_xx = torch.pow(l_11, 2)
      _p_yy = torch.pow(l_22, 2)

      self.p_xx = torch.pow(torch.cos(self.alpha), 2) * _p_xx + torch.pow(torch.sin(self.alpha), 2) * _p_yy
      self.p_yy = torch.pow(torch.sin(self.alpha), 2) * _p_xx + torch.pow(torch.cos(self.alpha), 2) * _p_yy
      self.p_yx = -torch.cos(self.alpha)*torch.sin(self.alpha)*_p_xx + torch.sin(self.alpha)*torch.cos(self.alpha)* _p_yy
      self.p_xy = self.p_yx

      # precision matrix
      if values.is_cuda:
        self.precision_mat = torch.cuda.FloatTensor([[self.p_xx, self.p_yx], [self.p_yx, self.p_yy]])
      else:
        self.precision_mat = torch.Tensor([[self.p_xx, self.p_xy], [self.p_yx, self.p_yy]])

      # positive defintie check
      eigvals = torch.eig(self.precision_mat)
      assert all(eigvals.eigenvalues[:,0] >= 0)

  def gaussian(self, dx: float, dy: float) -> float:
    """ Compute 2D gaussian

    Args:
        dx (float): offset from mean in x-direction
        dy (float): offset from mean in y-direction

    Returns:
        float: value of 2D Gaussian
    """
    if self.mode == 'np':
      value = np.exp(-(1/2)* ((dx * self.p_xx * dx) + (dy * self.p_yx * dx) +  (dx * self.p_yx * dy) + (dy * self.p_yy * dy)))
    elif self.mode == 'torch':
      value = torch.exp(-(1/2)* ((dx * self.p_xx * dx) + (dy * self.p_yx * dx) +  (dx * self.p_yx * dy) + (dy * self.p_yy * dy)))
    
    return value

  def gaussian_margin(self, x: np.ndarray, p: float=0.5) -> float:
    """ Compute the margin for a given probability.

    Args:
        x (np.ndarray): vector of length 1 of shape (2,)
        p (float, optional): probability margin. Defaults to 0.5.

    Returns:
        float: scale of vector x to reach the margin with probability p
    """
    if self.mode == 'np':
      margin = np.sqrt(- (2*np.log(p)) / (x[0]**2*self.p_xx + x[1]*self.p_yx*x[0] + x[0]*self.p_yx*x[1] + x[1]**2*self.p_yy))
    elif self.mode == 'torch':
      margin = torch.sqrt(- (2*torch.log(p)) / (x[0]**2*self.p_xx + x[1]*self.p_yx*x[0] + x[0]*self.p_yx*x[1] + x[1]**2*self.p_yy))

    x_margin = x[0] * margin
    y_margin = x[1] * margin
    check_margin = self.gaussian(x_margin, y_margin)
    if self.mode == 'np':
      assert np.abs(check_margin - p) < 1e-2
    elif self.mode == 'torch':
      assert torch.abs(check_margin - p) < 1e-2

    return margin

  def get_ellipse_params(self) -> Tuple[int, int, float, np.ndarray, np.ndarray]:
    """ Compute the ellipse param of this precision matrix

    Returns:
        Tuple[int, int, float]: height, width and angle
    """
    if not isinstance(self.precision_mat, np.ndarray):
      self.precision_mat = self.precision_mat.cpu().numpy()

    eigvals, eigvecs = np.linalg.eig(self.precision_mat)
    assert np.all(eigvals >= 0)

    # compute major and minor axis
    width = int(np.ceil(self.gaussian_margin(eigvecs[:,0]) * self.height))
    height = int(np.ceil(self.gaussian_margin(eigvecs[:,1]) * self.height))
    angle = np.arctan2(eigvecs[1,0], eigvecs[0,0])
    angle = np.arctan2(np.sin(self.alpha), np.cos(self.alpha))

    v1 = eigvecs[:,0] * width
    v2 = eigvecs[:,1] * height

    # HACK to compute width and height correctly 
    r = np.array([[np.cos(self.alpha), -np.sin(self.alpha)], [np.sin(self.alpha), np.cos(self.alpha)]])
    save_pxx = self.p_xx
    save_pyy = self.p_yy
    save_pyx = self.p_yx
    save_pxy = self.p_xy 
    pmat = r @ self.precision_mat @ r.T # transform basis
    self.p_xx = pmat[0,0]
    self.p_yy = pmat[1,1]
    self.p_yx = pmat[1,0]
    self.p_xy = pmat[0,1]
    width = int(np.ceil(self.gaussian_margin(np.array([1, 0])) * self.height)) # := scale of x-axis
    height = int(np.ceil(self.gaussian_margin(np.array([0, 1])) * self.height)) # := scale of y-axis
    self.p_xx = save_pxx
    self.p_yy = save_pyy
    self.p_yx = save_pyx
    self.p_xy = save_pxy

    return (height, width , angle, v1, v2)

class Cluster():
  """ Postprocessing to cluster parts and objects based on spatial embeddings.
  """
  def __init__(self,
               mode: str,
               width: int,
               height: int,
               n_classes:int,
               n_sigma: int,
               sigma_scale:float,
               alpha_scale: float,
               parts_area_thres: int,
               parts_score_thres: float,
               objects_area_thres: int,
               objects_score_thres: float,
               apply_offsets: bool = True):
    self.mode = mode # {'np', 'torch'}
    self.width = width
    self.height = height
    self.n_classes = n_classes
    self.n_sigma = n_sigma
    self.sigma_scale = sigma_scale
    self.alpha_scale = alpha_scale
    self.parts_area_thres = parts_area_thres
    self.parts_score_thres = parts_score_thres
    self.objects_area_thres = objects_area_thres
    self.objects_score_thres = objects_score_thres
    self.apply_offsets = apply_offsets

    # build coordinate map
    x_max = int(round(self.width / self.height))
    if self.mode == 'np':
      xm = np.linspace(0, x_max, self.width).reshape(1, 1, -1).repeat(self.height, axis=1)
      ym = np.linspace(0, 1, self.height).reshape(1, -1, 1).repeat(self.width, axis=2)
      self.xym = np.concatenate((xm, ym), axis=0) # (2 x h x w)
    elif self.mode == 'torch':
      xm = torch.linspace(0, x_max, self.width).view(1, 1, -1).expand(1, self.height, self.width)
      ym = torch.linspace(0, 1, self.height).view(1, -1, 1).expand(1, self.height, self.width)
      self.xym = torch.cat((xm, ym), 0) # (2 x h x w)

      if torch.cuda.is_available():
        self.xym = self.xym.cuda()
    else:
      assert False, 'Cluster mode is not valid - set it either to "np" or "torch".'

  def cluster(self, pred: Union[np.ndarray, torch.Tensor]) -> Dict:
    """ Perform clustering based on network predictions.

    Args:
        pred (Union[np.ndarray, torch.Tensor]): network predicition (chans, h, w)

        The instance's mode should be set 'np' or 'torch' accordingly.

    Returns:
        Dict: results of clustering
    """
    height, width = pred.shape[1], pred.shape[2]
    xym = self.xym[:, :height, :width]

    # compute indicies
    start_objects_spatial_emb = 0
    end_objects_spatial_emb = 2

    start_objects_sigma = end_objects_spatial_emb
    end_objects_sigma = start_objects_sigma + self.n_sigma

    start_parts_spatial_emb = end_objects_sigma
    end_parts_spatial_emb = start_parts_spatial_emb + 2

    start_parts_sigma = end_parts_spatial_emb
    end_parts_sigma = start_parts_sigma + self.n_sigma

    start_objects_seed = end_parts_sigma
    end_objects_seed = start_objects_seed + self.n_classes

    start_parts_seed = end_objects_seed
    end_parts_seed = start_parts_seed + self.n_classes

    # extract predictions
    if self.mode == 'np':
      parts_offsets = np.tanh(pred[start_parts_spatial_emb: end_parts_spatial_emb]) # (2, h, w)
      objects_offsets = np.tanh(pred[start_objects_spatial_emb: end_objects_spatial_emb]) # (2, h, w)

      parts_seed = sigmoid(pred[start_parts_seed: end_parts_seed]) # (n_classes, h, w)
      objects_seed = sigmoid(pred[start_objects_seed: end_objects_seed]) # (n_classes, h, w)
    elif self.mode == 'torch':
      parts_offsets = torch.tanh(pred[start_parts_spatial_emb: end_parts_spatial_emb]) # (2, h, w)
      objects_offsets = torch.tanh(pred[start_objects_spatial_emb: end_objects_spatial_emb]) # (2, h, w)

      parts_seed = torch.sigmoid(pred[start_parts_seed: end_parts_seed]) # (n_classes, h, w)
      objects_seed = torch.sigmoid(pred[start_objects_seed: end_objects_seed]) # (n_classes, h, w)

    if self.apply_offsets:
      parts_spatial_emb = xym + parts_offsets # (2, h, w)
      objects_spatial_emb = xym + parts_offsets + objects_offsets # (2, h, w)
    else:
      parts_spatial_emb = xym
      objects_spatial_emb = xym

    parts_sigma = pred[start_parts_sigma:end_parts_sigma] # (n_sigma, h, w)
    objects_sigma = pred[start_objects_sigma: end_objects_sigma] # (n_sigma, h, w)

    # --- clustering ---
    results = {}
    results["parts"] = defaultdict(list)
    results["objects"] = defaultdict(list)

    object_count = 0
    part_count = 0

    # start clustering of parts
    for cls_idx in range(self.n_classes):
      # get all foreground pixels
      parts_cls_mask = parts_seed[cls_idx] > 0.5 # (h, w)

      if parts_cls_mask.sum() > self.parts_area_thres:
        if self.mode == 'np':
          # get coords of pixels which belong to foreground
          part_xym_masked = xym[np.broadcast_to(parts_cls_mask, xym.shape)].reshape(2, -1) # (2, n)

          # get all predictions which belong to foreground
          parts_spatial_emb_masked = parts_spatial_emb[np.broadcast_to(parts_cls_mask, parts_spatial_emb.shape)].reshape(2, -1) # (2, n)
          objects_spatial_emb_masked = objects_spatial_emb[np.broadcast_to(parts_cls_mask, objects_spatial_emb.shape)].reshape(2, -1) # (2, n)
          parts_sigma_masked = parts_sigma[np.broadcast_to(parts_cls_mask, parts_sigma.shape)].reshape(self.n_sigma, -1) # (n_sigma, n)
          parts_seed_masked = parts_seed[cls_idx][parts_cls_mask] # (n, )

          # set all foreground pixels to be unclustered
          parts_unclustered = np.ones(parts_cls_mask.sum(), dtype=np.bool) # (n, )
        elif self.mode == 'torch':
          part_xym_masked = xym[parts_cls_mask.expand_as(xym)].reshape(2, -1) # (2, n)
 
          parts_spatial_emb_masked = parts_spatial_emb[parts_cls_mask.expand_as(parts_spatial_emb)].reshape(2, -1) # (2, n)
          objects_spatial_emb_masked = objects_spatial_emb[parts_cls_mask.expand_as(objects_spatial_emb)].reshape(2, -1) # (2, n)
          parts_sigma_masked = parts_sigma[parts_cls_mask.expand_as(parts_sigma)].reshape(self.n_sigma, -1) # (n_sigma, n)
          parts_seed_masked = parts_seed[cls_idx][parts_cls_mask] # (n, )

          # set all foreground pixels to be unclustered
          parts_unclustered = torch.ones(parts_cls_mask.sum()).byte() # (n, )
          if torch.cuda.is_available():
            parts_unclustered = parts_unclustered.cuda()

        # cluster each part
        while(parts_unclustered.sum()) > self.parts_area_thres:
          if self.mode == 'np':
            # get pixel with highest score which is still unclustered
            idx = (parts_seed_masked * parts_unclustered.astype(np.float32)).argmax().item()
          elif self.mode == 'torch':
            idx = (parts_seed_masked * parts_unclustered.float()).argmax().item()

          part_seed_score = parts_seed_masked[idx]
          if part_seed_score < self.parts_score_thres:
            break
          
          part_center = parts_spatial_emb_masked[: , idx] # (2, )
          part_sigmas = parts_sigma_masked[:, idx] # (n_sigma)
          part_prec_mat = PrecisionMatrix(self.mode, part_sigmas, self.sigma_scale, self.alpha_scale, height)

          # calculate gaussian distance between center to all foreground pixels
          delta = (parts_spatial_emb_masked - part_center.reshape(2, 1)) # (2, n)
          if self.mode == 'np':
            prob = np.exp(-(1/2) * (
                                    (delta[0] * part_prec_mat.p_xx * delta[0]) + \
                                    (delta[1] * part_prec_mat.p_yx * delta[0]) + \
                                    (delta[0] * part_prec_mat.p_xy * delta[1]) + \
                                    (delta[1] * part_prec_mat.p_yy * delta[1])
                                   )
                         ) # (n, )
          elif self.mode == 'torch':
            prob = torch.exp(-(1/2) * ((delta[0] * part_prec_mat.p_xx * delta[0]) + \
                                       (delta[1] * part_prec_mat.p_yx * delta[0]) + \
                                       (delta[0] * part_prec_mat.p_yx * delta[1]) + \
                                       (delta[1] * part_prec_mat.p_yy * delta[1])
                                      )) # (n, )
                
          part_proposal = ((prob * parts_unclustered) > 0.5)
          parts_unclustered[idx] = False # set center as clustered to avoid never ending loop
  
          if part_proposal.sum() > self.parts_area_thres:
            parts_unclustered[part_proposal] = False
            # binary mask for this part
            part_instance_map = np.zeros((height, width), dtype=np.bool)
            if self.mode == 'np':
              part_instance_map[parts_cls_mask] = part_proposal

              # spatial embeddings of this part
              part_spatial_emb = parts_spatial_emb_masked[np.broadcast_to(part_proposal, parts_spatial_emb_masked.shape)].reshape(2, -1)

              # spatial embeddings of this part w.r.t to object
              object_spatial_emb = objects_spatial_emb_masked[np.broadcast_to(part_proposal, objects_spatial_emb_masked.shape)].reshape(2, -1)

              # coordinates of this part in original image
              part_xym = part_xym_masked[np.broadcast_to(part_proposal, part_xym_masked.shape)].reshape(2, -1)
            elif self.mode == 'torch':
              if not isinstance(parts_cls_mask, np.ndarray):
                parts_cls_mask = parts_cls_mask.cpu().numpy().astype(np.bool)
              part_instance_map[parts_cls_mask] = part_proposal.cpu().numpy()

               # spatial embeddings of this part
              part_spatial_emb = parts_spatial_emb_masked[part_proposal.expand_as(parts_spatial_emb_masked)].reshape(2, -1).cpu().numpy()

              # spatial embeddings of this part w.r.t to object
              object_spatial_emb = objects_spatial_emb_masked[part_proposal.expand_as(objects_spatial_emb_masked)].reshape(2, -1).cpu().numpy()

              # coordinates of this part in original image
              part_xym = part_xym_masked[part_proposal.expand_as(part_xym_masked)].reshape(2, -1).cpu().numpy()

            part_count += 1
            part = {"part_mask": part_instance_map,
                    "part_embeddings": part_spatial_emb,
                    "object_embeddings": object_spatial_emb,
                    "part_coordinates": part_xym,
                    "part_score": part_seed_score,
                    "part_center": part_center,
                    "part_sigma": part_prec_mat,
                    "part_id": part_count,
                    "belongs_to_object": False}

            results["parts"][str(cls_idx)].append(part)
            
      # --- finished clustering parts of current cls - now let's merge them to objects ---
      # get all foreground pixels
      objects_cls_mask = objects_seed[cls_idx] > 0.5 # (h, w)
      if objects_cls_mask.sum() > self.objects_area_thres:
        if self.mode == 'np':
          # get coords of pixels which belong to (object) foreground
          objects_xym_masked = xym[np.broadcast_to(objects_cls_mask, xym.shape)].reshape(2, -1) # (2, n)
          
          # get all predictions which belong to (object) foreground
          objects_spatial_emb_masked = objects_spatial_emb[np.broadcast_to(objects_cls_mask, objects_spatial_emb.shape)].reshape(2, -1) # (2, n)
          objects_sigma_masked = objects_sigma[np.broadcast_to(objects_cls_mask, objects_sigma.shape)].reshape(self.n_sigma, -1) # (n_sigma, n)
          objects_seed_masked = objects_seed[cls_idx][objects_cls_mask] # (n, )

          # set all foreground pixels to be unclustered
          objects_unclustered = np.ones(objects_cls_mask.sum(), dtype=np.bool) # (n, )
        elif self.mode == 'torch':
          objects_xym_masked = xym[objects_cls_mask.expand_as(xym)].reshape(2, -1) # (2, n)

          objects_spatial_emb_masked = objects_spatial_emb[objects_cls_mask.expand_as(objects_spatial_emb)].reshape(2, -1) # (2, n)
          objects_sigma_masked = objects_sigma[objects_cls_mask.expand_as(objects_sigma)].reshape(self.n_sigma, -1) # (n_sigma, n)
          objects_seed_masked = objects_seed[cls_idx][objects_cls_mask] # (n, )

          # set all foreground pixels to be unclustered
          objects_unclustered = torch.ones(objects_cls_mask.sum()).byte() # (n, )
          if torch.cuda.is_available():
            objects_unclustered = objects_unclustered.cuda()

        while(objects_unclustered.sum()) > self.objects_area_thres:
          if self.mode == 'np':
            idx = (objects_seed_masked * objects_unclustered.astype(np.float32)).argmax().item()
          elif self.mode == 'torch':
            idx = (objects_seed_masked * objects_unclustered.float()).argmax().item()

          objects_seed_score = objects_seed_masked[idx]
          if objects_seed_score < self.objects_score_thres:
            break

          object_center = objects_spatial_emb_masked[:, idx]  # (2, )
          object_sigmas = objects_sigma_masked[:, idx] # (n_sigma)
          object_prec_mat = PrecisionMatrix(self.mode, object_sigmas, self.sigma_scale, self.alpha_scale, height)
          objects_unclustered[idx] = False

          object_has_parts = False
          object_area = 0.0

          # assign each part of same cls to an object
          object_part_idx = []
          for idx, part in enumerate(results["parts"][str(cls_idx)]):
            if part["belongs_to_object"]:
              # do not assign any object twice or more
              continue

            # calculate gaussian
            if self.mode == 'np':
              delta = part["object_embeddings"] - object_center.reshape(2, 1)
              prob = np.exp(-(1/2) * ((delta[0] * object_prec_mat.p_xx * delta[0]) + \
                                      (delta[1] * object_prec_mat.p_yx * delta[0]) + \
                                      (delta[0] * object_prec_mat.p_yx * delta[1]) + \
                                      (delta[1] * object_prec_mat.p_yy * delta[1])
                                     )) # (n, )
              # check if at least 50 percent of part embeddings are within the object
              ratio = (np.sum(prob > 0.5)) / (prob.shape[0])
            elif self.mode == 'torch':
              part_object_embeddings = torch.from_numpy(part["object_embeddings"])
              if torch.cuda.is_available:
                part_object_embeddings = part_object_embeddings.cuda()

              delta = part_object_embeddings - object_center.reshape(2, 1)
              prob = torch.exp(-(1/2) * ((delta[0] * object_prec_mat.p_xx * delta[0]) + \
                                         (delta[1] * object_prec_mat.p_yx * delta[0]) + \
                                         (delta[0] * object_prec_mat.p_yx * delta[1]) + \
                                         (delta[1] * object_prec_mat.p_yy * delta[1])
                                        )) # (n, )
              # check if at least 50 percent of part embeddings are within the object
              ratio = (torch.sum(prob > 0.5)).float() / float((prob.shape[0]))

            if ratio < 0.5:
              # the amount of embeddings within this cluster is not sufficient
              continue

            object_has_parts = True
            object_area += part["part_mask"].sum()
            object_part_idx.append(idx)
            part["belongs_to_object"] = True

            # set pixels which belong to current part to be clustered
            for j in range(part["part_coordinates"].shape[1]):
              part_coord = part["part_coordinates"][:, j].reshape(2, 1)
              if self.mode == 'np':
                # check if this coordinate is in the object foreground mask
                part_coord_in_fg_mask = np.all(objects_xym_masked == part_coord, axis=0)
                if np.any(part_coord_in_fg_mask):
                  objects_unclustered[part_coord_in_fg_mask.argmax()] = False
              elif self.mode == 'torch':
                part_coord = torch.Tensor(part_coord)
                part_coord = part_coord.to(objects_xym_masked.device)
                part_coord_in_fg_mask = (objects_xym_masked == part_coord).all(dim=0)
                if part_coord_in_fg_mask.any():
                  objects_unclustered[part_coord_in_fg_mask.argmax()] = False
                
          if object_has_parts:
            if object_area > self.objects_area_thres:
              object_count += 1
              if self.mode == 'np':
                obj = {"obj_part_indicies": object_part_idx,
                       "obj_score": objects_seed_score,
                       "obj_center": object_center,
                       "obj_sigma": object_prec_mat,
                       "object_id": object_count}
              elif self.mode == 'torch':
                obj = {"obj_part_indicies": object_part_idx,
                       "obj_score": objects_seed_score.cpu().numpy(),
                       "obj_center": object_center.cpu().numpy(),
                       "obj_sigma": object_prec_mat,
                       "object_id": object_count}
  
              results["objects"][str(cls_idx)].append(obj)
              

    return objects_seed, parts_seed, objects_offsets, parts_offsets, objects_sigma, parts_sigma, results

  def convert_results_to_coco(self, results: Dict) -> Tuple[Dict,Dict]:
    """ Convert results to pass them to coco evaluator.
    """
    boxes = []
    scores = []
    labels = []
    masks = []
    object_preds = {}

    part_boxes = []
    part_scores = []
    part_labels = []
    part_maks = []
    part_preds = {}

    for cls_id in results["objects"].keys():
      for obj in results["objects"][cls_id]:
        score = float(obj['obj_score'])
        scores.append(score)
        
        #TODO This +1 is here kind of unsuspected but needed s.t. categories start with 1
        # however this is not in agreement with cls_id which start with 0 -> HACKY but fine until we have more classes
        label = int(cls_id) + 1
        labels.append(label)

        # loop over all parts which belong to current object
        obj_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for part_idx in obj["obj_part_indicies"]:
          # --- now lets gather the predictions for this parts ---
          part_mask = results["parts"][cls_id][part_idx]["part_mask"]
          part_maks.append(part_mask)

          x_tl, y_tl, w, h = bounding_box_from_mask(part_mask)
          part_boxes.append([x_tl, y_tl, w, h])
          
          part_score = float(results["parts"][cls_id][part_idx]["part_score"])
          part_scores.append(part_score)

          part_labels.append(label)

          # update objects mask
          obj_mask += part_mask
        
        masks.append(obj_mask)

        x_top_left, y_top_left, box_width, box_height = bounding_box_from_mask(obj_mask)
        boxes.append([x_top_left, y_top_left, box_width, box_height])
        
    object_preds["boxes"] = boxes
    object_preds["scores"] = scores 
    object_preds["labels"] = labels
    object_preds["masks"] = masks

    part_preds["boxes"] = part_boxes
    part_preds["scores"] = part_scores 
    part_preds["labels"] = part_labels
    part_preds["masks"] = part_maks

    return object_preds, part_preds

  def draw_part_map(self, results: Dict, cls_idx: str) -> np.ndarray:
    """ Create a map of all predicted parts.

    This map contains consecutive numbers for each part.

    Args:
        results (Dict): predicition from self.cluster()
        cls_idx (str): class index of part to be drawn

    Returns:
        np.ndarray: part map
    """
    part_map = np.zeros((self.height, self.width), dtype=np.uint8)
  
    n_objs = len(results["objects"][cls_idx])
    part_counter = 0
    for i in range(n_objs):
      obj_part_ids = results["objects"][cls_idx][i]['obj_part_indicies']
      for obj_part_id in obj_part_ids:
        part_counter += 1
        part_mask = results["parts"][cls_idx][obj_part_id]['part_mask'].astype(np.uint8)
        part_map += (part_mask * part_counter)

    return part_map

class Visualizer():
  """ Visualize network predictions and results.
  """
  def __init__(self, path_train_imgs, path_val_imgs, path_test_imgs, n_classes: int, colors: Dict, im_width: int, im_height: int, alpha: float=0.5):
    self.path_train_imgs = path_train_imgs
    self.path_val_imgs = path_val_imgs
    self.path_test_imgs = path_test_imgs
    self.n_classes = n_classes
    self.colors = colors
    self.status = None
    self.im_width = im_width
    self.im_height = im_height

    alpha = alpha

    # What size does the figure need to be in inches to fit the image?
    dpi = plt.rcParams['figure.dpi']
    figsize = self.im_width / float(dpi), self.im_height / float(dpi)
    plt.rcParams['figure.figsize'] = figsize # set default size of plots

  def set_status(self, status: str):
    """ Set status to train or val

    Args:
        status (str): status description {'train', 'val', 'test'}
    """
    status = status.lower()
    assert (status == 'train') or (status == "val") or (status == "test")

    self.status = status

  def plot_sigmas(self, path: str, name: str, objects_sigma_map: np.ndarray, parts_sigma_map: np.ndarray, alpha: float=1.0):
    """ Visualize the sigma maps of objects and parts.

    Args:
        path (str): path to folder which contains all network predictions
        name (str): ilename of prediction, e.g., 00024.pred
        objects_sigma_map (np.ndarray): sigma map of objects (n_sigmas, h, w)
        parts_sigma_map (np.ndarray):  sigma map of parts (n_sigmas, h, w)
        alpha (float, optional): Transparency of plot. Defaults to 0.5.
    """
    name = name.split(".")[0]

    # load input image
    if self.status == 'train':
      img_path = os.path.join(self.path_train_imgs, name + '.png')
    if self.status == 'val':
      img_path = os.path.join(self.path_val_imgs, name + '.png')
    if self.status == 'test':
      img_path = os.path.join(self.path_test_imgs, name + '.png')

    img = skimage.io.imread(img_path)
    img = cv2.resize(img, (self.im_width, self.im_height), interpolation=cv2.INTER_LINEAR)

    n_sigmas = objects_sigma_map.shape[0]
    assert n_sigmas == parts_sigma_map.shape[0]
    
    # --- objects ---
    for i in range(n_sigmas):
      fig_objects, ax_objects = plt.subplots(nrows=1, ncols=1)
      ax_objects.axis('off')
      # ax_objects.imshow(img, interpolation="bicubic")
      ax_objects.imshow(objects_sigma_map[i], alpha=alpha, interpolation="bicubic")

      path_object_sigmas = os.path.join(path, name + "_objects_sigmas_{:04d}.png".format(i))
      fig_objects.tight_layout()
      fig_objects.savefig(path_object_sigmas, transparent=True)
      plt.close(fig_objects)

    # --- parts ---
    for j in range(n_sigmas):
      fig_parts, ax_parts = plt.subplots(nrows=1, ncols=1)
      ax_parts.axis('off')
      # ax_parts.imshow(img, interpolation="bicubic")
      ax_parts.imshow(parts_sigma_map[j], alpha=alpha, interpolation="bicubic")

      path_part_sigmas = os.path.join(path, name + "_parts_sigmas_{:04d}.png".format(j))
      fig_parts.tight_layout()
      fig_parts.savefig(path_part_sigmas, transparent=True)
      plt.close(fig_parts)
    plt.close('all')

  def plot_seed(self, path: str, name: str, objects_seed_map: np.ndarray, parts_seed_map: np.ndarray, alpha: float=1.0):
    """ Visualize the seed maps of objects and parts.

    Args:
        path (str): path to folder which contains all network predictions
        name (str): filename of prediction, e.g., 00024.pred
        objects_seed_map (np.ndarray): seed map of objects (n_classes, h, w)
        parts_seed_map (np.ndarray): seed map of parts (n_classes, h, w)
    """
    name = name.split(".")[0]

    # load input image
    if self.status == 'train':
      img_path = os.path.join(self.path_train_imgs, name + '.png')
    if self.status == 'val':
      img_path = os.path.join(self.path_val_imgs, name + '.png')
    if self.status == 'test':
      img_path = os.path.join(self.path_test_imgs, name + '.png')
    img = skimage.io.imread(img_path)

    # sanity checks
    objects_seed_max_val = np.max(objects_seed_map)
    objects_seed_min_val = np.min(objects_seed_map)
    parts_seed_max_val = np.max(parts_seed_map)
    parts_seed_min_val = np.min(parts_seed_map)
    assert  objects_seed_max_val <= 1, "encountered invalid value in seed map"
    assert  objects_seed_min_val >= 0, "encountered invalid value in seed map"
    assert  parts_seed_max_val <= 1, "encountered invalid value in seed map"
    assert  parts_seed_min_val >= 0, "encountered invalid value in seed map"

    # --- objects ---
    fig_objects, ax_objects = plt.subplots(nrows=1, ncols=self.n_classes)
    if self.n_classes > 1:
      for cls_idx in range(self.n_classes):
        ax_objects[cls_idx].axis('off')
        # ax_objects[cls_idx].imshow(img, interpolation="bicubic")
        ax_objects[cls_idx].imshow(objects_seed_map[cls_idx], vmin=0, vmax=1, alpha=alpha, interpolation="bicubic")
    else:
      ax_objects.axis('off')
      # ax_objects.imshow(img, interpolation="bicubic")
      ax_objects.imshow(objects_seed_map[0], vmin=0, vmax=1, alpha=alpha, interpolation="bicubic")

    path_object_seed = os.path.join(path, name + "_objects_seed.png")
    fig_objects.tight_layout()
    fig_objects.savefig(path_object_seed, transparent=True)

    # --- parts ---
    fig_parts, ax_parts = plt.subplots(nrows=1, ncols=self.n_classes)

    if self.n_classes > 1:
      for cls_idx in range(self.n_classes):
        ax_parts[cls_idx].axis('off')
        # ax_parts[cls_idx].imshow(img, interpolation="bicubic")
        ax_parts[cls_idx].imshow(parts_seed_map[cls_idx], vmin=0, vmax=1, alpha=alpha, interpolation="bicubic")
    else:
      ax_parts.axis('off')
      # ax_parts.imshow(img, interpolation="bicubic")
      ax_parts.imshow(parts_seed_map[0], vmin=0, vmax=1, alpha=alpha, interpolation="bicubic")

    path_part_seed = os.path.join(path, name + "_parts_seed.png")
    fig_parts.tight_layout()
    fig_parts.savefig(path_part_seed, transparent=True)
    plt.close('all')

  def plot_embeddings(self, path: str, name: str, results: Dict, part_offsets: np.ndarray, objects_offsets: np.ndarray, alpha: float=1.0):
    """ Visualize the embeddings of objects and parts.

    Args:
        path (str): path to folder which contains all network predictions
        name (str): filename of prediction, e.g., 00024.pred
        results (Dict): cluster results
        part_offsets (np.ndarray): predicted offsetmap for parts (2, h, w), 1st channel contains x-offsets and 2nd y-offsets
        objects_offsets (np.ndarray): predicted offsetmap for objects (2, h, w)
    """
    name = name.split(".")[0]

    # load input image
    if self.status == 'train':
      img_path = os.path.join(self.path_train_imgs, name + '.png')
    if self.status == 'val':
      img_path = os.path.join(self.path_val_imgs, name + '.png')
    if self.status == 'test':
      img_path = os.path.join(self.path_test_imgs, name + '.png')
    img = skimage.io.imread(img_path)
    img = cv2.resize(img, (self.im_width, self.im_height), interpolation=cv2.INTER_LINEAR)

    # --- part embeddings ---
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.axis('off')
    ax1.imshow(img, interpolation="bicubic")

    parts_offsets_x = part_offsets[0]
    parts_offsets_y = part_offsets[1]
    parts_orientation = np.arctan2(parts_offsets_y, parts_offsets_x)
    ax1.imshow(parts_orientation, alpha=alpha, interpolation="bicubic")

    path_part_offsets = os.path.join(path, name + "_parts_offsets.png")
    fig.tight_layout()
    fig.savefig(path_part_offsets, transparent=True)
    plt.close(fig)

    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.imshow(img, alpha=0.0, interpolation="bicubic")
    ax1.axis('off')
    for cls_idx in results["parts"].keys():
      for part in results["parts"][cls_idx]:

        random_fc = (np.random.rand(), np.random.rand(), np.random.rand(), 0.15)
        random_ec = (random_fc[0], random_fc[1], random_fc[2], 1.0)

        # part cluster
        part_embeddings_x = np.around(part["part_embeddings"][0] * self.im_height)
        part_embeddings_y = np.around(part["part_embeddings"][1] * self.im_height)
        ax1.scatter(part_embeddings_x, part_embeddings_y, s = 1, color = random_ec)

        # connect original coords to spatial embeddings
        part_coords_x = np.around(part["part_coordinates"][0] * self.im_height)
        part_coords_y = np.around(part["part_coordinates"][1] * self.im_height)
        sample_ids = np.random.choice(part_coords_x.size, 3)
        for sample_id in sample_ids:
          random_color = (np.random.rand(), np.random.rand(), np.random.rand(), 1)
          # ax1.plot([part_embeddings_x[sample_id], part_coords_x[sample_id]], [part_embeddings_y[sample_id], part_coords_y[sample_id]], color=random_color)
          # ax1.scatter(part_coords_x[sample_id], part_coords_y[sample_id], s = 1.0, color='w')

        # center
        part_center_x = int(np.round(part["part_center"][0] * self.im_height))
        part_center_y = int(np.round(part["part_center"][1] * self.im_height))
        part_center = patches.Circle((part_center_x, part_center_y), radius=3, facecolor=self.colors[cls_idx], edgecolor="none")
        ax1.add_patch(part_center)

        # ellipse
        part_ellip_height, part_ellip_width, part_ellip_angle, v1, v2 = part["part_sigma"].get_ellipse_params()
        # ax1.text(part_center_x, part_center_y, "{:3.2f}".format(part_ellip_angle * (180 / np.pi)))
        part_ellip = patches.Ellipse((part_center_x, part_center_y), width=2*(part_ellip_width), height=2*(part_ellip_height), angle=-part_ellip_angle * (180/np.pi), facecolor="none", edgecolor=self.colors[cls_idx], linestyle='--', linewidth=2)
        # ax1.plot([part_center_x, part_center_x + v1[0]], [part_center_y, part_center_y + v1[1]], color='green')
        # ax1.plot([part_center_x, part_center_x + v2[0]], [part_center_y, part_center_y + v2[1]], color='green')
        ax1.add_patch(part_ellip)

    path_part_embeddings = os.path.join(path, name + "_parts_embeddings.png")
    fig.tight_layout()
    fig.savefig(path_part_embeddings, transparent=True)
    plt.close('all')

    # --- object embeddings ---
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.axis('off')
    ax1.imshow(img, interpolation="bicubic")

    objects_offsets_x = objects_offsets[0]
    objects_offsets_y = objects_offsets[1]
    objects_orientation = np.arctan2(objects_offsets_x, objects_offsets_y)
    ax1.imshow(objects_orientation, alpha=alpha, interpolation="bicubic")

    path_obj_offsets = os.path.join(path, name + "_objects_offsets.png")
    fig.tight_layout()
    fig.savefig(path_obj_offsets, transparent=True)
    plt.close(fig)

    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.imshow(img, alpha=0.0, interpolation="bicubic")
    ax1.axis('off')

    for cls_idx in results["objects"].keys():
      for obj in results["objects"][cls_idx]:
        obj_color_random = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        # loop over all parts which belong to current object
        for part_idx in obj["obj_part_indicies"]:
          obj_part_embeddings = results["parts"][cls_idx][part_idx]["object_embeddings"]
          part_embeddings = results["parts"][cls_idx][part_idx]["part_embeddings"]
          part_coords = results["parts"][cls_idx][part_idx]["part_coordinates"]

          # object cluster
          obj_part_embeddings_x = np.around(obj_part_embeddings[0] * self.im_height)
          obj_part_embeddings_y = np.around(obj_part_embeddings[1] * self.im_height)
          ax1.scatter(obj_part_embeddings_x, obj_part_embeddings_y, s = 1, color=obj_color_random, zorder=0)
     
          # part embeddings
          part_embeddings_x = np.around(part_embeddings[0] * self.im_height)
          part_embeddings_y = np.around(part_embeddings[1] * self.im_height)
    
          # connect original coords to spatial embeddings
          part_coords_x = np.around(part_coords[0] * self.im_height)
          part_coords_y = np.around(part_coords[1] * self.im_height)
      
          sample_ids = np.random.choice(part_coords_x.size, 1)
          for sample_id in sample_ids:
            color_random = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            
            # ax1.plot([part_embeddings_x[sample_id], part_coords_x[sample_id]], [part_embeddings_y[sample_id], part_coords_y[sample_id]], color=color_random, zorder=1)
            # ax1.plot([part_embeddings_x[sample_id], obj_part_embeddings_x[sample_id]], [part_embeddings_y[sample_id], obj_part_embeddings_y[sample_id]], color=color_random, zorder=2)
            
            start = patches.Circle((part_coords_x[sample_id], part_coords_y[sample_id]), radius=3, facecolor="#FFFFFF", edgecolor="none")
            end = patches.Circle((part_embeddings_x[sample_id], part_embeddings_y[sample_id]), radius=3, facecolor="#FFFFFF", edgecolor="none")
            # ax1.add_patch(start)
            # ax1.add_patch(end)
            # ax1.scatter(part_coords_x[sample_id], part_coords_y[sample_id], s = 2.0, color='w')
            # ax1.scatter(part_embeddings_x[sample_id], part_embeddings_y[sample_id], s = 2.0, color='w')

        # center
        obj_center_x = int(np.round(obj["obj_center"][0] * self.im_height))
        obj_center_y = int(np.round(obj["obj_center"][1] * self.im_height))
        obj_center = patches.Circle((obj_center_x, obj_center_y), radius=3, facecolor=self.colors[cls_idx], edgecolor="none")
        ax1.add_patch(obj_center)

        # ellipse
        obj_ellip_height, obj_ellip_width, obj_ellip_angle, v1, v2 = obj["obj_sigma"].get_ellipse_params()
        obj_ellip = patches.Ellipse((obj_center_x, obj_center_y), width=2*(obj_ellip_width), height=2*(obj_ellip_height), angle=-obj_ellip_angle * (180/np.pi), facecolor="none", edgecolor=self.colors[cls_idx], linestyle='--', linewidth=2)
        ax1.add_patch(obj_ellip)
            
      path_obj_embeddings = os.path.join(path, name + "_objects_embeddings.png")
      fig.tight_layout()
      fig.savefig(path_obj_embeddings, transparent=True)
      plt.close('all')

  def plot_instances(self, path: str, name: str, results: Dict, alpha: float=0.75):
    """ Visualize the predicted object and part instances.

    Args:
        path (str): path to folder which contains all network predictions
        name (str): filename of prediction, e.g., 00024.pred
        results (Dict): cluster results
    """
    name = name.split(".")[0]

    # load input image
    if self.status == 'train':
      img_path = os.path.join(self.path_train_imgs, name + '.png')
    if self.status == 'val':
      img_path = os.path.join(self.path_val_imgs, name + '.png')
    if self.status == 'test':
      img_path = os.path.join(self.path_test_imgs, name + '.png')
    img = skimage.io.imread(img_path)
    img = cv2.resize(img, (self.im_width, self.im_height), interpolation=cv2.INTER_LINEAR)

    # --- parts ---
    fig, ax1 = plt.subplots(nrows=1, ncols=1)

    for cls_idx in results["parts"].keys():
      for part in results["parts"][cls_idx]:
        # mask
        cnts = get_contours(part["part_mask"])
        random_fc = (np.random.rand(), np.random.rand(), np.random.rand(), 0.15)
        random_ec = (random_fc[0], random_fc[1], random_fc[2], 1.0)
        for cnt in cnts:
          polygon = np.array(cnt).reshape(-1, 2)
          poly = patches.Polygon(polygon, color=random_ec, fill=True, edgecolor=None)
          ax1.add_patch(poly)

    ax1.axis('off')
    ax1.imshow(img, interpolation="bicubic")
    # ax1.imshow(parts_canvas, alpha=alpha)
    path_parts = os.path.join(path, name + "_parts.png")
    fig.tight_layout()
    fig.savefig(path_parts, transparent=True)
    plt.close('all')

    # --- objects ---
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    
    for cls_idx in results["objects"].keys():
      for obj in results["objects"][cls_idx]:
        objects_canvas = np.zeros((self.im_height, self.im_width), dtype=np.uint8)

        # loop over all parts which belong to current object
        for idx, part_idx in enumerate(obj["obj_part_indicies"]):
          objects_canvas += (results["parts"][cls_idx][part_idx]["part_mask"].astype(np.uint8) * (idx + 1))
        
        # mask
        cnts = get_contours(objects_canvas)
        random_fc = (np.random.rand(), np.random.rand(), np.random.rand(), 0.15)
        random_ec = (random_fc[0], random_fc[1], random_fc[2], 1.0)
        for cnt in cnts:
          polygon = np.array(cnt).reshape(-1, 2)
          poly = patches.Polygon(polygon, color=random_ec, fill=True, edgecolor=None)
          ax1.add_patch(poly)

        # bbox
        # x_tl, y_tl, w, h = bounding_box_from_mask(objects_canvas)
        # rect = patches.Rectangle((x_tl, y_tl), w , h, facecolor='none', ec=random_ec, linewidth=1)
        # ax1.add_patch(rect)

    ax1.axis('off')
    ax1.imshow(img, interpolation="bicubic")
    # ax1.imshow(objects_canvas, alpha=alpha)
    path_obj = os.path.join(path, name + "_objects.png")
    fig.tight_layout()
    fig.savefig(path_obj, transparent=True)
    plt.close('all')

class TensorboardLogger():
  """ Tensorboard logging.
  """
  def __init__(self, log_path: str, log_dir: str):
    """ Constructor.

    Args:
        log_path (str): path to folder
        log_dir (str): directory name, e.g., 'train' or 'val
    """
    self.path = os.path.join(log_path, log_dir)
    self.writer = SummaryWriter(self.path)

  def dump(self, info: Dict[str, float], epoch: int):
    """ Add scalar data to logger.

    Args:
        info (Dict[str, float]): scalar data
        epoch (int): global_step
    """
    for key, value in info.items():
        self.writer.add_scalar(key, value, epoch)

    #TODO: This is not working - maybe version of tensorboard is too old?
    # self.writer.flush()

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def nms(seed: torch.Tensor, kernel: int=1) -> torch.Tensor:
  """ Non-maximum supression (nms).

  Args:
      seed (torch.Tensor): input tensor (N, C, H, W)
      kernel (int, optional): kernel sized of max pool op. Defaults to 1.

  Returns:
      torch.Tensor: tensor after applying nms (N, C, H, W)
  """
  converted_from_np = False
  if isinstance(seed, np.ndarray):
    seed = torch.Tensor(seed)
    converted_from_np = True

  pad = (kernel - 1) // 2

  hmax = nn.functional.max_pool2d(seed, (kernel, kernel), stride=1, padding=pad)

  keep = (hmax == seed).float()

  if converted_from_np:
    seed = seed.numpy()
    keep = keep.numpy()

  return seed * keep