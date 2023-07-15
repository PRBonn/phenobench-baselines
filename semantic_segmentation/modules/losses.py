import math
import pdb
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

# --------------------------------------- CROSS ENTROPY ---------------------------------------
  
class CrossEntropy(nn.Module):

  def __init__(self, weights: Optional[List] = None):
    super().__init__()

    if weights is not None:
      self.weights = torch.Tensor(weights)
    else:
      self.weights = None

  def forward(self, inputs: torch.Tensor, target: torch.Tensor, mode: str,
              mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ Compute cross entropy loss.

    Args:
        inputs (torch.Tensor): Unnormalized input tensor (logits) of shape [B x C x H x W]
        target (torch.Tensor): Ground-truth target tensor of shape [B x H x W]
        mode (str): train, val, or test
        mask_keep (Optional[torch.Tensor], optional): Mask of pixels of shape [B x H x W] which should be kept during loss computation (1 := keep, 0 := ignore). Defaults to None := keep all.

    Returns:
        torch.Tensor: loss value (scalar)
    """
    assert mode in ['train', 'val', 'test']

    if mask_keep is not None:
      target[mask_keep == False] = 0

    # get the number of classes and device
    batch_size, num_classes, height, width = inputs.shape
    input_device = inputs.device

    # convert logits to softmax probabilities
    probs = functional.softmax(inputs, dim=1)  # [N x n_classes x H x W]
    del inputs

    # apply one-hot encoding to ground truth annotations
    target_one_hot = to_one_hot(target, int(num_classes))  # [N x n_classes x H x W]
    target_one_hot = target_one_hot.bool()
    del target

    # prepare to ignore certain pixels which should not be considered during loss computation
    if mask_keep is None:
      # consider all pixels to compute the loss
      mask_keep = torch.ones((batch_size, 1, height, width), dtype=torch.bool, device=input_device) # [N x 1 x H x W]
    else:
      # get the dimension correctly
      mask_keep = mask_keep.unsqueeze(1)  # [N x 1 x H x W]
 
    # set ignore pixels to false
    target_one_hot = target_one_hot * mask_keep

    # if mode == 'train':
    #   for i in range(batch_size):
    #     # randomly mask soil annotations s.t. they are not considered in the loss
    #     soil_mask = target_one_hot[i,0,:,:]
    #     n_soil_pixel = float(torch.sum(soil_mask))
    #     n_soil_pixel_to_remove = int(0.95 * n_soil_pixel) # <- remove 95% of all soil annotations
        
    #     row_idx, col_idx = list(torch.where(soil_mask == 1))
    #     assert len(row_idx) == n_soil_pixel
    #     rand_perm = torch.randperm(len(row_idx))[:n_soil_pixel_to_remove]

    #     row_idx = row_idx[rand_perm]
    #     col_idx = col_idx[rand_perm]
    #     soil_mask[(row_idx, col_idx)] = 0 # <- set these soil annotations to ignore
    #     assert torch.sum(soil_mask) != n_soil_pixel

    #     # randomly mask crop annotations s.t. they are not considered in the loss
    #     crop_mask = target_one_hot[i,1,:,:]
    #     n_crop_pixel = float(torch.sum(crop_mask))
    #     n_crop_pixel_to_remove = int(0.25 * n_crop_pixel) # <- remove 30% of all crop annotations
        
    #     if n_crop_pixel_to_remove > 0:
    #       row_idx, col_idx = list(torch.where(crop_mask == 1))
    #       assert len(row_idx) == n_crop_pixel
    #       rand_perm = torch.randperm(len(row_idx))[:n_crop_pixel_to_remove]

    #       row_idx = row_idx[rand_perm]
    #       col_idx = col_idx[rand_perm]
    #       crop_mask[(row_idx, col_idx)] = 0 # <- set these crop annotations to ignore
    #       assert torch.sum(crop_mask) != n_crop_pixel

    #     # randomly mask weed annotations s.t. they are not considered in the loss
    #     weed_mask = target_one_hot[i,2,:,:]
    #     n_weed_pixel = float(torch.sum(weed_mask))
    #     n_weed_pixel_to_remove = int(0.05 * n_weed_pixel) # <- remove 5% of all crop annotations
    #     if n_weed_pixel_to_remove > 0:
    #       row_idx, col_idx = list(torch.where(weed_mask == 1))
    #       assert len(row_idx) == n_weed_pixel
    #       rand_perm = torch.randperm(len(row_idx))[:n_weed_pixel_to_remove]

    #       row_idx = row_idx[rand_perm]
    #       col_idx = col_idx[rand_perm]
    #       weed_mask[(row_idx, col_idx)] = 0 # <- set these crop annotations to ignore
    #       assert torch.sum(weed_mask) != n_weed_pixel

    # gather the predicited probabilities of each ground truth category
    probs_gathered = probs[target_one_hot]  # M = N * (H * W) entries

    # make sure that probs are numerically stable when passed to log function: log(0) -> inf
    probs_gathered = torch.clip(probs_gathered, 1e-12, 1.0)

    # compute loss
    losses = -torch.log(probs_gathered)  # M = N * (H * W) entries
    del probs_gathered

    assert losses.shape[0] == torch.sum(mask_keep)
    del mask_keep
    
    # create weight matrix
    if self.weights is not None:
      if input_device != self.weights.device:
        self.weights = self.weights.to(input_device)

      weight_matrix = (target_one_hot.permute(0, 2, 3, 1) * self.weights).permute(0, 3, 1, 2)  # [N x n_classes x H x W]
      weights_gathered = weight_matrix[target_one_hot]  # M = N * (H * W) entries
      assert torch.all(weights_gathered > 0)

      # compute weighted loss for each prediction
      losses *= weights_gathered

    return torch.mean(losses)

# --------------------------------------- Generalized-Jensen–Shannon Divergence -------------------------
def gjs_div_loss(p1_logits: torch.Tensor, p2_logits: torch.Tensor, p3_logits: torch.Tensor) -> torch.Tensor:
  
  p1_probs = functional.softmax(p1_logits, dim=1) # [B x C x H x W]
  p2_probs = functional.softmax(p2_logits, dim=1)
  p3_probs = functional.softmax(p3_logits, dim=1)

  m_probs = (p1_probs + p2_probs + p3_probs) / 3.0 # [B x C x H x W]
  m_probs = torch.clamp(m_probs, 1e-7, 1.0).log()

  loss1 = functional.kl_div(input=m_probs, target=p1_probs, reduction='none', log_target=False) # [B x C x H x W]
  loss1 = torch.sum(loss1, dim=1) # [B x H x W]

  loss2 = functional.kl_div(input=m_probs, target=p2_probs, reduction='none', log_target=False)
  loss2 = torch.sum(loss2, dim=1) # [B x H x W]

  loss3 = functional.kl_div(input=m_probs, target=p3_probs, reduction='none', log_target=False)
  loss3 = torch.sum(loss3, dim=1) # [B x H x W]

  loss = (loss1 + loss2 + loss3) / 3.0 # [B x H x W]
  loss = loss.mean()

  return loss

  # p1_probs = functional.softmax(p1_logits, dim=1)
  # p1_probs = torch.clamp(p1_probs, 1e-12, 1)

  # p2_probs = functional.softmax(p2_logits, dim=1)
  # p2_probs = torch.clamp(p2_probs, 1e-12, 1)

  # p3_probs = functional.softmax(p3_logits, dim=1)
  # p3_probs = torch.clamp(p3_probs, 1e-12, 1)

  # m_probs = (p1_probs + p2_probs + p3_probs) / 3 # be aware of floating-point arithmetic (1/3)

  # kl_p1_m = p1_probs * torch.log(p1_probs / m_probs)  # [B, C, H, W]
  # kl_p1_m = torch.sum(kl_p1_m, dim=1)  # [B, H, W]

  # kl_p2_m = p2_probs * torch.log(p2_probs / m_probs)  # [B, C, H, W]
  # kl_p2_m = torch.sum(kl_p2_m, dim=1)  # [B, H, W]

  # kl_p3_m = p3_probs * torch.log(p3_probs / m_probs)  # [B, C, H, W]
  # kl_p3_m = torch.sum(kl_p3_m, dim=1)  # [B, H, W]

  # gjs = (kl_p1_m + kl_p2_m + kl_p3_m) / 3
  # gjs[gjs < 0] = 0.0 # we perform this operation to prevent issues due to floating-point rounding erros

  # loss = torch.mean(gjs)

  # assert loss >= 0, f"Invalid loss for js divergence: {loss}"  # lower bound
  # assert loss <= math.log(3), f"Invalid loss for gjs divergence: {loss}"  # upper bound

  # return loss

# --------------------------------------- Jensen–Shannon Divergence -------------------------
def js_div_loss(p_logits: torch.Tensor, q_logits: torch.Tensor, mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
  """ Compute Jensen–Shannon divergence.

  Args:
      p (torch.Tensor): 1st distributions of shape [B x C x H x W]
      q (torch.Tensor): 2nd distributions of shape [B x C x H x W]
      mask_keep (Optional[torch.Tensor], optional): Mask of pixels of shape [B x H x W] which should be kept during loss computation (1 := keep, 0 := ignore). Defaults to None.

  Returns:
      torch.Tensor: loss value
  """
  p_probs = functional.softmax(p_logits, dim=1)
  q_probs = functional.softmax(q_logits, dim=1)
  m_probs = 0.5 * (p_probs + q_probs)

  p_probs = torch.clamp(p_probs, 1e-12, 1)
  q_probs = torch.clamp(q_probs, 1e-12, 1)
  m_probs = torch.clamp(m_probs, 1e-12, 1)

  kl_p_m = p_probs * torch.log(p_probs / m_probs) # [B, C, H, W]
  kl_p_m = torch.sum(kl_p_m, dim=1)  # [B, H, W]

  kl_q_m = q_probs * torch.log(q_probs / m_probs) # [B, C, H, W]
  kl_q_m = torch.sum(kl_q_m, dim=1) # [B, H, W]

  # compute Jensen–Shannon divergence
  js_p_q = (0.5 * kl_p_m) + (0.5 * kl_q_m) # [B, H, W]

  if mask_keep is not None:
    js_p_q = js_p_q[mask_keep] # [M] where M is number of pixel which should be kept according to mask_keep (i.e. torch.sum(mask_keep) = M)

  loss = torch.mean(js_p_q)

  assert loss >= 0, f"Invalid loss for js divergence: {loss}" # lower bound
  assert loss <= math.log(2), f"Invalid loss for js divergence: {loss}" # upper bound

  return loss

# --------------------------------------- Kullback–Leibler Divergence -------------------------

def kl_div_loss(x_logits_pred: torch.Tensor, x_logits_true: torch.Tensor, mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
  """ Compute KL-Divergence.

  There are different ways to compute the Kullback-Leibler Divergence.
  We refer to https://machinelearningmastery.com/divergence-between-probability-distributions/ for more information.

  Args:
      x_logits_pred (torch.Tensor): Source distributions of shape [B x C x H x W]
      x_logits_true (torch.Tensor): Target distributions of shape [B x C x H x W]
      mask_keep (Optional[torch.Tensor], optional): Mask of pixels of shape [B x H x W] which should be kept during loss computation (1 := keep, 0 := ignore). Defaults to None.

  Returns:
      torch.Tensor: loss value
  """
  x_pred = functional.softmax(x_logits_pred, dim=1)
  x_true = functional.softmax(x_logits_true, dim=1)

  x_pred = torch.clamp(x_pred, 1e-12, 1)
  x_true = torch.clamp(x_true, 1e-12, 1)

  loss = x_true * torch.log((x_true)/(x_pred)) # [B, C, H, W]
  loss = torch.sum(loss, dim=1) # [B, H, W]

  if mask_keep is not None:
    loss = loss[mask_keep] # [M] where M is number of pixel which should be kept according to mask_keep (i.e. torch.sum(mask_keep) = M)

  loss = torch.mean(loss)

  assert loss >= 0, f"Invalid loss for kl divergence: {loss}"

  return loss

# --------------------------------------- UTILS -----------------------------------------------
def get_div_loss_weight(current_epoch: int, max_epochs: int) -> float:
  """ We increase the weight of the divergence losses linearly with increasig number of epochs.

  Note, that the weight is alway in [0.0, 1.0].
  We increase the weight until a predefined number of epochs is reached and set it to 1.0 afterwards.

  Args:
      current_epoch (int): current epoch
      max_epochs (int): increase the weight linearly until max_epochs is reached - afterwards we set the weight to 1.0

  Returns:
      float: value in [0.0, 1.0]
  """
  max_weight = 12.0 # 15.0
  if current_epoch > (max_epochs):
    return 1.0 * max_weight
  
  weight = ((1 / max_epochs) * current_epoch) * max_weight
  assert weight >= 0.0
  assert weight <= max_weight

  return weight

def to_one_hot(tensor: torch.Tensor, n_classes: int) -> torch.Tensor:
  """ Convert tensor to its one hot encoded version.

  Props go to https://github.com/PRBonn/bonnetal/blob/master/train/common/onehot.py

  Args:
      tensor (torch.Tensor): ground truth tensor of shape [N x H x W]
      n_classes (int): number of classes

  Returns:
      torch.Tensor: one hot tensor of shape [N x n_classes x H x W]
  """
  if len(tensor.size()) == 1:
    b = tensor.size(0)
    if tensor.is_cuda:
      one_hot = torch.zeros(b, n_classes, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
    else:
      one_hot = torch.zeros(b, n_classes).scatter_(1, tensor.unsqueeze(1), 1)
  elif len(tensor.size()) == 2:
    n, b = tensor.size()
    if tensor.is_cuda:
      one_hot = torch.zeros(n, n_classes, b, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
    else:
      one_hot = torch.zeros(n, n_classes, b).scatter_(1, tensor.unsqueeze(1), 1)
  elif len(tensor.size()) == 3:
    n, h, w = tensor.size()
    if tensor.is_cuda:
      one_hot = torch.zeros(n, n_classes, h, w, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
    else:
      one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.unsqueeze(1), 1)
  return one_hot

def get_criterion(cfg) -> nn.Module:
  loss_name = cfg['train']['loss']

  if loss_name == 'xentropy':
    weights = cfg['train']['class_weights']

    return CrossEntropy(weights)
