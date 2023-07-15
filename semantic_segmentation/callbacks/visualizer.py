import colorsys
import os
import pdb
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import skimage.io
import torch
from pytorch_lightning.callbacks import Callback
from torchvision import transforms


def blend_images(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
  """ Blend two images 

  Args:
      image1 (np.ndarray): 1st image of shape [H x W x 3]
      image2 (np.ndarray): 2nd image of shape [H x W x 3]
      alpha (float, optional): strength of blending for 1st image. Defaults to 0.5.

  Returns:
      np.ndarray: blended image of shape [H x W x 3]
  """
  assert alpha <= 1.0
  assert alpha >= 0.0
  assert image1.shape == image2.shape

  image1 = image1.astype(np.float32)
  image2 = image2.astype(np.float32)

  blended = alpha * image1 + (1 - alpha) * image2
  blended = np.round(blended).astype(np.uint8)

  return blended

class BasicVisualizer(ABC):
  """ Basic representation of a visualizer. """

  @abstractmethod
  def save_visualize_batch(self, path_to_dir: str, outputs: Dict[str, Any], batch: Dict[str, Any]) -> None:
    """ Save and visualize all predictions of a certain batch

    Args:
        path_to_dir (str): path to output directory
        outputs (Dict[str, Any]): model outputs
        batch (Dict[str, Any]): input batch
        epoch (int): current epoch
    """
    raise NotImplementedError

class InputImageVisualizer(BasicVisualizer):

  def __init__(self) -> None:
    self.tensor_to_pil_img = transforms.ToPILImage()

  def save_visualize_batch(self, path_to_dir: str, outputs: Dict[str, Any], batch: Dict[str, Any]) -> None:    
    path_to_dir = os.path.join(path_to_dir, 'input_images')
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir)
    
    with torch.no_grad():
      imgs = batch['input_image_before_norm']
      fnames = batch['fname']
      for input_image_before_norm, filename in zip(imgs, fnames):
        raw_pil_img = self.tensor_to_pil_img(input_image_before_norm)

        fpath = os.path.join(path_to_dir, filename)
        raw_pil_img.save(fpath)

class SemanticMapVisualizer(BasicVisualizer):
  """ Basic representation of a visualizer. """

  def __init__(self, classes_to_colors: Dict[int, List[int]]) -> None:
    self.classes_to_colors = classes_to_colors

  def save_visualize_batch(self, path_to_dir: str, outputs: Dict[str, Any], batch: Dict[str, Any]) -> None:
    path_to_dir = os.path.join(path_to_dir, 'semantic_maps')
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir)

    with torch.no_grad():
      logits = outputs['logits']
      fnames = batch['fname']
      for logits_single_img, filename in zip(logits, fnames):
        pred = torch.argmax(logits_single_img, dim=0)

        canvas = torch.zeros((3, pred.shape[0], pred.shape[1]))
        for class_id, class_color in self.classes_to_colors.items():
          mask = (pred == class_id)
          canvas[:, mask] = torch.Tensor(class_color).unsqueeze(1)

        canvas = canvas.cpu().numpy().astype(np.uint8)
        canvas = canvas.transpose(1, 2, 0)

        fpath = os.path.join(path_to_dir, filename)
        skimage.io.imsave(fpath, canvas, check_contrast=False)

class GroundTruthVisualizer(BasicVisualizer):
  """ Basic representation of a visualizer. """

  def __init__(self, classes_to_colors: Dict[int, List[int]]) -> None:
    self.classes_to_colors = classes_to_colors

  def save_visualize_batch(self, path_to_dir: str, outputs: Dict[str, Any], batch: Dict[str, Any]) -> None:
    path_to_dir = os.path.join(path_to_dir, 'ground_truth')
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir)

    with torch.no_grad():
      annos = batch['anno']
      fnames = batch['fname']
      for anno, filename in zip(annos, fnames):
        canvas = torch.zeros((3, anno.shape[0], anno.shape[1]))
        for class_id, class_color in self.classes_to_colors.items():
          mask = (anno == class_id)
          canvas[:, mask] = torch.Tensor(class_color).unsqueeze(1)

        canvas = canvas.cpu().numpy().astype(np.uint8)
        canvas = canvas.transpose(1, 2, 0)

        fpath = os.path.join(path_to_dir, filename)
        skimage.io.imsave(fpath, canvas, check_contrast=False)

class SemanticsOverlayCorrectIncorrectVisualizer(BasicVisualizer):
  """ Basic representation of a visualizer. """

  def __init__(self, classes_to_colors: Dict[int, List[int]]) -> None:
    self.tensor_to_pil_img = transforms.ToPILImage()
    self.classes_to_colors = classes_to_colors

  def save_visualize_batch(self, path_to_dir: str, outputs: Dict[str, Any], batch: Dict[str, Any]) -> None:
    path_to_dir = os.path.join(path_to_dir, 'semantic_overlays_correct_incorrect')
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir)

    with torch.no_grad():
      imgs = batch['input_image_before_norm']
      logits = outputs['logits']
      annos = batch['anno']
      fnames = batch['fname']
      for input_image_before_norm, logits_single_img, anno, filename in zip(imgs, logits, annos, fnames):
        pred = torch.argmax(logits_single_img, dim=0)
        
        canvas = torch.zeros_like(logits_single_img)
        for class_id, class_color in self.classes_to_colors.items():
          mask_correct_pred = (anno == class_id) & (pred == class_id) 
          mask_false_pred = (anno != class_id) & (pred == class_id) 
          
          # visualize correct predictions
          color_correct_tensor = torch.Tensor(class_color).unsqueeze(1).to(canvas.device)
          canvas[:, mask_correct_pred] = color_correct_tensor

          # visualize incorrect predictions
          h, s, v = colorsys.rgb_to_hsv(class_color[0], class_color[1], class_color[2])
          if v == 0:
            v = 127
          r, g, b = colorsys.hsv_to_rgb(h, s * 0.6, v)
          color_incorrect_tensor = torch.Tensor([r,g,b]).unsqueeze(1).to(canvas.device)
          canvas[:, mask_false_pred] = color_incorrect_tensor

        canvas = canvas.cpu().numpy().astype(np.uint8)
        canvas = canvas.transpose(1, 2, 0)

        raw_pil_img = self.tensor_to_pil_img(input_image_before_norm)
        image = np.array(raw_pil_img)
        overlay = blend_images(image, canvas)

        fpath = os.path.join(path_to_dir, filename)
        skimage.io.imsave(fpath, overlay, check_contrast=False)

class SemanticOverlayVisualizer(BasicVisualizer):
  """ Basic representation of a visualizer. """

  def __init__(self, classes_to_colors: Dict[int, List[int]]) -> None:
    self.tensor_to_pil_img = transforms.ToPILImage()
    self.classes_to_colors = classes_to_colors

  def save_visualize_batch(self, path_to_dir: str, outputs: Dict[str, Any], batch: Dict[str, Any]) -> None:
    path_to_dir = os.path.join(path_to_dir, 'semantic_overlays')
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir)

    with torch.no_grad():
      imgs = batch['input_image_before_norm']
      logits = outputs['logits']
      fnames = batch['fname']
      for input_image_before_norm, logits_single_img, filename in zip(imgs, logits, fnames):
        pred = torch.argmax(logits_single_img, dim=0)

        canvas = torch.zeros_like(logits_single_img)
        for class_id, class_color in self.classes_to_colors.items():
          mask = (pred == class_id)
          color_tensor = torch.Tensor(class_color).unsqueeze(1).to(canvas.device)
          canvas[:, mask] = color_tensor

        canvas = canvas.cpu().numpy().astype(np.uint8)
        canvas = canvas.transpose(1, 2, 0)

        raw_pil_img = self.tensor_to_pil_img(input_image_before_norm)
        image = np.array(raw_pil_img)
        overlay = blend_images(image, canvas, alpha=0.65)

        fpath = os.path.join(path_to_dir, filename)
        skimage.io.imsave(fpath, overlay, check_contrast=False)

def get_visualizers(cfg: Dict) -> List[BasicVisualizer]:
  visualizers = []

  try:
    cfg['visualizers'].keys()
  except KeyError:
    return visualizers
    
  for visualizer_name in cfg['visualizers'].keys():
    if visualizer_name == 'input_image_visualizer':
      visualizer = InputImageVisualizer()
      visualizers.append(visualizer)

    if visualizer_name == 'semantic_map_visualizer':
      classes_to_colors = cfg['visualizers'][visualizer_name]['classes_to_colors']
      visualizer = SemanticMapVisualizer(classes_to_colors)

      visualizers.append(visualizer)

    if visualizer_name == 'semantic_overlay_visualizer':
      classes_to_colors = cfg['visualizers'][visualizer_name]['classes_to_colors']
      visualizer = SemanticOverlayVisualizer(classes_to_colors)

      visualizers.append(visualizer)

    if visualizer_name == 'ground_truth_visualizer':
      classes_to_colors = cfg['visualizers'][visualizer_name]['classes_to_colors']
      visualizer = GroundTruthVisualizer(classes_to_colors)

      visualizers.append(visualizer)

    if visualizer_name == 'semantic_overlay_correct_incorrect_visualizer':
      classes_to_colors = cfg['visualizers'][visualizer_name]['classes_to_colors']
      visualizer = SemanticsOverlayCorrectIncorrectVisualizer(classes_to_colors)

      visualizers.append(visualizer)

  return visualizers


class VisualizerCallback(Callback):
  """ Callback to visualize semantic segmentation.
  """

  def __init__(self, visualizers: List[BasicVisualizer], vis_train_every_x_epochs: int = 1, vis_val_every_x_epochs: int = 1):
    """ Constructor.

    Args:
        vis_train_every_x_epochs (int): Frequency of train visualizations. Defaults to 1.
        vis_val_every_x_epochs (int): Frequency of val visualizations. Defaults to 1.
    """
    super().__init__()
    self.visualizers = visualizers
    self.vis_train_every_x_epochs = vis_train_every_x_epochs
    self.vis_val_every_x_epochs = vis_val_every_x_epochs

  def on_train_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    # visualize
    epoch = trainer.current_epoch
    if (epoch % self.vis_train_every_x_epochs) == 0 and (epoch != 0):
      path = os.path.join(trainer.log_dir, 'train', 'visualize', f'epoch-{epoch:06d}')

      for visualizer in self.visualizers:
        visualizer.save_visualize_batch(path, outputs, batch)

  def on_validation_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    # visualize
    epoch = trainer.current_epoch

    if ((epoch + 1) % self.vis_val_every_x_epochs) == 0 and (epoch != 0):
      path = os.path.join(trainer.log_dir, 'val', 'visualize', f'epoch-{epoch:06d}')

      for visualizer in self.visualizers:
        visualizer.save_visualize_batch(path, outputs, batch)

  def on_test_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    # visualize
    path = os.path.join(trainer.log_dir, 'visualize')

    for visualizer in self.visualizers:
      visualizer.save_visualize_batch(path, outputs, batch)

  # def on_predict_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
  #   # visualize
  #   path = os.path.join(trainer.log_dir, 'visualize')

  #   for visualizer in self.visualizers:
  #     visualizer.save_visualize_batch(path, outputs, batch)
