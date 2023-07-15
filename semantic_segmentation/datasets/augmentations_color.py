""" Define a set of geometric color data augmentations which can be applied to the input image.

Note that the annotations is not affected in any way.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
import torch
import torchvision.transforms.functional as functional
from numpy import imag
from torchvision import transforms

import datasets.augmentations_augmix as augmentations_augmix


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  """ Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
  """
  if not isinstance(image, torch.Tensor):
    raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

  if len(image.shape) < 3 or image.shape[-3] != 3:
    raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

  max_rgb, argmax_rgb = image.max(-3)
  min_rgb, argmin_rgb = image.min(-3)
  deltac = max_rgb - min_rgb

  v = max_rgb
  s = deltac / (max_rgb + eps)

  deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
  rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

  h1 = (bc - gc)
  h2 = (rc - bc) + 2.0 * deltac
  h3 = (gc - rc) + 4.0 * deltac

  h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
  h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
  h = (h / 6.0) % 1.0
  h = 2. * math.pi * h  # we return 0/2pi output

  return torch.stack((h, s, v), dim=-3)

def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
  """ Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
  if not isinstance(image, torch.Tensor):
    raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

  if len(image.shape) < 3 or image.shape[-3] != 3:
    raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

  h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
  s: torch.Tensor = image[..., 1, :, :]
  v: torch.Tensor = image[..., 2, :, :]

  hi: torch.Tensor = torch.floor(h * 6) % 6
  f: torch.Tensor = ((h * 6) % 6) - hi
  one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
  p: torch.Tensor = v * (one - s)
  q: torch.Tensor = v * (one - f * s)
  t: torch.Tensor = v * (one - (one - f) * s)

  hi = hi.long()
  indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
  out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
  out = torch.gather(out, -3, indices)

  return out

class MyColorAugmentation(ABC):
  """ General representation of a color augmentation.
  """
  @abstractmethod
  def __call__(self, image:torch.Tensor) -> torch.Tensor:
    """ Apply a certain color augmentation to an input image.

    Args:
        image (torch.Tensor): input image of shape [C x H x W]

    Returns:
        torch.Tensor: augmented image
    """
    raise NotImplementedError

class RandomGlobalSaturation(MyColorAugmentation):
  def __init__(self, min_saturation_factor: float = 1, max_saturation_factor: float = 1, prob: float = 0.5):
    assert min_saturation_factor >= 0
    assert max_saturation_factor >= min_saturation_factor

    assert prob >= 0.0
    assert prob <= 1.0

    self.min_saturation_factor = min_saturation_factor
    self.max_saturation_factor = max_saturation_factor
    self.prob = prob

  def __call__(self, image: torch.Tensor) -> torch.Tensor:
    """ See: https://pytorch.org/vision/0.8/transforms.html#functional-transforms
    """
    if random.random() <= self.prob:
      rand_saturation_factor = random.uniform(self.min_saturation_factor, self.max_saturation_factor)
      image = functional.adjust_saturation(image, saturation_factor=rand_saturation_factor)
      
    assert torch.all(image >= 0)
    assert torch.all(image <= 1.0)

    return image

class RandomGlobalHue(MyColorAugmentation):
  def __init__(self, min_hue_factor: float = 0.0, max_hue_factor: float = 0.0, prob: float = 0.5):
    assert min_hue_factor >= -0.5
    assert max_hue_factor <= 0.5
    assert max_hue_factor >= min_hue_factor

    assert prob >= 0.0
    assert prob <= 1.0

    self.min_hue_factor = min_hue_factor
    self.max_hue_factor = max_hue_factor
    self.prob = prob

  def __call__(self, image: torch.Tensor) -> torch.Tensor:
    """ See: https://pytorch.org/vision/0.8/transforms.html#functional-transforms
    """
    if random.random() <= self.prob:
      rand_hue_factor = random.uniform(self.min_hue_factor, self.max_hue_factor)
      image = functional.adjust_hue(image, hue_factor=rand_hue_factor)
      
    assert torch.all(image >= 0)
    assert torch.all(image <= 1.0)

    return image

class RandomGlobalBrightness(MyColorAugmentation):
  def __init__(self, min_brightness_factor: float = 1, max_brightness_factor: float = 1, prob: float = 0.5):
    assert min_brightness_factor >= 0
    assert min_brightness_factor <= max_brightness_factor

    assert prob >= 0.0
    assert prob <= 1.0

    self.min_brightness_factor = min_brightness_factor
    self.max_brightness_factor = max_brightness_factor
    self.prob = prob

  def __call__(self, image: torch.Tensor) -> torch.Tensor:
    """ See: https://pytorch.org/vision/0.8/transforms.html#functional-transforms
    """
    if random.random() <= self.prob:
      rand_brightness_factor = random.uniform(self.min_brightness_factor, self.max_brightness_factor)
      image = functional.adjust_brightness(image, brightness_factor=rand_brightness_factor)
      
    assert torch.all(image >= 0)
    assert torch.all(image <= 1.0)

    return image

class RandomGlobalContrast(MyColorAugmentation):
  def __init__(self, min_contrast_factor: float = 1, max_contrast_factor: float = 1, prob: float = 0.5):
    assert min_contrast_factor >= 0
    assert min_contrast_factor <= max_contrast_factor

    assert prob >= 0.0
    assert prob <= 1.0

    self.min_contrast_factor = min_contrast_factor
    self.max_contrast_factor = max_contrast_factor
    self.prob = prob

  def __call__(self, image: torch.Tensor) -> torch.Tensor:
    """ See: https://pytorch.org/vision/0.8/transforms.html#functional-transforms
    """
    if random.random() <= self.prob:
      rand_contrast_factor = random.uniform(self.min_contrast_factor, self.max_contrast_factor)
      image = functional.adjust_contrast(image, contrast_factor=rand_contrast_factor)

    assert torch.all(image >= 0)
    assert torch.all(image <= 1.0)
      
    return image

class RandomGaussianColorJitter(MyColorAugmentation):
  """ Add small random noise to each RGB pixel to obtain invariance against some camera distortions.

  This augmentation is proposed in: http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera18iv.pdf
  """
  def __init__(self, mean: float = 0.0, std: float = 0.0, prob: float = 0.5):
    self.mean = mean
    self.std = std
    self.prob = prob

    assert prob >= 0.0
    assert prob <= 1.0

  def __call__(self, image: torch.Tensor) -> torch.Tensor:
    if random.random() <= self.prob:
      image_gaussian_noise = (torch.randn(image.shape) * self.std) + self.mean
      image = image + image_gaussian_noise
      image = torch.clamp(image, 0.0, 1.0)
      
    assert torch.all(image >= 0)
    assert torch.all(image <= 1.0)

    return image

class AugMixAugmentator(MyColorAugmentation):
  """ Perform augmentations proposed by https://github.com/google-research/augmix

  Note: 
  Technically, this operation may also include some geometric transformation.
  However, we find it convenient to put this augmentation method here since the
  operations involved do not change the image dimensions - accordingly, we do
  not need to transform the annotations in the same way (i.e. the semantic
  information is preserved).
  """
  def __init__(self):
    # We use the default parameters used for imagenet - see source code:
    
    self.aug_prob_coeff: float = 1.0  # Probability distribution coefficients
    self.mixture_width: int = 3  # Number of augmentation chains to mix per augmented example
    self.mixture_depth: int = -1  # Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]
    self.aug_severity: int = 1  # Severity of base augmentation operators

    self.tensor_to_pil = transforms.ToPILImage()
    self.pil_image_to_tensor = transforms.ToTensor()

  def __call__(self, image: torch.Tensor) -> torch.Tensor:
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmentations_augmix.augmentations_all # augmentations.augmentations

    ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
    m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

    mix = torch.zeros_like(image)
    for i in range(self.mixture_width):
      image_aug = self.tensor_to_pil(image).copy()
      depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
      for _ in range(depth):
        op = np.random.choice(aug_list)
        image_aug = op(image_aug, self.aug_severity)
      # Preprocessing commutes since all coefficients are convex
      mix += ws[i] * self.pil_image_to_tensor(image_aug)

    mixed = (1 - m) * image + m * mix
    return mixed

def get_color_augmentations(cfg, stage: str) -> List[Callable]:
  assert stage in ['train', 'val', 'test', 'predict']

  color_augmentations = []
  try:
    cfg[stage]['color_data_augmentations'].keys()
  except KeyError:
    return color_augmentations

  for color_aug_name in cfg[stage]['color_data_augmentations'].keys():
    if color_aug_name == 'random_global_brightness':
      min_brightness_factor = cfg[stage]['color_data_augmentations'][color_aug_name]['min_brightness_factor']
      max_brightness_factor = cfg[stage]['color_data_augmentations'][color_aug_name]['max_brightness_factor']
      augmentor = RandomGlobalBrightness(min_brightness_factor, max_brightness_factor)
      color_augmentations.append(augmentor)

    if color_aug_name == 'random_global_contrast':
      min_contrast_factor = cfg[stage]['color_data_augmentations'][color_aug_name]['min_contrast_factor']
      max_contrast_factor = cfg[stage]['color_data_augmentations'][color_aug_name]['max_contrast_factor']
      augmentor = RandomGlobalContrast(min_contrast_factor, max_contrast_factor)
      color_augmentations.append(augmentor)

    if color_aug_name == 'random_global_saturation':
      min_saturation_factor = cfg[stage]['color_data_augmentations'][color_aug_name]['min_saturation_factor']
      max_saturation_factor = cfg[stage]['color_data_augmentations'][color_aug_name]['max_saturation_factor']
      augmentor = RandomGlobalSaturation(min_saturation_factor, max_saturation_factor)
      color_augmentations.append(augmentor)

    if color_aug_name == 'random_gaussian_color_jitter':
      mean = cfg[stage]['color_data_augmentations'][color_aug_name]['mean']
      std = cfg[stage]['color_data_augmentations'][color_aug_name]['std']
      augmentor = RandomGaussianColorJitter(mean, std)
      color_augmentations.append(augmentor)

    if color_aug_name == 'random_global_hue':
      min_hue_factor = cfg[stage]['color_data_augmentations'][color_aug_name]['min_hue_factor']
      max_hue_factor = cfg[stage]['color_data_augmentations'][color_aug_name]['max_hue_factor']
      augmentor = RandomGlobalHue(min_hue_factor, max_hue_factor)
      color_augmentations.append(augmentor)

    if color_aug_name == 'augmix':
      augmentor = AugMixAugmentator()
      color_augmentations.append(augmentor)

  assert len(color_augmentations) == len(cfg[stage]['color_data_augmentations'].keys())

  return color_augmentations
