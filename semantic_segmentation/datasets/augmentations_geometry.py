""" Define a set of geometric data augmentations which can be applied to the input image and its corresponding annotations.

This is relevant for the task of semantic segmentation since the input image and its annotation need to be treated in the same way.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms


class GeometricDataAugmentation(ABC):
  """ General transformation which can be applied simultaneously to the input image and its corresponding anntations.
  """

  @abstractmethod
  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Apply a geometric transformation to a given image and its corresponding annotation.

    Args:
      image (torch.Tensor): input image to be transformed.
      anno (torch.Tensor): annotation to be transformed.

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: transformed image and its corresponding annotation
    """
    raise NotImplementedError

class RandomTranslationTransform(GeometricDataAugmentation):
  """ Randomly translate an image and its annoation.
  """

  def __init__(self, min_translation: int = 0, max_translation: int = 0):
    self.min_translation = min_translation
    self.max_translation = max_translation
    assert max_translation >= min_translation
    
  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    tx = random.randint(self.min_translation, self.max_translation + 1)
    ty = random.randint(self.min_translation, self.max_translation + 1)

    image_translated = functional.affine(image, angle=0, translate=[tx,ty], scale=1.0, shear=[0,0], interpolation=transforms.InterpolationMode.BILINEAR)
    anno_translated = functional.affine(anno, angle=0, translate=[tx,ty], scale=1.0, shear=[0,0], interpolation=transforms.InterpolationMode.NEAREST)

    return image_translated, anno_translated

class RandomRotationTransform(GeometricDataAugmentation):
  """ Randomly rotate an image and its annotation by a random angle.
  """
  def __init__(self, min_angle_in_deg :float = 0, max_angle_in_deg: float = 360):
    assert min_angle_in_deg >= 0
    assert max_angle_in_deg <= 360 
    assert min_angle_in_deg > max_angle_in_deg

    self.min_angle_in_deg = min_angle_in_deg
    self.max_angle_in_deg = max_angle_in_deg

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    angle = random.uniform(self.min_angle_in_deg, self.max_angle_in_deg)

    image_rotated = functional.rotate(image, angle=angle, interpolation=transforms.InterpolationMode.BILINEAR)
    anno_rotated = functional.rotate(anno, angle=angle, interpolation=transforms.InterpolationMode.NEAREST)

    return image_rotated, anno_rotated

class RandomGaussianRotationTransform(GeometricDataAugmentation):
  """ Randomly rotate an image and its annotation by a random angle specified by a gaussian.
  """
  def __init__(self, mean:float = 0.0, std:float = 0.0):
    """ Randomly rotate an image and its annotation by a random angle specified by a gaussian.

    Args:
        mean (float, optional): mean rotation angle in radians. Defaults to 0.0.
        std (float, optional): standard deviation of mean rotation angle in radians. Defaults to 0.0.
    """
    self.mean = mean
    self.std = std

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    angle = random.gauss(self.mean, self.std) # radians
    angle = angle * (180 / math.pi)

    image_rotated = functional.rotate(image, angle=angle, interpolation=transforms.InterpolationMode.BILINEAR)
    anno_rotated = functional.rotate(anno, angle=angle, interpolation=transforms.InterpolationMode.NEAREST)

    return image_rotated, anno_rotated

class RandomHorizontalFlipTransform(GeometricDataAugmentation):
  """ Apply random horizontal flipping.
  """

  def __init__(self, prob: float = 0.5):
    """ Apply random horizontal flipping.

    Args:
        prob (float, optional): probability of the image being flipped. Defaults to 0.5.
    """
    assert prob >= 0.0
    assert prob <= 1.0
    self.prob = prob

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    if random.random() <= self.prob:
      image_hz_flipped = functional.hflip(image)
      anno_hz_flipped = functional.hflip(anno)

      return image_hz_flipped, anno_hz_flipped
    else:
      return image, anno

class RandomVerticalFlipTransform(GeometricDataAugmentation):
  """ Apply random vertical flipping.
  """

  def __init__(self, prob: float = 0.5):
    """ Apply random vertical flipping.

    Args:
        prob (float, optional): probability of the image being flipped. Defaults to 0.5.
    """
    assert prob >= 0.0
    assert prob <= 1.0
    self.prob = prob

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    if random.random() <= self.prob:
      image_v_flipped = functional.vflip(image)
      anno_v_flipped = functional.vflip(anno)

      return image_v_flipped, anno_v_flipped
    else:
      return image, anno

class CenterCropTransform(GeometricDataAugmentation):
  """ Extract a patch from the image center.
  """

  def __init__(self, crop_height: Optional[int] = None, crop_width: Optional[int] = None):
    """ Set height and width of cropping region.

    Args:
        crop_height (Optional[int], optional): Height of cropping region. Defaults to None.
        crop_width (Optional[int], optional): Width of cropping region. Defaults to None.
    """
    self.crop_height = crop_height
    self.crop_width = crop_width

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    if (self.crop_height is None) or (self.crop_width is None):
      return image, anno

    img_chans, img_height, img_width = image.shape[:3]
    anno_chans = anno.shape[0]

    if (self.crop_width > img_width):
      raise ValueError("Width of cropping region must not be greather than img width")
    if (self.crop_height > img_height):
      raise ValueError("Height of cropping region must not be greather than img height.")

    image_cropped = functional.center_crop(image, [self.crop_height, self.crop_width])
    anno_cropped = functional.center_crop(anno, [self.crop_height, self.crop_width])

    assert image_cropped.shape[0] == img_chans, "Cropped image has an unexpected number of channels."
    assert image_cropped.shape[1] == self.crop_height, "Cropped image has not the desired size."
    assert image_cropped.shape[2] == self.crop_width, "Cropped image has not the desired width."

    assert anno_cropped.shape[0] == anno_chans, "Cropped anno has an unexpected number of channels."
    assert anno_cropped.shape[1] == self.crop_height, "Cropped anno has not the desired size."
    assert anno_cropped.shape[2] == self.crop_width, "Cropped anno has not the desired width."

    return image_cropped, anno_cropped

class RandomCropTransform(GeometricDataAugmentation):
  """ Extract a random patch from a given image and its corresponding annnotation.
    """

  def __init__(self, crop_height: Optional[int] = None, crop_width: Optional[int] = None):
    """ Set height and width of cropping region.

      Args:
          crop_height (Optional[int], optional): Height of cropping region. Defaults to None.
          crop_width (Optional[int], optional): Width of cropping region. Defaults to None.
      """
    self.crop_height = crop_height
    self.crop_width = crop_width

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    if (self.crop_height is None) or (self.crop_width is None):
      return image, anno

    img_chans, img_height, img_width = image.shape[:3]
    anno_chans = anno.shape[0]

    if (self.crop_width > img_width):
      raise ValueError(f"Width of cropping region must not be greather than img width: {self.crop_width} vs {img_width}")
    if (self.crop_height > img_height):
      raise ValueError(f"Height of cropping region must not be greather than img height: {self.crop_height} vs {img_height}.")

    max_x = img_width - self.crop_width
    x_start = random.randint(0, max_x)

    max_y = img_height - self.crop_height
    y_start = random.randint(0, max_y)

    assert (x_start + self.crop_width) <= img_width, "Cropping region (width) exceeds image dims."
    assert (y_start + self.crop_height) <= img_height, "Cropping region (height) exceeds image dims."

    image_cropped = functional.crop(image, y_start, x_start, self.crop_height, self.crop_width)
    anno_cropped = functional.crop(anno, y_start, x_start, self.crop_height, self.crop_width)

    assert image_cropped.shape[0] == img_chans, "Cropped image has an unexpected number of channels."
    assert image_cropped.shape[1] == self.crop_height, "Cropped image has not the desired size."
    assert image_cropped.shape[2] == self.crop_width, "Cropped image has not the desired width."

    assert anno_cropped.shape[0] == anno_chans, "Cropped anno has an unexpected number of channels."
    assert anno_cropped.shape[1] == self.crop_height, "Cropped anno has not the desired size."
    assert anno_cropped.shape[2] == self.crop_width, "Cropped anno has not the desired width."

    return image_cropped, anno_cropped

class MyRandomScaleTransform(GeometricDataAugmentation):
  """ Apply random overall scaling.
  """
  def __init__(self, min_scale: float, max_scale: float, prob: float = 0.5):
    """ Apply random overall scaling.

    Args:
        min_scale (float): minimum scaling factor.
        max_scale (float): maximum scaling factor.
        prob (float, optional): probability of the image being scaled. Defaults to 0.5.
    """
    assert prob >= 0.0
    assert prob <= 1.0
    assert min_scale > 0.0
    assert max_scale >= min_scale
    
    self.min_scale = min_scale
    self.max_scale = max_scale
    self.prob = prob

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    img_chans, img_height, img_width = image.shape[0], image.shape[1], image.shape[2]
    anno_chans = anno.shape[0]

    if random.random() < self.prob:
      random_scale = random.uniform(self.min_scale, self.max_scale)
      
      image = functional.affine(image, angle=0, translate=[0,0], scale=random_scale, shear=[0,0], interpolation=transforms.InterpolationMode.BILINEAR)
      anno = functional.affine(anno, angle=0, translate=[0,0], scale=random_scale, shear=[0,0], interpolation=transforms.InterpolationMode.NEAREST)
      
    assert img_chans == image.shape[0]
    assert img_height == image.shape[1]
    assert img_width == image.shape[2]

    assert anno_chans == anno.shape[0]
    assert img_height == anno.shape[1]
    assert img_width == anno.shape[2]

    return image, anno

class MyRandomAspectRatioTransform(GeometricDataAugmentation):
  """ Rescaling the image and its annotation in one dimension (width or height) to diversify the aspect ratio.
  """
  def __init__(self, min_scale: float, max_scale: float, prob: float = 0.5):
    """ Rescaling the image and its annotation in one dimension (width or height) to diversify the aspect ratio.

    This augmentation is proposed in: http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera18iv.pdf

    Args:
        min_scale (float): minimum scaling factor.
        max_scale (float): maximum scaling factor.
        prob (float, optional): probability of the image being scaled. Defaults to 0.5.
    """
    assert prob >= 0.0
    assert prob <= 1.0
    assert min_scale > 0.0
    assert max_scale >= min_scale
    
    self.min_scale = min_scale
    self.max_scale = max_scale
    self.prob = prob

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    img_chans, img_height, img_width = image.shape[0], image.shape[1], image.shape[2]
    anno_chans = anno.shape[0]

    if random.random() < self.prob:
      random_scale = random.uniform(self.min_scale, self.max_scale)
      if random.random() > 0.5:
        # specify new height
        img_height_new = int(round(img_height * random_scale))
        img_width_new = img_width # unchanged
      else:
        # specify new width
        img_height_new = img_height # unchanged
        img_width_new = int(round(img_width * random_scale))

      image = functional.resize(image, size=[img_height_new, img_width_new], interpolation=transforms.InterpolationMode.BILINEAR)
      anno = functional.resize(anno, size=[img_height_new, img_width_new], interpolation=transforms.InterpolationMode.NEAREST)
    
    assert img_chans == image.shape[0]
    assert anno_chans == anno.shape[0]

    return image, anno

class MyRandomShearTransform(GeometricDataAugmentation):
  """ Apply random shear along x- and y-axis.
  """
  def __init__(self, max_x_shear: float, max_y_shear: float, prob: float = 0.5):
    """ Apply random shear.

    Args:
        x_shear (float): maximum shear along x-axis (in degrees).
        y_shear (float): maximum shear along y-axis (in degrees).
        prob (float, optional): probability of the image being sheared. Defaults to 0.5.
    """
    assert prob >= 0.0
    assert prob <= 1.0
    
    self.max_x_shear = max_x_shear
    self.max_y_shear = max_y_shear
    self.prob = prob

  def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # dimension of each input should be identical
    assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
    assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

    img_chans, img_height, img_width = image.shape[0], image.shape[1], image.shape[2]
    anno_chans = anno.shape[0]

    if random.random() < self.prob:
      x_shear = random.uniform(-self.max_x_shear, self.max_x_shear)
      
      image = functional.affine(image, angle=0, translate=[0,0], scale=1.0, shear=[x_shear,0], interpolation=transforms.InterpolationMode.BILINEAR)
      anno = functional.affine(anno, angle=0, translate=[0,0], scale=1.0, shear=[x_shear,0], interpolation=transforms.InterpolationMode.NEAREST)
    
    if random.random() < self.prob:
      y_shear = random.uniform(-self.max_y_shear, self.max_y_shear)
      
      image = functional.affine(image, angle=0, translate=[0,0], scale=1.0, shear=[0, y_shear], interpolation=transforms.InterpolationMode.BILINEAR)
      anno = functional.affine(anno, angle=0, translate=[0,0], scale=1.0, shear=[0, y_shear], interpolation=transforms.InterpolationMode.NEAREST)

    assert img_chans == image.shape[0]
    assert img_height == image.shape[1]
    assert img_width == image.shape[2]

    assert anno_chans == anno.shape[0]
    assert img_height == anno.shape[1]
    assert img_width == anno.shape[2]

    return image, anno

# class MyPaddingTransform(GeometricDataAugmentation):
#   """ Apply padding.
#   """
#   def __init__(self, padding_size: int, mode: str):
#     """ Apply padding.

#     Args:
#         padding_size (int): amount of padding on all sides of the imagee.
#         mode (str): 'edge', 'reflect' or 'symmetric'
#     """
#     assert padding_size < 0
#     assert (padding_size % 2) == 0
#     assert mode in ['edge', 'reflect', 'symmetric']

#     self.padding_size = padding_size
#     self.mode = mode

#   def __call__(self, image: torch.Tensor, anno: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     # dimension of each input should be identical
#     assert image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
#     assert image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

#     img_chans, img_height, img_width = image.shape[0], image.shape[1], image.shape[2]
#     anno_chans = anno.shape[0]

#     image = functional.pad(image, [self.padding_size], padding_mode=self.mode)
#     anno = functional.pad(anno, [self.padding_size], padding_mode=self.mode)
    
#     assert img_chans == image.shape[0]
#     assert img_height == (image.shape[1] + (2 * self.padding_size))
#     assert img_width == (image.shape[2] + (2 * self.padding_size))

#     assert anno_chans == anno.shape[0]
#     assert img_height == (anno.shape[1] + (2 * self.padding_size))
#     assert img_width == (anno.shape[2] + (2 * self.padding_size))

#     return image, anno


def get_geometric_augmentations(cfg, stage: str) -> List[GeometricDataAugmentation]:
  assert stage in ['train', 'val', 'test', 'predict']

  geometric_augmentations = []

  for tf_name in cfg[stage]['geometric_data_augmentations'].keys():
    # if tf_name == 'pad':
    #   padding_size = cfg[stage]['geometric_data_augmentations'][tf_name]['padding_size']
    #   padding_mode = cfg[stage]['geometric_data_augmentations'][tf_name]['padding_mode']
    #   augmentor = MyPaddingTransform(padding_size, padding_mode)
    #   geometric_augmentations.append(augmentor)

    if tf_name == 'random_hflip':
      augmentor = RandomHorizontalFlipTransform()
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_vflip':
      augmentor = RandomVerticalFlipTransform()
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_translation':
      min_translation = cfg[stage]['geometric_data_augmentations'][tf_name]['min_translation']
      max_translation = cfg[stage]['geometric_data_augmentations'][tf_name]['max_translation']
      augmentor = RandomTranslationTransform(min_translation, max_translation)
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_rotate_gaussian':
      mean = cfg[stage]['geometric_data_augmentations'][tf_name]['mean']
      std = cfg[stage]['geometric_data_augmentations'][tf_name]['std']
      augmentor = RandomGaussianRotationTransform(mean, std)
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_rotate':
      min_angle_in_deg = cfg[stage]['geometric_data_augmentations'][tf_name]['min_angle_in_deg']
      max_angle_in_deg = cfg[stage]['geometric_data_augmentations'][tf_name]['max_angle_in_deg']
      augmentor = RandomRotationTransform(min_angle_in_deg, max_angle_in_deg)
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_shear':
      max_x_shear = cfg[stage]['geometric_data_augmentations'][tf_name]['max_x_shear']
      max_y_shear = cfg[stage]['geometric_data_augmentations'][tf_name]['max_y_shear']
      augmentor = MyRandomShearTransform(max_x_shear, max_y_shear)
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_scale':
      min_scale = cfg[stage]['geometric_data_augmentations'][tf_name]['min_scale']
      max_scale = cfg[stage]['geometric_data_augmentations'][tf_name]['max_scale']
      augmentor = MyRandomScaleTransform(min_scale, max_scale)
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_aspect_ratio':
      min_ap_scale = cfg[stage]['geometric_data_augmentations'][tf_name]['min_scale']
      max_ap_scale = cfg[stage]['geometric_data_augmentations'][tf_name]['max_scale']
      augmentor = MyRandomAspectRatioTransform(min_ap_scale, max_ap_scale)
      geometric_augmentations.append(augmentor)

    if tf_name == 'center_crop':
      crop_height = cfg[stage]['geometric_data_augmentations'][tf_name]['height']
      crop_width = cfg[stage]['geometric_data_augmentations'][tf_name]['width']
      augmentor = CenterCropTransform(crop_height, crop_width)
      geometric_augmentations.append(augmentor)

    if tf_name == 'random_crop':
      crop_height = cfg[stage]['geometric_data_augmentations'][tf_name]['height']
      crop_width = cfg[stage]['geometric_data_augmentations'][tf_name]['width']
      augmentor = RandomCropTransform(crop_height, crop_width)
      geometric_augmentations.append(augmentor)

  assert len(geometric_augmentations) == len(cfg[stage]['geometric_data_augmentations'].keys())
  return geometric_augmentations