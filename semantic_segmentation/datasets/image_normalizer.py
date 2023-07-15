from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from torchvision import transforms
import datasets.common as common


class ImageNormalizer(ABC):

  @abstractmethod
  def normalize(self, image: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


class NoImageNormalizer(ImageNormalizer):
  """ Do not perform any normalization the the image
  """

  def __init__(self) -> None:
    pass

  def normalize(self, image: torch.Tensor) -> torch.Tensor:
    return image


class GlobalImageNormalizer(ImageNormalizer):
  """ Normalize an image by a global mean and a global std.
  """

  def __init__(self, rgb_means: List[float], rgb_stds: List[float]) -> None:
    """ Define the global rgb means and stds.

    We assume that mean and stds are provided channel-wise and
    are computed based on normalized images: img_normalized = image / 255.
    Thus, each mean is in [0, 1].

    Args:
        rgb_means (List[float]): [red_mean, green_mean, blue_mean]
        rgb_stds (List[float]): [red_std, green_std, blue_std]
    """
    self.rgb_means = rgb_means
    assert any([value <= 1.0 for value in self.rgb_means])
    assert any([value >= 0.0 for value in self.rgb_means])

    self.rgb_stds = rgb_stds
    assert any([value >= 0.0 for value in self.rgb_stds])
    self.normalizer = transforms.Normalize(mean=self.rgb_means, std=self.rgb_stds)

  def normalize(self, image: torch.Tensor) -> torch.Tensor:
    assert torch.all(image <= 1.0)
    assert torch.all(image >= 0.0)

    image_normalized = self.normalizer(image)

    return image_normalized


class SingleImageNormalizer(ImageNormalizer):
  """ Normalize an image by its mean and std such that it has zero mean and a standard deviation of 1.
  """

  def __init__(self):
    pass

  def transform_torch_img_channel(self, channel: torch.Tensor, mean: float, std: float):
    """ Compute z-score. """
    channel_normalized = (channel - mean) / (std + 1e-17)

    return channel_normalized

  def normalize(self, image: torch.Tensor) -> torch.Tensor:
    # assert torch.max(image) <= 1.0, f"{torch.max(image)}"
    # assert torch.min(image) >= 0.0, f"{torch.min(image)}"

    # get channels
    r_chan, g_chan, b_chan = common.image_rgb_split(image)

    # get channel stats
    r_mean, r_std, g_mean, g_std, b_mean, b_std = common.image_stats(image)

    # standardize each channel
    r_chan_normed = self.transform_torch_img_channel(r_chan, r_mean, r_std)
    g_chan_normed = self.transform_torch_img_channel(g_chan, g_mean, g_std)
    b_chan_normed = self.transform_torch_img_channel(b_chan, b_mean, b_std)

    image_normalized = common.image_rgb_stack(r_chan_normed, g_chan_normed, b_chan_normed)

    return image_normalized


class SingleImageGlobalNormalizer(ImageNormalizer):
  """ Normalize an image by its global mean and standard deviation.
  """

  def __init__(self):
    pass

  def normalize(self, image: torch.Tensor) -> torch.Tensor:
    # assert torch.max(image) <= 1.0, f"{torch.max(image)}"
    # assert torch.min(image) >= 0.0, f"{torch.min(image)}"

    # get mean across all channels 
    img_mean = torch.mean(image)
    img_std = torch.std(image)

    image_normalized = (image - img_mean) / img_std

    return image_normalized


def get_image_normalizer(cfg: Dict):
  name = cfg['data']['image_normalizer']['name']

  if name is None:
    return NoImageNormalizer()

  if name == "single_image_normalizer":
    return SingleImageNormalizer()

  if name == "single_image_global_normalizer":
    return SingleImageGlobalNormalizer()

  if name == "global_image_normalizer":
    rgb_means = cfg['data']['image_normalizer']['rgb_means']
    rgb_stds = cfg['data']['image_normalizer']['rgb_stds']
    
    return GlobalImageNormalizer(rgb_means, rgb_stds)

  assert False, "You need to specify an image normalizer."
