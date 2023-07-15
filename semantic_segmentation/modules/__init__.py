import pdb
from typing import Dict

import torch.nn as nn

from modules.deeplab.modeling import deeplabv3plus_resnet50
from modules.erfnet.erfnet_modified import ERFNetModel
from modules.unet.unet_model import UNet
from modules.losses import get_criterion


def get_backbone(cfg: Dict) -> nn.Module:
  num_classes = cfg['backbone']['num_classes']
  pretrained = cfg['backbone']['pretrained']
  if cfg['backbone']['name'] == 'erfnet':
    return ERFNetModel(num_classes, pretrained=pretrained)

  if cfg['backbone']['name'] == 'unet':
    return UNet(num_classes)

  if cfg['backbone']['name'] == 'deeplabv3plus_resnet50':
    return deeplabv3plus_resnet50(num_classes, output_stride=16, pretrained_backbone=pretrained)

  raise ValueError("The requested backbone is not supported.")



