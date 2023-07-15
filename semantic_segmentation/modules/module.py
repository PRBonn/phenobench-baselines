import os
import pdb
from typing import Any, Dict, List, Optional, Tuple

import oyaml as yaml
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from modules.losses import get_div_loss_weight, js_div_loss, gjs_div_loss, kl_div_loss

# from modules.deeplab.modeling import deeplabv3plus_resnet50
# from modules.unet import UNet

class SegmentationNetwork(pl.LightningModule):

  def __init__(self, network: nn.Module, 
                     criterion: nn.Module, 
                     learning_rate: float, 
                     weight_decay: float, 
                     train_step_settings: Optional[List[str]] = None,
                     val_step_settings: Optional[List[str]] = None,
                     test_step_settings: Optional[List[str]] = None,
                     predict_step_settings: Optional[List[str]] = None,
                     ckpt_path=None):
    super().__init__()

    self.network = network
    self.criterion = criterion
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay

    self.save_hyperparameters("learning_rate", "weight_decay")

    # evaluation metrics for all classes
    self.metric_train_iou = torchmetrics.JaccardIndex(self.network.num_classes, reduction=None)
    self.metric_val_iou = torchmetrics.JaccardIndex(self.network.num_classes, reduction=None)
    self.metric_test_iou = torchmetrics.JaccardIndex(self.network.num_classes, reduction=None)

    if train_step_settings is not None:
      self.train_step_settings = train_step_settings
    else:
      self.train_step_settings = []

    if val_step_settings is not None:
      self.val_step_settings = val_step_settings
    else:
      self.val_step_settings = []

    if test_step_settings is not None:
      self.test_step_settings = test_step_settings
    else:
      self.test_step_settings = []

    if predict_step_settings is not None:
      self.predict_step_settings = predict_step_settings
    else:
      self.predict_step_settings = []

    if ckpt_path is not None:
      print("Load pretrained weights of backbone.")
      ckpt_dict = torch.load(ckpt_path)
      self.load_state_dict(ckpt_dict['state_dict'], strict=False)

  def compute_xentropy_loss(self, logits: torch.Tensor, y: torch.Tensor, mode: str, mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ Compute cross entropy loss based on logits and ground-truths.

    Args:
        logits (torch.Tensor): logits [B x num_classes x H x W]
        y (torch.Tensor): ground-truth [B x H x W]
        mode (str): train, val, or test
        mask_keep (Optional[torch.Tensor], optional): 1 := consider annotation, 0 := do not consider annotation [B x H x W]. Defaults to None.

    Returns:
        torch.Tensor: loss
    """

    return self.criterion(logits, y, mode=mode, mask_keep=mask_keep)

  def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
    """ Forward pass of backbone network.

    Args:
        img_batch (torch.Tensor): input image(s) [B x C x H x W]

    Returns:
        torch.Tensor, torch.Tensor: predictions of segmentation head [B x num_classes x H x W]
    """
    output = self.network.forward(img_batch)
    
    return output

  def training_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
    if not self.train_step_settings:
      raise ValueError('You need to specify the settings for the training step.')

    processed_steps = []
    if 'regular' in self.train_step_settings:
      logits_regular = self.forward(batch['input_image'])

      mask_keep_regular = batch['anno'] != 255
      loss_regular = self.compute_xentropy_loss(logits_regular, batch['anno'], mode='train', mask_keep=mask_keep_regular)

      processed_steps.append('regular')
      # update training metrics
      pred = torch.argmax(logits_regular.detach(), dim=1)
      self.metric_train_iou(pred , batch['anno'])

    if (self.trainer.current_epoch == 0) and (batch_idx == 0):
      print("Processed steps during training: ", processed_steps)

    # accumulate all losses
    my_local_loss_values = {var_name: var_value for var_name, var_value in locals().items() if var_name.startswith('loss')}
    loss = torch.sum(torch.stack(list(my_local_loss_values.values())))
    
    out_dict = {'loss': loss, 'logits': logits_regular, 'anno': batch['anno']}
    out_dict.update(my_local_loss_values)
    del my_local_loss_values
    
    return out_dict
   
  def training_epoch_end(self, training_step_outputs: List) -> None:
    epoch = self.trainer.current_epoch

    # compute loss(es) over all batches and log
    dict_keys = training_step_outputs[0].keys()
    for key in dict_keys:
      if key.startswith('loss'):
        loss_accumulated = torch.stack([x[key] for x in training_step_outputs])
        loss_avg = loss_accumulated.mean().detach()
        self.logger.experiment.add_scalars(f'{key}', {'train': loss_avg}, epoch)
   
    # the following logging is required *ModelCheckpoint* to work as expected
    losses = torch.stack([x['loss'] for x in training_step_outputs])
    train_loss_avg = losses.mean().detach()
    self.log("train_loss", train_loss_avg, on_epoch=True, sync_dist=False)

    # compute final metrics over all batches
    iou_per_class = self.metric_train_iou.compute().detach()
    mIoU = iou_per_class.mean()
    self.metric_train_iou.reset()

    for class_index, iou_class in enumerate(iou_per_class):
      self.logger.experiment.add_scalars(f"iou_class_{class_index}", {'train': iou_class}, epoch)
    self.logger.experiment.add_scalars("mIoU", {'train': mIoU}, epoch)
    self.log("train_mIoU", mIoU, on_epoch=True, sync_dist=False)

    # path_to_classwise_iou_dir = os.path.join(self.trainer.log_dir, 'train', 'evaluation', 'iou-classwise', f'epoch-{epoch:06d}')
    # save_iou_metric(iou_per_class, path_to_classwise_iou_dir)

  def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
    if not self.val_step_settings:
      raise ValueError('You need to specify the settings for the validation step.')
    assert len(self.val_step_settings) == 1

    # predictions
    if 'regular' in self.val_step_settings:
      logits = self.forward(batch['input_image'])

    # objective
    loss = self.compute_xentropy_loss(logits, batch['anno'], mode='val')

    # update validation metrics
    self.metric_val_iou(torch.argmax(logits, dim=1), batch['anno'])

    return {'loss': loss, 'logits': logits, 'anno': batch['anno']}

  def validation_epoch_end(self, validation_step_outputs: List) -> None:
    # compute loss over all batches
    losses = torch.stack([x['loss'] for x in validation_step_outputs])
    val_loss_avg = losses.mean()

    # logging
    epoch = self.trainer.current_epoch
    self.logger.experiment.add_scalars('loss', {'val': val_loss_avg}, epoch)
    self.log("val_loss", val_loss_avg, on_epoch=True, sync_dist=False)

    # compute final metrics over all batches
    iou_per_class = self.metric_val_iou.compute()
    mIoU = iou_per_class.mean() 
    self.metric_val_iou.reset()

    for class_index, iou_class in enumerate(iou_per_class):
      self.logger.experiment.add_scalars(f"iou_class_{class_index}", {'val': iou_class}, epoch)
    self.logger.experiment.add_scalars("mIoU", {'val': mIoU}, epoch)
    self.log("val_mIoU", mIoU, on_epoch=True, sync_dist=False)

    path_to_classwise_iou_dir = os.path.join(self.trainer.log_dir, 'val', 'evaluation', 'iou-classwise', f'epoch-{epoch:06d}')
    save_iou_metric(iou_per_class, path_to_classwise_iou_dir)

  def test_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
    if not self.test_step_settings:
      raise ValueError('You need to specify the settings for the test step.')
    assert len(self.test_step_settings) == 1

    # predictions
    if 'regular' in self.test_step_settings:
      logits = self.forward(batch['input_image'])
    
    pred_cls = torch.argmax(logits, dim=1)

    # regular evaluation for all classes
    self.metric_test_iou(pred_cls, batch['anno'])

    return {'logits': logits, 'anno': batch['anno']}

  def test_epoch_end(self, test_step_outputs: List) -> None:
    # compute final metrics over all batches
    iou_per_class = self.metric_test_iou.compute()
    mIoU = round(float(iou_per_class.mean().cpu()), 3)
    print(f"Test mIoU: {mIoU}")
    self.metric_test_iou.reset()

    epoch = self.trainer.current_epoch
    
    path_to_classwise_iou_dir = os.path.join(self.trainer.log_dir, 'evaluation', 'iou-classwise',  f'epoch-{epoch:06d}')
    save_iou_metric(iou_per_class, path_to_classwise_iou_dir)

  def lr_scaling(self, current_epoch: int) -> float:
    warm_up_epochs = 16
    if current_epoch <= warm_up_epochs:
      lr_scale = current_epoch / warm_up_epochs
    else:
      lr_scale = pow((1 - ((current_epoch - (warm_up_epochs + 1)) / (self.trainer.max_epochs - (warm_up_epochs + 1)))), 3.0)

    return lr_scale

  def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LambdaLR]]:
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    # lambda1 = lambda epoch: pow((1 - (epoch / self.trainer.max_epochs)), 3.0) # 1.25
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_scaling)

    return [optimizer], [scheduler]

def save_iou_metric(metrics: torch.Tensor, path_to_dir: str) -> None:
  if not os.path.exists(path_to_dir):
    os.makedirs(path_to_dir)

  iou_info = {}
  for cls_index, iou_metric in enumerate(metrics):
    iou_info[f'class_{cls_index}'] = round(float(iou_metric), 5)

  iou_info['mIoU'] = round(float(metrics.mean()), 5)

  fpath = os.path.join(path_to_dir, "iou.yaml")
  with open(fpath, 'w') as ostream:
    yaml.dump(iou_info, ostream)
