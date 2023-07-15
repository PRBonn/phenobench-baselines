""" Train semantic segmentation model.
"""
import argparse
from calendar import c
import os
import pdb
import time
from typing import Dict

import git
import oyaml as yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from callbacks import (ConfigCallback, PostprocessorrCallback,
                       VisualizerCallback, get_postprocessors, get_visualizers)
from datasets import get_data_module
from modules import get_backbone, get_criterion, module


def get_git_commit_hash() -> str:
  repo = git.Repo(search_parent_directories=True)
  sha = repo.head.object.hexsha

  return sha

def parse_args() -> Dict[str, str]:
  parser = argparse.ArgumentParser()
  parser.add_argument("--export_dir", required=True, help="Path to export dir which saves logs, metrics, etc.")
  parser.add_argument("--config", required=True, help="Path to configuration file (*.yaml)")
  parser.add_argument("--ckpt_path", required=False, default=None, help="Provide *.ckpt file to continue training.")
  parser.add_argument("--resume", required=False, action='store_true') # implies default = False

  args = vars(parser.parse_args())

  return args

def load_config(path_to_config_file: str) -> Dict:
  assert os.path.exists(path_to_config_file)

  with open(path_to_config_file) as istream:
    config = yaml.safe_load(istream)

  return config

def main():
  args = parse_args()

  cfg = load_config(args['config'])
  cfg['git-commit'] = get_git_commit_hash()

  if cfg.get('seed') is None:
    seed_val = int(time.time())
    cfg['seed'] = seed_val
  else:
    seed_val = cfg['seed'] 
  pl.utilities.seed.seed_everything(seed_val)

  datasetmodule = get_data_module(cfg)
  criterion = get_criterion(cfg)

  # define backbone
  network = get_backbone(cfg)

  if (args['ckpt_path'] is not None) and (not args['resume']):
    seg_module = module.SegmentationNetwork(network, 
                                            criterion, 
                                            cfg['train']['learning_rate'], 
                                            cfg['train']['weight_decay'], 
                                            train_step_settings = cfg['train']['step_settings'], 
                                            val_step_settings = cfg['val']['step_settings'],
                                            ckpt_path = args['ckpt_path'])
  else:
    seg_module = module.SegmentationNetwork(network, 
                                            criterion, 
                                            cfg['train']['learning_rate'], 
                                            cfg['train']['weight_decay'], 
                                            train_step_settings = cfg['train']['step_settings'],
                                            val_step_settings = cfg['val']['step_settings'])

  # Add callbacks
  lr_monitor = LearningRateMonitor(logging_interval='epoch')
  checkpoint_saver_val_loss = ModelCheckpoint(
      monitor='val_loss', filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_loss:.4f}', mode='min', save_last=True)
  checkpoint_saver_val_mIoU = ModelCheckpoint(
      monitor='val_mIoU', filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mIoU:.4f}', mode='max', save_last=False)
  checkpoint_saver_train_loss = ModelCheckpoint(
      monitor='train_loss', filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_loss:.4f}', mode='min', save_last=False)
  checkpoint_saver_train_mIoU = ModelCheckpoint(
      monitor='train_mIoU', filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mIoU:.4f}', mode='max', save_last=False)
  
  my_checkpoint_savers = [var_value for var_name, var_value in locals().items() if var_name.startswith('checkpoint_saver')]
  
  visualizer_callback = VisualizerCallback(get_visualizers(cfg), cfg['train']['vis_train_every_x_epochs'], cfg['val']['vis_val_every_x_epochs'])
  postprocessor_callback = PostprocessorrCallback(
      get_postprocessors(cfg), cfg['train']['postprocess_train_every_x_epochs'], cfg['val']['postprocess_val_every_x_epochs'])
  config_callback = ConfigCallback(cfg)

  # Setup trainer
  trainer = Trainer(
      benchmark=cfg['train']['benchmark'],
      gpus=cfg['train']['n_gpus'],
      default_root_dir=args['export_dir'],
      max_epochs=cfg['train']['max_epoch'],
      check_val_every_n_epoch=cfg['val']['check_val_every_n_epoch'],
      callbacks=[*my_checkpoint_savers,
                 lr_monitor, 
                 visualizer_callback, 
                 postprocessor_callback, 
                 config_callback])

  if args['ckpt_path'] is None:
    print("Train from scratch.")
    trainer.fit(seg_module, datasetmodule)
  elif (args['ckpt_path'] is not None) and (not args['resume']):
    print("Load pretrained model weights but other params (e.g. learning rate) start from scratch.")
    trainer.fit(seg_module, datasetmodule)
  elif (args['ckpt_path'] is not None) and args['resume']:
    print("Load pretrained model weights and resume training.")
    trainer.fit(seg_module, datasetmodule, ckpt_path=args['ckpt_path'])
  else:
    raise RuntimeError("Can't train any model since the settings are invalid.")


if __name__ == '__main__':
  main()
