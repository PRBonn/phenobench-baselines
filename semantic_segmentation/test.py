""" Validate semantic segmentation model.
"""
import argparse
import os
import pdb
from typing import Dict

import oyaml as yaml
from pytorch_lightning import Trainer

from callbacks import (ConfigCallback, PostprocessorrCallback,
                       VisualizerCallback, get_postprocessors, get_visualizers)
from datasets import get_data_module
from modules import get_backbone, get_criterion, module


def parse_args() -> Dict[str, str]:
  parser = argparse.ArgumentParser()
  parser.add_argument("--export_dir", required=True, help="Path to export dir which saves logs, metrics, etc.")
  parser.add_argument("--config", required=True, help="Path to configuration file (*.yaml)")
  parser.add_argument("--ckpt_path", required=True, help="Provide *.ckpt file to continue training.")

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

  datasetmodule = get_data_module(cfg)
  criterion = get_criterion(cfg)

  # define backbone
  network = get_backbone(cfg)

  seg_module = module.SegmentationNetwork(network, 
                                          criterion, 
                                          cfg['train']['learning_rate'],
                                          cfg['train']['weight_decay'],
                                          test_step_settings=cfg['test']['step_settings'])

  # Add callbacks
  visualizer_callback = VisualizerCallback(get_visualizers(cfg), cfg['train']['vis_train_every_x_epochs'])
  postprocessor_callback = PostprocessorrCallback(
      get_postprocessors(cfg), cfg['train']['postprocess_train_every_x_epochs'])
  config_callback = ConfigCallback(cfg)

  # Setup trainer
  trainer = Trainer(default_root_dir=args['export_dir'],
                    gpus=cfg['test']['n_gpus'],
                    callbacks=[visualizer_callback, postprocessor_callback, config_callback])
  trainer.test(seg_module, datasetmodule, ckpt_path=args['ckpt_path'])


if __name__ == '__main__':
  main()
