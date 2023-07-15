from typing import Dict

import pytorch_lightning as pl

from .pdc import PDCModule


def get_data_module(cfg: Dict) -> pl.LightningDataModule:
  dataset_name = cfg['data']['name']
  if dataset_name == 'PDC':
    return PDCModule(cfg)
  else:
    assert False, "There is no parser for: {dataset_name}."
