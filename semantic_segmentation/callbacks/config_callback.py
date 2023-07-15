import os
from typing import Dict

import oyaml as yaml
from pytorch_lightning.callbacks import Callback


class ConfigCallback(Callback):
  """ Callback to save the config file of a model.
  """

  def __init__(self, cfg: Dict):
    self.cfg = cfg

  def setup(self, trainer, pl_module, stage=None) -> None:
    export_dir = os.path.join(trainer.log_dir, 'configuration')
    if not os.path.exists(export_dir):
      os.makedirs(export_dir)

    fpath = os.path.join(export_dir, 'config.yaml')
    with open(fpath, 'w') as ostream:
      yaml.dump(self.cfg, ostream)
