import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import tifffile
import torch
import torch.nn.functional as functional
import skimage.io
from pytorch_lightning.callbacks import Callback


class Postprocessor(ABC):
  """ Basic representation of postprocessor. """

  @abstractmethod
  def process_logits(self, logits: torch.Tensor) -> Union[torch.Tensor, None]:
    """ Perform a post-processing of the logits predicted by a semantic segmentation network.

    Args:
        logits (torch.Tensor): logits of shape [batch_size x num_classes x H x W]

    Returns:
        torch.Tensor: post-processed logits of shape [batch_size x ? x H x W] (? := depends on implementation)
    """
    raise NotImplementedError

  @abstractmethod
  def process_embeddings(self, embeddings: torch.Tensor) -> Union[torch.Tensor, None]:
    """ Perform a post-processing of the embeddigns computed by the network

    Args:
        embeddings (torch.Tensor): embeddings of shape [batch_size x emb_dim x H x W]

    Returns:
        torch.Tensor: post-processed embeddings of shape [batch_size x ? x H x W] (? := depends on implementation)
    """
    raise NotImplementedError

  @abstractmethod
  def save_postprocessed_logits(self, processed_logits: Union[torch.Tensor, None], path_to_dir: str, fnames: List[str]) -> None:
    """ Save processed outputs created by this class.

    Args:
        processed_logits (torch.Tensor): post-processed output of shape [batch_size x ? x H x W] (? := depends on implementation)
        path_to_dir (str): path to output directory
        fnames (List[str]): raw filenames with fileformat (len(fnames)==batch_size)
    """
    raise NotImplementedError

  @abstractmethod
  def save_postprocessed_embeddings(self, processed_embeddings: Union[torch.Tensor, None], path_to_dir: str, fnames: List[str]) -> None:
    """ Save processed outputs created by this class.

    Args:
        processed_logits (torch.Tensor): post-processed output of shape [batch_size x ? x H x W] (? := depends on implementation)
        path_to_dir (str): path to output directory
        fnames (List[str]): raw filenames with fileformat (len(fnames)==batch_size)
    """
    raise NotImplementedError

class KeepLogitsPostprocessor(Postprocessor):
  """ Just keep the predicted logits and do not perform any further processing."""

  def __init__(self):
    self.name = 'logits'

  def process_logits(self, logits: torch.Tensor) -> Union[torch.Tensor, None]:
    """ This method just returns the logits and does not perform any processing.

    Args:
        logits (torch.Tensor): logits of shape [batch_size x num_classes x H x W]

    Returns:
        torch.Tensor: logits of shape [batch_size x num_classes x H x W]
    """
    assert len(logits.shape) == 4

    return logits

  def process_embeddings(self, embeddings: torch.Tensor) -> Union[torch.Tensor, None]:
    return

  def save_postprocessed_logits(self, processed_logits: Union[torch.Tensor, None], path_to_dir: str, fnames: List[str]) -> None:
    if processed_logits is None:
      return 

    path_to_dir = os.path.join(path_to_dir, self.name)
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir, exist_ok=True)

    if not (processed_logits.device == torch.device('cpu')):
      processed_logits = processed_logits.cpu()  # [batch_size x num_classes x H x W]

    with torch.no_grad():
      processed_logits = processed_logits.numpy().astype(np.float32)

    # save each batch to disk
    for i, output in enumerate(processed_logits):
      fname = fnames[i].split('.')[0] + ".tif"
      fpath = os.path.join(path_to_dir, fname)

      assert len(output.shape) == 3
      tifffile.imsave(fpath, output)

  def save_postprocessed_embeddings(self, processed_embeddings: Union[torch.Tensor, None], path_to_dir: str, fnames: List[str]) -> None:
    return

class KeepEmbeddingsPostprocessor(Postprocessor):
  def __init__(self) -> None:
    self.name = 'embeddings'
  
  def process_logits(self, logits: torch.Tensor) -> Union[torch.Tensor, None]:
    return 

  def process_embeddings(self, embeddings: torch.Tensor) -> Union[torch.Tensor, None]:
    assert len(embeddings.shape) == 4

    return embeddings

  def save_postprocessed_logits(self, processed_logits: Union[torch.Tensor, None], path_to_dir: str, fnames: List[str]) -> None:
    return

  def save_postprocessed_embeddings(self, processed_embeddings: Union[torch.Tensor, None], path_to_dir: str, fnames: List[str]) -> None:
    if processed_embeddings is None:
      return 

    path_to_dir = os.path.join(path_to_dir, self.name)
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir, exist_ok=True)

    if not (processed_embeddings.device == torch.device('cpu')):
        processed_embeddings = processed_embeddings.cpu()  # [batch_size x num_classes x H x W]

    with torch.no_grad():
      processed_embeddings = processed_embeddings.numpy()

    # save each batch to disk
    for i, output in enumerate(processed_embeddings):
      fname = fnames[i].split('.')[0] + ".npy"
      fpath = os.path.join(path_to_dir, fname)

      assert len(output.shape) == 3
      np.save(fpath, output)

class ProbablisticSoftmaxPostprocessor(Postprocessor):
  """ Convert the predicted logits into softmax probabilities. """

  def __init__(self):
    self.name = 'probabilities'

  def process_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """ Convert the predicted logits into softmax probabilities.

    Args:
        logits (torch.Tensor): logits of shape [batch_size x num_classes x H x W]

    Returns:
        torch.Tensor: class probabilities of shape [batch_size x num_classes x H x W]
    """
    assert len(logits.shape) == 4

    softmax_probs = functional.softmax(logits, dim=1)  # [batch_size x num_classes x H x W]

    return softmax_probs

  def process_embeddings(self, embeddings: torch.Tensor) -> Union[torch.Tensor, None]:
    return

  def save_postprocessed_logits(self, processed_logits: Union[torch.Tensor, None], path_to_dir: str, fnames: List[str]) -> None:
    """ Save predicted probabilites for each image as a tiff image.

    Args:
        processed_logits (torch.Tensor): post-processed logits of shape [batch_size x num_classes x H x W]
        path_to_dir (str): path to output directory
        fnames (List[str]): raw filenames with fileformat
    """
    if processed_logits is None:
      return 

    assert len(fnames) == int(processed_logits.shape[0])

    path_to_dir = os.path.join(path_to_dir, self.name)
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir, exist_ok=True)

    if not (processed_logits.device == torch.device('cpu')):
      processed_logits = processed_logits.cpu()  # [batch_size x num_classes x H x W]

    with torch.no_grad():
      processed_logits = processed_logits.numpy().astype(np.float32)

    # save each batch to disk
    for i, output in enumerate(processed_logits):
      # output is of shape [H x W x num_classes]
      fname = fnames[i].split('.')[0] + ".tif"
      fpath = os.path.join(path_to_dir, fname)

      assert len(output.shape) == 3
      tifffile.imsave(fpath, output)

  def save_postprocessed_embeddings(self, processed_embeddings: torch.Tensor, path_to_dir: str, fnames: List[str]) -> None:
    return

class ArgMaxClassPostprocessor(Postprocessor):
  """ Convert the predicted logits into class ids. """

  def __init__(self):
    self.name = 'arg_max_class'

  def process_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """ Convert the predicted logits into class ids.

    Args:
        logits (torch.Tensor): logits of shape [batch_size x num_classes x H x W]

    Returns:
        torch.Tensor: class probabilities of shape [batch_size x H x W]
    """
    assert len(logits.shape) == 4

    classes = torch.argmax(logits, dim=1)  # [batch_size x H x W]

    return classes

  def process_embeddings(self, embeddings: torch.Tensor) -> Union[torch.Tensor, None]:
    return

  def save_postprocessed_logits(self, processed_logits: Union[torch.Tensor, None], path_to_dir: str, fnames: List[str]) -> None:
    """ Save predicted probabilites for each image as a tiff image.

    Args:
        processed_logits (torch.Tensor): post-processed logits of shape [batch_size x H x W]
        path_to_dir (str): path to output directory
        fnames (List[str]): raw filenames with fileformat
    """
    if processed_logits is None:
      return 

    assert len(fnames) == int(processed_logits.shape[0])

    path_to_dir = os.path.join(path_to_dir, self.name)
    if not os.path.exists(path_to_dir):
      os.makedirs(path_to_dir, exist_ok=True)

    if not (processed_logits.device == torch.device('cpu')):
      processed_logits = processed_logits.cpu()  # [batch_size x num_classes x H x W]

    with torch.no_grad():
      processed_logits = processed_logits.numpy().astype(np.uint8)

    # save each batch to disk
    for i, output in enumerate(processed_logits):
      # output is of shape [H x W x num_classes]
      fname = fnames[i].split('.')[0] + ".png"
      fpath = os.path.join(path_to_dir, fname)

      assert len(output.shape) == 2
      skimage.io.imsave(fpath, output, check_contrast=False)

  def save_postprocessed_embeddings(self, processed_embeddings: torch.Tensor, path_to_dir: str, fnames: List[str]) -> None:
    return


def get_postprocessors(cfg: Dict) -> List[Postprocessor]:
  postprocessors = []

  try:
    cfg['postprocessors'].keys()
  except KeyError:
    return postprocessors

  for postprocessors_name in cfg['postprocessors'].keys():
    if postprocessors_name == 'keep_logits_postprocessor':
      postprocessor = KeepLogitsPostprocessor()
      postprocessors.append(postprocessor)
    if postprocessors_name == 'probablistic_softmax_postprocessor':
      postprocessor = ProbablisticSoftmaxPostprocessor()
      postprocessors.append(postprocessor)
    if postprocessors_name == 'keep_embeddings_postprocessor':
      postprocessor = KeepEmbeddingsPostprocessor()
      postprocessors.append(postprocessor)
    if postprocessors_name == 'arg_max_classes_postprocessor':
      postprocessor = ArgMaxClassPostprocessor()
      postprocessors.append(postprocessor)

  return postprocessors


class PostprocessorrCallback(Callback):
  """ Callback to visualize semantic segmentation.
  """

  def __init__(self, postprocessors: List[Postprocessor], postprocess_train_every_x_epochs: int = 1, postprocess_val_every_x_epochs: int = 1):
    """ Constructor.

    Args:
        postprocess_train_every_x_epochs (int): Frequency of train postprocessing. Defaults to 1.
        postprocess_val_every_x_epochs (int): Frequency of val postprocessing. Defaults to 1.
    """
    super().__init__()
    self.postprocessors = postprocessors
    self.postprocess_train_every_x_epochs = postprocess_train_every_x_epochs
    self.postprocess_val_every_x_epochs = postprocess_val_every_x_epochs

  def on_train_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    # visualize
    epoch = trainer.current_epoch
    if (epoch % self.postprocess_train_every_x_epochs) == 0 and (epoch != 0):
      path = os.path.join(trainer.log_dir, 'train', 'postprocess', f'epoch-{epoch:06d}')

      for postprocessor in self.postprocessors:
        # process logits
        processed_logits = postprocessor.process_logits(outputs['logits'])
        postprocessor.save_postprocessed_logits(processed_logits, path, batch['fname'])

        # process embeddings
        if 'embeddings' in outputs.keys():
          processed_embeddings = postprocessor.process_embeddings(outputs['embeddings'])
          postprocessor.save_postprocessed_embeddings(processed_embeddings, path, batch['fname'])

  def on_validation_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    # visualize
    epoch = trainer.current_epoch
    if ((epoch + 1) % self.postprocess_val_every_x_epochs) == 0 and (epoch != 0):
     path = os.path.join(trainer.log_dir, 'val', 'postprocess', f'epoch-{epoch:06d}')

     for postprocessor in self.postprocessors:
       processed_logits = postprocessor.process_logits(outputs['logits'])
       postprocessor.save_postprocessed_logits(processed_logits, path, batch['fname'])

       # process embeddings
       if 'embeddings' in outputs.keys():
         processed_embeddings = postprocessor.process_embeddings(outputs['embeddings'])
         postprocessor.save_postprocessed_embeddings(processed_embeddings, path, batch['fname'])

  def on_test_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
    # visualize
    path = os.path.join(trainer.log_dir, 'postprocess')

    for postprocessor in self.postprocessors:
      processed_logits = postprocessor.process_logits(outputs['logits'])
      postprocessor.save_postprocessed_logits(processed_logits, path, batch['fname'])

      # process embeddings
      if 'embeddings' in outputs.keys():
        processed_embeddings = postprocessor.process_embeddings(outputs['embeddings'])
        postprocessor.save_postprocessed_embeddings(processed_embeddings, path, batch['fname'])

  # def on_predict_batch_end(self, trainer, pl_module, outputs: Dict[str, Any], batch, batch_idx, dataloader_idx):
  #   # visualize
  #   path = os.path.join(trainer.log_dir, 'postprocess')

  #   for postprocessor in self.postprocessors:
  #     processed_logits = postprocessor.process_logits(outputs['logits'])
  #     postprocessor.save_postprocessed_logits(processed_logits, path, batch['fname'])

  #     # process embeddings
  #     if 'embeddings' in outputs.keys():
  #       processed_embeddings = postprocessor.process_embeddings(outputs['embeddings'])
  #       postprocessor.save_postprocessed_embeddings(processed_embeddings, path, batch['fname'])
