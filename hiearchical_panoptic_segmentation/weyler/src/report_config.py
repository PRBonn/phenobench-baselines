""" Configuration file.
"""
import copy

args = dict(
  train_img_dir='</path/to/dataset>/train/images',
  
  val_img_dir='</path/to/dataset>/val/images',

  test_img_dir='</path/to/dataset>/test/images',

  report_dir = '<path/to/log/directoy>/reports',

  type='test',
  
  width = 1024,
  height = 1024,

  n_classes = 1,
  n_sigma = 3,
  apply_offsets = True,

  sigma_scale = 11.0,
  alpha_scale = 11.0,

  parts_area_thres=32,
  parts_score_thres=0.7,

  objects_area_thres = 64,
  objects_score_thres = 0.7,

  cls_colors = {"0" : "#ff0000", "1" : "#1eff00"}
)

def get_args():
  return copy.deepcopy(args)