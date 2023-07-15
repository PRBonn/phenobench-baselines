""" Perform automated postprocessing step to cluster individual crop leaf and plant instances.
"""
import os

import h5py
import numpy as np
import skimage.exposure
import skimage.io

import report_config
from utils.myutils import (Cluster, Visualizer, bounding_box_from_mask,
                           get_all_predictions)

args = report_config.get_args()

path_train_reports = os.path.join(args["report_dir"], 'train')
path_val_reports = os.path.join(args["report_dir"], 'val')
path_test_reports = os.path.join(args["report_dir"], 'test')

try:
  train_preds = get_all_predictions(path_train_reports)
except FileNotFoundError:
  pass
try:
  val_preds = get_all_predictions(path_val_reports)
except FileNotFoundError:
  pass
try:
  test_preds = get_all_predictions(path_test_reports)
except FileNotFoundError:
  pass

cluster = Cluster('np',
                  args['width'],
                  args['height'],
                  args['n_classes'],
                  args['n_sigma'],
                  args['sigma_scale'],
                  args['alpha_scale'],
                  args['parts_area_thres'],
                  args['parts_score_thres'],
                  args['objects_area_thres'],
                  args['objects_score_thres'],
                  args['apply_offsets'])

vis = Visualizer(args['train_img_dir'], args['val_img_dir'], args['test_img_dir'], args['n_classes'], args['cls_colors'], args['width'], args['height'])
vis.set_status(args['type'])

epoch = ''
if args['type'] == 'train':
  preds = train_preds
elif args['type'] == 'val':
  preds = val_preds
elif args['type'] == 'test':
  preds = test_preds
else:
  raise ValueError

print("# preds: ", len(preds))
for val_pred in preds:
  print(f"=> report {args['type']} ", val_pred.path, val_pred.pred)
  
  img_name = val_pred.pred.split(".")[0] # filename of current image
  current_epoch = val_pred.path.split('/')[-1] # e.g. '0127'
  if current_epoch != '-001':
    continue
  if current_epoch != epoch:
    epoch = current_epoch

    if 'hdf5_ground_truth' in locals():
      hdf5_ground_truth.close()
    if 'hdf5_predictions' in locals():
      hdf5_predictions.close()    

    # create export directories for current epoch
    export_dir_gt = os.path.join(path_val_reports, epoch, 'patches', 'ground_truth')
    if not os.path.exists(export_dir_gt):
      os.makedirs(export_dir_gt)
    
    export_dir_pred = os.path.join(path_val_reports, epoch, 'patches', 'pred')
    if not os.path.exists(export_dir_pred):
      os.makedirs(export_dir_pred)

    export_dir_obj_instances = os.path.join(path_val_reports, epoch, 'instances', 'objects')
    if not os.path.exists(export_dir_obj_instances):
      os.makedirs(export_dir_obj_instances)

    export_dir_part_instances = os.path.join(path_val_reports, epoch, 'instances', 'parts')
    if not os.path.exists(export_dir_part_instances):
      os.makedirs(export_dir_part_instances)

    # create hdf5 files to store ground truth and predictions patches
    hdf5_ground_truth = h5py.File(os.path.join(export_dir_gt, 'ground_truth.h5'), 'w')
    hdf5_predictions = h5py.File(os.path.join(export_dir_pred, 'predictions.h5'), 'w')

  pred = val_pred.load()
  objects_seed, parts_seed, objects_offsets, parts_offsets, objects_sigma, parts_sigma, results = cluster.cluster(pred)

  vis.plot_sigmas(val_pred.path, val_pred.pred, objects_sigma, parts_sigma)
  vis.plot_seed(val_pred.path, val_pred.pred, objects_seed, parts_seed)
  vis.plot_embeddings(val_pred.path, val_pred.pred, results, parts_offsets, objects_offsets)
  vis.plot_instances(val_pred.path, val_pred.pred, results)

  part_map = cluster.draw_part_map(results, cls_idx='0')

  # save part instance maps to disk
  part_instances_map = np.zeros((args['height'], args['width']), dtype=np.uint8)
  for part_id, part_results in enumerate(results['parts']['0']):
    part_mask = part_results['part_mask']
    part_instances_map[part_mask] = part_id + 1

  skimage.io.imsave(os.path.join(export_dir_part_instances, img_name + ".png"), part_instances_map, check_contrast=False)
  
  # save object instance maps to disk
  object_instances_map = np.zeros((args['height'], args['width']), dtype=np.uint8)
  for obj_id, obj_results in enumerate(results['objects']['0']):
    part_ids = obj_results['obj_part_indicies']
    for part_id in part_ids:
      part_mask = results['parts']['0'][part_id]['part_mask']
      object_instances_map[part_mask] = obj_id + 1  

  skimage.io.imsave(os.path.join(export_dir_obj_instances, img_name + ".png"), object_instances_map, check_contrast=False)
