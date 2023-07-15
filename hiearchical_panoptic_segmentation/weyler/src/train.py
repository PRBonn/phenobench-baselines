"""
Train model.
"""
import io
import pdb
import os
import shutil

import torch
import torch.nn as nn
from tqdm import tqdm

import report_config
import train_config
from common import convert_model
from criterions.my_loss import SpatialEmbLoss
from datasets import get_dataset
from models import get_model
from utils.myutils import AverageMeter, Cluster, TensorboardLogger

torch.backends.cudnn.benchmark = True

args = train_config.get_args()
report_args = report_config.get_args()

if args['save']:
  if not os.path.exists(args['save_dir']):
    os.makedirs(args['save_dir'])

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# train dataloader
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])

val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=False, drop_last=True, num_workers=args['val_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['n_sigma'])

# set criterion
criterion = SpatialEmbLoss(**args['loss_opts'],
                           **args['image'],
                           n_classes=args['model']['kwargs']['num_classes'][1]//2,
                           sigma_scale=args['sigma_scale'],
                           alpha_scale=args['alpha_scale'])


def lambda_(epoch):
  return pow((1-((epoch)/args['n_epochs'])), 0.9)

# resume model
start_epoch = 0
best_iou = 0
resume = False
if args['resume_path'] is not None and os.path.exists(args['resume_path']):
  with open(args['resume_path'], 'rb') as f:
    buffer = io.BytesIO(f.read())
  state = torch.load(buffer, map_location=device)
  start_epoch = state['epoch'] + 1
  model.load_state_dict(state['model_state_dict'])
  optim_state_dict = state['optim_state_dict']
  resume = True
  print('Resuming model from {}'.format(args['resume_path']))

# multi-gpu   
n_gpus = 0
if args['cuda']:
  n_gpus = torch.cuda.device_count()
if n_gpus > 1:
  print("Let's use #{} GPUs for our model".format(torch.cuda.device_count()))
  model = nn.DataParallel(model)
  model = convert_model(model) # sync batch norm
model = model.to(device)

if n_gpus > 1:
  print("Let's use #{} GPUs for our criterion".format(torch.cuda.device_count()))
  criterion = nn.DataParallel(criterion)
criterion = criterion.to(device)

# set optimizer
optimizer = torch.optim.Adam([{'params': model.parameters(),
                               'lr': args['lr'],
                               'weight_decay': args['w_decay']}])
if resume:
  optimizer.load_state_dict(optim_state_dict)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda_)

# clustering
cluster = Cluster('np', 
                  report_args['width'],
                  report_args['height'],
                  report_args['n_classes'],
                  report_args['n_sigma'],
                  report_args['sigma_scale'],
                  report_args['alpha_scale'],
                  report_args['parts_area_thres'],
                  report_args['parts_score_thres'],
                  report_args['objects_area_thres'],
                  report_args['objects_score_thres'],
                  report_args['apply_offsets'])

# Logger
tb_train_logger = TensorboardLogger(args['log_dir'], 'train')
train_info = {"loss" : 0, 'iou/object_iou': 0, 'iou/part_iou': 0, "lr": None}

tb_val_logger = TensorboardLogger(args['log_dir'], 'val')
val_info = {"loss" : 0, 'iou/object_iou': 0, 'iou/part_iou': 0}

def train(epoch, device):
  # define average meters
  loss_meter = AverageMeter()
  iou_meter_obj = AverageMeter()
  iou_meter_parts = AverageMeter()

  # empty the cache to train now
  # if n_gpus > 0:
  #   torch.cuda.empty_cache()

  # put model into training mode
  model.train()

  for param_group in optimizer.param_groups:
    print('learning rate: {}'.format(param_group['lr']))

  for sample in tqdm(train_dataset_it):
    print("=> train")
    img = sample['image']

    global_instances = sample['global_instances'].squeeze(1)  # [batch x H x W]
    global_labels = sample['global_labels'].squeeze(1)

    parts_instances = sample['parts_instances'].squeeze(1)
    parts_labels = sample['parts_labels'].squeeze(1)

    try:
      stem_anno = sample['stem_anno'].squeeze(1) # [batch x H x W]
      stem_anno = stem_anno.to(device)
    except KeyError:
      stem_anno = None
      pass

    img = img.to(device)
    global_instances = global_instances.to(device)
    global_labels = global_labels.to(device)
    parts_instances = parts_instances.to(device)
    parts_labels = parts_labels.to(device)

    outputs = model(img)

    loss = criterion(outputs, global_instances, global_labels, parts_instances,  parts_labels, stem_anno, **args['loss_w'], iou=True, iou_meter_obj=iou_meter_obj, iou_meter_parts=iou_meter_parts)
    loss = loss.mean() # only required if nn.DataParallel is used
    loss_meter.update(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_info["loss"] = loss_meter.avg
  train_info["iou/object_iou"] = iou_meter_obj.avg
  train_info["iou/part_iou"] = iou_meter_parts.avg
  train_info["lr"] = optimizer.param_groups[0]["lr"]

def val(epoch, device, only_eval=False):
  # define meters
  loss_meter = AverageMeter()
  iou_meter_obj = AverageMeter()
  iou_meter_parts = AverageMeter()

  # empty the cache to train now
  # if n_gpus > 0:
  #   torch.cuda.empty_cache()

  # put model into eval mode
  model.eval()
  
  with torch.no_grad():
    for sample in tqdm(val_dataset_it):
      print("=> ", val_dataset_it.dataset.type)
      img = sample['image']
      img_names = sample['im_name']

      img = img.to(device)

      outputs = model(img)

      if ((epoch % args['report_epoch'] == 0) and (epoch > 0)) or only_eval:
        batch_size = img.shape[0]
        path_to_dir = os.path.join(args['log_dir'], 'reports', val_dataset_it.dataset.type, "{:04d}".format(epoch))
        if not os.path.isdir(path_to_dir):
          os.makedirs(path_to_dir)

        for j in range(batch_size):
          im_filename, _ = os.path.basename(sample['im_name'][j]).split(".")
          im_name = img_names[j]
          im_id = os.path.basename(im_name)

          im_predictions = outputs[j].detach().cpu().numpy()
          im_predictions.tofile(os.path.join(path_to_dir, im_filename + ".pred"))

          with open(os.path.join(path_to_dir, im_filename + ".meta"), "w") as ostream:
            channels = im_predictions.shape[0]
            im_height = im_predictions.shape[1]
            im_width = im_predictions.shape[2]
            dtype = im_predictions.dtype
            ostream.write("{:d}\n{:d}\n{:d}\n{}".format(channels, im_height, im_width, dtype))

    if ((epoch % args['report_epoch'] == 0) and (epoch > 0)) or only_eval:
      pass


  val_info["loss"] = loss_meter.avg
  val_info["iou/object_iou"] = iou_meter_obj.avg
  val_info["iou/part_iou"] = iou_meter_parts.avg

def save_checkpoint(state, epoch: int, is_best=False, name='checkpoint'):
  print('=> saving checkpoint')
  name = '{}_{:04d}.pth'.format(name, epoch)
  filename = os.path.join(args['save_dir'], name)
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, os.path.join(
        args['save_dir'], 'best_iou_model.pth'))

if args["only_eval"]:
  print("=> Only evaluate model - no training")

  val(epoch=-1, device=device, only_eval=args["only_eval"])

  print('===> val loss: {:.2f}'.format(val_info["loss"]))
else:
  print("=> Start to train the network")
  for epoch in range(start_epoch, args['n_epochs']):
    print('Starting epoch {}'.format(epoch))
    scheduler.step(epoch)

    train(epoch, device)
    if ((epoch % args['report_epoch'] == 0) and (epoch > 0)):
      val(epoch, device)

    tb_train_logger.dump(train_info, epoch)
    tb_val_logger.dump(val_info, epoch)

    print('===> train loss: {:.2f}'.format(train_info["loss"]))
    print('===> val loss: {:.2f}'.format(val_info["loss"]))

    if args['save'] and ((epoch % args['report_epoch'] == 0) and (epoch > 0)):
      if n_gpus > 1:
        state = {"epoch": epoch,
                 "model_state_dict": model.module.state_dict(),
                 "optim_state_dict": optimizer.state_dict()}
      else:
        state = {"epoch": epoch,
                 "model_state_dict": model.state_dict(),
                 "optim_state_dict": optimizer.state_dict()}
      save_checkpoint(state, epoch=epoch)
