import torchvision
import torch.nn as nn
from torchmetrics import JaccardIndex
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision
import torch
import os
import torch.nn as nn
from torchmetrics import JaccardIndex
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
import torchvision.ops as tops
import yaml

class FasterRCNN(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.n_classes = cfg['train']['n_classes']
        self.epochs = cfg['train']['max_epoch'] 
        self.batch_size = cfg['train']['batch_size']

        self.ap = MeanAveragePrecision(box_format='xyxy', num_classes=self.n_classes, reduction='none')
        self.network = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=None, progress=True, num_classes = self.n_classes)
        self.network = self.network.float().cuda()

        self.prob_th = cfg['val']['prob_th']
        self.overlapping_th = cfg['val']['nms_th']

        self.ckpt_dir, self.tboard_dir = self.set_up_logging_directories(cfg)
        self.writer = SummaryWriter(log_dir=self.tboard_dir)
        self.log_val_predictions = True

    def forward(self, batch):
        # moving everything to cuda here to avoid stalling when workers != 0
        for b in range(len(batch['targets'])):
            batch['image'][b] = batch['image'][b].cuda()
            for k in batch['targets'][b]:
                batch['targets'][b][k] = batch['targets'][b][k].cuda()
        out = self.network(batch['image'], batch['targets'])
        return out 

    def getLoss(self, out):
        loss = out['loss_classifier'] + out['loss_box_reg'] + out['loss_objectness'] + out['loss_rpn_box_reg']
        return loss

    def training_step(self, batch):
        out = self.forward(batch)
        loss = self.getLoss(out)
        return loss

    def on_validation_start(self):
        if self.log_val_predictions:
            self.img_with_box = []

    def validation_step(self, batch):
        # moving everything to cuda here to avoid stalling when workers != 0
        for b in range(len(batch['targets'])):
            batch['image'][b] = batch['image'][b].cuda()
            for k in batch['targets'][b]:
                batch['targets'][b][k] = batch['targets'][b][k].cuda()
        out = self.network(batch['image'])

        # here start the postprocessing 
        b = len(batch['targets'])

        for b_idx in range(b):

            scores = out[b_idx]['scores']
            boxes = out[b_idx]['boxes']
            labels = out[b_idx]['labels']

            # non maximum suppression
            refined = tops.nms(boxes, scores, self.overlapping_th)
            refined_boxes = boxes[refined]
            refined_scores = scores[refined]
            refined_labels = labels[refined]

            # keeping only high scores
            high_scores = refined_scores > self.prob_th

            # if any scores are above self.prob_th we can compute metrics
            if high_scores.sum():
                surviving_boxes = refined_boxes[high_scores]
                surviving_scores = refined_scores[high_scores]
                surviving_labels = refined_labels[high_scores]
                
                surviving_dict = {}
                surviving_dict['boxes'] = surviving_boxes.cuda()
                surviving_dict['labels'] = surviving_labels.cuda()
                surviving_dict['scores'] = surviving_scores.cuda()
            
            # if not populate prediction dict with empty tensor to get 0 for ap and ap_ins
            # define zero sem and ins masks for iou metric
            else:
                surviving_dict = {}
                surviving_dict['boxes'] = torch.empty((0, 4)).cuda()
                surviving_dict['labels'] = torch.empty(0).cuda()
                surviving_dict['scores'] = torch.empty(0).cuda()

            self.ap.update([surviving_dict], [batch['targets'][b_idx]])
            
            if self.log_val_predictions:
                import matplotlib.pyplot as plt
                import cv2
                labels = surviving_dict['labels']
                bbox = surviving_dict['boxes'].long()
                img = batch['image'][b_idx].cpu().permute(1, 2, 0).numpy()
                for i in range(bbox.shape[0]):
                    # import ipdb;ipdb.set_trace()
                    color = (1,0,0) if labels[i] == 2 else (0,1,0)
                    img = cv2.rectangle(
                        img, (bbox[i][0].item(), bbox[i][1].item()), (bbox[i][2].item(), bbox[i][3].item()), color, 3)
                    # import ipdb;ipdb.set_trace()
                self.img_with_box.append(img)

    def test_step(self, batch):
        # moving everything to cuda here to avoid stalling when workers != 0
        for b in range(len(batch['targets'])):
            batch['image'][b] = batch['image'][b].cuda()
            for k in batch['targets'][b]:
                batch['targets'][b][k] = batch['targets'][b][k].cuda()
        out = self.network(batch['image'])

        # here start the postprocessing 
        b = len(batch['targets'])
        predictions_dictionaries = []

        for b_idx in range(b):

            scores = out[b_idx]['scores']
            boxes = out[b_idx]['boxes']
            labels = out[b_idx]['labels']

            # non maximum suppression
            refined = tops.nms(boxes, scores, self.overlapping_th)
            refined_boxes = boxes[refined]
            refined_scores = scores[refined]
            refined_labels = labels[refined]

            # keeping only high scores
            high_scores = refined_scores > self.prob_th

            # if any scores are above self.prob_th we can compute metrics
            if high_scores.sum():
                surviving_boxes = refined_boxes[high_scores]
                surviving_scores = refined_scores[high_scores]
                surviving_labels = refined_labels[high_scores]
                
                surviving_dict = {}
                surviving_dict['boxes'] = surviving_boxes.cuda()
                surviving_dict['labels'] = surviving_labels.cuda()
                surviving_dict['scores'] = surviving_scores.cuda()
            
            # if not populate prediction dict with empty tensor to get 0 for ap and ap_ins
            # define zero sem and ins masks for iou metric
            else:
                surviving_dict = {}
                surviving_dict['boxes'] = torch.empty((0, 4)).cuda()
                surviving_dict['labels'] = torch.empty(0).cuda()
                surviving_dict['scores'] = torch.empty(0).cuda()

            predictions_dictionaries.append(surviving_dict)

        return predictions_dictionaries

    def compute_metrics(self):
        ap = self.ap.compute()
        self.ap.reset()
        return ap

        
    @staticmethod
    def set_up_logging_directories(cfg):
        os.makedirs(cfg['checkpoint'], exist_ok = True) 
        os.makedirs(cfg['tensorboard'], exist_ok = True) 

        versions = os.listdir(cfg['checkpoint'])
        versions.sort()

        if len(versions) == 0:
            current_version = 0

        else:
            max_v = 0
            for fname in versions:
                if os.path.isdir(os.path.join(cfg['checkpoint'],fname)):
                    tmp_v = int(fname.split('_')[1])
                    if tmp_v > max_v:
                        max_v = tmp_v

            current_version = max_v  + 1

        new_dir = 'version_{}'.format(current_version)
        ckpt = os.path.join(cfg['checkpoint'],new_dir)
        tboard = os.path.join(cfg['tensorboard'],new_dir)
        os.makedirs(ckpt, exist_ok = True) 
        os.makedirs(tboard, exist_ok = True) 
        # save cfg here
        cfg_path = os.path.join(ckpt, 'cfg.yaml')
        with open(cfg_path, 'w') as f: 
            yaml.dump(cfg, f, default_flow_style=False)

        return ckpt, tboard
