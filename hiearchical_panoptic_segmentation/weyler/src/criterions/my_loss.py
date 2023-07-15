""" Compute loss functions.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from criterions.lovasz_losses import lovasz_hinge


def gaussian(embeddings, center, p_xx, p_yy, p_yx):
  """ Compute gaussian function."""
  delta = embeddings - center # 2 x h x w
  probs = torch.exp(-(1/2) * (
                             (delta[0] * p_xx * delta[0]) +  \
                             (delta[1] * p_yx * delta[0]) +  \
                             (delta[0] * p_yx * delta[1]) +  \
                             (delta[1] * p_yy * delta[1])
                            )
                  ).unsqueeze(0) # 1 x h x w
  return probs

class SpatialEmbLoss(nn.Module):

    def __init__(self,  label_ids, im_width, im_height, class_weights=1, to_center=True, apply_offsets=True, n_sigma=1, n_classes=1, sigma_scale=5.5, alpha_scale=1.0):
        super().__init__()

        print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, class_weights: {}'.format(
            to_center, n_sigma, class_weights))

        self.label_ids = label_ids
        self.im_width = im_width
        self.im_height = im_height
        self.to_center = to_center
        self.apply_offsets = apply_offsets
        self.n_classes = n_classes
        self.n_sigma = n_sigma
        self.sigma_scale = sigma_scale
        self.alpha_scale = alpha_scale
        self.class_weights = class_weights

        # instance id assigned to backgound
        self.bg_instance_id = 0

        # global coordinate map
        x_max = int(round(im_width / im_height))
        xm = torch.linspace(0, x_max, self.im_width).view(
            1, 1, -1).expand(1, self.im_height, self.im_width) # x-coords in rows
        ym = torch.linspace(0, 1, self.im_height).view(
            1, -1, 1).expand(1, self.im_height, self.im_width) # y-coords in cols
        xym = torch.cat((xm, ym), 0) # 2 x self.im_height x self.im_width

        self.register_buffer("xym", xym) # add to state dict

    def forward(self, prediction, obj_instances, obj_labels, parts_instances, parts_labels, stem_anno, w_inst=1, w_var=10, w_seed=1, w_offset=0, iou=False, iou_meter_obj=None, iou_meter_parts=None):

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)
        
        # adapt coordinate map to given image size
        xym = self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w
        loss = 0

        for batch in range(0, batch_size):
            # get predictions
            start_idx_parts = 2 + self.n_sigma
            start_idx_obj_cls = start_idx_parts + 2 + self.n_sigma
            start_idx_parts_cls = start_idx_obj_cls + self.n_classes

            if self.apply_offsets:
                parts_spatial_emb = xym + torch.tanh(prediction[batch, (start_idx_parts) : (start_idx_parts + 2)])  # 2 x h x w
            else:
                parts_spatial_emb = xym
            
            parts_sigma = prediction[batch, (start_idx_parts + 2) : (start_idx_parts + 2 + self.n_sigma)]  # n_sigma x h x w

            if self.apply_offsets:
                obj_spatial_emb = parts_spatial_emb.detach() + torch.tanh(prediction[batch, 0:2]) # 2 x h x w
            else:
                obj_spatial_emb = xym

            obj_sigma = prediction[batch, 2 : (2+self.n_sigma)]  # n_sigma x h x w
             
            obj_seed_map = torch.sigmoid(prediction[batch, start_idx_obj_cls : start_idx_obj_cls + self.n_classes])  # n_classes x h x w
            parts_seed_map = torch.sigmoid(prediction[batch, start_idx_parts_cls : start_idx_parts_cls + self.n_classes])  # n_classes x h x w
            
            # loss accumulators
            obj_var_loss = 0
            obj_instance_loss = 0
            obj_seed_loss = 0
            obj_count = 0
            obj_offsets_reg = 0
            
            parts_var_loss = 0
            parts_instance_loss = 0
            parts_seed_loss = 0
            parts_obj_count = 0
            parts_offset_reg = 0

            # get obj ground-truth
            img_obj_instances = obj_instances[batch]  # h x w
            img_obj_labels = obj_labels[batch]  # h x w
            
            img_obj_instance_ids = img_obj_instances.unique()
            img_obj_instance_ids = img_obj_instance_ids[img_obj_instance_ids != 0] # remove background

            img_parts_instances = parts_instances[batch]  # h x w
            img_parts_labels = parts_labels[batch]  # h x w
            
            # regress obj bg to zero (bg := background)
            for channel, label_id in enumerate(self.label_ids):
                bg_mask = (~(img_obj_labels == label_id)) # h x w
                if bg_mask.sum() > 0:
                    obj_seed_loss += F.mse_loss(input=obj_seed_map[channel][bg_mask], target=torch.zeros_like(obj_seed_map[channel][bg_mask]), reduction='sum')
            del bg_mask

            # regress parts bg to zero
            for channel, label_id in enumerate(self.label_ids):
                bg_mask = (~(img_parts_labels == label_id)) # h x w
                if bg_mask.sum() > 0:
                    parts_seed_loss += F.mse_loss(input=parts_seed_map[channel][bg_mask], target=torch.zeros_like(parts_seed_map[channel][bg_mask]), reduction='sum')
            del bg_mask

            for obj_id in img_obj_instance_ids:
                obj_in_mask = img_obj_instances.eq(obj_id)  # h x w

                # get all parts which belong to current object ...
                obj_parts_ids = img_parts_instances[obj_in_mask].unique()
                # ... but do not consider background with instance id = 0
                if self.bg_instance_id in obj_parts_ids:
                    obj_parts_ids = obj_parts_ids[1:]

                for obj_part_id in obj_parts_ids:
                    obj_part_in_mask = img_parts_instances.eq(obj_part_id) # h x w

                    # calculate center of attraction ...
                    if self.to_center:
                        # ... based on center of mass
                        part_center = (xym[obj_part_in_mask.expand_as(xym)].view(2, -1)).mean(1).view(2, 1, 1)  # 2 x 1 x 1
                    else:
                        # ... based on center of spatial embeddings
                        part_center = parts_spatial_emb[obj_part_in_mask.expand_as(parts_spatial_emb)].view(
                            2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

                    # calculate sigma
                    part_sigma_in = parts_sigma[obj_part_in_mask.expand_as(parts_sigma)].view(self.n_sigma, -1)

                    part_s = part_sigma_in.mean(1).view(self.n_sigma, 1, 1)   # n_sigma x 1 x 1
 
                    parts_var_loss += torch.mean(torch.sum(torch.pow(part_sigma_in - part_s.detach().view(self.n_sigma, 1), 2), 0))

                    # diagonals
                    l_11 = torch.exp(part_s[0] * self.sigma_scale) # [1x1]
                    try:
                        l_22 = torch.exp(part_s[1] * self.sigma_scale)
                    except IndexError:
                        l_22 = l_11

                    # rotation
                    try:
                        alpha = (part_s[2] * self.alpha_scale) * (math.pi / 2)
                    except IndexError:
                        alpha = torch.zeros_like(l_11)

                    # rotate precision matrix -> P = R' * P * R
                    p_xx = torch.pow(torch.cos(alpha), 2) * l_11 + torch.pow(torch.sin(alpha), 2) * l_22
                    p_yy = torch.pow(torch.sin(alpha), 2) * l_11 + torch.pow(torch.cos(alpha), 2) * l_22
                    p_yx = -torch.cos(alpha)*torch.sin(alpha) * l_11 + torch.sin(alpha)*torch.cos(alpha) * l_22

                    # calculate gaussian
                    probs = gaussian(parts_spatial_emb, part_center, p_xx, p_yy, p_yx)
                    
                    # apply lovasz-hinge loss
                    parts_instance_loss += lovasz_hinge(probs*2-1, obj_part_in_mask.unsqueeze(0))

                    # calculate part iou
                    if iou:
                        iou_meter_parts.update(calculate_iou((probs > 0.5), obj_part_in_mask))

                    d = (parts_spatial_emb[obj_part_in_mask.expand_as(parts_spatial_emb)].view(2,-1) - xym[obj_part_in_mask.expand_as(xym)].view(2,-1))
                    parts_offset_reg += torch.mean(torch.sum(torch.pow(d, 2), 0))

                    # get label id of current instance
                    part_label = torch.unique(img_parts_labels[obj_part_in_mask])
                    assert part_label.nelement() == 1
                    part_label = part_label[-1].item()
                    channel = self.label_ids.index(part_label)
                        
                    # seed loss
                    parts_seed_loss += self.class_weights[channel] * F.mse_loss(input=parts_seed_map[channel][obj_part_in_mask], 
                                                                                      target=probs[obj_part_in_mask.unsqueeze(0)].detach(), 
                                                                                      reduction='sum')

                    parts_obj_count += 1
                
                # Don’t hold onto tensors and variables you don’t need anymore
                # referring to https://pytorch.org/docs/stable/notes/faq.html
                try:
                    del obj_part_in_mask
                except UnboundLocalError:
                    pass

                try: 
                    del probs
                except UnboundLocalError:
                    pass

                # --- parts are done ---
                # --- now compute losses for current object ---
                
                # calculate center of attraction ...
                if self.to_center:
                    # ... based on center of mass
                    obj_center = (xym[obj_in_mask.expand_as(xym)].view(2, -1)).mean(1).view(2, 1, 1)  # 2 x 1 x 1
                else:
                    # ... based on stem position
                    center_mask = (stem_anno[batch] == obj_id) # h x w
                    obj_center = (xym[center_mask.expand_as(xym)].view(2, -1)).mean(1).view(2, 1, 1)

                # calculate sigma
                obj_sigma_in = obj_sigma[obj_in_mask.expand_as(
                    obj_sigma)].view(self.n_sigma, -1)

                obj_s = obj_sigma_in.mean(1).view(
                    self.n_sigma, 1, 1)   # n_sigma x 1 x 1

                # calculate var loss before exp
                obj_var_loss += torch.mean(torch.sum(torch.pow(obj_sigma_in - obj_s.detach().view(self.n_sigma, 1), 2), 0))

                # diagonals
                l_11 = torch.exp(obj_s[0] * self.sigma_scale) # [1x1]
                try:
                    l_22 = torch.exp(obj_s[1] * self.sigma_scale)
                except IndexError:
                    l_22 = l_11

                # rotation
                try:
                    alpha = (obj_s[2] * self.alpha_scale) * (math.pi / 2)
                except IndexError:
                    alpha = torch.zeros_like(l_11)

                # rotate precision matrix -> P = R' * P * R
                p_xx = torch.pow(torch.cos(alpha), 2) * l_11 + torch.pow(torch.sin(alpha), 2) * l_22
                p_yy = torch.pow(torch.sin(alpha), 2) * l_11 + torch.pow(torch.cos(alpha), 2) * l_22
                p_yx = -torch.cos(alpha)*torch.sin(alpha) * l_11 + torch.sin(alpha)*torch.cos(alpha) * l_22

                # calculate gaussian
                probs = gaussian(obj_spatial_emb, obj_center, p_xx, p_yy, p_yx)
                # apply lovasz-hinge loss
                obj_instance_loss += lovasz_hinge(probs*2-1, obj_in_mask.unsqueeze(0))

                # calculate part iou
                if iou:
                    iou_meter_obj.update(calculate_iou(probs > 0.5, obj_in_mask))
                    
                d = (obj_spatial_emb[obj_in_mask.expand_as(obj_spatial_emb)].view(2,-1) - parts_spatial_emb[obj_in_mask.expand_as(parts_spatial_emb)].view(2,-1).detach())
                obj_offsets_reg += torch.mean(torch.sum(torch.pow(d, 2), 0))
                
                # get label id of current instance
                obj_label = torch.unique(img_obj_labels[obj_in_mask])

                assert obj_label.nelement() == 1
                obj_label = obj_label[0].item()
                channel = self.label_ids.index(obj_label)

                # seed loss
                obj_seed_loss += self.class_weights[channel] * F.mse_loss(input=obj_seed_map[channel][obj_in_mask], 
                                                                                target=probs[obj_in_mask.unsqueeze(0)].detach(), 
                                                                                reduction='sum')
                obj_count += 1
                
                # Don’t hold onto tensors and variables you don’t need anymore
                del obj_in_mask
                del probs

            if parts_obj_count > 0:
                parts_instance_loss /= parts_obj_count
                parts_var_loss /= parts_obj_count

            if obj_count > 0:
                obj_instance_loss /= obj_count
                obj_var_loss /= obj_count

            obj_seed_loss = obj_seed_loss / (height * width)
            parts_seed_loss = parts_seed_loss / (height * width)
            
            loss += w_inst * (obj_instance_loss + parts_instance_loss) + \
                    w_var * (obj_var_loss + parts_var_loss) + \
                    w_seed * (obj_seed_loss + parts_seed_loss) + \
                    w_offset * (parts_offset_reg + obj_offsets_reg)

        loss = loss / (batch+1)

        return loss

def calculate_iou(pred, label): 
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
