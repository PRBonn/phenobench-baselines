"""
Set training options.
"""
import copy
import os

import torch
from utils import mytransforms as my_transforms

DATASET_DIR=os.environ.get('DATASET_DIR')

args = dict(

    cuda=True,

    save=True,
    save_dir= '<path/to/save/directoy>',
    resume_path='<path/to/checkpoint.pth>',

    only_eval=True, # set to False if you want to train a new model

    log_dir= '<path/to/log/directoy>',

    train_dataset = {
        'name': 'mydataset',
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': 'train',
            'size': None,
            'stems': False,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image', 'global_instances', 'global_labels', 'parts_instances', 'parts_labels'],
                        'type': [torch.FloatTensor, torch.ByteTensor, torch.ByteTensor, torch.ByteTensor, torch.ByteTensor],
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': 8
    },

    val_dataset = {
        'name': 'mydataset',
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': 'test', # 'val' or 'test'
            'stems': False,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image'],
                        'type': [torch.FloatTensor]
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': 8
    },

    image = {
        'im_width': 1024,
        'im_height': 1024,
    },

    model = {
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [2*5,1*2],
            'batch_norm': True,
            'instance_norm': False,
        }
    }, 

    lr= 1e-3,
    w_decay=0,
    n_epochs=512,
    report_epoch=127, # every x epochs, report train & validation set

    # fix model params
    sigma_scale = 11.0,
    alpha_scale = 11.0,

    # loss options
    loss_opts={
        'to_center': True,
        'apply_offsets': True,
        'n_sigma': 3,
        'class_weights': [10],
        'label_ids': [1]
    },

    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
        'w_offset': 0,
    },
)

def get_args():
  return copy.deepcopy(args)