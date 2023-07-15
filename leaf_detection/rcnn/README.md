# FasterRCNN for Leaf Detection

The pretrained model can be downloaded at this [link](https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_detection/last.pt)

## Install
`conda env create -f environment.yml`

## Train
`python train.py -c configs/fasterrcnn_leaves.yaml`


In the config file we set path for saving checkpoints, logging on tensorboard and the path to train and validation set. 

## Test
`python test.py -c configs/fasterrcnn_leaves.yaml -w <PATH-TO-WEIGHTS>  -o <OUTPUT-DIR>`