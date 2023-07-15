# Mask2Former

This project contains code to train and evaluate Mask2Former on the Phenobench dataset.

## How to run
- Install docker and nvidia-docker2
- Clone this repo to a folder on your pc
- Add the data and the output paths to the ```Makefile``` 
- Execute ```make build```
- Execute ```make train_plants``` to train the panoptic segmentation task.
- Execute ```make train_leaves``` to train the leaf instance segmentation task.
- Execute ```make predict_plants``` to do inference for the panoptic segmentation task.
- Execute ```make predict_leaves``` to do inference for the leaf instance segmentation task.