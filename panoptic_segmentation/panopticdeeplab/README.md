# Panoptic Deeplab

This project contains code to train and evaluate Panoptic Deeplab on the Phenobench dataset.

## How to run
- Install docker and nvidia-docker2
- Clone this repo to a folder on your pc
- Add the data and the output paths to the ```Makefile``` 
- Execute ```make build```
- Execute ```make train_plants``` to train the panoptic segmentation task.
- Execute ```make predict_plants``` to do inference for the panoptic segmentation task.