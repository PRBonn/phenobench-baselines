# Mask2Former

This project contains code to train and evaluate Mask2Former on the Phenobench dataset.

## How to run
- Install docker and nvidia-docker2
- Clone this repo to a folder on your pc
- Prepare plant data by running ```python src/data_preprocessing/prepare_dataset_plants.py --dataset-folder PHENOBENCH_PATH``` and ```python src/data_preprocessing/prepare_coco_semantic_annos_from_panoptic_annos_plants.py``` after replacing PHENOBENCH_PATH with the path to the downloaded dataset.
- Prepare leaf data by running ```python src/data_preprocessing/prepare_dataset_leaves.py --dataset-folder PHENOBENCH_PATH``` and ```python src/data_preprocessing/prepare_coco_semantic_annos_from_panoptic_annos_leaves.py``` after replacing PHENOBENCH_PATH with the path to the downloaded dataset.
- Add the data and the output paths to the ```Makefile``` 
- Execute ```make build```
- Execute ```make train_plants``` to train the panoptic segmentation task.
- Execute ```make train_leaves``` to train the leaf instance segmentation task.
- Execute ```make predict_plants``` to do inference for the panoptic segmentation task.
- Execute ```make predict_leaves``` to do inference for the leaf instance segmentation task.