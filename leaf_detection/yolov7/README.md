# YOLOv7 for Leaf Detection


## Install 
### Requirements
+ We assume you have an NVIDIA GPU. 
+ You should have Docker installed. 
+ Tested on docker: Ubuntu 18.04, CUDA 11.3, CuDNN 8, PyTorch 1.10.0.


### Installation steps
1. Download dataset from the [challenge website](https://www.phenobench.org/dataset.html), 
extract the data to your desired PARENT_DIR_PATH (the filepath of the dataset).
1. Initialise yolov7 git submodule:
    ```sh
     git submodule init
     git submodule update
    ```
1. In the Makefile, set the data_dir to the $PARENT_DIR_PATH. Alternatively, you can set the environment variable accordingly and the makefile use this.
1. Edit the ./Dockerfile to set the TORCH_CUDA_ARCH_LIST to match your GPU's compute capability
1. Set up the docker image using the Makefile:
    ```sh
    make build
    ```

## Converting the labels into YOLO format
You can run the script to make the conversions via make:   
```sh
make convert_annos2yolo
```
+ make sure that targets do not exist beforehand
+ Labels in YOLO format will be saved to ${PARENT_DIR_PATH}/YOLO_format
+ Images will also be copied into ${PARENT_DIR_PATH}/YOLO_format
<!-- + [TODO] For the sanity check, we are working on a visualiser so you can visualise the converted labels with the code from [this repo](https://github.com/yuelinn/yolo-labels-python-visualiser.git) which is git submoduled at ./yolo-labels-python-visualiser -->
  

## Inference and Visualisation
You can reproduce our results using the weights we trained.
1. Download [pretrained weights](https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/YOLOv7/yolov7_leaf_detection.pt).
1. Put the downloaded/desired weights file at ${PARENT_DIR_PATH}/src/weights. Sadly, there is no nice way to specify which weights you want to use (for now (?!)). You can change the path manually in the makefile if you want to run inference with the non-default weights. <!-- [TODO] specify which weights to use in make command (default will be the downloaded weights) -->
1. Run make command  <!-- TODO check what happens to the mounting when the paths get weird-->
    ```sh 
    export IN_IMGS_DIR="./data/PhenoBench/train/images"
    make infer 
    ```
1. The visualisation of the predictions and the .txt labels files are saved to ./src/runs/detect/yolov7\*/exp. 


## Train
1. Run the training in docker via make
    ```sh
    make train
    ```
1. Training data and weights will be written to ./src/runs/train/yolov7\*  
Hint: You may need to change the batch size (default=16). Unfortunately, you need to make the change directly in the Makefile (for now(?)).
<!-- TODO set batch size from env var or some config file -->


## Evaluation
For the val set:
1. Run inference on val set images:
    ```sh 
    export IN_IMGS_DIR="./data/PhenoBench/val/images"
    make val 
    ```
1. Use the [PhenoBench dev kit](https://github.com/PRBonn/phenobench.git) to run evaluation and get the performance metrics.

For the test set:
1. Run inference on test set images:
    ```sh 
    export IN_IMGS_DIR="./data/PhenoBench/test/images"
    make val
    ```
1. Use the [PhenoBench: Plant Detection CodaLab](https://codalab.lisn.upsaclay.fr/competitions/14178#learn_the_details-overview) to run evaluation and get the performance metrics on the test set.

Note: the difference between inference and evaluation is that in inference, the confidence is thresholded for better visualisation but we use all predictions for evaluation.

<!-- TODO move to docker config -->


## Acknowledgements
+ YOLOv7 code is git submoduled from [the official repo](https://github.com/WongKinYiu/yolov7)
+ Dockerfile shamelessly stolen from Elias Marks who may have stolen it from someone else.
