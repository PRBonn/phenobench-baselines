## Setup

```bash
conda create -n phenobench_weyler python=3.7.9
conda activate phenobench_weyler
pip install -r ./requirements.txt
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```

## Training and Testing
First, specify the path to the dataset as an environment variable
```bash 
export DATASET_DIR=/path/to/dataset
```

Second, open the file ```train_config.py``` and specify the directories. 

In case you want to train a new model set the variable ```only_eval``` to ```False```.

If you want to perform inference on the ```val``` or ```test``` split set this variable to ```True```.
To compute predictions on the ```val``` split set ```val_dataset['kwargs']['type_']='val' ```.
But if you want to compute predictions on the ```test``` split set ```val_dataset['kwargs']['type_']='test' ```.
In both cases, you need to specify a valid ```resume_path``` which point to the trained model. In case of training
you should set this path to ```None```.


Afterwards, run the following command to train a new model or to perform inference on the ```val``` or ```test``` split:
```python
python src/train.py
```

In case of inference, the next step is to set the variables in ```report_config.py```.
First, specify all directories and also if you want to get the predictions for the ```val``` or ```test``` split by setting the ```type``` variable. 

Next, run the following command to obtain the final predictions for each image:
```python
python src/report.py
```
This command saves all predictions in the folder specified in the variable ```report_dir``` in  ```report_config.py```.

## Pretrained Model
We provide the weights of a pretrained model [here](https://www.ipb.uni-bonn.de/html/projects/phenobench/hierarchical/weyler/weyler_checkpoint_0381.pth).