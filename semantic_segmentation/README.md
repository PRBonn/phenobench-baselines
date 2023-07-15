# Semantic Segmentation

## Setup
```bash
conda create -n phenobench_semseg python=3.8
conda activate phenobench_semseg
pip install -r ./setup/requirements.txt
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install setuptools==59.5.0
```
Please note that the cuda version depends on your local machine.

Next, change the path to the dataset in the following configuration files
```bash
./config/config_erfnet.yaml
./config/config_deeplab.yaml
```

## Train ERFNet

```python 
python train.py --config ./config/config_erfnet.yaml --export_dir <path-to-export-directory>
```

## Train DeepLabV3+

```python 
python train.py --config ./config/config_deeplab.yaml --export_dir <path-to-export-directory>
```

## Test ERFNet

```python
python test.py --config ./config/config_erfnet.yaml --ckpt_path <path-to-export-ckpt> --export_dir <path-to-export-directory>
```

## Test DeepLabV3+

```python
python test.py --config ./config/config_deeplab.yaml --ckpt_path <path-to-export-ckpt> --export_dir <path-to-export-directory>
```

## Pretrained Model
We provide the weights of pretrained models:
- Please find the weights of ERFNet [here](https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-erfnet.ckpt)
- Please find the weights of DeepLabV3+ [here](https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-deeplab.ckpt)
