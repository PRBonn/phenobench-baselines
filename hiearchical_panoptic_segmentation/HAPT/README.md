# Hierarchical Approach for Joint Semantic, Plant Instance, and Leaf Instance Segmentation in the Agricultural Domain 

This README contains the instructions to download and run the code. 

## Instructions 

1. Download the code via 

```
git clone https://github.com/PRBonn/HAPT.git
```

2. Enter in the code folder and install the requirements 

```
cd HAPT
pip install -r requirements.yml
```

3. Download the dataloader for the PhenoBench dataset from this repository, the file is named `PhenoBenchDataset.py`, and put it in the folder `HAPT/datasets`

4. Download the config file `PhenoBenchConfig.yaml` from this repository, put it in the folder `HAPT/config` and set the correct path to the dataset

5. Download the weights for the PhenoBench dataset from [here](https://drive.google.com/drive/folders/1BctpWMAALU0l6pTvo1e6Mxs8PWplNioT?usp=sharing) 

6. In the file `HAPT/train_hapt.py` change line 8 from `import datasets.datasets as datasets` to `import datasets.PhenoBenchDataset as datasets`

7. You can train the network running 

```
python train_hapt.py --config path/to/config/file --weights path/to/weights
```

This will start training the network, loading the weights but not the optimizer state.

## Citation

If you use this code, please cite the original paper! 

```
@inproceedings{roggiolani2023icra-hajs,
  title={Hierarchical Approach for Joint Semantic, Plant Instance, and Leaf Instance Segmentation in the Agricultural Domain},
  author={Roggiolani, Gianmarco and Sodano, Matteo and Guadagnino, Tiziano and Magistri, Federico and Behley, Jens and Stachniss, Cyrill},
  booktitle={Proc. of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
``` 
