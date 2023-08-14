# PhenoBench Baselines

[PhenoBench](https://www.phenobench.org) is a large dataset and benchmarks for the semantic interpretation of images of real agricultural fields. The [benchmark tasks](https://www.phenobench.org/benchmarks.html) cover semantic segmentation of crops and weeds, panoptic segmentation of plants, leaf instance segmentation, detection of plants and leaves, and the novel task of hierarchical panoptic segmentation for jointly identifying plants and leaves.

## Implementations

In this repository, we provide code and instructions to reproduce the baseline results reported in our [paper](https://arxiv.org/pdf/2306.04557.pdf). Please see the sub-folders of the individual tasks for further instructions, configurations, and setup of the baselines.

**Important:** These are copies from the original repositories. We tried to make it transparent where the code originated with a link to the original repository. Thus, leave a :star: at their repository if you use their code.

## Checkpoints

To allow to produce the exact same results as reported in the paper, we also provide pre-trained models for our baseline methods. Thus, one can use the inference code to reproduce the results. For an overview of the pre-trained models, i.e., checkpoints of the final models.

<details>
<summary>Checkpoints</summary>
<table>
<tr><th>Task</th><th>Approach</th><th>Checkpoint</th></tr>
<tr><td>Semantic Segmentation</td><td>ERFNet</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-erfnet.ckpt">Download</a></td></tr>
<tr><td>Semantic Segmentation</td><td>DeepLabV3+</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/semantic-seg-deeplab.ckpt">Download</a></td></tr>
<tr><td></td><td></td><td></td></tr>
<tr><td>Panoptic Segmentation</td><td>Mask R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/panoptic_segmentation/last.pt">Download</a></td></tr>
<tr><td>Panoptic Segmentation</td><td>Panoptic DeepLab</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/PanopticDeeplab/model.pth">Download</a></td></tr>
<tr><td>Panoptic Segmentation</td><td>Mask2Former</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/Mask2former/model.pth">Download</a></td></tr>
<tr><td></td><td></td><td></td></tr>
<tr><td>Leaf Instance Segmentation</td><td>Mask R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_instance_segmentation/last.pt">Download</a></td></tr>
<tr><td>Leaf Instance Segmentation</td><td>Mask2Former</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_instance_segmentation/Mask2former/model.pth">Download</a></td></tr>
<tr><td></td><td></td><td></td></tr>
<tr><td>Hierarchical Panoptic Segmentation</td><td>Weyler et al.</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/hierarchical/weyler/weyler_checkpoint_0381.pth">Download</a></td></tr>
<tr><td>Hierarchical Panoptic Segmentation</td><td>HAPT</td><td><a href="https://drive.google.com/drive/folders/1BctpWMAALU0l6pTvo1e6Mxs8PWplNioT?usp=sharing">Download</a></td></tr>
<tr><td></td><td></td><td></td></tr>
<tr><td>Plant Detection</td><td>Faster R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/plant_detection/last.pt">Download</a></td></tr>
<tr><td>Plant Detection</td><td>Mask R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/plant_detection/last.pt">Download</a></td></tr>
<tr><td>Plant Detection</td><td>YOLOv7</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/plant_detection/YOLOv7/yolov7_plant_detection.pt">Download</a></td></tr>
<tr><td></td><td></td><td></td></tr>
<tr><td>Leaf Detection</td><td>Faster R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_detection/last.pt">Download</a></td></tr>
<tr><td>Leaf Detection</td><td>Mask R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/rcnn/leaf_detection/last.pt">Download</a></td></tr>
<tr><td>Leaf Detection</td><td>YOLOv7</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/YOLOv7/yolov7_leaf_detection.pt">Download</a></td></tr>
</table>
</details>

## Predictions

For comparison and rendering of detailed results, we also provide the submissions files of the predictions to the individual CodaLab competitions. These can be used with the code in the [PhenoBench development kit](https://www.github.com/PRBonn/phenobench) to visualize the results of the baseline and could be valuable for additional images and qualitative comparisons in papers.

Please refer to the [PhenoBench development kit](https://www.github.com/PRBonn/phenobench) for further details on the visualization of the results and additional information on the usage of the provided tools.

<details>
<summary>Predictions</summary>
<table>
<tr><th>Task</th><th>Approach</th><th>Validation</th><th>Test</th></tr>
<tr><td>Semantic Segmentation</td><td>ERFNet</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/predictions/erfnet/erfnet-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/predictions/erfnet/erfnet-test.zip">Download</a></td></tr>
<tr><td>Semantic Segmentation</td><td>DeepLabV3+</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/predictions/deeplab/deeplab-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/predictions/deeplab/deeplab-test.zip">Download</a></td></tr>
<tr><td></td><td></td><td></td><td></td></tr>
<tr><td>Panoptic Segmentation</td><td>Mask R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/predictions/maskrcnn-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/predictions/maskrcnn-test.zip">Download</a></td></tr>
<tr><td>Panoptic Segmentation</td><td>Panoptic DeepLab</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/predictions/panopticdeeplab-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/predictions/panopticdeeplab-test.zip">Download</a></td></tr>
<tr><td>Panoptic Segmentation</td><td>Mask2Former</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/predictions/mask2former-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/panoptic_segmentation/predictions/mask2former-test.zip">Download</a></td></tr>
<tr><td></td><td></td><td></td><td></td></tr>
<tr><td>Leaf Instance Segmentation</td><td> Mask R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_instance_segmentation/predictions/maskrcnn-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_instance_segmentation/predictions/maskrcnn-test.zip">Download</a></td></tr>
<tr><td>Leaf Instance Segmentation</td><td>Mask2Former</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_instance_segmentation/predictions/mask2former-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_instance_segmentation/predictions/maskrcnn-test.zip">Download</a></td></tr>
<tr><td></td><td></td><td></td><td></td></tr>
<tr><td>Hierarchical Panoptic Segmentation</td><td>Weyler et al.</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/hierarchical/predictions/weyler-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/hierarchical/predictions/weyler-test.zip">Download</a></td></tr>
<tr><td>Hierarchical Panoptic Segmentation</td><td>HAPT</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/hierarchical/predictions/hapt-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/hierarchical/predictions/hapt-test.zip">Download</a></td></tr>
<tr><td></td><td></td><td></td><td></td></tr>
<tr><td>Plant Detection</td><td>Faster R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/plant_detection/predictions/fastrcnn-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/plant_detection/predictions/fastrcnn-test.zip">Download</a></td></tr>
<tr><td>Plant Detection</td><td>Mask R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/plant_detection/predictions/maskrcnn-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/plant_detection/predictions/maskrcnn-test.zip">Download</a></td></tr>
<tr><td>Plant Detection</td><td>YOLOv7</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/plant_detection/predictions/yolov7-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/plant_detection/predictions/yolov7-test.zip">Download</a></td></tr>
<tr><td></td><td></td><td></td><td></td></tr>
<tr><td>Leaf Detection</td><td>Faster R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/predictions/fastrcnn-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/predictions/fastrcnn-test.zip">Download</a></td></tr>
<tr><td>Leaf Detection</td><td>Mask R-CNN</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/predictions/maskrcnn-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/predictions/maskrcnn-test.zip">Download</a></td></tr>
<tr><td>Leaf Detection</td><td>YOLOv7</td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/predictions/yolov7-val.zip">Download</a></td><td><a href="https://www.ipb.uni-bonn.de/html/projects/phenobench/leaf_detection/predictions/yolov7-test.zip">Download</a></td></tr>
</table>
</details>

## License

Please see the licenses of the particular sub-folders.

## Citation

If you use the specific baseline code, please cite the corresponding paper. We added a file `CITATION.md` that includes the suggested BibTeX entry for the baseline code.

If you use our dataset, then you should cite our paper [PDF](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/weyler2023arxiv.pdf):

```
@article{weyler2023dataset,
  author = {Jan Weyler and Federico Magistri and Elias Marks and Yue Linn Chong and Matteo Sodano 
    and Gianmarco Roggiolani and Nived Chebrolu and Cyrill Stachniss and Jens Behley},
  title = {{PhenoBench --- A Large Dataset and Benchmarks for Semantic Image Interpretation
    in the Agricultural Domain}},
  journal = {arXiv preprint},
  year = {2023}
}
```

