import sys
sys.path.insert(0, "Mask2Former")
import tempfile
from pathlib import Path
import numpy as np
import cv2
import os
import torch
# import cog
# import ipdb;ipdb.set_trace()  # fmt: skip
import argparse
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from mask2former import add_maskformer2_config

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from tqdm import tqdm

class Predictor():
    def __init__(self, config_file, weights, out_path):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_file) #"Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        self.predictor = DefaultPredictor(cfg)
        # self.coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
        self.out_path = out_path


    # @cog.input(
    #     "image",
    #     type=Path,
    #     help="Input image for segmentation. Output will be the concatenation of Panoptic segmentation (top), "
    #          "instance segmentation (middle), and semantic segmentation (bottom).",
    # )
    def predict(self, image):
        im = cv2.imread(str(image))
        outputs = self.predictor(im)
        # v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),
        #                                       outputs["panoptic_seg"][1]).get_image()
        # v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        # v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
        # result = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]
        # out_path = Path(tempfile.mkdtemp()) / "out.png"
        semantics = torch.argmax(outputs['sem_seg'], dim=0).cpu()
        instances = outputs['panoptic_seg'][0].cpu()
        instances[semantics==0] = 0
        visualize = False
        if visualize:
            plt.imshow(semantics)
            plt.show()
            plt.imshow(instances)
            plt.show()
        cv2.imwrite(os.path.join(self.out_path,"semantics", os.path.basename(image)), semantics.numpy())
        cv2.imwrite(os.path.join(self.out_path,"instances", os.path.basename(image)), instances.numpy())
        # return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file",
                        dest="cfgFile",
                        help="path to the config file.",
                        default=None,
                        type=str)
    parser.add_argument("--model-weights",
                        dest="modelWeights",
                        help="path to the model weights.",
                        default=None,
                        type=str)
    parser.add_argument("--testset-folder",
                        dest="testsetFolder",
                        help="path to the images folder of the test set",
                        default=None,
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default=None,
                        type=str)
    args = parser.parse_args()
    
    if not os.path.exists(os.path.join(args.outputFolder, "semantics")):
        os.makedirs(os.path.join(args.outputFolder, "semantics"))
    if not os.path.exists(os.path.join(args.outputFolder, "instances")):
        os.makedirs(os.path.join(args.outputFolder, "instances"))
    
    predictor = Predictor(config_file=args.cfgFile, weights=args.modelWeights, out_path=args.outputFolder)
    for image in tqdm(os.listdir(args.testsetFolder)):
        if not image.endswith(".png"):
            continue
        predictor.predict(os.path.join(args.testsetFolder, image))

# call the main
if __name__ == "__main__":
    main()
