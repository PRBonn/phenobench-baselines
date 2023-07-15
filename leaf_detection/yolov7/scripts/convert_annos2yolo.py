#!/usr/bin/env python3
"""Script to convert PDC annotations to YOLO format annotations.
"""

import argparse
import os
import shutil

from tqdm import tqdm
import numpy as np
import cv2

def convert_one_split(in_dir, out_dir, split_tag, is_verbose=False):
    """does the conversion for given split tag. Also copies images over.
    in_dir(str)   : filepath of the parent dir of phenobench
    out_dir(str)  : filepath of the parent dir where you want the yolo format annotations to be saved at 
    split_tag(str): split name. i.e., "train" or "val". 
                    "test" is not supported since you should not have access to the test labels ;)
    """
    split_in_dir=os.path.join(in_dir, split_tag)
    sem_dir = os.path.join(split_in_dir, "semantics")
    id_dir = os.path.join(split_in_dir, "plant_instances")

    output_dir = os.path.join(out_dir, split_tag, "labels")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting labels for {split_tag} split. Writing to {output_dir}...")

    for img_name in tqdm(os.listdir(sem_dir)):
        sem_path = os.path.join(sem_dir, img_name)
        id_path = os.path.join(id_dir, img_name)
        one_anno(id_path, output_dir)

    img_in_dir = os.path.join(split_in_dir, "images")
    img_out_dir = os.path.join(out_dir, split_tag, "images")
    print(f"Copying images over to {img_out_dir}...")
    os.makedirs(img_out_dir, exist_ok=True)
    for filename in tqdm(os.listdir(img_in_dir)):
        shutil.copy2(os.path.join(img_in_dir, filename),
                     os.path.join(img_out_dir,filename)
                     )


def one_anno(id_path, out_dir):
    out_file_path = os.path.join(out_dir, os.path.basename(id_path).split('.')[0]+".txt")
    # print(f"Write to file {out_file_path}")
    out_file = open(out_file_path, "x")

    id_mask = cv2.imread(id_path, cv2.IMREAD_UNCHANGED)
    img_width_x = id_mask.shape[1]
    img_height_y = id_mask.shape[0]

    plant_ids = np.unique(id_mask)

    for plant_id in plant_ids:
        if plant_id == 0:
            continue # skip because this is just bg
        # create an instance 
        if not np.any(id_mask==plant_id):
            print("no semantics found.")
            print(f"{sem_path} {plant_id}")
            continue
        # assert np.all(plant_class_mask == plant_class_mask[0]), f"{sem_path}: Error on plant id {plant_id} each plant id must be associated with only one class (crop or weed)"
        plant_class = 0

        plant_indices = np.where(id_mask==plant_id)

        plant_max_x = plant_indices[1].max()
        plant_min_x = plant_indices[1].min()

        plant_max_y = plant_indices[0].max()
        plant_min_y = plant_indices[0].min()

        plant_center_x = int((plant_max_x + plant_min_x)/2)
        plant_center_y = int((plant_max_y + plant_min_y)/2)

        plant_width_x = plant_max_x - plant_min_x
        plant_height_y = plant_max_y - plant_min_y

        out_file.write(f"{plant_class} {plant_center_x/img_width_x} {plant_center_y/img_height_y} {plant_width_x/img_width_x} {plant_height_y/img_height_y}\n")


    out_file.close()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog = "convert_annos2yolo",
                description = "This script converts the annotations of PhenoBench to the YOLO format.",
                epilog = "Example usage:\npython3 convert_annos2yolo.py --input_dir ./data/PhenoBench --output_dir ./data/phenobench-yolo"
                )

    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    convert_one_split(
            in_dir=args.input_dir,
            out_dir=args.output_dir, 
            split_tag="train"
            )

    convert_one_split(
            in_dir=args.input_dir,
            out_dir=args.output_dir, 
            split_tag="val"
            )

