import argparse
from email.mime import base
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import skimage.io


def is_png(fname: str) -> bool:
  return fname.endswith('.png')


def get_png_filenames(path_to_dir: str) -> List[str]:
  filenames = []
  for fname in os.listdir(path_to_dir):
    if is_png(fname):
      filenames.append(fname)

  return filenames


def parse_args() -> Dict[str, str]:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_dir", required=True, help='Path to dataset directory.')
  parser.add_argument("--export_dir", required=True, help='Path to export directory.')
  args = vars(parser.parse_args())

  if not os.path.exists(args['export_dir']):
    os.makedirs(args['export_dir'])

  return args


def main():
  args = parse_args()

  path_to_imgs = os.path.join(args['dataset_dir'], 'images', 'rgb')
  path_to_anno = os.path.join(args['dataset_dir'], 'annotations')

  fnames_images = get_png_filenames(path_to_imgs)
  fnames_images.sort()

  fnames_annos = get_png_filenames(path_to_anno)
  fnames_annos.sort()

  assert len(fnames_images) == len(fnames_annos)

  for fname_img, fname_anno in zip(fnames_images, fnames_annos):
    img = skimage.io.imread(os.path.join(path_to_imgs, fname_img))
    img_h, img_w = img.shape[:2]

    anno = skimage.io.imread(os.path.join(path_to_anno, fname_anno))
    mask_10000 = (anno == 10000)
    anno[mask_10000] = 1
    if len(anno.shape) > 2:
      anno = anno[:, :, 0]

    assert np.max(anno) <= 2
    assert np.min(anno) >= 0

    basename = fname_img.split('.')[0]
    fpath_out = os.path.join(args['export_dir'], basename)

    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    mask_crop = anno == 1
    mask_weed = anno == 2

    canvas[mask_crop, :] = (0, 255, 0)
    canvas[mask_weed, :] = (255, 0, 0)

    fig, ax = plt.subplots(1, 1, dpi=300)
    ax.imshow(img)
    ax.imshow(canvas, alpha=0.5)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(fpath_out, transparent=True)
    plt.close('all')


if __name__ == '__main__':
  main()