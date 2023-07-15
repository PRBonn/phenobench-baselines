import yaml
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image, ImageFile
import numpy as np
import torch
import torchvision.transforms as transforms
from segmentation.data.transforms import PanopticTargetGenerator as PTG
import cv2
import matplotlib.pyplot as plt

from phenorob_challenge_tools.dataloader import PhenoRobPlantsBase

ImageFile.LOAD_TRUNCATED_IMAGES = True


def my_collate_function(items):
    N = len(items)
    batch = {}
    batch["target"] = {}
    batch["instances"] = torch.zeros(
        N, items[0]["image"].shape[-2], items[0]["image"].shape[-1]
    )
    batch["ignore_mask"] = torch.zeros(
        N, items[0]["image"].shape[-2], items[0]["image"].shape[-1], dtype=torch.bool
    )
    batch["image"] = torch.zeros(
        N, 3, items[0]["image"].shape[-2], items[0]["image"].shape[-1]
    )
    batch["target"]["semantic"] = torch.zeros(
        N,
        items[0]["target"]["semantic"].shape[-2],
        items[0]["target"]["semantic"].shape[-1],
        dtype=torch.long,
    )
    batch["target"]["foreground"] = torch.zeros(
        N,
        items[0]["target"]["foreground"].shape[-2],
        items[0]["target"]["foreground"].shape[-1],
    )
    batch["target"]["center"] = torch.zeros(
        N,
        items[0]["target"]["center"].shape[-3],
        items[0]["target"]["center"].shape[-2],
        items[0]["target"]["center"].shape[-1],
    )
    batch["target"]["offset"] = torch.zeros(
        N,
        items[0]["target"]["offset"].shape[-3],
        items[0]["target"]["offset"].shape[-2],
        items[0]["target"]["offset"].shape[-1],
    )
    batch["target"]["semantic_weights"] = torch.zeros(
        N,
        items[0]["target"]["semantic_weights"].shape[-2],
        items[0]["target"]["semantic_weights"].shape[-1],
    )
    batch["target"]["center_weights"] = torch.zeros(
        N,
        items[0]["target"]["center_weights"].shape[-2],
        items[0]["target"]["center_weights"].shape[-1],
    )
    batch["target"]["offset_weights"] = torch.zeros(
        N,
        items[0]["target"]["offset_weights"].shape[-2],
        items[0]["target"]["offset_weights"].shape[-1],
    )
    center_points = []
    names = []
    for i in range(N):
        batch["image"][i] = items[i]["image"]
        batch["instances"][i] = items[i]["instances"]
        batch["ignore_mask"][i] = items[i]["ignore_mask"]
        batch["target"]["semantic"][i] = items[i]["target"]["semantic"]
        batch["target"]["foreground"][i] = items[i]["target"]["foreground"]
        batch["target"]["center"][i] = items[i]["target"]["center"]
        batch["target"]["offset"][i] = items[i]["target"]["offset"]
        batch["target"]["semantic_weights"][i] = items[i]["target"]["semantic_weights"]
        batch["target"]["center_weights"][i] = items[i]["target"]["center_weights"]
        batch["target"]["offset_weights"][i] = items[i]["target"]["offset_weights"]
        center_points.append(items[i]["target"]["center_points"])
        names.append(items[i]["image_name"])
    batch["target"]["center_points"] = center_points
    batch["image_name"] = names
    return batch


def get_DLs(cfg, inst_type):
    # data_train = SugarBeets(cfg["DATASET"]["ROOT"], "train", overfit=cfg["DATASET"]["OVERFIT"])
    # data_val = SugarBeets(cfg["DATASET"]["ROOT"], "val", overfit=cfg["DATASET"]["OVERFIT"])
    data_train = PhenoRobPlantsDeeplab(
        os.path.join(cfg["DATASET"]["ROOT"], "train"), inst_type, overfit=cfg["DATASET"]["OVERFIT"]
    )
    data_val = PhenoRobPlantsDeeplab(
        os.path.join(cfg["DATASET"]["ROOT"], "val"), inst_type, overfit=cfg["DATASET"]["OVERFIT"]
    )
    train_loader = DataLoader(
        data_train,
        batch_size=cfg["TRAIN"]["IMS_PER_BATCH"],
        num_workers=cfg["WORKERS"],
        shuffle=True,
        pin_memory=True,
        collate_fn=my_collate_function,
    )
    val_loader = DataLoader(
        data_val,
        batch_size=cfg["TRAIN"]["IMS_PER_BATCH"],
        num_workers=cfg["WORKERS"],
        shuffle=False,
        pin_memory=True,
        collate_fn=my_collate_function,
    )

    return train_loader, val_loader

def get_test_DL(cfg, inst_type):
    # data_train = SugarBeets(cfg["DATASET"]["ROOT"], "train", overfit=cfg["DATASET"]["OVERFIT"])
    # data_val = SugarBeets(cfg["DATASET"]["ROOT"], "val", overfit=cfg["DATASET"]["OVERFIT"])
    data_test = PhenoRobPlantsDeeplab(
        os.path.join(cfg["DATASET"]["ROOT"], "test"), inst_type, overfit=cfg["DATASET"]["OVERFIT"]
    )
    test_loader = DataLoader(
        data_test,
        batch_size=cfg["TRAIN"]["IMS_PER_BATCH"],
        num_workers=cfg["WORKERS"],
        shuffle=True,
        pin_memory=True,
        collate_fn=my_collate_function,
    )

    return test_loader


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]


def id2rgb(id):
    return np.array([id % 256, id // 256, id // 256 // 256])


#################################################
################## Data loader ##################
#################################################
class Transform:
    def __init__(self):
        self.transform_img = transforms.Compose(
            [
                transforms.Resize((128, 256), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

        self.transform_labels = transforms.Compose(
            [transforms.Resize((128, 256), transforms.InterpolationMode.NEAREST)]
        )

    def __call__(self, data):
        new_data = {}
        new_data["image"] = self.transform_img(data["image"])
        new_data["instances"] = np.asarray(self.transform_labels(data["instances"]))
        return new_data


class SugarBeets(Dataset):
    def __init__(self, datapath, mode, overfit=False):
        super().__init__()

        self.datapath = datapath
        self.mode = mode
        self.overfit = overfit

        if self.overfit:
            self.datapath += "/images/train"
        else:
            self.datapath += "/images/" + mode

        self.all_imgs = [
            os.path.join(self.datapath, x)
            for x in os.listdir(self.datapath)
            if ".png" in x
        ]
        self.all_imgs.sort()
        # self.all_imgs = self.all_imgs[0]

        global_annotations_path = os.path.join(
            self.datapath.replace("images", "annos"), "global"
        )
        parts_annotations_path = os.path.join(
            self.datapath.replace("images", "annos"), "parts"
        )
        self.global_instance_list = [
            os.path.join(global_annotations_path, x)
            for x in os.listdir(global_annotations_path)
            if ".semantic" in x
        ]
        self.parts_instance_list = [
            os.path.join(parts_annotations_path, x)
            for x in os.listdir(parts_annotations_path)
            if ".semantic" in x
        ]
        self.global_instance_list.sort()
        self.parts_instance_list.sort()

        # self.global_instance_list = self.global_instance_list[0]
        # self.parts_instance_list = self.parts_instance_list[0]
        self.transform = Transform()
        self.target_gen = PTG(ignore_label=255, rgb2id=rgb2id, thing_list=[1])

        self.len = len(self.all_imgs)

    def __getitem__(self, index):
        # TRAINING MODE
        # load image
        sample = {}

        image = Image.open(self.all_imgs[index])
        width, height = image.size
        sample["image"] = image
        sample["im_name"] = self.all_imgs[index]

        global_annos = np.fromfile(self.global_instance_list[index], dtype=np.uint32)
        global_annos = global_annos.reshape(height, width)

        parts_annos = np.fromfile(self.parts_instance_list[index], dtype=np.uint32)
        parts_annos = parts_annos.reshape(height, width)

        global_labels = global_annos & 0xFFFF  # get lower 16-bits
        instances = global_annos >> 16  # get upper 16-bits

        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        global_instance_ids = np.unique(instances)[1:]  # no background
        instances_successive = np.zeros_like(instances)
        for idx, id_ in enumerate(global_instance_ids):
            instance_mask = instances == id_
            instances_successive[instance_mask] = idx + 1
        instances = instances_successive

        assert (
            np.max(instances) <= 255
        ), "Currently we do not suppot more than 255 instances in an image"

        parts_labels = parts_annos & 0xFFFF  # get lower 16-bits
        parts_instances = parts_annos >> 16  # get upper 16-bits
        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        parts_instance_ids = np.unique(parts_instances)[1:]  # no background
        parts_instances_successive = np.zeros_like(parts_instances)
        for idx, id_ in enumerate(parts_instance_ids):
            instance_mask = parts_instances == id_
            parts_instances_successive[instance_mask] = idx + 1
        parts_instances = parts_instances_successive

        assert (
            np.max(parts_instances) <= 255
        ), "Currently we do not suppot more than 255 instances in an image"

        # global_labels = Image.fromarray(np.uint8(global_labels))
        instances = Image.fromarray(np.uint8(instances))
        # parts_labels = Image.fromarray(np.uint8(parts_labels))
        parts_instances = Image.fromarray(np.uint8(parts_instances))

        sample["instances"] = instances
        # sample['global_labels'] = global_labels

        sample["parts_instances"] = parts_instances
        # sample['parts_labels'] = parts_labels

        # transform
        if self.transform is not None:
            sample = self.transform(sample)

        n_inst = np.max(sample["instances"])
        panoptic = sample["instances"]
        segments = []
        for i in range(1, n_inst + 1):
            current_dic = {}
            current_dic["id"] = i
            current_dic["category_id"] = 1
            current_dic["area"] = (sample["instances"] == i).sum()
            current_dic["bbox"] = self.masks_to_boxes((sample["instances"] == i))
            current_dic["iscrowd"] = False
            segments.append(current_dic)

        target = self.target_gen(panoptic, segments)

        sample["target"] = target
        return sample

    def __len__(self):
        if self.overfit:
            return 4
        return self.len

    def masks_to_boxes(self, masks):
        """
        Compute the bounding boxes around the provided masks.

        Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            masks (Tensor[N, H, W]): masks to transform where N is the number of masks
                and (H, W) are the spatial dimensions.

        Returns:
            Tensor[N, 4]: bounding boxes
        """
        if masks.sum() == 0:
            return np.zeros((0, 4), dtype=np.float)

        bounding_boxes = np.zeros((4), dtype=np.float)

        y, x = np.where(masks != 0)

        bounding_boxes[0] = np.min(x)
        bounding_boxes[1] = np.min(y)
        bounding_boxes[2] = np.max(x)
        bounding_boxes[3] = np.max(y)

        return bounding_boxes


class GrowliFlower(Dataset):
    def __init__(self, datapath, mode, overfit=False):
        super().__init__()

        self.datapath = datapath
        self.mode = mode
        # self.target = target
        self.overfit = overfit

        if self.overfit:
            self.datapath += "/images/Train"
        else:
            self.datapath += "/images/" + mode

        self.all_imgs = [
            os.path.join(self.datapath, x)
            for x in os.listdir(self.datapath)
            if ".jpg" in x
        ]
        self.all_imgs.sort()
        # self.all_imgs = self.all_imgs[0]

        global_annotations_path = os.path.join(
            self.datapath.replace("images", "labels"), "maskPlants"
        )
        parts_annotations_path = os.path.join(
            self.datapath.replace("images", "labels"), "maskLeaves"
        )
        void_annotations_path = os.path.join(
            self.datapath.replace("images", "labels"), "maskVoid"
        )

        self.global_instance_list = [
            os.path.join(global_annotations_path, x)
            for x in os.listdir(global_annotations_path)
        ]
        self.parts_instance_list = [
            os.path.join(parts_annotations_path, x)
            for x in os.listdir(parts_annotations_path)
        ]
        self.void_instance_list = [
            os.path.join(void_annotations_path, x)
            for x in os.listdir(void_annotations_path)
        ]

        self.global_instance_list.sort()
        self.parts_instance_list.sort()
        self.void_instance_list.sort()

        # self.global_instance_list = self.global_instance_list[0]
        # self.parts_instance_list = self.parts_instance_list[0]
        self.transform = transforms.ToTensor()
        self.target_gen = PTG(ignore_label=255, rgb2id=rgb2id, thing_list=[1])

        self.len = len(self.all_imgs)

    def __getitem__(self, index):
        # TRAINING MODE
        # load image
        sample = {}

        image = Image.open(self.all_imgs[index])
        width, height = image.size
        sample["image"] = self.transform(image)

        global_annos = np.array(Image.open(self.global_instance_list[index])).astype(
            np.int32
        )
        parts_annos = np.array(Image.open(self.parts_instance_list[index])).astype(
            np.int32
        )
        void_annos = np.array(Image.open(self.void_instance_list[index])).astype(
            np.int32
        )

        sample["global_instances"] = np.logical_not(void_annos) * global_annos
        sample["parts_instances"] = parts_annos

        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        global_instance_ids = np.unique(sample["global_instances"])[1:]  # no background
        global_instances_successive = np.zeros_like(sample["global_instances"])
        for idx, id_ in enumerate(global_instance_ids):
            instance_mask = sample["global_instances"] == id_
            global_instances_successive[instance_mask] = idx + 1
        global_instances = global_instances_successive

        assert (
            np.max(global_instances) <= 255
        ), "Currently we do not suppot more than 255 instances in an image"

        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        parts_instance_ids = np.unique(sample["parts_instances"])[1:]  # no background
        parts_instances_successive = np.zeros_like(sample["parts_instances"])
        for idx, id_ in enumerate(parts_instance_ids):
            instance_mask = sample["parts_instances"] == id_
            parts_instances_successive[instance_mask] = idx + 1
        parts_instances = parts_instances_successive

        assert (
            np.max(parts_instances) <= 255
        ), "Currently we do not suppot more than 255 instances in an image"

        # global_labels = Image.fromarray(np.uint8(global_labels))
        # global_instances = Image.fromarray(np.uint8(global_instances))
        # parts_labels = Image.fromarray(np.uint8(parts_labels))
        # parts_instances = Image.fromarray(np.uint8(parts_instances))

        sample["global_instances"] = self.transform(global_instances)
        sample["parts_instances"] = self.transform(parts_instances)
        n_inst = torch.max(sample["global_instances"])
        panoptic = global_instances  # sample['global_instances']
        segments = []
        for i in range(1, n_inst + 1):
            current_dic = {}
            current_dic["id"] = i
            current_dic["category_id"] = 1
            current_dic["area"] = (sample["global_instances"] == i).sum()
            current_dic["bbox"] = self.masks_to_boxes((sample["global_instances"] == i))
            current_dic["iscrowd"] = False
            segments.append(current_dic)

        target = self.target_gen(panoptic, segments)

        sample["target"] = target
        return sample

    def __len__(self):
        if self.overfit:
            return 4
        return self.len

    def masks_to_boxes(self, masks):
        """
        Compute the bounding boxes around the provided masks.

        Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            masks (Tensor[N, H, W]): masks to transform where N is the number of masks
                and (H, W) are the spatial dimensions.

        Returns:
            Tensor[N, 4]: bounding boxes
        """
        if masks.sum() == 0:
            return torch.zeros((0, 4), dtype=torch.float)

        bounding_boxes = torch.zeros((4), dtype=torch.float)
        y, x = torch.where(masks.squeeze() != 0)

        bounding_boxes[0] = torch.min(x)
        bounding_boxes[1] = torch.min(y)
        bounding_boxes[2] = torch.max(x)
        bounding_boxes[3] = torch.max(y)

        return bounding_boxes


class PhenoRobPlantsDeeplab(PhenoRobPlantsBase):
    def __init__(self, data_path, inst_type, overfit=False):
        super().__init__(data_path, overfit=overfit)
        self.target_gen = PTG(ignore_label=0, rgb2id=rgb2id, thing_list=[1, 2])
        self.inst_type = inst_type

    def __getitem__(self, index):
        raw_sample = self.get_sample(index)
        # self.visualize_sample(raw_sample)
        sample = {}
        sample["image"] = raw_sample["images"]
        sample["ignore_mask"] = raw_sample["ignore_mask"]
        if self.inst_type == "leaves":
            sample["instances"] = raw_sample["leaf_instances"]
        elif self.inst_type == "plants":
            sample["instances"] = raw_sample["plant_instances"]
        else:
            assert False
        # n_inst = torch.max(sample["instances"])
        instance_list = sample["instances"].unique()[1:]
        panoptic = sample["instances"].cpu().numpy()
        segments = []
        for inst in instance_list:
            instance_mask = sample["instances"] == inst
            current_dic = {}
            current_dic["id"] = inst
            inst_categories = raw_sample["semantics"][instance_mask]
            assert (inst_categories == inst_categories[0]).all()
            current_dic["category_id"] = (
                inst_categories[0] if inst_categories[0] < 3 else 0
            )
            current_dic["area"] = (instance_mask).sum()
            current_dic["bbox"] = self.masks_to_boxes(instance_mask)
            current_dic["iscrowd"] = False
            segments.append(current_dic)

        target = self.target_gen(panoptic, segments)
        # ignore ignored pixels
        target["center_weights"][sample["ignore_mask"]] = 0
        target["offset_weights"][sample["ignore_mask"]] = 0

        sample["target"] = target
        sample["image_name"] = raw_sample["image_name"]
        return sample
    
    def visualize_sample(self, sample):
        plt.imshow(sample["images"].permute(1,2,0))
        plt.imshow(sample["plant_instances"], alpha=0.5)
        plt.show()
        plt.imshow(sample["images"].permute(1,2,0))
        plt.imshow(sample["leaf_instances"], alpha=0.5)
        plt.show()
        

    def masks_to_boxes(self, masks):
        """
        Compute the bounding boxes around the provided masks.

        Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            masks (Tensor[N, H, W]): masks to transform where N is the number of masks
                and (H, W) are the spatial dimensions.

        Returns:
            Tensor[N, 4]: bounding boxes
        """
        if masks.sum() == 0:
            return torch.zeros((0, 4), dtype=torch.float)

        bounding_boxes = torch.zeros((4), dtype=torch.float)
        y, x = torch.where(masks.squeeze() != 0)

        bounding_boxes[0] = torch.min(x)
        bounding_boxes[1] = torch.min(y)
        bounding_boxes[2] = torch.max(x)
        bounding_boxes[3] = torch.max(y)

        return bounding_boxes


# class PhenoRobPlantsBase(Dataset):
#     def __init__(self, data_path, overfit=False):
#         super().__init__()

#         self.data_path = data_path
#         self.overfit = overfit

#         self.image_list = [
#             x for x in os.listdir(os.path.join(self.data_path, "images")) if ".png" in x
#         ]

#         self.image_list.sort()

#         self.len = len(self.image_list)

#         # preload the data to memory
#         self.field_list = os.listdir(self.data_path)
#         self.data_frame = self.load_data(
#             self.data_path, self.image_list, self.field_list
#         )

#     @staticmethod
#     def load_data(data_path, image_list, field_list):
#         data_frame = {}
#         for field in field_list:
#             data_frame[field] = []
#             for image in image_list:
#                 image = cv2.imread(
#                     os.path.join(os.path.join(data_path, field), image),
#                     cv2.IMREAD_UNCHANGED,
#                 )
#                 if len(image.shape) > 2:
#                     sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                     sample = torch.tensor(sample)
#                 else:
#                     sample = torch.tensor(image.astype("int16"))

#                 data_frame[field].append(sample)
#         return data_frame

#     def __getitem__(self, index):
#         sample = {}

#         for field in self.field_list:
#             sample[field] = self.data_frame[field][index]

#         # make instances sequential
#         sample["plant_instances"] = torch.unique(
#             sample["plant_instances"], return_inverse=True
#         )[1]
#         sample["leaf_instances"] = torch.unique(
#             sample["leaf_instances"], return_inverse=True
#         )[1]

#         return sample
