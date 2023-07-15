from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassJaccardIndex
from .panoptic_quality import PanopticQuality
import torch


class PhenorobChallengeEvaluator:
    def __init__(self, n_classes) -> None:
        self.iou = MulticlassJaccardIndex(
            num_classes=n_classes, average="none"  # , ignore_index=0
        )
        self.ap = MeanAveragePrecision(
            box_format="xyxy", num_classes=n_classes, reduction="none"
        )
        self.ap_ins = MeanAveragePrecision(
            box_format="xyxy",
            num_classes=n_classes,
            reduction="none",
            iou_type="segm",
        )
        self.pq = PanopticQuality()
        self.pq_list = []
        self.class_pq_list = []
        self.class_sq_list = []
        self.class_rq_list = []
        self.pq_list = []

    def update(self, semantic_pred, semantic_gt, instance_pred=None, instance_gt=None):
        semantic_pred = semantic_pred.detach().cpu()
        semantic_gt = semantic_gt.detach().cpu()

        self.update_iou(semantic_pred, semantic_gt)

        if instance_pred != None:
            instance_pred = instance_pred.detach().cpu()
            instance_gt = instance_gt.detach().cpu()
            self.update_pq(semantic_pred, semantic_gt, instance_pred, instance_gt)

    def update_iou(self, semantic_pred, semantic_gt):
        self.iou.update(semantic_pred, semantic_gt)

    def update_pq(self, semantic_pred, semantic_gt, instance_pred, instance_gt):
        self.pq.reset()
        pq_class, sq_class, rq_class = self.pq.compute_pq(
            semantic_pred,
            semantic_gt,
            instance_pred,
            instance_gt,
        )
        # self.pq_list.append(pq_image)
        self.class_pq_list.append(pq_class)
        self.class_sq_list.append(sq_class)
        self.class_rq_list.append(rq_class)

    def compute(self):
        metrics = {}
        iou = self.iou.compute()
        metrics["iou_soil"] = iou[0]
        metrics["iou_crop"] = iou[1]
        metrics["iou_weed"] = iou[2]
        self.class_pq_list = [x for x in self.class_pq_list if len(x) == 2]
        self.class_sq_list = [x for x in self.class_sq_list if len(x) == 2]
        self.class_rq_list = [x for x in self.class_rq_list if len(x) == 2]        
        class_pq = torch.tensor(self.class_pq_list).sum(dim=0) / len(self.class_pq_list)
        class_sq = torch.tensor(self.class_sq_list).sum(dim=0) / len(self.class_sq_list)
        class_rq = torch.tensor(self.class_rq_list).sum(dim=0) / len(self.class_rq_list)
        self.class_pq_list = []
        self.class_sq_list = []
        self.class_rq_list = []
        
        if len(self.class_pq_list) < 1:
            metrics["pq"] = 0
            metrics["pq_crop"] = 0
            metrics["pq_weed"] = 0
            metrics["sq_crop"] = 0
            metrics["sq_weed"] = 0
            metrics["rq_crop"] = 0
            metrics["rq_weed"] = 0
            
            return metrics
        
        metrics["pq"] = class_pq.mean()
        metrics["pq_crop"] = class_pq[0]
        metrics["pq_weed"] = class_pq[1]
        metrics["sq_crop"] = class_sq[0]
        metrics["sq_weed"] = class_sq[1]
        metrics["rq_crop"] = class_rq[0]
        metrics["rq_weed"] = class_rq[1]
        return metrics
