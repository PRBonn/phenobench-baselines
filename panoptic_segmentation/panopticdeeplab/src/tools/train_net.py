# ------------------------------------------------------------------------------
# Training code.
# Example command:
# python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --cfg PATH_TO_CONFIG_FILE
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import logging
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

import _init_paths
from fvcore.common.file_io import PathManager
from segmentation.config import config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.utils import comm
from segmentation.solver import build_optimizer, build_lr_scheduler
from segmentation.solver import get_lr_group_id
from segmentation.utils import save_debug_images
from segmentation.utils import AverageMeter
from segmentation.model.post_processing.instance_post_processing import (
    get_instance_segmentation,
    get_panoptic_segmentation,
)
from segmentation.utils.utils import get_loss_info_str, to_cuda, get_module
from tools.datasets import get_DLs

# from torchmetrics import IoU
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from tools.panoptic_quality import PanopticQuality
from panoptic_quality import PanopticQuality as pq
import matplotlib.pyplot as plt
from matplotlib import cm

from tqdm import tqdm
from phenorob_challenge_tools.evaluator import PhenorobChallengeEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation network")

    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument(
        "--inst_type", help="which kind of instance to segment", required=True, type=str
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger = logging.getLogger("segmentation")
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=config.OUTPUT_DIR, distributed_rank=args.local_rank)

    logger.info(pprint.pformat(args))
    logger.info(config)

    evaluator = PhenorobChallengeEvaluator(n_classes=3)

    colormap = cm.get_cmap("tab20b")
    colormap.colorbar_extend = True
    viridis = cm.get_cmap("viridis")
    diverging_cm = cm.get_cmap("PiYG")

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device("cuda:{}".format(args.local_rank))
    # device = "cpu"

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    # build model
    model = build_segmentation_model_from_cfg(config)
    logger.info("Model:\n{}".format(model))

    logger.info(
        "Rank of current process: {}. World size: {}".format(
            comm.get_rank(), comm.get_world_size()
        )
    )

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    # summary(model)

    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # data_loader = build_train_loader_from_cfg(config)
    train_dl, val_dl = get_DLs(config, args.inst_type)
    optimizer = build_optimizer(config, model)
    # lr_scheduler = build_lr_scheduler(config, optimizer)
    # train_loader_iter = iter(train_dl)
    # val_loader_iter = iter(val_dl)

    start_epoch = 0
    max_epoch = config.TRAIN.MAX_ITER
    best_param_group_id = get_lr_group_id(optimizer)

    # # initialize model
    # if os.path.isfile(config.MODEL.WEIGHTS):
    #     model_weights = torch.load(config.MODEL.WEIGHTS)
    #     get_module(model, distributed).load_state_dict(model_weights, strict=False)
    #     logger.info('Pre-trained model from {}'.format(config.MODEL.WEIGHTS))
    # elif not config.MODEL.BACKBONE.PRETRAINED:
    #     if os.path.isfile(config.MODEL.BACKBONE.WEIGHTS):
    #         pretrained_weights = torch.load(config.MODEL.BACKBONE.WEIGHTS)
    #         get_module(model, distributed).backbone.load_state_dict(pretrained_weights, strict=False)
    #         logger.info('Pre-trained backbone from {}'.format(config.MODEL.BACKBONE.WEIGHTS))
    #     else:
    #         logger.info('No pre-trained weights for backbone, training from scratch.')

    # # load model
    # if config.TRAIN.RESUME:
    #     model_state_file = os.path.join(config.OUTPUT_DIR, 'checkpoint.pth.tar')
    #     if os.path.isfile(model_state_file):
    #         checkpoint = torch.load(model_state_file)
    #         start_iter = checkpoint['start_iter']
    #         get_module(model, distributed).load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         logger.info('Loaded checkpoint (starting from iter {})'.format(checkpoint['start_iter']))

    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    # Debug output.
    if config.DEBUG.DEBUG:
        debug_out_dir = os.path.join(config.OUTPUT_DIR, "debug_train")
        PathManager.mkdirs(debug_out_dir)

    # Train loop.
    try:
        writer = SummaryWriter(config.OUTPUT_DIR)

        accumulated_train_loss = 0.0
        accumulated_val_loss = 0.0

        # iou = torch.tensor(
        #     [0.0, 0.0]
        # ).cuda()  # IoU(num_classes=2, reduction="none").cuda()
        accumulated_miou = torch.tensor([0.0, 0.0]).cuda()
        accumulated_pq = 0.0
        best_pq = 0.0

        for epoch in tqdm(range(start_epoch, 5000)):
            print("Training epoch", epoch)
            model.train()
            for i, data in tqdm(enumerate(train_dl), total=len(train_dl), leave=False):
                if not distributed:
                    data = to_cuda(data, device)
                image = data.pop("image")
                out_dict = model(image, data["target"])

                loss = out_dict["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Get lr.
                lr = optimizer.param_groups[best_param_group_id]["lr"]
                # lr_scheduler.step()

                accumulated_train_loss += loss.detach().cpu().item()
            #     with torch.no_grad():
            #         semantic_pred = torch.argmax(torch.softmax(out_dict['semantic'], dim=1), dim=1)
            #         center_pred = out_dict['center']
            #         offset_pred = out_dict['offset']
            #         iou_batch = iou_t(semantic_pred, data['target']['semantic'])
            #         accumulated_miou_t += iou_batch
            #         panoptic_quality_t = pq()
            #         for index in range(image.shape[0]):
            #             instance_pred, _ = get_instance_segmentation(semantic_pred[index].unsqueeze(0), center_pred[index].unsqueeze(0), offset_pred[index].unsqueeze(0), [1])
            #             panoptic_quality_t.reset()
            #             pq_image, _ = panoptic_quality_t.panoptic_quality_forward(semantic_pred[index].unsqueeze(0), data['target']['semantic'][index].unsqueeze(0), instance_pred, data['global_instances'][index].unsqueeze(0))
            #             accumulated_pq_t += pq_image
            #             if pq_image < 0: import ipdb; ipdb.set_trace()

            # writer.add_scalar('Loss/train_loss', accumulated_train_loss / len(train_dl), epoch)
            # accumulated_miou_t /= len(train_dl)
            # accumulated_pq_t /= train_dl.dataset.__len__()
            # writer.add_scalars("Metrics/iou_t", {'soil': accumulated_miou_t[0], 'plants': accumulated_miou_t[1]}, epoch)
            # writer.add_scalar("Metrics/pq_t", accumulated_pq_t, epoch)

            # model.eval()
            if epoch > 29 and epoch % 10 == 0:
                with torch.no_grad():
                    for i, data in enumerate(val_dl):
                        # data
                        if not distributed:
                            data = to_cuda(data, device)
                            image = data.pop("image")
                        out_dict = model(image, data["target"])
                        loss = out_dict["loss"]
                        semantic_pred = torch.argmax(
                            torch.softmax(out_dict["semantic"], dim=1), dim=1
                        )
                        center_pred = out_dict["center"]
                        offset_pred = out_dict["offset"]

                        # if epoch >= 100 and epoch % 20 == 0:
                        #     import ipdb;ipdb.set_trace()  # fmt: skip

                        iou_batch = torch.tensor(
                            [0.0, 0.0]
                        ).cuda()  # iou(semantic_pred, data["target"]["semantic"])
                        accumulated_miou += iou_batch

                        accumulated_val_loss += loss.detach().cpu().item()

                        panoptic_quality = pq()
                        for index in range(image.shape[0]):
                            if (
                                data["target"]["semantic"][index].sum() == 0
                                and semantic_pred.sum() == 0
                            ):
                                accumulated_pq += 1.0
                            elif (
                                data["target"]["semantic"][index].sum() == 0
                            ):  # there are no plants but we are predicting at least one
                                accumulated_pq += 0.0
                            else:
                                # instance_pred, _ = get_instance_segmentation(
                                #     semantic_pred[index].unsqueeze(0),
                                #     center_pred[index].unsqueeze(0),
                                #     offset_pred[index].unsqueeze(0),
                                #     [1, 2],
                                # )
                                panoptic_pred, _ = get_panoptic_segmentation(
                                    sem=semantic_pred[index].unsqueeze(0),
                                    ctr_hmp=center_pred[index].unsqueeze(0),
                                    offsets=offset_pred[index].unsqueeze(0),
                                    thing_list=[1, 2],
                                    label_divisor=1000,
                                    stuff_area=config.POST_PROCESSING.STUFF_AREA,
                                    void_label=257,
                                    threshold=config.POST_PROCESSING.CENTER_THRESHOLD,
                                    top_k=config.POST_PROCESSING.TOP_K_INSTANCE,
                                )
                                instance_pred = torch.unique(
                                    panoptic_pred, return_inverse=True
                                )[1]

                                panoptic_quality.reset()
                                pq_image, _ = panoptic_quality.panoptic_quality_forward(
                                    semantic_pred[index].unsqueeze(0),
                                    data["target"]["semantic"][index].unsqueeze(0),
                                    instance_pred,
                                    data["instances"][index].unsqueeze(0),
                                )
                                masked_inst_preds = instance_pred[0]
                                masked_inst_preds[data["ignore_mask"][index]] = 0
                                masked_inst_gt = data["instances"][index]
                                masked_inst_gt[data["ignore_mask"][index]] = 0
                                evaluator.update(
                                    semantic_pred[index].unsqueeze(0),
                                    data["target"]["semantic"][index].unsqueeze(0),
                                    masked_inst_preds.unsqueeze(0),
                                    masked_inst_gt.unsqueeze(0),
                                )

                                accumulated_pq += pq_image
                                if pq_image < 0:
                                    import ipdb

                                    ipdb.set_trace()

                    metrics = evaluator.compute()

                    writer.add_scalar(
                        "Loss/val_loss", accumulated_val_loss / len(val_dl), epoch
                    )
                    writer.add_scalar(
                        "Loss/offset_loss",
                        model.loss_meter_dict["Offset loss"].val,
                        epoch,
                    )

                    for key in metrics:
                        writer.add_scalar(
                            "Metrics/" + key,
                            metrics[key],
                            epoch,
                        )
                    if epoch % 10 == 0:
                        instance_viz = colormap(instance_pred.squeeze().long().cpu())[
                            :, :, :3
                        ]
                        import ipdb;ipdb.set_trace()  # fmt: skip
                        writer.add_image(
                            "Instance seg",
                            instance_viz,
                            epoch,
                            dataformats="HWC",
                        )
                        semantic_viz = colormap(
                            semantic_pred[-1].squeeze().long().cpu()
                        )[:, :, :3]
                        writer.add_image(
                            "Semantic seg",
                            semantic_viz,
                            epoch,
                            dataformats="HWC",
                        )
                        center_viz = viridis((center_pred[-1][0]/ center_pred[-1].max()).cpu())[
                            :, :, :3
                        ]
                        writer.add_image(
                            "Center pred",
                            center_viz,
                            epoch,
                            dataformats="HWC",
                        )
                        normalized_offsets = offset_pred / (offset_pred.abs().max() * 2)
                        normalized_offsets += 0.5
                        # normalized_offsets[normalized_offsets < 0] = (
                        #     normalized_offsets[normalized_offsets < 0] + 1
                        # )
                        offset_vizy = diverging_cm(
                            normalized_offsets.squeeze()[-1][0].cpu()
                        )[:, :, :3]
                        writer.add_image(
                            "Offset predy",
                            offset_vizy,
                            epoch,
                            dataformats="HWC",
                        )
                        offset_vizx = diverging_cm(
                            normalized_offsets.squeeze()[-1][1].cpu()
                        )[:, :, :3]
                        writer.add_image(
                            "Offset predx",
                            offset_vizx,
                            epoch,
                            dataformats="HWC",
                        )
                    # accumulated_miou /= len(val_dl)
                    accumulated_pq /= val_dl.dataset.__len__()
                    # writer.add_scalar(
                    #     "Metrics/iou_plants",
                    #     accumulated_miou[1],
                    #     epoch,
                    # )
                    # writer.add_scalar(
                    #     "Metrics/iou_soil",
                    #     accumulated_miou[0],
                    #     epoch,
                    # )
                    # writer.add_scalar("Metrics/pq", accumulated_pq, epoch)

                    if accumulated_pq > best_pq:
                        best_pq = accumulated_pq
                        torch.save(
                            get_module(model, distributed).state_dict(),
                            os.path.join(config.OUTPUT_DIR, "best.pth"),
                        )

                    accumulated_train_loss *= 0
                    accumulated_val_loss *= 0
                    accumulated_miou *= 0
                    accumulated_pq *= 0

                    writer.flush()

        writer.close()

    except Exception:
        logger.exception("Exception during training:")
        raise
    finally:
        if comm.is_main_process():
            torch.save(
                get_module(model, distributed).state_dict(),
                os.path.join(config.OUTPUT_DIR, "final_state.pth"),
            )
        logger.info("Training finished.")


if __name__ == "__main__":
    main()
