import click
import os
import torch
from os.path import join, dirname, abspath
from torch.utils.data import DataLoader
from dataloaders.datasets import Plants, collate_pdc
import models
import yaml
import time
from tqdm import tqdm


def save_model(model, epoch, optim, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        }, name)

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/cfg.yaml'))
@click.option('--percentage',
              '-p',
              type=float,
              help='percentage of training data to be used for training, if train_list specified in the cfg file',
              default=1.0)
def main(config, percentage):
    cfg = yaml.safe_load(open(config))
    cfg['data']['percentage'] = percentage

    train_dataset = Plants(
        datapath=cfg['data']['train'], overfit=cfg['train']['overfit'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc,
                              shuffle=True, drop_last=False, persistent_workers=True, pin_memory=True, num_workers=cfg['train']['workers'])
    if cfg['train']['overfit']:
        val_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False,
                                drop_last=False, persistent_workers=True, pin_memory=True, num_workers=cfg['train']['workers'])
    else:
        val_dataset = Plants(
            datapath=cfg['data']['val'], overfit=cfg['train']['overfit'])
        val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False,
                                drop_last=False, persistent_workers=True, pin_memory=True, num_workers=cfg['train']['workers'])

    model = models.get_model(cfg)
    optim = torch.optim.AdamW(model.network.parameters(), lr=cfg['train']['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

    best_map_det = 0
    best_map_ins = 0
    best_iou = 0
    best_pq = 0

    with torch.autograd.set_detect_anomaly(True):
        n_iter = 0  # used for tensorboard
        for e in range(cfg['train']['max_epoch']):
            model.network.train()
            start = time.time()
            for idx, item in enumerate(iter(train_loader)):

                optim.zero_grad()
                loss = model.training_step(item)
                loss.backward()
                optim.step()

                it_time = time.time() - start 
                print('Epoch: {}/{} -- Step: {}/{} -- Loss: {} -- Lr: {} -- Time: {}'.format(
                    e, cfg['train']['max_epoch'], idx*cfg['train']['batch_size'], len(
                        train_dataset), loss.item(),  scheduler.get_lr()[0], it_time
                ))
                model.writer.add_scalar('Loss/Train/', loss.detach().cpu().item(), n_iter)
                n_iter += 1
                start = time.time()
            
            scheduler.step()
            name = os.path.join(model.ckpt_dir,'last.pt')
            save_model(model, e, optim, name)
            
            model.network.eval()
            model.on_validation_start()
            for idx, item in enumerate(tqdm(val_loader)):
                with torch.no_grad():
                    model.validation_step(item)

            ap_detection, ap_instance, iou = model.compute_metrics()
            model.writer.add_scalar('Metric/Val/mAP_detection', ap_detection['map'].item(), n_iter)
            model.writer.add_scalar('Metric/Val/mAP_instance', ap_instance['map'].item(), n_iter)
            model.writer.add_scalar('Metric/Val/mIoU', iou.item(), n_iter)

            if model.log_val_predictions:
                from matplotlib import cm
                ins = model.img_with_masks[:4]
                sem = model.img_with_sem[:4]
                bbs = model.img_with_box[:4]
                for batch_idx in range(len(ins)):
                    colormap = cm.get_cmap('tab20b')
                    ins_transformed = colormap(ins[batch_idx].long().cpu())[:, :, :3]
                    sem_transformed = colormap(sem[batch_idx].long().cpu())[:, :, :3]
                    model.writer.add_image("Instances/" + "b" + str(batch_idx), ins_transformed, n_iter, dataformats='HWC')
                    model.writer.add_image("Semantics/" + "b" + str(batch_idx), sem_transformed, n_iter, dataformats='HWC')
                    model.writer.add_image("Boxes/" + "b" + str(batch_idx), bbs[batch_idx], n_iter, dataformats='HWC')

            # checking improvements on validation set
            if ap_detection['map'].item() >= best_map_det:
                name = os.path.join(model.ckpt_dir,'best_detection_map.pt')
                save_model(model, e, optim, name)
            if ap_instance['map'].item() >= best_map_ins:
                name = os.path.join(model.ckpt_dir,'best_instance_map.pt')
                save_model(model, e, optim, name)
            if iou.item() >= best_iou:
                name = os.path.join(model.ckpt_dir,'best_miou.pt')
                save_model(model, e, optim, name)
        
        
if __name__ == "__main__":
    main()
