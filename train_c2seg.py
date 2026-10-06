"""
Training script for C2Seg.
Usage:
  # Stage I - CaCL pretraining
  python train_c2seg.py --stage pretrain --config configs/c2seg.yaml

  # Stage II - segmentation fine-tuning
  python train_c2seg.py --stage train --config configs/c2seg.yaml

  # Evaluation
  python train_c2seg.py --stage eval --config configs/c2seg.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import load_config
from utils import set_seed, get_device, save_checkpoint, load_checkpoint
from utils.logger import get_logger
from utils.metrics import AverageMeter
from utils.seg_metrics import dice_score, miou_score, hd95, assd
from models.c2seg import C2Seg
from models.losses import CaCLLoss, BCEDiceLoss
from datasets.c2seg_dataset import C2SegDataset


# ---------------------------------------------------------------------------
# Stage I: CaCL pretraining
# ---------------------------------------------------------------------------
def pretrain(cfg):
    import time
    from tqdm import tqdm

    logger = get_logger('c2seg_pretrain', cfg['logging']['log_dir'])
    device = get_device()
    set_seed(cfg['project']['seed'])
    pcfg = cfg['pretrain']

    dataset = C2SegDataset(cfg['data']['train_dir'],
                           image_size=cfg['data']['image_size'],
                           augment=True)
    loader  = DataLoader(dataset, batch_size=pcfg['batch_size'],
                         shuffle=True, num_workers=cfg['data']['num_workers'],
                         pin_memory=True, drop_last=True)

    model = C2Seg(**cfg['model']).to(device)
    cacl  = CaCLLoss(temperature=pcfg['temperature'], eps=pcfg['eps'])

    # only train visual encoder + projections, language encoder is frozen
    params = [p for n, p in model.named_parameters()
              if 'lang_enc' not in n and p.requires_grad]
    trainable_params = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    optimizer = torch.optim.Adam(params, lr=pcfg['lr'],
                                 weight_decay=pcfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pcfg['epochs'])

    ckpt_dir = Path(cfg['logging']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {len(dataset)} samples | Batch size: {pcfg['batch_size']} | Steps/epoch: {len(loader)}")
    logger.info(f"Config: τ={pcfg['temperature']}, ρ={pcfg['eps']}, lr={pcfg['lr']}")
    logger.info("-" * 60)

    best_loss = float('inf')
    for epoch in range(pcfg['epochs']):
        model.train()
        loss_meter = AverageMeter()
        epoch_start = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{pcfg['epochs']}", leave=False)
        for batch_idx, (images, _, texts) in enumerate(pbar):
            images = images.to(device)

            # single forward pass — v, l projected; l_raw from frozen CLIP (no grad)
            v, l, l_raw = model.get_contrastive_embeddings(images, texts)

            loss = cacl(v, l, l_raw)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.size(0))

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{loss_meter.avg:.4f}'})

        scheduler.step()
        epoch_time = time.time() - epoch_start
        lr_current = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch [{epoch+1:02d}/{pcfg['epochs']}] "
            f"Loss: {loss_meter.avg:.4f} | "
            f"LR: {lr_current:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint({'epoch': epoch+1, 'model': model.state_dict(), 'best_loss': best_loss},
                            str(ckpt_dir / 'c2seg_pretrain_best.pth'))

        # Save checkpoint periodically
        if (epoch + 1) % cfg['logging']['save_every'] == 0:
            save_checkpoint({'epoch': epoch+1, 'model': model.state_dict()},
                            str(ckpt_dir / f'c2seg_pretrain_epoch{epoch+1}.pth'))

    # Save final checkpoint
    save_checkpoint({'epoch': pcfg['epochs'], 'model': model.state_dict(), 'best_loss': best_loss},
                    pcfg['checkpoint'])
    logger.info("-" * 60)
    logger.info(f"Pretrain done. Best loss: {best_loss:.4f}")
    logger.info(f"Saved to {pcfg['checkpoint']}")
    return model


# ---------------------------------------------------------------------------
# Stage II: segmentation fine-tuning
# ---------------------------------------------------------------------------
def train(cfg, pretrained_ckpt: str = None):
    logger = get_logger('c2seg_train', cfg['logging']['log_dir'])
    device = get_device()
    set_seed(cfg['project']['seed'])
    tcfg = cfg['train']

    train_ds = C2SegDataset(cfg['data']['train_dir'],
                            image_size=cfg['data']['image_size'], augment=True)
    val_ds   = C2SegDataset(cfg['data']['val_dir'],
                            image_size=cfg['data']['image_size'], augment=False)

    train_loader = DataLoader(train_ds, batch_size=tcfg['batch_size'],
                              shuffle=True,  num_workers=cfg['data']['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['eval']['batch_size'],
                              shuffle=False, num_workers=cfg['data']['num_workers'])

    model = C2Seg(**cfg['model']).to(device)

    # load pretrained weights if available
    ckpt = pretrained_ckpt or cfg['pretrain']['checkpoint']
    if Path(ckpt).exists():
        load_checkpoint(ckpt, model)
        logger.info(f"Loaded pretrained weights from {ckpt}")

    criterion = BCEDiceLoss(bce_weight=tcfg['bce_weight'])

    # visual encoder uses smaller lr (fine-tune), rest uses full lr
    visual_params  = list(model.visual_enc.parameters())
    other_params   = [p for n, p in model.named_parameters()
                      if not n.startswith('visual_enc') and
                         not n.startswith('lang_enc') and
                         p.requires_grad]
    optimizer = torch.optim.Adam([
        {'params': visual_params, 'lr': tcfg['lr'] * 0.1},   # small lr for visual encoder
        {'params': other_params,  'lr': tcfg['lr']},
    ], weight_decay=tcfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg['epochs'])

    ckpt_dir = Path(cfg['logging']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    for epoch in range(tcfg['epochs']):
        model.train()
        loss_meter = AverageMeter()

        for images, masks, texts in train_loader:
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images, texts)
            loss   = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            if tcfg.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg['grad_clip'])
            optimizer.step()
            loss_meter.update(loss.item(), images.size(0))

        scheduler.step()

        # validation
        metrics = evaluate(model, val_loader, device, cfg['eval']['threshold'])
        logger.info(
            f"Epoch [{epoch+1}/{tcfg['epochs']}] Loss: {loss_meter.avg:.4f} | "
            f"Dice: {metrics['dice']:.2f}% | mIoU: {metrics['miou']:.2f}% | "
            f"HD95: {metrics['hd95']:.2f} | ASSD: {metrics['assd']:.2f}"
        )

        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            save_checkpoint({'epoch': epoch+1, 'model': model.state_dict(),
                             'best_dice': best_dice},
                            str(ckpt_dir / 'c2seg_best.pth'))

        if (epoch + 1) % cfg['logging']['save_every'] == 0:
            save_checkpoint({'epoch': epoch+1, 'model': model.state_dict()},
                            str(ckpt_dir / f'c2seg_epoch{epoch+1}.pth'))

    logger.info(f"Training done. Best Dice: {best_dice:.2f}%")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    dice_m = AverageMeter()
    miou_m = AverageMeter()
    hd95_m = AverageMeter()
    assd_m = AverageMeter()

    with torch.no_grad():
        for images, masks, texts in loader:
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images, texts)
            B = images.size(0)
            dice_m.update(dice_score(logits, masks, threshold), 1)
            miou_m.update(miou_score(logits, masks, threshold), 1)
            hd95_m.update(hd95(logits, masks, threshold), 1)
            assd_m.update(assd(logits, masks, threshold), 1)

    return {'dice': dice_m.avg, 'miou': miou_m.avg,
            'hd95': hd95_m.avg, 'assd': assd_m.avg}


def eval_mode(cfg):
    logger = get_logger('c2seg_eval', cfg['logging']['log_dir'])
    device = get_device()

    test_ds = C2SegDataset(cfg['data']['test_dir'],
                           image_size=cfg['data']['image_size'], augment=False)
    loader  = DataLoader(test_ds, batch_size=cfg['eval']['batch_size'],
                         shuffle=False, num_workers=cfg['data']['num_workers'])

    model = C2Seg(**cfg['model']).to(device)
    load_checkpoint(cfg['eval']['checkpoint'], model)

    metrics = evaluate(model, loader, device, cfg['eval']['threshold'])
    logger.info(
        f"Test Results | Dice: {metrics['dice']:.2f}% | mIoU: {metrics['miou']:.2f}% | "
        f"HD95: {metrics['hd95']:.2f} | ASSD: {metrics['assd']:.2f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',  required=True, choices=['pretrain', 'train', 'eval'])
    p.add_argument('--config', default='configs/c2seg.yaml')
    p.add_argument('--pretrained', default=None, help='Path to pretrained ckpt for stage II')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = load_config(args.config)

    if args.stage == 'pretrain':
        pretrain(cfg)
    elif args.stage == 'train':
        train(cfg, pretrained_ckpt=args.pretrained)
    elif args.stage == 'eval':
        eval_mode(cfg)
