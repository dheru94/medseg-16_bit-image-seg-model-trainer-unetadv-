# evaluate.py
"""
Evaluate a trained checkpoint on a dataset and print metrics.

Usage (standalone):
    python evaluate.py --model unet --checkpoint checkpoints/best.pth \
                       --img-dir data/images --mask-dir data/masks

Also imported by train.py for mid-epoch/end-of-epoch validation.
"""
import argparse
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dice_score import dice_coeff, multiclass_dice_coeff
from utils.data_loading import SegmentationDataset
from models import build_model


@torch.inference_mode()
def evaluate(net, dataloader, device, amp: bool) -> float:
    """
    Compute mean Dice score over a dataloader.
    Called from train.py during validation rounds.
    """
    net.eval()
    total_dice  = 0.0
    num_batches = len(dataloader)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches,
                          desc='Validation', unit='batch', leave=False):
            images, masks_true = batch['image'], batch['mask']

            images     = images.to(device, dtype=torch.float32)
            if device.type == 'cuda':
                images = images.to(memory_format=torch.channels_last)
            masks_true = masks_true.to(device, dtype=torch.float32)

            masks_pred = net(images)
            # Unwrap deep supervision list — use the final (strongest) head
            if isinstance(masks_pred, (list, tuple)):
                masks_pred = masks_pred[-1]

            if net.n_classes == 1:
                assert masks_true.min() >= 0 and masks_true.max() <= 1
                pred_bin   = (torch.sigmoid(masks_pred) > 0.5).float()
                masks_true = masks_true.unsqueeze(1)
                total_dice += dice_coeff(pred_bin, masks_true, reduce_batch_first=False)
            else:
                masks_true_long = masks_true.long()
                assert masks_true_long.min() >= 0 and masks_true_long.max() < net.n_classes, \
                    f'Mask value out of range: min={masks_true_long.min()}, max={masks_true_long.max()}, n_classes={net.n_classes}'
                pred_oh    = F.one_hot(masks_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                true_oh    = F.one_hot(masks_true_long, net.n_classes).permute(0, 3, 1, 2).float()

                # Calculate dice for EACH class separately, then average
                dice_per_class = []
                for c in range(1, net.n_classes):  # Skip background (class 0)
                    pred_c = pred_oh[:, c]
                    true_c = true_oh[:, c]
                    # Calculate dice for this class
                    inter = 2 * (pred_c * true_c).sum()
                    sets_sum = pred_c.sum() + true_c.sum()
                    if sets_sum > 0:
                        dice_c = (inter + 1e-6) / (sets_sum + 1e-6)
                        dice_per_class.append(dice_c)

                # Average across all foreground classes
                if dice_per_class:
                    total_dice += torch.stack(dice_per_class).mean()

    net.train()
    return total_dice / max(num_batches, 1)


@torch.inference_mode()
def full_evaluate(net, dataloader, device, amp: bool, n_classes: int):
    """
    Full evaluation with per-class IoU, precision, recall, F1, Dice.
    Supports both binary (n_classes=1) and multiclass (n_classes>1).
    """
    net.eval()
    num_batches = len(dataloader)

    # Accumulators — one slot per class (binary uses index 0 only)
    n = max(n_classes, 1)
    tp = torch.zeros(n)
    fp = torch.zeros(n)
    fn = torch.zeros(n)
    dice_total = 0.0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc='Evaluating'):
            images, masks_true = batch['image'], batch['mask']
            images     = images.to(device, dtype=torch.float32)
            if device.type == 'cuda':
                images = images.to(memory_format=torch.channels_last)
            masks_true = masks_true.to(device, dtype=torch.float32)
            masks_pred = net(images)
            # Unwrap deep supervision list — use the final (strongest) head
            if isinstance(masks_pred, (list, tuple)):
                masks_pred = masks_pred[-1]

            # ── Binary ────────────────────────────────────────────────────────
            if n_classes == 1:
                pred = (torch.sigmoid(masks_pred.squeeze(1)) > 0.5).float()
                true = masks_true
                tp[0] += (pred * true).sum().item()
                fp[0] += (pred * (1 - true)).sum().item()
                fn[0] += ((1 - pred) * true).sum().item()
                dice_total += dice_coeff(
                    pred.unsqueeze(1), true.unsqueeze(1),
                    reduce_batch_first=False
                ).item()

            # ── Multiclass ────────────────────────────────────────────────────
            else:
                masks_true_long = masks_true.long()

                # Safety clamp — avoid index errors
                masks_true_long = masks_true_long.clamp(0, n_classes - 1)

                pred_classes = masks_pred.argmax(dim=1)          # (B, H, W)

                pred_oh = F.one_hot(pred_classes,  n_classes).permute(0, 3, 1, 2).float()
                true_oh = F.one_hot(masks_true_long, n_classes).permute(0, 3, 1, 2).float()

                # Per-class TP / FP / FN
                for c in range(n_classes):
                    p = pred_oh[:, c]
                    t = true_oh[:, c]
                    tp[c] += (p * t).sum().item()
                    fp[c] += (p * (1 - t)).sum().item()
                    fn[c] += ((1 - p) * t).sum().item()

                # Mean Dice (excluding background class 0)
                dice_total += multiclass_dice_coeff(
                    pred_oh[:, 1:], true_oh[:, 1:],
                    reduce_batch_first=False
                ).item()

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    precision_per = tp / (tp + fp + 1e-6)
    recall_per    = tp / (tp + fn + 1e-6)
    f1_per        = 2 * precision_per * recall_per / (precision_per + recall_per + 1e-6)
    iou_per       = tp / (tp + fp + fn + 1e-6)

    # Mean over foreground classes only (skip background=0 for multiclass)
    if n_classes == 1:
        precision = precision_per[0].item()
        recall    = recall_per[0].item()
        f1        = f1_per[0].item()
        iou       = iou_per[0].item()
        per_class = None
    else:
        precision = precision_per[1:].mean().item()
        recall    = recall_per[1:].mean().item()
        f1        = f1_per[1:].mean().item()
        iou       = iou_per[1:].mean().item()
        per_class = {
            f'class_{c}': {
                'iou':       iou_per[c].item(),
                'precision': precision_per[c].item(),
                'recall':    recall_per[c].item(),
                'f1':        f1_per[c].item(),
            }
            for c in range(n_classes)
        }

    dice = dice_total / max(num_batches, 1)

    return dict(
        dice      = dice,
        iou       = iou,
        precision = precision,
        recall    = recall,
        f1        = f1,
        per_class = per_class,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='MedSeg — Evaluate a checkpoint')
    p.add_argument('--model',        type=str,   required=True, help='unet | unetpp | deeplab')
    p.add_argument('--checkpoint',   type=str,   required=True)
    p.add_argument('--img-dir',      type=str,   required=True)
    p.add_argument('--mask-dir',     type=str,   required=True)
    p.add_argument('--mask-channel', type=int,   default=3)
    p.add_argument('--scale',        type=float, default=0.5)
    p.add_argument('--img-size',     type=int,   nargs=2, metavar=('W', 'H'), dest='img_size',
                   help='Fixed output size e.g. 512 512 (overrides scale)')
    p.add_argument('--batch-size',   type=int,   default=1)
    p.add_argument('--amp',          action='store_true', default=False)
    p.add_argument('--num-workers',  type=int,   default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt       = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_channels = ckpt.get('n_channels', 1)
    n_classes  = ckpt.get('n_classes',  1)
    epoch      = ckpt.get('epoch',      '?')
    val_dice   = ckpt.get('val_dice',   '?')

    logging.info(f'Checkpoint  — epoch: {epoch}, val_dice: {val_dice}')
    logging.info(f'Model config — n_channels: {n_channels}, n_classes: {n_classes}')

    model = build_model(args.model, n_channels=n_channels, n_classes=n_classes)
    model.load_state_dict(ckpt['model_state'])
    model.to(memory_format=torch.channels_last).to(device)
    logging.info(f'Loaded {args.model} from {args.checkpoint}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    img_size = tuple(args.img_size) if args.img_size else (512, 512)

    dataset = SegmentationDataset(
        images_dir   = args.img_dir,
        mask_dir     = args.mask_dir,
        scale        = args.scale,
        mask_channel = args.mask_channel,
        img_size     = img_size,
    )

    # Warn if dataset classes don't match checkpoint
    n_detected = len(dataset.mask_values)
    if n_detected != n_classes:
        logging.warning(
            f'Checkpoint has n_classes={n_classes} but dataset has '
            f'{n_detected} unique values: {dataset.mask_values}. '
            f'Metrics may be wrong — did you use the right checkpoint?'
        )

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = full_evaluate(model, loader, device, args.amp, n_classes)

    print('\n' + '═' * 45)
    print('  Evaluation Results')
    print('═' * 45)
    print(f'  {"dice":<14}: {metrics["dice"]:.4f}')
    print(f'  {"iou":<14}: {metrics["iou"]:.4f}')
    print(f'  {"precision":<14}: {metrics["precision"]:.4f}')
    print(f'  {"recall":<14}: {metrics["recall"]:.4f}')
    print(f'  {"f1":<14}: {metrics["f1"]:.4f}')

    if metrics['per_class']:
        print('─' * 45)
        print('  Per-Class Breakdown')
        print('─' * 45)
        for cls, m in metrics['per_class'].items():
            print(f'  {cls}:')
            for k, v in m.items():
                print(f'    {k:<12}: {v:.4f}')

    print('═' * 45)