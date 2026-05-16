# train.py
"""
Train UNet / UNet++ / DeepLab v3+ on segmentation data.

Usage:
    python train.py --model unet --epochs 150 --batch-size 2 --lr 1e-4 --amp --img-size 640 640 --n-channels 1 --classes 4
    python train.py --model unetpp --epochs 150 --batch-size 2 --lr 1e-4 --amp --img-size 640 640 --n-channels 1 --classes 4
    python train.py --model deeplab --epochs 150 --batch-size 2 --lr 1e-4 --amp --img-size 640 640 --n-channels 1 --classes 4
"""
import argparse
import logging
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models import build_model
from utils.data_loading import SegmentationDataset
from utils.dice_score import dice_loss
from evaluate import evaluate


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss with numerical stability fix.
    - Clamps exp() input to prevent overflow -> NaN
    - Supports per-class inverse-frequency weights
    - gamma=2.0 standard; use 1.0 if NaN persists
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce    = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        # clamp max=88 prevents exp() overflow -> NaN/Inf
        pt    = torch.exp(-ce.clamp(max=88.0))
        focal = ((1 - pt) ** self.gamma * ce).mean()
        return focal


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str = 'configs/default.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def log_metrics(metrics: dict, log_path: Path):
    line = ', '.join(f'{k}: {v}' for k, v in metrics.items())
    with open(log_path, 'a') as f:
        f.write(line + '\n')


def save_checkpoint(model, dataset, epoch: int, val_dice: float,
                    checkpoint_dir: Path, best_dice: float,
                    keep_best_only: bool) -> float:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch':       epoch,
        'val_dice':    val_dice,
        'model_state': model.state_dict(),
        'mask_values': dataset.mask_values,
        'n_channels':  model.n_channels,
        'n_classes':   model.n_classes,
    }

    # Always save latest
    torch.save(state, checkpoint_dir / 'latest.pth')

    # Save per-epoch (unless keep_best_only)
    if not keep_best_only:
        torch.save(state, checkpoint_dir / f'epoch_{epoch:03d}_dice_{val_dice:.4f}.pth')

    # Save best
    if val_dice > best_dice:
        torch.save(state, checkpoint_dir / 'best.pth')
        logging.info(f'  ★ New best checkpoint! Dice: {val_dice:.4f}')
        return val_dice

    return best_dice


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(cfg: dict, device: torch.device):

    # ── Validate data directories ─────────────────────────────────────────────
    img_dir_path = Path(cfg['img_dir'])
    mask_dir_path = Path(cfg['mask_dir'])

    if not img_dir_path.exists():
        raise FileNotFoundError(f'Images directory does not exist: {img_dir_path}')
    if not mask_dir_path.exists():
        raise FileNotFoundError(f'Masks directory does not exist: {mask_dir_path}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = SegmentationDataset(
        images_dir   = cfg['img_dir'],
        mask_dir     = cfg['mask_dir'],
        scale        = cfg['scale'],
        mask_channel = cfg['mask_channel'],
        augment      = True,
        img_size     = tuple(cfg.get('img_size', [512, 512])),
    )

    # ── Auto-correct n_classes from actual mask data ───────────────────────
    n_detected = len(dataset.mask_values)
    if cfg['n_classes'] != n_detected:
        logging.warning(
            f'n_classes in config is {cfg["n_classes"]} but dataset has '
            f'{n_detected} unique mask values: {dataset.mask_values}. '
            f'Auto-correcting to {n_detected}.'
        )
        cfg['n_classes'] = n_detected

    n_val   = int(len(dataset) * cfg['val_percent'])
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    loader_args  = dict(batch_size=cfg['batch_size'], num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_set,   shuffle=False, drop_last=True, **loader_args)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        name       = cfg['model'],
        n_channels = cfg['n_channels'],
        n_classes  = cfg['n_classes'],
        bilinear   = cfg.get('bilinear', False),
        dropout    = cfg.get('dropout', 0.0),  # Disable dropout for small dataset
    ).to(memory_format=torch.channels_last).to(device)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_dice   = 0.0
    if cfg.get('load'):
        ckpt = torch.load(cfg['load'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_dice   = ckpt.get('val_dice', 0.0)
        if 'n_classes' in ckpt:
            cfg['n_classes'] = ckpt['n_classes']
        if 'n_channels' in ckpt:
            cfg['n_channels'] = ckpt['n_channels']
        logging.info(f'Resumed from {cfg["load"]} (epoch {start_epoch - 1}, dice {best_dice:.4f}), n_classes={cfg["n_classes"]}, n_channels={cfg["n_channels"]}')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if cfg['model'] == 'deeplab':
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'],
                                  momentum=cfg['momentum'], foreach=True)

    # ── Scheduler — only steps once per epoch, patience=10, factor=0.5 ───────
    # FIX: was stepping 6x per epoch (5 mid-epoch + 1 end) which killed LR fast
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'max',
        patience = 10,       # wait 10 full epochs before reducing
        factor   = 0.5,      # halve LR (not 10x drop)
        min_lr   = 1e-6,     # never go below 1e-6
    )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'])

    # ── Loss with class balancing ─────────────────────────────────────────────
    if cfg['n_classes'] > 1:
        # Count pixels per class across the training set
        logging.info('Computing class weights from training set (inverse frequency)...')
        class_counts = torch.zeros(cfg['n_classes'])
        for item in tqdm(train_set, desc='Counting pixels', leave=False):
            mask = item['mask'].long()
            for c in range(cfg['n_classes']):
                class_counts[c] += (mask == c).sum()

        logging.info(f'Pixel counts per class: {class_counts.long().tolist()}')

        # Inverse frequency — rare defect classes get much higher weight
        class_weights = 1.0 / (class_counts.float() + 1)
        class_weights = class_weights / class_weights.sum() * cfg['n_classes']
        class_weights = class_weights.clamp(max=10.0)
        class_weights = class_weights.to(device)
        logging.info(f'Class weights: {[round(w, 4) for w in class_weights.tolist()]}')

        # Use CrossEntropyLoss like YOLO (simpler, more stable)
        # Added ignore_index=0 to focus learning on defect classes
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)

    else:
        pos_weight = torch.tensor([cfg['pos_weight']]).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_path = Path(cfg['log_dir']) / f"{cfg['model']}_training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(f"=== MedSeg Training Log ===\n")
        f.write(f"Model: {cfg['model']}, Channels: {cfg['n_channels']}, "
                f"Classes: {cfg['n_classes']}\n")
        f.write(f"Train: {n_train}, Val: {n_val}, Device: {device}\n\n")

    logging.info(f'''
    ╔══════════════════════════════════╗
    ║        MedSeg Training           ║
    ╠══════════════════════════════════╣
    ║  Model:    {cfg["model"]:<22} ║
    ║  Channels: {cfg["n_channels"]:<22} ║
    ║  Classes:  {cfg["n_classes"]:<22} ║
    ║  Epochs:   {cfg["epochs"]:<22} ║
    ║  Batch:    {cfg["batch_size"]:<22} ║
    ║  LR:       {cfg["lr"]:<22} ║
    ║  Train:    {n_train:<22} ║
    ║  Val:      {n_val:<22} ║
    ║  Device:   {str(device):<22} ║
    ║  AMP:      {str(cfg["amp"]):<22} ║
    ╚══════════════════════════════════╝
    ''')

    global_step = 0
    nan_count   = 0

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{cfg["epochs"]}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, (
                    f'Model expects {model.n_channels} channels '
                    f'but got {images.shape[1]}. Check n_channels in config.'
                )

                images     = images.to(device, dtype=torch.float32)
                if device.type == 'cuda':
                    images = images.to(memory_format=torch.channels_last)
                true_masks = true_masks.to(device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu',
                                    enabled=cfg['amp']):
                    preds = model(images)

                    if cfg['n_classes'] == 1:
                        pred  = preds.squeeze(1)
                        probs = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)
                        loss  = criterion(pred, true_masks)
                        loss += 2 * dice_loss(probs, true_masks, multiclass=False)
                    else:
                        true_long = true_masks.long()
                        # Safety clamp — prevent out-of-range class index crashes
                        true_long = true_long.clamp(0, cfg['n_classes'] - 1)

                        # Simple CrossEntropyLoss like YOLO uses
                        loss = criterion(preds, true_long)

                    # DEBUG: Print model predictions every 50 steps
                    if global_step % 50 == 0:
                        with torch.no_grad():
                            pred_classes = preds.argmax(dim=1)
                            unique_preds = pred_classes.unique().tolist()
                            unique_true = true_long.unique().tolist() if cfg['n_classes'] > 1 else []
                            logging.info(f'DEBUG - Pred classes: {unique_preds}, True classes: {unique_true}')

                # ── NaN / Inf guard ───────────────────────────────────────────
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    logging.warning(
                        f'NaN/Inf loss at step {global_step} — skipping batch '
                        f'({nan_count} total skips this run)'
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if nan_count > 50:
                        logging.error(
                            'More than 50 NaN batches — training is unstable. '
                            'Try: lower --lr (e.g. 5e-5) or --focal-gamma 1.0'
                        )
                    continue
                else:
                    nan_count = 0   # reset on clean batch

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clipping'])
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss  += loss.item()
                num_batches += 1
                global_step += 1

                pbar.update(images.shape[0])
                pbar.set_postfix(loss=f'{loss.item():.4f}')

                if global_step % cfg['log_interval'] == 0:
                    log_metrics({'step': global_step, 'epoch': epoch,
                                 'loss': round(loss.item(), 4)}, log_path)

                # ── Mid-epoch validation (monitoring only — NO scheduler step) ─
                div = n_train // (5 * cfg['batch_size'])
                if div > 0 and global_step % div == 0:
                    val_dice = evaluate(model, val_loader, device, cfg['amp'])
                    # NOTE: scheduler.step() intentionally NOT called here
                    # Calling it mid-epoch was causing LR to drop 10x every epoch
                    logging.info(f'  Val Dice: {val_dice:.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
                    log_metrics({'step': global_step, 'epoch': epoch,
                                 'lr':       optimizer.param_groups[0]['lr'],
                                 'val_dice': round(float(val_dice), 4)}, log_path)

        # ── End of epoch — scheduler steps ONCE here only ─────────────────────
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
        val_dice = evaluate(model, val_loader, device, cfg['amp'])
        scheduler.step(val_dice)   # ← only place scheduler is called

        logging.info(
            f'Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f} '
            f'| LR: {optimizer.param_groups[0]["lr"]:.2e}'
        )
        log_metrics({
            'epoch':    epoch,
            'avg_loss': round(avg_loss, 4),
            'val_dice': round(float(val_dice), 4),
            'lr':       optimizer.param_groups[0]['lr'],
        }, log_path)

        if epoch % cfg['save_every'] == 0:
            best_dice = save_checkpoint(
                model, dataset, epoch, float(val_dice),
                Path(cfg['checkpoint_dir']), best_dice,
                cfg['keep_best_only']
            )

    logging.info(f'Training complete. Best Val Dice: {best_dice:.4f}')
    logging.info(f'Best checkpoint saved to: {cfg["checkpoint_dir"]}/best.pth')


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='MedSeg — Train segmentation models')
    p.add_argument('--config',       type=str,   default='configs/default.yaml')
    p.add_argument('--model',        type=str,   help='unet | unetpp | deeplab')
    p.add_argument('--epochs',       '-e', type=int)
    p.add_argument('--batch-size',   '-b', type=int,   dest='batch_size')
    p.add_argument('--lr',           '-l', type=float)
    p.add_argument('--scale',        '-s', type=float)
    p.add_argument('--img-dir',      type=str)
    p.add_argument('--mask-dir',     type=str)
    p.add_argument('--mask-channel', type=int,   dest='mask_channel',
                   help='Which channel is the mask (default 3 = alpha)')
    p.add_argument('--n-channels',   type=int,   dest='n_channels')
    p.add_argument('--classes',      '-c', type=int,   dest='n_classes')
    p.add_argument('--load',         '-f', type=str,   help='Resume from checkpoint')
    p.add_argument('--amp',          action='store_true', default=False)
    p.add_argument('--no-amp',       action='store_false', dest='amp')
    p.add_argument('--bilinear',     action='store_true', default=False)
    p.add_argument('--img-size',     type=int,   nargs=2, metavar=('W', 'H'), dest='img_size',
                   help='Fixed output size: W H (e.g. 512 512)')
    p.add_argument('--focal-gamma',  type=float, dest='focal_gamma', default=2.0,
                   help='Focal loss gamma — lower to 1.0 if NaN persists (default 2.0)')
    p.add_argument('--dropout',     type=float, default=0.2,
                   help='Dropout probability at bottleneck (default 0.2, reduce for small datasets)')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load config then override with any CLI args provided
    cfg = load_config(args.config)
    skip_keys = {'config', 'amp', 'bilinear'}
    for k, v in vars(args).items():
        if k in skip_keys:
            continue
        if v is not None:
            cfg[k] = v

    # Only override booleans if explicitly passed on CLI
    if '--amp' in sys.argv:
        cfg['amp'] = True
    if '--no-amp' in sys.argv:
        cfg['amp'] = False
    if '--bilinear' in sys.argv:
        cfg['bilinear'] = True

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info('Note: MPS (Apple Silicon) has limited FP16 support. Consider using CPU for stability.')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device: {device}')

    try:
        train_model(cfg, device)
    except torch.cuda.OutOfMemoryError:
        logging.error('Out of memory! Try reducing --batch-size or --img-size.')
        raise
