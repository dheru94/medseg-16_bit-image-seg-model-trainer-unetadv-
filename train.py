##############
# python train.py --model unetadv --epochs 100 --batch-size 1 --lr 1e-4 --scale 1.0 --img-dir D:\itarsi\mask_data\images --mask-dir D:\itarsi\mask_data\masks --test-img-dir D:\itarsi\mask_data\test\image --test-mask-dir D:\itarsi\mask_data\test\mask --n-channels 1 --classes 4 --img-size 640 640 --focal-gamma 2.0 --dice-weight 2.0 --dice-warmup-epochs 5 --dropout 0.3 --amp


#########################











# train.py
"""
Train UNet / UNet++ / DeepLab v3+ / UNetAdv on segmentation data.

Usage examples:
    # Stratified random split (recommended when you have no separate test set)
    python train.py --model unetadv --epochs 50 --batch-size 1 --lr 1e-4 \
                    --amp --img-size 512 512 --n-channels 1 --classes 4

    # Provide a dedicated test set (images guaranteed to contain all classes)
    python train.py --model unetadv --epochs 50 --batch-size 1 --lr 1e-4 \
                    --amp --img-size 512 512 --n-channels 1 --classes 4 \
                    --test-img-dir data/test/images --test-mask-dir data/test/masks

Deep-supervision note (UNetAdv):
    When deep_supervision=True the model returns a list of 4 logit tensors.
    train.py computes a weighted sum of losses across all heads:
        loss = 0.1*L(out1) + 0.2*L(out2) + 0.3*L(out3) + 0.4*L(out4)
    This lets early heads learn coarse features while the final head is
    optimised most strongly. At inference, only the last head is used.

Stratified split note:
    random_split() is replaced with stratified_split(), which groups images
    by the set of classes they contain and splits each group proportionally.
    This guarantees every rare class (1 / 2 / 3) appears in train, val, AND
    test — critical when class 2 has only ~11K pixels total.
    If you supply --test-img-dir / --test-mask-dir, the stratified split is
    used only for train/val; your external test set is loaded as-is.
"""

import argparse
import logging
import sys
import yaml
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models import build_model
from utils.data_loading import SegmentationDataset
from utils.dice_score import dice_loss
from evaluate import evaluate


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy background pixels so the model focuses
    on hard rare foreground classes.

    gamma=2.0 is the safe default.
    Only go to 3.0 if classes 1/2/3 are still ignored after 10+ epochs.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce.clamp(max=88.0))   # clamp prevents exp() overflow
        return ((1 - pt) ** self.gamma * ce).mean()


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
    torch.save(state, checkpoint_dir / 'latest.pth')
    if not keep_best_only:
        torch.save(state, checkpoint_dir / f'epoch_{epoch:03d}_dice_{val_dice:.4f}.pth')
    if val_dice > best_dice:
        torch.save(state, checkpoint_dir / 'best.pth')
        logging.info(f'  ★ New best checkpoint! Dice: {val_dice:.4f}')
        return val_dice
    return best_dice


# ── Stratified split ──────────────────────────────────────────────────────────

def stratified_split(dataset: SegmentationDataset,
                     val_frac: float,
                     test_frac: float,
                     seed: int = 42):
    """
    Split dataset indices into train / val / test while keeping the class
    distribution as balanced as possible across all three splits.

    Strategy
    --------
    1. For every image, compute the *frozenset* of classes present in its mask.
       This is the image's "stratum key".
    2. Within each stratum, shuffle and split proportionally into
       train / val / test.
    3. Concatenate the per-stratum splits.

    Why frozenset of classes?
    -------------------------
    With severe imbalance (class 2 ≈ 0.01 % of pixels) a random split can
    easily send ALL images containing class 2 into training, leaving val and
    test with zero class-2 examples. Stratifying by class presence prevents
    this collapse.

    Edge cases
    ----------
    - If a stratum has too few images to contribute at least 1 sample to each
      split, all samples from that stratum go to training and a warning is
      logged. You will also see a "classes missing from val/test" warning at
      the end so you know to either collect more data or reduce val/test_frac.
    """
    rng = torch.Generator().manual_seed(seed)

    # ── Step 1: assign each image to a stratum ────────────────────────────────
    strata: dict[frozenset, list[int]] = defaultdict(list)
    logging.info('Stratifying dataset by class presence…')
    for idx in tqdm(range(len(dataset)), desc='Scanning masks', leave=False):
        mask = dataset[idx]['mask'].long()          # (H, W) or (1, H, W)
        classes_present = frozenset(mask.unique().tolist())
        strata[classes_present].append(idx)

    logging.info('Strata (class-set → image count):')
    for k, v in sorted(strata.items(), key=lambda x: -len(x[1])):
        logging.info(f'  {set(k)}: {len(v)} images')

    train_ids, val_ids, test_ids = [], [], []

    # ── Step 2: split each stratum proportionally ─────────────────────────────
    for key, indices in strata.items():
        n = len(indices)

        # Shuffle within stratum
        perm = torch.randperm(n, generator=rng).tolist()
        indices = [indices[i] for i in perm]

        n_test  = max(1, round(n * test_frac))  if n >= 3 else 0
        n_val   = max(1, round(n * val_frac))   if n >= 3 else 0
        n_val   = min(n_val,  n - n_test - 1)   # always keep ≥1 for train
        n_val   = max(n_val,  0)

        if n < 3:
            # Too few to split — put everything in train
            logging.warning(
                f'Stratum {set(key)} has only {n} image(s); '
                f'all assigned to training.'
            )
            train_ids.extend(indices)
            continue

        test_ids.extend(indices[:n_test])
        val_ids.extend(indices[n_test: n_test + n_val])
        train_ids.extend(indices[n_test + n_val:])

    # ── Step 3: sanity-check coverage ─────────────────────────────────────────
    def classes_in(ids):
        s = set()
        for i in ids:
            mask = dataset[i]['mask'].long()
            s |= set(mask.unique().tolist())
        return s

    all_classes = set().union(*strata.keys())
    val_classes  = classes_in(val_ids)
    test_classes = classes_in(test_ids)
    missing_val  = all_classes - val_classes
    missing_test = all_classes - test_classes

    if missing_val:
        logging.warning(
            f'Classes {missing_val} are absent from the validation split! '
            f'Consider collecting more images that contain these classes, '
            f'or reduce --val-percent.'
        )
    if missing_test:
        logging.warning(
            f'Classes {missing_test} are absent from the test split! '
            f'Consider using --test-img-dir with a manually curated test set.'
        )
    if not missing_val and not missing_test:
        logging.info(
            f'✓ All classes {all_classes} present in train, val, and test splits.'
        )

    logging.info(
        f'Split sizes — train: {len(train_ids)}, '
        f'val: {len(val_ids)}, test: {len(test_ids)}'
    )
    return train_ids, val_ids, test_ids


# ── Loss computation ──────────────────────────────────────────────────────────

DS_WEIGHTS = [0.1, 0.2, 0.3, 0.4]


def compute_loss(criterion, preds, true_masks, true_long,
                 n_classes: int, dice_weight: float):
    """
    Dispatch to single or deep-supervision loss.
    preds: single logit tensor OR list of 4 tensors (UNetAdv deep supervision).
    """
    if isinstance(preds, (list, tuple)):
        total = torch.tensor(0.0, device=true_masks.device)
        for w, p in zip(DS_WEIGHTS, preds):
            total = total + w * _single_loss(
                criterion, p, true_masks, true_long, n_classes, dice_weight)
        return total
    else:
        return _single_loss(criterion, preds, true_masks,
                            true_long, n_classes, dice_weight)


def _single_loss(criterion, pred, true_masks, true_long,
                 n_classes: int, dice_weight: float):
    """
    Focal loss  +  foreground-only Dice loss.

    ── Why foreground-only Dice? ────────────────────────────────────────────
    Your dataset:
        class 0 (background): 79,323,984 pixels  (~98%)
        class 1:                  285,325 pixels
        class 2:                   11,063 pixels
        class 3:                   71,364 pixels

    If Dice includes background, the model can achieve a near-perfect Dice
    score by predicting ALL background — collapsing Val Dice to 0.0000.
    Excluding background ([:, 1:]) forces Dice to only measure foreground
    overlap, which is what actually matters for segmentation quality.
    ─────────────────────────────────────────────────────────────────────────
    """
    if n_classes == 1:
        # Binary: one channel, use sigmoid + BCE + Dice
        p     = pred.squeeze(1)
        probs = torch.sigmoid(p).clamp(1e-6, 1 - 1e-6)
        return (criterion(p, true_masks)
                + dice_weight * dice_loss(probs, true_masks, multiclass=False))
    else:
        # Multiclass: Focal loss + foreground-only Dice
        focal   = criterion(pred, true_long)

        probs   = F.softmax(pred, dim=1).clamp(1e-6, 1 - 1e-6)
        true_oh = F.one_hot(true_long, n_classes).permute(0, 3, 1, 2).float()

        # [:, 1:] skips background (class 0) — prevents collapse to all-background
        fg_dice = dice_loss(probs[:, 1:], true_oh[:, 1:], multiclass=True)

        return focal + dice_weight * fg_dice


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(cfg: dict, device: torch.device):

    # ── Validate directories ──────────────────────────────────────────────────
    for key in ('img_dir', 'mask_dir'):
        p = Path(cfg[key])
        if not p.exists():
            raise FileNotFoundError(f'{key} does not exist: {p}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = SegmentationDataset(
        images_dir   = cfg['img_dir'],
        mask_dir     = cfg['mask_dir'],
        scale        = cfg['scale'],
        mask_channel = cfg['mask_channel'],
        augment      = True,
        img_size     = tuple(cfg.get('img_size', [512, 512])),
    )

    # Auto-correct n_classes from actual mask data
    n_detected = len(dataset.mask_values)
    if cfg['n_classes'] != n_detected:
        logging.warning(
            f'n_classes={cfg["n_classes"]} but dataset has {n_detected} unique '
            f'mask values {dataset.mask_values}. Auto-correcting.'
        )
        cfg['n_classes'] = n_detected

    # ── Split: external test set OR stratified split ──────────────────────────
    has_external_test = cfg.get('test_img_dir') and cfg.get('test_mask_dir')

    if has_external_test:
        # ── Validate external test directories ────────────────────────────────
        for key in ('test_img_dir', 'test_mask_dir'):
            p = Path(cfg[key])
            if not p.exists():
                raise FileNotFoundError(f'{key} does not exist: {p}')

        # Use stratified split only for train / val
        train_ids, val_ids, _ = stratified_split(
            dataset,
            val_frac  = cfg['val_percent'],
            test_frac = 0.0,          # no test portion taken from main dataset
            seed      = 42,
        )

        # Load the separate, manually curated test set (no augmentation)
        test_dataset = SegmentationDataset(
            images_dir   = cfg['test_img_dir'],
            mask_dir     = cfg['test_mask_dir'],
            scale        = cfg['scale'],
            mask_channel = cfg['mask_channel'],
            augment      = False,     # never augment test data
            img_size     = tuple(cfg.get('img_size', [512, 512])),
        )

        # Verify class coverage of the external test set
        test_classes = set()
        for i in range(len(test_dataset)):
            mask = test_dataset[i]['mask'].long()
            test_classes |= set(mask.unique().tolist())

        all_classes = set(range(cfg['n_classes']))
        missing = all_classes - test_classes
        if missing:
            logging.warning(
                f'External test set is missing classes {missing}! '
                f'Dice scores for those classes will be trivially 0 or NaN. '
                f'Add images containing those classes to your test set.'
            )
        else:
            logging.info(
                f'✓ External test set covers all {cfg["n_classes"]} classes.'
            )

        logging.info(
            f'Using external test set: {cfg["test_img_dir"]} '
            f'({len(test_dataset)} images)'
        )

    else:
        # ── Stratified three-way split from the single dataset ────────────────
        logging.info(
            'No external test set provided — using stratified three-way split.'
        )
        train_ids, val_ids, test_ids = stratified_split(
            dataset,
            val_frac  = cfg['val_percent'],
            test_frac = cfg.get('test_percent', 0.1),
            seed      = 42,
        )
        test_dataset = Subset(dataset, test_ids)

    train_set = Subset(dataset, train_ids)
    val_set   = Subset(dataset, val_ids)

    n_train = len(train_set)
    n_val   = len(val_set)
    n_test  = len(test_dataset)

    loader_args  = dict(batch_size=cfg['batch_size'], num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set,    shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_set,      shuffle=False, drop_last=True, **loader_args)
    test_loader  = DataLoader(test_dataset, shuffle=False, **loader_args)

    # ── Save test indices so you can reproduce the exact split later ──────────
    split_path = Path(cfg['checkpoint_dir']) / 'data_split.yaml'
    split_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml as _yaml
    with open(split_path, 'w') as f:
        _yaml.dump({
            'train_indices': train_ids,
            'val_indices':   val_ids,
            'test_source':   cfg.get('test_img_dir', 'internal_split'),
            'test_indices':  [] if has_external_test else list(test_ids),
        }, f)
    logging.info(f'Data split saved to {split_path}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        name       = cfg['model'],
        n_channels = cfg['n_channels'],
        n_classes  = cfg['n_classes'],
        bilinear   = cfg.get('bilinear', False),
        dropout    = cfg.get('dropout', 0.0),
    ).to(memory_format=torch.channels_last).to(device)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_dice   = 0.0
    if cfg.get('load'):
        ckpt = torch.load(cfg['load'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_dice   = ckpt.get('val_dice', 0.0)
        if 'n_classes'  in ckpt: cfg['n_classes']  = ckpt['n_classes']
        if 'n_channels' in ckpt: cfg['n_channels'] = ckpt['n_channels']
        logging.info(
            f'Resumed from {cfg["load"]} '
            f'(epoch {start_epoch - 1}, dice {best_dice:.4f})'
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = cfg['lr'],
        weight_decay = cfg.get('weight_decay', 1e-4),
    )

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'max',
        patience = 10,
        factor   = 0.5,
        min_lr   = 1e-6,
    )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'])

    # ── Class-weighted Focal Loss ─────────────────────────────────────────────
    if cfg['n_classes'] > 1:
        logging.info('Computing class weights (inverse frequency)…')
        class_counts = torch.zeros(cfg['n_classes'])
        for item in tqdm(train_set, desc='Counting pixels', leave=False):
            mask = item['mask'].long()
            for c in range(cfg['n_classes']):
                class_counts[c] += (mask == c).sum()

        logging.info(f'Pixel counts per class: {class_counts.long().tolist()}')

        # sqrt-inverse frequency — smoother than raw inverse for extreme imbalance
        class_weights = 1.0 / (class_counts.float().sqrt() + 1)
        class_weights = class_weights / class_weights.sum() * cfg['n_classes']
        class_weights = class_weights.clamp(max=20.0).to(device)
        logging.info(f'Class weights: {[round(w, 4) for w in class_weights.tolist()]}')

        criterion = FocalLoss(
            gamma  = cfg.get('focal_gamma', 2.0),
            weight = class_weights,
        )
    else:
        pos_weight = torch.tensor([cfg['pos_weight']]).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_path = Path(cfg['log_dir']) / f"{cfg['model']}_training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        f.write('=== MedSeg Training Log ===\n')
        f.write(f"Model: {cfg['model']}, Ch: {cfg['n_channels']}, "
                f"Classes: {cfg['n_classes']}\n")
        f.write(f"FocalGamma: {cfg.get('focal_gamma', 2.0)}, "
                f"DiceWeight: {cfg.get('dice_weight', 2.0)}\n")
        f.write(f"Train: {n_train}, Val: {n_val}, Test: {n_test}, "
                f"Device: {device}\n\n")

    logging.info(f'''
    ╔══════════════════════════════════╗
    ║        MedSeg Training           ║
    ╠══════════════════════════════════╣
    ║  Model:      {cfg["model"]:<20} ║
    ║  Channels:   {cfg["n_channels"]:<20} ║
    ║  Classes:    {cfg["n_classes"]:<20} ║
    ║  Epochs:     {cfg["epochs"]:<20} ║
    ║  Batch:      {cfg["batch_size"]:<20} ║
    ║  LR:         {cfg["lr"]:<20} ║
    ║  FocalGamma: {cfg.get("focal_gamma", 2.0):<20} ║
    ║  DiceWeight: {cfg.get("dice_weight", 2.0):<20} ║
    ║  Train:      {n_train:<20} ║
    ║  Val:        {n_val:<20} ║
    ║  Test:       {n_test:<20} ║
    ║  Device:     {str(device):<20} ║
    ║  AMP:        {str(cfg["amp"]):<20} ║
    ╚══════════════════════════════════╝
    ''')

    global_step = 0
    nan_count   = 0
    dice_weight_max = cfg.get('dice_weight', 2.0)

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        # Dice warmup: ramp from 0 → full weight over first 5 epochs.
        # Prevents Dice from overwhelming Focal loss before model has
        # learned any basic class structure.
        warmup_epochs = cfg.get('dice_warmup_epochs', 5)
        dice_weight   = dice_weight_max * min(1.0, epoch / warmup_epochs)

        with tqdm(total=n_train,
                  desc=f'Epoch {epoch}/{cfg["epochs"]}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, (
                    f'Model expects {model.n_channels} ch, '
                    f'got {images.shape[1]}. Check n_channels in config.'
                )

                images     = images.to(device=device, dtype=torch.float32,
                                       memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(
                    device.type if device.type != 'mps' else 'cpu',
                    enabled=cfg['amp']
                ):
                    preds = model(images)

                    true_long = None
                    if cfg['n_classes'] > 1:
                        true_long = true_masks.long().clamp(0, cfg['n_classes'] - 1)

                    loss = compute_loss(
                        criterion, preds, true_masks,
                        true_long, cfg['n_classes'],
                        dice_weight=dice_weight,
                    )

                # ── Diagnostics every 50 steps ────────────────────────────────
                if global_step % 50 == 0:
                    with torch.no_grad():
                        p_ins    = preds[-1] if isinstance(preds, (list, tuple)) else preds
                        pred_cls = p_ins.argmax(dim=1)
                        logging.info(
                            f'DEBUG step {global_step} '
                            f'(dice_w={dice_weight:.2f}) — '
                            f'pred: {sorted(pred_cls.unique().tolist())}'
                            + (f', true: {sorted(true_long.unique().tolist())}'
                               if true_long is not None else '')
                        )

                # ── NaN / Inf guard ───────────────────────────────────────────
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    logging.warning(
                        f'NaN/Inf loss at step {global_step} — skipping '
                        f'({nan_count} total skips)'
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if nan_count > 50:
                        logging.error(
                            'More than 50 NaN batches — training is unstable. '
                            'Try: lower --lr (e.g. 5e-5) or --focal-gamma 1.0'
                        )
                    continue

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg['gradient_clipping'])
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss  += loss.item()
                num_batches += 1
                global_step += 1

                pbar.update(images.shape[0])
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                 dice_w=f'{dice_weight:.2f}')

                if global_step % cfg['log_interval'] == 0:
                    log_metrics({'step': global_step, 'epoch': epoch,
                                 'loss': round(loss.item(), 4),
                                 'dice_w': round(dice_weight, 4)}, log_path)

                # ── Mid-epoch validation (monitor only) ───────────────────────
                div = n_train // (5 * cfg['batch_size'])
                if div > 0 and global_step % div == 0:
                    val_dice = evaluate(model, val_loader, device, cfg['amp'])
                    logging.info(
                        f'  Val Dice: {val_dice:.4f} '
                        f'| LR: {optimizer.param_groups[0]["lr"]:.2e} '
                        f'| dice_w: {dice_weight:.2f}'
                    )
                    log_metrics({'step': global_step, 'epoch': epoch,
                                 'lr':       optimizer.param_groups[0]['lr'],
                                 'val_dice': round(float(val_dice), 4)}, log_path)

        # ── End-of-epoch ──────────────────────────────────────────────────────
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
        val_dice = evaluate(model, val_loader, device, cfg['amp'])
        scheduler.step(val_dice)

        logging.info(
            f'Epoch {epoch} | Avg Loss: {avg_loss:.4f} '
            f'| Val Dice: {val_dice:.4f} '
            f'| dice_w: {dice_weight:.2f} '
            f'| LR: {optimizer.param_groups[0]["lr"]:.2e}'
        )
        log_metrics({
            'epoch':    epoch,
            'avg_loss': round(avg_loss, 4),
            'val_dice': round(float(val_dice), 4),
            'dice_w':   round(dice_weight, 4),
            'lr':       optimizer.param_groups[0]['lr'],
        }, log_path)

        if epoch % cfg['save_every'] == 0:
            best_dice = save_checkpoint(
                model, dataset, epoch, float(val_dice),
                Path(cfg['checkpoint_dir']), best_dice, cfg['keep_best_only']
            )

    # ── Final evaluation on test set ──────────────────────────────────────────
    logging.info('=' * 60)
    logging.info('Running final evaluation on TEST set…')
    test_dice_per_class = evaluate_per_class(
        model, test_loader, device, cfg['amp'], cfg['n_classes']
    )
    mean_fg_dice = test_dice_per_class[1:].mean().item()  # exclude background

    logging.info('TEST SET RESULTS:')
    for c, d in enumerate(test_dice_per_class.tolist()):
        label = 'background' if c == 0 else f'class {c}'
        logging.info(f'  {label:>12}: Dice = {d:.4f}')
    logging.info(f'  {"mean fg":>12}: Dice = {mean_fg_dice:.4f}')

    log_metrics({
        'test_dice_bg': round(test_dice_per_class[0].item(), 4),
        **{f'test_dice_c{c}': round(test_dice_per_class[c].item(), 4)
           for c in range(1, cfg['n_classes'])},
        'test_dice_mean_fg': round(mean_fg_dice, 4),
    }, log_path)

    logging.info(f'Training complete. Best Val Dice: {best_dice:.4f}')
    logging.info(f'Best checkpoint: {cfg["checkpoint_dir"]}/best.pth')


# ── Per-class Dice evaluation ─────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_per_class(model, loader, device, amp: bool,
                       n_classes: int) -> torch.Tensor:
    """
    Returns a 1-D tensor of shape (n_classes,) with per-class Dice scores.
    Dice for class c = 2 * |pred_c ∩ true_c| / (|pred_c| + |true_c|).
    A class that is absent from both pred and true gets Dice = 1.0 (perfect).
    A class absent only from true but predicted gets Dice = 0.0 (penalised).
    """
    model.eval()
    intersection = torch.zeros(n_classes)
    sum_pred     = torch.zeros(n_classes)
    sum_true     = torch.zeros(n_classes)

    for batch in tqdm(loader, desc='Test evaluation', leave=False):
        images     = batch['image'].to(device=device, dtype=torch.float32,
                                       memory_format=torch.channels_last)
        true_masks = batch['mask'].to(device=device, dtype=torch.long)
        true_masks = true_masks.clamp(0, n_classes - 1)

        with torch.autocast(
            device.type if device.type != 'mps' else 'cpu', enabled=amp
        ):
            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[-1]   # use final head for deep supervision

        preds = logits.argmax(dim=1)   # (B, H, W)

        for c in range(n_classes):
            pred_c = (preds     == c)
            true_c = (true_masks == c)
            intersection[c] += (pred_c & true_c).sum().cpu().float()
            sum_pred[c]     += pred_c.sum().cpu().float()
            sum_true[c]     += true_c.sum().cpu().float()

    dice = torch.zeros(n_classes)
    for c in range(n_classes):
        denom = sum_pred[c] + sum_true[c]
        if denom == 0:
            dice[c] = 1.0   # class absent from both pred and GT → trivially perfect
        else:
            dice[c] = (2.0 * intersection[c] / denom)

    model.train()
    return dice


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='MedSeg — Train segmentation models')
    p.add_argument('--config',            type=str,   default='configs/default.yaml')
    p.add_argument('--model',             type=str,   help='unet | unetpp | unetadv | deeplab')
    p.add_argument('--epochs',            '-e', type=int)
    p.add_argument('--batch-size',        '-b', type=int,   dest='batch_size')
    p.add_argument('--lr',                '-l', type=float)
    p.add_argument('--scale',             '-s', type=float)
    p.add_argument('--img-dir',           type=str)
    p.add_argument('--mask-dir',          type=str)
    p.add_argument('--mask-channel',      type=int,   dest='mask_channel')
    p.add_argument('--n-channels',        type=int,   dest='n_channels')
    p.add_argument('--classes',           '-c', type=int,   dest='n_classes')
    p.add_argument('--load',              '-f', type=str)
    p.add_argument('--amp',               action='store_true',  default=False)
    p.add_argument('--no-amp',            action='store_false', dest='amp')
    p.add_argument('--bilinear',          action='store_true',  default=False)
    p.add_argument('--img-size',          type=int, nargs=2, metavar=('W', 'H'), dest='img_size')
    p.add_argument('--focal-gamma',       type=float, dest='focal_gamma', default=None,
                   help='Focal loss gamma (default 2.0)')
    p.add_argument('--dice-weight',       type=float, dest='dice_weight', default=None,
                   help='Max Dice loss weight (default 2.0)')
    p.add_argument('--dice-warmup-epochs',type=int,   dest='dice_warmup_epochs', default=None,
                   help='Epochs to ramp Dice weight 0→max (default 5)')
    p.add_argument('--dropout',           type=float, default=0.0)
    # ── External test set (optional) ──────────────────────────────────────────
    p.add_argument('--test-img-dir',  type=str, dest='test_img_dir',  default=None,
                   help='Directory of test images (curated, covers all classes)')
    p.add_argument('--test-mask-dir', type=str, dest='test_mask_dir', default=None,
                   help='Directory of test masks  (curated, covers all classes)')
    p.add_argument('--test-percent',  type=float, dest='test_percent', default=None,
                   help='Fraction of data held out as test when no external set given '
                        '(default 0.1)')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    cfg = load_config(args.config)

    skip_keys = {'config', 'amp', 'bilinear'}
    for k, v in vars(args).items():
        if k in skip_keys:
            continue
        if v is not None:
            cfg[k] = v

    if '--amp'      in sys.argv: cfg['amp']      = True
    if '--no-amp'   in sys.argv: cfg['amp']      = False
    if '--bilinear' in sys.argv: cfg['bilinear'] = True

    cfg.setdefault('focal_gamma',        2.0)
    cfg.setdefault('dice_weight',        2.0)
    cfg.setdefault('dice_warmup_epochs', 5)
    cfg.setdefault('test_percent',       0.1)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info('MPS detected — consider CPU if training is unstable.')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device: {device}')

    try:
        train_model(cfg, device)
    except torch.cuda.OutOfMemoryError:
        logging.error('Out of memory! Reduce --batch-size or --img-size.')
        raise






































# # train.py
# """
# Train UNet / UNet++ / DeepLab v3+ / UNetAdv on segmentation data.

# Usage examples:
#     python train.py --model unetadv --epochs 50 --batch-size 1 --lr 1e-4 \
#                     --amp --img-size 512 512 --n-channels 1 --classes 4

# Deep-supervision note (UNetAdv):
#     When deep_supervision=True the model returns a list of 4 logit tensors.
#     train.py computes a weighted sum of losses across all heads:
#         loss = 0.1*L(out1) + 0.2*L(out2) + 0.3*L(out3) + 0.4*L(out4)
#     This lets early heads learn coarse features while the final head is
#     optimised most strongly. At inference, only the last head is used.
# """

# import argparse
# import logging
# import sys
# import yaml
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm

# from models import build_model
# from utils.data_loading import SegmentationDataset
# from utils.dice_score import dice_loss
# from evaluate import evaluate


# # ── Focal Loss ────────────────────────────────────────────────────────────────

# class FocalLoss(nn.Module):
#     """
#     Focal Loss — down-weights easy background pixels so the model focuses
#     on hard rare foreground classes.

#     gamma=2.0 is the safe default.
#     Only go to 3.0 if classes 1/2/3 are still ignored after 10+ epochs.
#     """
#     def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
#         super().__init__()
#         self.gamma  = gamma
#         self.weight = weight

#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
#         pt = torch.exp(-ce.clamp(max=88.0))   # clamp prevents exp() overflow
#         return ((1 - pt) ** self.gamma * ce).mean()


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def load_config(path: str = 'configs/default.yaml') -> dict:
#     with open(path) as f:
#         return yaml.safe_load(f)


# def log_metrics(metrics: dict, log_path: Path):
#     line = ', '.join(f'{k}: {v}' for k, v in metrics.items())
#     with open(log_path, 'a') as f:
#         f.write(line + '\n')


# def save_checkpoint(model, dataset, epoch: int, val_dice: float,
#                     checkpoint_dir: Path, best_dice: float,
#                     keep_best_only: bool) -> float:
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     state = {
#         'epoch':       epoch,
#         'val_dice':    val_dice,
#         'model_state': model.state_dict(),
#         'mask_values': dataset.mask_values,
#         'n_channels':  model.n_channels,
#         'n_classes':   model.n_classes,
#     }
#     torch.save(state, checkpoint_dir / 'latest.pth')
#     if not keep_best_only:
#         torch.save(state, checkpoint_dir / f'epoch_{epoch:03d}_dice_{val_dice:.4f}.pth')
#     if val_dice > best_dice:
#         torch.save(state, checkpoint_dir / 'best.pth')
#         logging.info(f'  ★ New best checkpoint! Dice: {val_dice:.4f}')
#         return val_dice
#     return best_dice


# # ── Loss computation ──────────────────────────────────────────────────────────

# DS_WEIGHTS = [0.1, 0.2, 0.3, 0.4]


# def compute_loss(criterion, preds, true_masks, true_long,
#                  n_classes: int, dice_weight: float):
#     """
#     Dispatch to single or deep-supervision loss.
#     preds: single logit tensor OR list of 4 tensors (UNetAdv deep supervision).
#     """
#     if isinstance(preds, (list, tuple)):
#         total = torch.tensor(0.0, device=true_masks.device)
#         for w, p in zip(DS_WEIGHTS, preds):
#             total = total + w * _single_loss(
#                 criterion, p, true_masks, true_long, n_classes, dice_weight)
#         return total
#     else:
#         return _single_loss(criterion, preds, true_masks,
#                             true_long, n_classes, dice_weight)


# def _single_loss(criterion, pred, true_masks, true_long,
#                  n_classes: int, dice_weight: float):
#     """
#     Focal loss  +  foreground-only Dice loss.

#     ── Why foreground-only Dice? ────────────────────────────────────────────
#     Your dataset:
#         class 0 (background): 79,323,984 pixels  (~98%)
#         class 1:                  285,325 pixels
#         class 2:                   11,063 pixels
#         class 3:                   71,364 pixels

#     If Dice includes background, the model can achieve a near-perfect Dice
#     score by predicting ALL background — collapsing Val Dice to 0.0000.
#     Excluding background ([:, 1:]) forces Dice to only measure foreground
#     overlap, which is what actually matters for segmentation quality.
#     ─────────────────────────────────────────────────────────────────────────
#     """
#     if n_classes == 1:
#         # Binary: one channel, use sigmoid + BCE + Dice
#         p     = pred.squeeze(1)
#         probs = torch.sigmoid(p).clamp(1e-6, 1 - 1e-6)
#         return (criterion(p, true_masks)
#                 + dice_weight * dice_loss(probs, true_masks, multiclass=False))
#     else:
#         # Multiclass: Focal loss + foreground-only Dice
#         focal   = criterion(pred, true_long)

#         probs   = F.softmax(pred, dim=1).clamp(1e-6, 1 - 1e-6)
#         true_oh = F.one_hot(true_long, n_classes).permute(0, 3, 1, 2).float()

#         # [:, 1:] skips background (class 0) — prevents collapse to all-background
#         fg_dice = dice_loss(probs[:, 1:], true_oh[:, 1:], multiclass=True)

#         return focal + dice_weight * fg_dice


# # ── Training loop ─────────────────────────────────────────────────────────────

# def train_model(cfg: dict, device: torch.device):

#     # ── Validate directories ──────────────────────────────────────────────────
#     for key in ('img_dir', 'mask_dir'):
#         p = Path(cfg[key])
#         if not p.exists():
#             raise FileNotFoundError(f'{key} does not exist: {p}')

#     # ── Dataset ───────────────────────────────────────────────────────────────
#     dataset = SegmentationDataset(
#         images_dir   = cfg['img_dir'],
#         mask_dir     = cfg['mask_dir'],
#         scale        = cfg['scale'],
#         mask_channel = cfg['mask_channel'],
#         augment      = True,
#         img_size     = tuple(cfg.get('img_size', [512, 512])),
#     )

#     # Auto-correct n_classes from actual mask data
#     n_detected = len(dataset.mask_values)
#     if cfg['n_classes'] != n_detected:
#         logging.warning(
#             f'n_classes={cfg["n_classes"]} but dataset has {n_detected} unique '
#             f'mask values {dataset.mask_values}. Auto-correcting.'
#         )
#         cfg['n_classes'] = n_detected

#     n_val   = int(len(dataset) * cfg['val_percent'])
#     n_train = len(dataset) - n_val
#     train_set, val_set = random_split(
#         dataset, [n_train, n_val],
#         generator=torch.Generator().manual_seed(42)
#     )

#     loader_args  = dict(batch_size=cfg['batch_size'], num_workers=0, pin_memory=True)
#     train_loader = DataLoader(train_set, shuffle=True,  **loader_args)
#     val_loader   = DataLoader(val_set,   shuffle=False, drop_last=True, **loader_args)

#     # ── Model ─────────────────────────────────────────────────────────────────
#     model = build_model(
#         name       = cfg['model'],
#         n_channels = cfg['n_channels'],
#         n_classes  = cfg['n_classes'],
#         bilinear   = cfg.get('bilinear', False),
#         dropout    = cfg.get('dropout', 0.0),
#     ).to(memory_format=torch.channels_last).to(device)

#     # ── Resume from checkpoint ────────────────────────────────────────────────
#     start_epoch = 1
#     best_dice   = 0.0
#     if cfg.get('load'):
#         ckpt = torch.load(cfg['load'], map_location=device, weights_only=False)
#         model.load_state_dict(ckpt['model_state'])
#         start_epoch = ckpt.get('epoch', 0) + 1
#         best_dice   = ckpt.get('val_dice', 0.0)
#         if 'n_classes'  in ckpt: cfg['n_classes']  = ckpt['n_classes']
#         if 'n_channels' in ckpt: cfg['n_channels'] = ckpt['n_channels']
#         logging.info(
#             f'Resumed from {cfg["load"]} '
#             f'(epoch {start_epoch - 1}, dice {best_dice:.4f})'
#         )

#     # ── Optimizer ─────────────────────────────────────────────────────────────
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr           = cfg['lr'],
#         weight_decay = cfg.get('weight_decay', 1e-4),
#     )

#     # ── LR Scheduler ─────────────────────────────────────────────────────────
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode     = 'max',
#         patience = 10,
#         factor   = 0.5,
#         min_lr   = 1e-6,
#     )

#     grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'])

#     # ── Class-weighted Focal Loss ─────────────────────────────────────────────
#     if cfg['n_classes'] > 1:
#         logging.info('Computing class weights (inverse frequency)…')
#         class_counts = torch.zeros(cfg['n_classes'])
#         for item in tqdm(train_set, desc='Counting pixels', leave=False):
#             mask = item['mask'].long()
#             for c in range(cfg['n_classes']):
#                 class_counts[c] += (mask == c).sum()

#         logging.info(f'Pixel counts per class: {class_counts.long().tolist()}')

#         # sqrt-inverse frequency — smoother than raw inverse for extreme imbalance
#         class_weights = 1.0 / (class_counts.float().sqrt() + 1)
#         class_weights = class_weights / class_weights.sum() * cfg['n_classes']
#         class_weights = class_weights.clamp(max=20.0).to(device)
#         logging.info(f'Class weights: {[round(w, 4) for w in class_weights.tolist()]}')

#         criterion = FocalLoss(
#             gamma  = cfg.get('focal_gamma', 2.0),
#             weight = class_weights,
#         )
#     else:
#         pos_weight = torch.tensor([cfg['pos_weight']]).to(device)
#         criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     # ── Logging setup ─────────────────────────────────────────────────────────
#     log_path = Path(cfg['log_dir']) / f"{cfg['model']}_training.log"
#     log_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(log_path, 'w') as f:
#         f.write('=== MedSeg Training Log ===\n')
#         f.write(f"Model: {cfg['model']}, Ch: {cfg['n_channels']}, "
#                 f"Classes: {cfg['n_classes']}\n")
#         f.write(f"FocalGamma: {cfg.get('focal_gamma', 2.0)}, "
#                 f"DiceWeight: {cfg.get('dice_weight', 2.0)}\n")
#         f.write(f"Train: {n_train}, Val: {n_val}, Device: {device}\n\n")

#     logging.info(f'''
#     ╔══════════════════════════════════╗
#     ║        MedSeg Training           ║
#     ╠══════════════════════════════════╣
#     ║  Model:      {cfg["model"]:<20} ║
#     ║  Channels:   {cfg["n_channels"]:<20} ║
#     ║  Classes:    {cfg["n_classes"]:<20} ║
#     ║  Epochs:     {cfg["epochs"]:<20} ║
#     ║  Batch:      {cfg["batch_size"]:<20} ║
#     ║  LR:         {cfg["lr"]:<20} ║
#     ║  FocalGamma: {cfg.get("focal_gamma", 2.0):<20} ║
#     ║  DiceWeight: {cfg.get("dice_weight", 2.0):<20} ║
#     ║  Train:      {n_train:<20} ║
#     ║  Val:        {n_val:<20} ║
#     ║  Device:     {str(device):<20} ║
#     ║  AMP:        {str(cfg["amp"]):<20} ║
#     ╚══════════════════════════════════╝
#     ''')

#     global_step = 0
#     nan_count   = 0
#     dice_weight_max = cfg.get('dice_weight', 2.0)

#     for epoch in range(start_epoch, cfg['epochs'] + 1):
#         model.train()
#         epoch_loss  = 0.0
#         num_batches = 0

#         # Dice warmup: ramp from 0 → full weight over first 5 epochs.
#         # Prevents Dice from overwhelming Focal loss before model has
#         # learned any basic class structure.
#         warmup_epochs = cfg.get('dice_warmup_epochs', 5)
#         dice_weight   = dice_weight_max * min(1.0, epoch / warmup_epochs)

#         with tqdm(total=n_train,
#                   desc=f'Epoch {epoch}/{cfg["epochs"]}', unit='img') as pbar:
#             for batch in train_loader:
#                 images, true_masks = batch['image'], batch['mask']

#                 assert images.shape[1] == model.n_channels, (
#                     f'Model expects {model.n_channels} ch, '
#                     f'got {images.shape[1]}. Check n_channels in config.'
#                 )

#                 images     = images.to(device=device, dtype=torch.float32,
#                                        memory_format=torch.channels_last)
#                 true_masks = true_masks.to(device=device, dtype=torch.float32)

#                 with torch.autocast(
#                     device.type if device.type != 'mps' else 'cpu',
#                     enabled=cfg['amp']
#                 ):
#                     preds = model(images)

#                     true_long = None
#                     if cfg['n_classes'] > 1:
#                         true_long = true_masks.long().clamp(0, cfg['n_classes'] - 1)

#                     loss = compute_loss(
#                         criterion, preds, true_masks,
#                         true_long, cfg['n_classes'],
#                         dice_weight=dice_weight,
#                     )

#                 # ── Diagnostics every 50 steps ────────────────────────────────
#                 if global_step % 50 == 0:
#                     with torch.no_grad():
#                         p_ins    = preds[-1] if isinstance(preds, (list, tuple)) else preds
#                         pred_cls = p_ins.argmax(dim=1)
#                         logging.info(
#                             f'DEBUG step {global_step} '
#                             f'(dice_w={dice_weight:.2f}) — '
#                             f'pred: {sorted(pred_cls.unique().tolist())}'
#                             + (f', true: {sorted(true_long.unique().tolist())}'
#                                if true_long is not None else '')
#                         )

#                 # ── NaN / Inf guard ───────────────────────────────────────────
#                 if torch.isnan(loss) or torch.isinf(loss):
#                     nan_count += 1
#                     logging.warning(
#                         f'NaN/Inf loss at step {global_step} — skipping '
#                         f'({nan_count} total skips)'
#                     )
#                     optimizer.zero_grad(set_to_none=True)
#                     if nan_count > 50:
#                         logging.error(
#                             'More than 50 NaN batches — training is unstable. '
#                             'Try: lower --lr (e.g. 5e-5) or --focal-gamma 1.0'
#                         )
#                     continue

#                 optimizer.zero_grad(set_to_none=True)
#                 grad_scaler.scale(loss).backward()
#                 grad_scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(),
#                                                cfg['gradient_clipping'])
#                 grad_scaler.step(optimizer)
#                 grad_scaler.update()

#                 epoch_loss  += loss.item()
#                 num_batches += 1
#                 global_step += 1

#                 pbar.update(images.shape[0])
#                 pbar.set_postfix(loss=f'{loss.item():.4f}',
#                                  dice_w=f'{dice_weight:.2f}')

#                 if global_step % cfg['log_interval'] == 0:
#                     log_metrics({'step': global_step, 'epoch': epoch,
#                                  'loss': round(loss.item(), 4),
#                                  'dice_w': round(dice_weight, 4)}, log_path)

#                 # ── Mid-epoch validation (monitor only) ───────────────────────
#                 div = n_train // (5 * cfg['batch_size'])
#                 if div > 0 and global_step % div == 0:
#                     val_dice = evaluate(model, val_loader, device, cfg['amp'])
#                     logging.info(
#                         f'  Val Dice: {val_dice:.4f} '
#                         f'| LR: {optimizer.param_groups[0]["lr"]:.2e} '
#                         f'| dice_w: {dice_weight:.2f}'
#                     )
#                     log_metrics({'step': global_step, 'epoch': epoch,
#                                  'lr':       optimizer.param_groups[0]['lr'],
#                                  'val_dice': round(float(val_dice), 4)}, log_path)

#         # ── End-of-epoch ──────────────────────────────────────────────────────
#         avg_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
#         val_dice = evaluate(model, val_loader, device, cfg['amp'])
#         scheduler.step(val_dice)

#         logging.info(
#             f'Epoch {epoch} | Avg Loss: {avg_loss:.4f} '
#             f'| Val Dice: {val_dice:.4f} '
#             f'| dice_w: {dice_weight:.2f} '
#             f'| LR: {optimizer.param_groups[0]["lr"]:.2e}'
#         )
#         log_metrics({
#             'epoch':    epoch,
#             'avg_loss': round(avg_loss, 4),
#             'val_dice': round(float(val_dice), 4),
#             'dice_w':   round(dice_weight, 4),
#             'lr':       optimizer.param_groups[0]['lr'],
#         }, log_path)

#         if epoch % cfg['save_every'] == 0:
#             best_dice = save_checkpoint(
#                 model, dataset, epoch, float(val_dice),
#                 Path(cfg['checkpoint_dir']), best_dice, cfg['keep_best_only']
#             )

#     logging.info(f'Training complete. Best Val Dice: {best_dice:.4f}')
#     logging.info(f'Best checkpoint: {cfg["checkpoint_dir"]}/best.pth')


# # ── CLI ───────────────────────────────────────────────────────────────────────

# def get_args():
#     p = argparse.ArgumentParser(description='MedSeg — Train segmentation models')
#     p.add_argument('--config',            type=str,   default='configs/default.yaml')
#     p.add_argument('--model',             type=str,   help='unet | unetpp | unetadv | deeplab')
#     p.add_argument('--epochs',            '-e', type=int)
#     p.add_argument('--batch-size',        '-b', type=int,   dest='batch_size')
#     p.add_argument('--lr',                '-l', type=float)
#     p.add_argument('--scale',             '-s', type=float)
#     p.add_argument('--img-dir',           type=str)
#     p.add_argument('--mask-dir',          type=str)
#     p.add_argument('--mask-channel',      type=int,   dest='mask_channel')
#     p.add_argument('--n-channels',        type=int,   dest='n_channels')
#     p.add_argument('--classes',           '-c', type=int,   dest='n_classes')
#     p.add_argument('--load',              '-f', type=str)
#     p.add_argument('--amp',               action='store_true',  default=False)
#     p.add_argument('--no-amp',            action='store_false', dest='amp')
#     p.add_argument('--bilinear',          action='store_true',  default=False)
#     p.add_argument('--img-size',          type=int, nargs=2, metavar=('W', 'H'), dest='img_size')
#     p.add_argument('--focal-gamma',       type=float, dest='focal_gamma', default=None,
#                    help='Focal loss gamma (default 2.0)')
#     p.add_argument('--dice-weight',       type=float, dest='dice_weight', default=None,
#                    help='Max Dice loss weight (default 2.0)')
#     p.add_argument('--dice-warmup-epochs',type=int,   dest='dice_warmup_epochs', default=None,
#                    help='Epochs to ramp Dice weight 0→max (default 5)')
#     p.add_argument('--dropout',           type=float, default=0.0)
#     return p.parse_args()


# if __name__ == '__main__':
#     args = get_args()
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#     cfg = load_config(args.config)

#     skip_keys = {'config', 'amp', 'bilinear'}
#     for k, v in vars(args).items():
#         if k in skip_keys:
#             continue
#         if v is not None:
#             cfg[k] = v

#     if '--amp'      in sys.argv: cfg['amp']     = True
#     if '--no-amp'   in sys.argv: cfg['amp']     = False
#     if '--bilinear' in sys.argv: cfg['bilinear'] = True

#     cfg.setdefault('focal_gamma',        2.0)
#     cfg.setdefault('dice_weight',        2.0)
#     cfg.setdefault('dice_warmup_epochs', 5)

#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     elif torch.backends.mps.is_available():
#         device = torch.device('mps')
#         logging.info('MPS detected — consider CPU if training is unstable.')
#     else:
#         device = torch.device('cpu')
#     logging.info(f'Using device: {device}')

#     try:
#         train_model(cfg, device)
#     except torch.cuda.OutOfMemoryError:
#         logging.error('Out of memory! Reduce --batch-size or --img-size.')
#         raise
































# # train.py
# """
# Train UNet / UNet++ / DeepLab v3+ / UNetAdv on segmentation data.

# Usage examples:
#     python train.py --model unetadv --epochs 150 --batch-size 2 --lr 1e-4 \
#                     --amp --img-size 640 640 --n-channels 1 --classes 4

#     python train.py --model unet    --epochs 150 --batch-size 2 --lr 1e-4 \
#                     --amp --img-size 640 640 --n-channels 1 --classes 4

# Deep-supervision note (UNetAdv):
#     When deep_supervision=True the model returns a list of 4 logit tensors.
#     train.py computes a weighted sum of losses across all heads:
#         loss = 0.1*L(out1) + 0.2*L(out2) + 0.3*L(out3) + 0.4*L(out4)
#     This lets early heads learn coarse features while the final head is
#     optimised most strongly.  At inference, only the last head is used.
# """

# import argparse
# import logging
# import sys
# import yaml
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm

# from models import build_model
# from utils.data_loading import SegmentationDataset
# from utils.dice_score import dice_loss
# from evaluate import evaluate


# # ── Focal Loss ────────────────────────────────────────────────────────────────

# class FocalLoss(nn.Module):
#     """
#     Focal Loss — down-weights easy background pixels so the model focuses
#     on hard, rare foreground classes (e.g. class 2 with only 11k pixels).

#     gamma=2.0 standard; use 3.0 when rare classes dominate the problem.
#     Supports per-class inverse-frequency weights.
#     """
#     def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
#         super().__init__()
#         self.gamma  = gamma
#         self.weight = weight

#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
#         pt = torch.exp(-ce.clamp(max=88.0))   # clamp prevents exp() overflow
#         return ((1 - pt) ** self.gamma * ce).mean()


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def load_config(path: str = 'configs/default.yaml') -> dict:
#     with open(path) as f:
#         return yaml.safe_load(f)


# def log_metrics(metrics: dict, log_path: Path):
#     line = ', '.join(f'{k}: {v}' for k, v in metrics.items())
#     with open(log_path, 'a') as f:
#         f.write(line + '\n')


# def save_checkpoint(model, dataset, epoch: int, val_dice: float,
#                     checkpoint_dir: Path, best_dice: float,
#                     keep_best_only: bool) -> float:
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     state = {
#         'epoch':       epoch,
#         'val_dice':    val_dice,
#         'model_state': model.state_dict(),
#         'mask_values': dataset.mask_values,
#         'n_channels':  model.n_channels,
#         'n_classes':   model.n_classes,
#     }

#     torch.save(state, checkpoint_dir / 'latest.pth')

#     if not keep_best_only:
#         torch.save(state, checkpoint_dir / f'epoch_{epoch:03d}_dice_{val_dice:.4f}.pth')

#     if val_dice > best_dice:
#         torch.save(state, checkpoint_dir / 'best.pth')
#         logging.info(f'  ★ New best checkpoint! Dice: {val_dice:.4f}')
#         return val_dice

#     return best_dice


# # ── Loss computation ──────────────────────────────────────────────────────────

# # Deep supervision weights — more weight to deeper / later heads.
# DS_WEIGHTS = [0.1, 0.2, 0.3, 0.4]


# def compute_loss(criterion, preds, true_masks, true_long,
#                  n_classes: int, dice_weight: float = 2.0):
#     """
#     Compute combined focal + dice loss.

#     preds : single logit tensor OR list of logit tensors (deep supervision).
#     Returns a scalar loss tensor.
#     """
#     if isinstance(preds, (list, tuple)):
#         # Deep supervision: weighted sum over all heads independently.
#         total = torch.tensor(0.0, device=true_masks.device)
#         for w, p in zip(DS_WEIGHTS, preds):
#             total = total + w * _single_loss(criterion, p, true_masks,
#                                               true_long, n_classes, dice_weight)
#         return total
#     else:
#         return _single_loss(criterion, preds, true_masks,
#                             true_long, n_classes, dice_weight)


# def _single_loss(criterion, pred, true_masks, true_long,
#                  n_classes: int, dice_weight: float):
#     """
#     FIX: multiclass now uses BOTH Focal loss + Dice loss.

#     Previously only used Focal loss for multiclass, which optimises
#     pixel-wise accuracy but ignores spatial overlap — causing erratic
#     Dice scores.  Dice loss forces the model to learn region shapes.
#     """
#     if n_classes == 1:
#         # Binary segmentation
#         p     = pred.squeeze(1)
#         probs = torch.sigmoid(p).clamp(1e-6, 1 - 1e-6)
#         return criterion(p, true_masks) + dice_weight * dice_loss(
#             probs, true_masks, multiclass=False
#         )
#     else:
#         # Multiclass segmentation
#         # FIX: add Dice loss on top of Focal loss
#         focal   = criterion(pred, true_long)

#         # Softmax probabilities for Dice — clamped for numerical safety
#         probs   = F.softmax(pred, dim=1).clamp(1e-6, 1 - 1e-6)

#         # One-hot encode ground truth for Dice
#         true_oh = F.one_hot(true_long, n_classes).permute(0, 3, 1, 2).float()

#         # Dice over ALL classes including background — then weighted sum
#         d_loss  = dice_loss(probs, true_oh, multiclass=True)

#         return focal + dice_weight * d_loss


# # ── Training loop ─────────────────────────────────────────────────────────────

# def train_model(cfg: dict, device: torch.device):

#     # ── Validate directories ──────────────────────────────────────────────────
#     for key in ('img_dir', 'mask_dir'):
#         p = Path(cfg[key])
#         if not p.exists():
#             raise FileNotFoundError(f'{key} does not exist: {p}')

#     # ── Dataset ───────────────────────────────────────────────────────────────
#     dataset = SegmentationDataset(
#         images_dir   = cfg['img_dir'],
#         mask_dir     = cfg['mask_dir'],
#         scale        = cfg['scale'],
#         mask_channel = cfg['mask_channel'],
#         augment      = True,
#         img_size     = tuple(cfg.get('img_size', [512, 512])),
#     )

#     # Auto-correct n_classes from actual mask data
#     n_detected = len(dataset.mask_values)
#     if cfg['n_classes'] != n_detected:
#         logging.warning(
#             f'n_classes={cfg["n_classes"]} but dataset has {n_detected} unique '
#             f'mask values {dataset.mask_values}. Auto-correcting.'
#         )
#         cfg['n_classes'] = n_detected

#     n_val   = int(len(dataset) * cfg['val_percent'])
#     n_train = len(dataset) - n_val
#     train_set, val_set = random_split(
#         dataset, [n_train, n_val],
#         generator=torch.Generator().manual_seed(42)
#     )

#     loader_args  = dict(batch_size=cfg['batch_size'], num_workers=0, pin_memory=True)
#     train_loader = DataLoader(train_set, shuffle=True,  **loader_args)
#     val_loader   = DataLoader(val_set,   shuffle=False, drop_last=True, **loader_args)

#     # ── Model ─────────────────────────────────────────────────────────────────
#     model = build_model(
#         name       = cfg['model'],
#         n_channels = cfg['n_channels'],
#         n_classes  = cfg['n_classes'],
#         bilinear   = cfg.get('bilinear', False),
#         dropout    = cfg.get('dropout', 0.0),
#     ).to(memory_format=torch.channels_last).to(device)

#     # ── Resume from checkpoint ────────────────────────────────────────────────
#     start_epoch = 1
#     best_dice   = 0.0
#     if cfg.get('load'):
#         ckpt = torch.load(cfg['load'], map_location=device, weights_only=False)
#         model.load_state_dict(ckpt['model_state'])
#         start_epoch = ckpt.get('epoch', 0) + 1
#         best_dice   = ckpt.get('val_dice', 0.0)
#         if 'n_classes' in ckpt:
#             cfg['n_classes'] = ckpt['n_classes']
#         if 'n_channels' in ckpt:
#             cfg['n_channels'] = ckpt['n_channels']
#         logging.info(
#             f'Resumed from {cfg["load"]} '
#             f'(epoch {start_epoch - 1}, dice {best_dice:.4f})'
#         )

#     # ── Optimizer ─────────────────────────────────────────────────────────────
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr           = cfg['lr'],
#         weight_decay = cfg.get('weight_decay', 1e-4),
#     )

#     # ── LR scheduler — one step per epoch ────────────────────────────────────
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode     = 'max',
#         patience = 10,
#         factor   = 0.5,
#         min_lr   = 1e-6,
#     )

#     grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'])

#     # ── Loss with class balancing ─────────────────────────────────────────────
#     if cfg['n_classes'] > 1:
#         logging.info('Computing class weights (inverse frequency)…')
#         class_counts = torch.zeros(cfg['n_classes'])
#         for item in tqdm(train_set, desc='Counting pixels', leave=False):
#             mask = item['mask'].long()
#             for c in range(cfg['n_classes']):
#                 class_counts[c] += (mask == c).sum()

#         logging.info(f'Pixel counts per class: {class_counts.long().tolist()}')

#         # FIX: stronger weighting for rare classes
#         # Use sqrt-inverse-frequency instead of raw inverse — less extreme but
#         # still strongly upweights class 2 (11k pixels vs 79M background)
#         class_weights = 1.0 / (class_counts.float().sqrt() + 1)
#         class_weights = class_weights / class_weights.sum() * cfg['n_classes']

#         # FIX: raised clamp ceiling from 10.0 → 20.0 so rare classes
#         # (class 2 = 11k pixels) are not capped and get proper upweighting
#         class_weights = class_weights.clamp(max=20.0).to(device)
#         logging.info(f'Class weights: {[round(w, 4) for w in class_weights.tolist()]}')

#         # FIX: use FocalLoss with gamma from config (default 3.0 for rare classes)
#         # Previously used CrossEntropyLoss which treats all misclassifications
#         # equally — FocalLoss down-weights easy background pixels so the model
#         # focuses on rare hard classes (class 2 = 11k pixels)
#         criterion = FocalLoss(
#             gamma  = cfg.get('focal_gamma', 3.0),
#             weight = class_weights,
#         )
#     else:
#         pos_weight = torch.tensor([cfg['pos_weight']]).to(device)
#         criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     # ── Logging setup ─────────────────────────────────────────────────────────
#     log_path = Path(cfg['log_dir']) / f"{cfg['model']}_training.log"
#     log_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(log_path, 'w') as f:
#         f.write('=== MedSeg Training Log ===\n')
#         f.write(f"Model: {cfg['model']}, Ch: {cfg['n_channels']}, "
#                 f"Classes: {cfg['n_classes']}\n")
#         f.write(f"FocalGamma: {cfg.get('focal_gamma', 3.0)}, "
#                 f"DiceWeight: {cfg.get('dice_weight', 2.0)}\n")
#         f.write(f"Train: {n_train}, Val: {n_val}, Device: {device}\n\n")

#     logging.info(f'''
#     ╔══════════════════════════════════╗
#     ║        MedSeg Training           ║
#     ╠══════════════════════════════════╣
#     ║  Model:      {cfg["model"]:<20} ║
#     ║  Channels:   {cfg["n_channels"]:<20} ║
#     ║  Classes:    {cfg["n_classes"]:<20} ║
#     ║  Epochs:     {cfg["epochs"]:<20} ║
#     ║  Batch:      {cfg["batch_size"]:<20} ║
#     ║  LR:         {cfg["lr"]:<20} ║
#     ║  FocalGamma: {cfg.get("focal_gamma", 3.0):<20} ║
#     ║  DiceWeight: {cfg.get("dice_weight", 2.0):<20} ║
#     ║  Train:      {n_train:<20} ║
#     ║  Val:        {n_val:<20} ║
#     ║  Device:     {str(device):<20} ║
#     ║  AMP:        {str(cfg["amp"]):<20} ║
#     ╚══════════════════════════════════╝
#     ''')

#     global_step = 0
#     nan_count   = 0

#     for epoch in range(start_epoch, cfg['epochs'] + 1):
#         model.train()
#         epoch_loss  = 0.0
#         num_batches = 0

#         with tqdm(total=n_train,
#                   desc=f'Epoch {epoch}/{cfg["epochs"]}', unit='img') as pbar:
#             for batch in train_loader:
#                 images, true_masks = batch['image'], batch['mask']

#                 assert images.shape[1] == model.n_channels, (
#                     f'Model expects {model.n_channels} ch, '
#                     f'got {images.shape[1]}. Check n_channels in config.'
#                 )

#                 images     = images.to(device=device, dtype=torch.float32,
#                                        memory_format=torch.channels_last)
#                 true_masks = true_masks.to(device=device, dtype=torch.float32)

#                 with torch.autocast(
#                     device.type if device.type != 'mps' else 'cpu',
#                     enabled=cfg['amp']
#                 ):
#                     preds = model(images)

#                     # Prepare long integer targets for multiclass
#                     true_long = None
#                     if cfg['n_classes'] > 1:
#                         true_long = true_masks.long().clamp(0, cfg['n_classes'] - 1)

#                     loss = compute_loss(
#                         criterion, preds, true_masks, true_long,
#                         cfg['n_classes'],
#                         dice_weight=cfg.get('dice_weight', 2.0),
#                     )

#                 # ── Diagnostics every 50 steps ────────────────────────────────
#                 if global_step % 50 == 0:
#                     with torch.no_grad():
#                         p_inspect = preds[-1] if isinstance(preds, (list, tuple)) else preds
#                         pred_cls  = p_inspect.argmax(dim=1)
#                         logging.info(
#                             f'DEBUG step {global_step} — '
#                             f'pred classes: {sorted(pred_cls.unique().tolist())}'
#                             + (f', true classes: {sorted(true_long.unique().tolist())}'
#                                if true_long is not None else '')
#                         )

#                 # ── NaN / Inf guard ───────────────────────────────────────────
#                 if torch.isnan(loss) or torch.isinf(loss):
#                     nan_count += 1
#                     logging.warning(
#                         f'NaN/Inf loss at step {global_step} — skipping batch '
#                         f'({nan_count} total skips this run)'
#                     )
#                     optimizer.zero_grad(set_to_none=True)
#                     if nan_count > 50:
#                         logging.error(
#                             'More than 50 NaN batches — training is unstable. '
#                             'Try: lower --lr (e.g. 5e-5) or --focal-gamma 1.0'
#                         )
#                     continue

#                 optimizer.zero_grad(set_to_none=True)
#                 grad_scaler.scale(loss).backward()
#                 grad_scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(),
#                                                cfg['gradient_clipping'])
#                 grad_scaler.step(optimizer)
#                 grad_scaler.update()

#                 epoch_loss  += loss.item()
#                 num_batches += 1
#                 global_step += 1

#                 pbar.update(images.shape[0])
#                 pbar.set_postfix(loss=f'{loss.item():.4f}')

#                 if global_step % cfg['log_interval'] == 0:
#                     log_metrics({'step': global_step, 'epoch': epoch,
#                                  'loss': round(loss.item(), 4)}, log_path)

#                 # ── Mid-epoch validation (monitor only — NO scheduler step) ───
#                 div = n_train // (5 * cfg['batch_size'])
#                 if div > 0 and global_step % div == 0:
#                     val_dice = evaluate(model, val_loader, device, cfg['amp'])
#                     logging.info(
#                         f'  Val Dice: {val_dice:.4f} '
#                         f'| LR: {optimizer.param_groups[0]["lr"]:.2e}'
#                     )
#                     log_metrics({'step': global_step, 'epoch': epoch,
#                                  'lr':       optimizer.param_groups[0]['lr'],
#                                  'val_dice': round(float(val_dice), 4)}, log_path)

#         # ── End-of-epoch: scheduler steps ONCE on end-of-epoch val ───────────
#         avg_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
#         val_dice = evaluate(model, val_loader, device, cfg['amp'])
#         scheduler.step(val_dice)

#         logging.info(
#             f'Epoch {epoch} | Avg Loss: {avg_loss:.4f} '
#             f'| Val Dice: {val_dice:.4f} '
#             f'| LR: {optimizer.param_groups[0]["lr"]:.2e}'
#         )
#         log_metrics({
#             'epoch':    epoch,
#             'avg_loss': round(avg_loss, 4),
#             'val_dice': round(float(val_dice), 4),
#             'lr':       optimizer.param_groups[0]['lr'],
#         }, log_path)

#         if epoch % cfg['save_every'] == 0:
#             best_dice = save_checkpoint(
#                 model, dataset, epoch, float(val_dice),
#                 Path(cfg['checkpoint_dir']), best_dice, cfg['keep_best_only']
#             )

#     logging.info(f'Training complete. Best Val Dice: {best_dice:.4f}')
#     logging.info(f'Best checkpoint: {cfg["checkpoint_dir"]}/best.pth')


# # ── CLI ───────────────────────────────────────────────────────────────────────

# def get_args():
#     p = argparse.ArgumentParser(description='MedSeg — Train segmentation models')
#     p.add_argument('--config',       type=str,   default='configs/default.yaml')
#     p.add_argument('--model',        type=str,   help='unet | unetpp | unetadv | deeplab')
#     p.add_argument('--epochs',       '-e', type=int)
#     p.add_argument('--batch-size',   '-b', type=int,   dest='batch_size')
#     p.add_argument('--lr',           '-l', type=float)
#     p.add_argument('--scale',        '-s', type=float)
#     p.add_argument('--img-dir',      type=str)
#     p.add_argument('--mask-dir',     type=str)
#     p.add_argument('--mask-channel', type=int,   dest='mask_channel',
#                    help='Which channel is the mask (default 3 = alpha)')
#     p.add_argument('--n-channels',   type=int,   dest='n_channels')
#     p.add_argument('--classes',      '-c', type=int,   dest='n_classes')
#     p.add_argument('--load',         '-f', type=str,   help='Resume from checkpoint')
#     p.add_argument('--amp',          action='store_true',  default=False)
#     p.add_argument('--no-amp',       action='store_false', dest='amp')
#     p.add_argument('--bilinear',     action='store_true',  default=False)
#     p.add_argument('--img-size',     type=int, nargs=2, metavar=('W', 'H'),
#                    dest='img_size',  help='Fixed output size: W H (e.g. 512 512)')
#     p.add_argument('--focal-gamma',  type=float, dest='focal_gamma', default=None,
#                    help='Focal loss gamma (default 3.0; try 2.0 for more balanced data)')
#     p.add_argument('--dice-weight',  type=float, dest='dice_weight', default=None,
#                    help='Weight of Dice loss term (default 2.0)')
#     p.add_argument('--dropout',      type=float, default=0.0,
#                    help='Bottleneck dropout probability (default 0.0)')
#     return p.parse_args()


# if __name__ == '__main__':
#     args = get_args()
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#     cfg = load_config(args.config)

#     # Override config with CLI args (skip booleans — handled separately below)
#     skip_keys = {'config', 'amp', 'bilinear'}
#     for k, v in vars(args).items():
#         if k in skip_keys:
#             continue
#         if v is not None:
#             cfg[k] = v

#     # Only override booleans when explicitly passed on CLI
#     if '--amp'      in sys.argv: cfg['amp']     = True
#     if '--no-amp'   in sys.argv: cfg['amp']     = False
#     if '--bilinear' in sys.argv: cfg['bilinear'] = True

#     # Default focal_gamma to 3.0 if not set anywhere
#     cfg.setdefault('focal_gamma', 3.0)
#     cfg.setdefault('dice_weight', 2.0)

#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     elif torch.backends.mps.is_available():
#         device = torch.device('mps')
#         logging.info('Note: MPS has limited FP16 support — consider CPU if unstable.')
#     else:
#         device = torch.device('cpu')
#     logging.info(f'Using device: {device}')

#     try:
#         train_model(cfg, device)
#     except torch.cuda.OutOfMemoryError:
#         logging.error('Out of memory! Reduce --batch-size or --img-size.')
#         raise
