# MedSeg — Medical & Satellite Image Segmentation

A production-ready segmentation toolkit supporting **16-bit RGBA** images (medical scans, satellite imagery).
Supports **UNet**, **UNet++**, **DeepLab v3+**, and **UNetAdv** (variable-attention UNet++) out of the box.

---

## Project Structure

```
medseg/
├── models/
│   ├── unet.py          # Classic UNet
│   ├── unetpp.py        # UNet++ (nested)
│   ├── deeplab.py       # DeepLab v3+
│   └── unetadv.py       # UNetAdv (variable-attention, deep supervision)
├── utils/
│   ├── data_loading.py  # 16-bit RGBA dataset loader
│   ├── dice_score.py    # Dice coefficient + loss
├── configs/
│   └── default.yaml     # All hyperparameters
├── scripts/
│   └── verify_dataset.py # Sanity-check your data before training
├── train.py             # Train a model (UNet/UNet++/UNetAdv/DeepLab)
├── train_1.py           # Train with pretrained encoders (SMP)
├── predict.py           # Run inference on new images
├── evaluate.py          # Evaluate a checkpoint on a dataset
├── test.py              # Full per-class evaluation with confusion matrix
├── check_prediction.py  # Visualize predictions with overlay + legend
├── diagnose_data.py     # Diagnose bit depth / normalization / class balance
└── requirements.txt
```

---

## Quickstart

### 1. Install
```bash
git clone https://github.com/yourname/medseg.git
cd medseg
pip install -r requirements.txt
```

### 2. Prepare your data
```
data/
├── images/    ← 16-bit RGBA PNG images
└── masks/     ← 16-bit RGBA PNG masks (alpha channel = segmentation)
```

### 3. Diagnose your data
```bash
python diagnose_data.py --img-dir data/images --mask-dir data/masks
```
Checks bit depth, normalization path (div by 65535 vs 255), class distribution, and dead-pixel statistics.

### 4. Verify your dataset
```bash
python scripts/verify_dataset.py --img-dir data/images --mask-dir data/masks
```

### 5. Train
```bash
# UNet (default)
python train.py --model unet --epochs 50 --batch-size 2 --lr 1e-5 --scale 0.5 --amp

# UNet++
python train.py --model unetpp --epochs 50 --batch-size 2 --lr 1e-5 --scale 0.5 --amp

# DeepLab v3+
python train.py --model deeplab --epochs 50 --batch-size 2 --lr 1e-4 --scale 0.5 --amp

# UNetAdv — variable attention, deep supervision, best for class imbalance
python train.py --model unetadv --epochs 50 --batch-size 1 --lr 1e-4 \
                --amp --img-size 512 512 --n-channels 1 --classes 4

# Resume from checkpoint
python train.py --model unet --load checkpoints/latest.pth

# Use an external test set (recommended for reliable reporting)
python train.py --model unetadv --epochs 50 --batch-size 1 --lr 1e-4 \
                --amp --img-size 512 512 --n-channels 1 --classes 4 \
                --test-img-dir data/test/images --test-mask-dir data/test/masks
```

### 6. Predict
```bash
# Single image
python predict.py --model unet --checkpoint checkpoints/best.pth \
                  --input image.png --output pred.png

# Folder of images
python predict.py --model unet --checkpoint checkpoints/best.pth \
                  --input data/images/ --output predictions/ --threshold 0.5

# Visualize with overlay + legend (confidence-gated)
python check_prediction.py --model unetadv --checkpoint checkpoints/best.pth \
                           --input data/images/ --save-dir overlays/ \
                           --conf-threshold 0.7
```

### 7. Evaluate
```bash
# Quick Dice score
python evaluate.py --model unet --checkpoint checkpoints/best.pth \
                   --img-dir data/images --mask-dir data/masks

# Full per-class metrics (Dice, IoU, precision, recall, confusion matrix)
python test.py --checkpoint checkpoints/best.pth \
               --img-dir data/test/images --mask-dir data/test/masks
```

### 8. Train with a pretrained encoder (SMP)
```bash
python train_1.py --img-dir data/images --mask-dir data/masks \
                  --encoder resnet34 --model unet --epochs 30 --batch-size 8
```

---

## Configuration (`configs/default.yaml`)

All settings live in `configs/default.yaml`. CLI args override config values.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `unet` | Architecture: `unet`, `unetpp`, `deeplab`, `unetadv` |
| `n_channels` | `4` | Input channels (4 = RGBA, 1 = grayscale, 3 = RGB) |
| `n_classes` | `1` | 1 = binary, >1 = multi-class |
| `bilinear` | `false` | UNet upsample mode (false = transposed conv) |
| `dropout` | `0.2` | Bottleneck dropout rate |
| `epochs` | `50` | Training epochs |
| `batch_size` | `2` | Batch size |
| `lr` | `1e-5` | Learning rate |
| `momentum` | `0.9` | RMSprop momentum |
| `weight_decay` | `1e-8` | Weight decay |
| `gradient_clipping` | `1.0` | Max gradient norm |
| `scale` | `0.5` | Image resize scale |
| `img_size` | `[512, 512]` | Fixed output size (W, H) — overrides scale |
| `val_percent` | `0.1` | Fraction used for validation |
| `amp` | `true` | Mixed precision (faster + less VRAM) |
| `mask_channel` | `3` | Which RGBA channel encodes the mask (3=alpha) |
| `pos_weight` | `2.0` | BCEWithLogitsLoss pos_weight (binary only) |
| `focal_gamma` | `2.0` | Focal loss gamma (multiclass) |
| `checkpoint_dir` | `checkpoints` | Checkpoint save directory |
| `save_every` | `1` | Save checkpoint every N epochs |
| `keep_best_only` | `false` | If true, only keep best.pth |
| `log_dir` | `logs` | Log directory |
| `log_interval` | `10` | Log batch loss every N steps |

---

## CLI Reference

Additional `train.py` CLI arguments not in the config table:

| Arg | Default | Description |
|-----|---------|-------------|
| `--load` / `-f` | — | Resume from checkpoint `.pth` |
| `--img-size` | `512 512` | Fixed W H output (overrides config) |
| `--focal-gamma` | `2.0` | Focal loss gamma |
| `--dice-weight` | `2.0` | Dice loss weight |
| `--dice-warmup-epochs` | `5` | Epochs to ramp Dice weight 0 → max |
| `--dropout` | `0.0` | Bottleneck dropout |
| `--bilinear` | — | Use bilinear upsample (UNet) |
| `--no-amp` | — | Disable mixed precision |
| `--test-img-dir` | — | External test set images |
| `--test-mask-dir` | — | External test set masks |
| `--test-percent` | `0.1` | Internal test fraction (no external set) |

Additional `test.py` CLI arguments:

| Arg | Default | Description |
|-----|---------|-------------|
| `--checkpoint` / `-f` | — | Path to `.pth` checkpoint |
| `--img-dir` | — | Test images directory |
| `--mask-dir` | — | Test masks directory |
| `--use-split-file` | — | Reuse split from `data_split.yaml` |
| `--batch-size` / `-b` | `1` | Batch size |
| `--scale` / `-s` | `1.0` | Image scale |
| `--mask-channel` | `0` | Mask channel index |
| `--img-size` | `512 512` | Fixed output size |
| `--amp` | — | Enable mixed precision |

Additional `check_prediction.py` CLI arguments:

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | — | Architecture name |
| `--checkpoint` | — | Path to `.pth` |
| `--input` | — | Image directory |
| `--save-dir` | — | Output directory for overlays |
| `--n-channels` | `1` | Input channels |
| `--classes` | `4` | Number of classes |
| `--img-size` | `640 640` | Model input size |
| `--conf-threshold` | `0.5` | Min softmax confidence (pixels below → background) |
| `--bilinear` | — | Use bilinear upsample |
| `--dropout` | `0.0` | Dropout rate |

---

## Model Notes

| Model | Best for | Speed | Accuracy |
|-------|----------|-------|----------|
| UNet | General segmentation | ⚡⚡⚡ | ⭐⭐⭐ |
| UNet++ | Fine structures (vessels) | ⚡⚡ | ⭐⭐⭐⭐ |
| DeepLab v3+ | Large images, satellite | ⚡⚡ | ⭐⭐⭐⭐ |
| UNetAdv | Fine structures + class imbalance | ⚡ | ⭐⭐⭐⭐⭐ |

UNetAdv features:
- **Variable attention** (SK blocks at each level)
- **Deep supervision** (4 decoder heads with weighted loss)
- **Higher capacity** (base filters: 32→64→128→256→512)

---

## Training Tips

- **16-bit RGBA**: All models accept 4-channel 16-bit input natively
- **Class imbalance** (tiny vessels in large image): use UNetAdv with `--classes 4 --focal-gamma 2.0 --dice-weight 2.0`
- **Out of memory**: reduce `--batch-size` to 1 or lower `--img-size` to `256 256`
- **Plateau early**: learning rate scheduler reduces LR automatically after 10 epochs of no improvement
- **Dice warmup**: Dice weight ramps from 0 → max over `--dice-warmup-epochs` (default 5) to avoid overwhelming Focal loss early
- **External test set**: use `--test-img-dir` / `--test-mask-dir` for a curated test set that covers all classes
- **NaN losses**: training auto-skips NaN batches (up to 50); if persistent, lower `--lr` to `5e-5` or `--focal-gamma` to `1.0`

---

## Feature Deep-Dives

### Stratified Split

`train.py` replaces `random_split` with `stratified_split()`, which groups images by the set of classes present in their masks and splits each group proportionally into train/val/test. This guarantees every rare class appears in all splits — critical when a class occupies only ~0.01% of pixels.

If you provide `--test-img-dir` / `--test-mask-dir`, the stratified split is used only for train/val and your external test set is loaded as-is.

The split is saved to `<checkpoint_dir>/data_split.yaml` for reproducibility. Reuse it later with `test.py --use-split-file`.

### Loss Function

**Binary:** BCEWithLogitsLoss (`--pos-weight 2.0`) + Dice loss.

**Multiclass:** Focal Loss (down-weights easy background pixels) + foreground-only Dice loss.

Foreground-only Dice excludes class 0 (background) from the Dice calculation, preventing the model from collapsing to all-background predictions when background dominates (~98% of pixels).

### Dice Warmup

Dice weight ramps linearly from `0` to `--dice-weight` (default 2.0) over the first `--dice-warmup-epochs` (default 5). This prevents Dice from overwhelming Focal loss before the model has learned basic class structure.

### Deep Supervision (UNetAdv)

UNetAdv returns a list of 4 logit tensors from intermediate decoder stages. The loss is a weighted sum:
```
loss = 0.1 * L(out1) + 0.2 * L(out2) + 0.3 * L(out3) + 0.4 * L(out4)
```
Early heads learn coarse features; the final head is optimised most strongly. At inference, only the last head is used.

### Resume Training

Use `--load checkpoints/latest.pth` to resume from a saved checkpoint. The optimizer state, epoch counter, and best validation Dice are restored. If the checkpoint has different `n_classes` or `n_channels`, those are auto-updated.

### Per-class Test Evaluation

`test.py` runs a full evaluation with:
- Per-class **Dice**, **IoU**, **Precision**, **Recall**
- **Confusion matrix** (rows = true class, cols = predicted)
- **Mean foreground Dice/IoU** (excludes background)
Results are saved to `<checkpoint_dir>/test_results.yaml`.

### Confidence Gating

`check_prediction.py` supports `--conf-threshold`: pixels whose max softmax probability falls below the threshold are remapped to background (class 0). This reduces false positives in low-confidence regions.

### Percentile Normalization

16-bit medical images often contain dead/bad pixels (sensor artifacts at the extreme black/white ends). `SegmentationDataset._normalize()` detects these (pixels ≤ 5 or ≥ 65530), excludes them from percentile computation, then clamps and normalizes valid pixels using the 1st–99th percentile range. Dead pixels are mapped to 0.0/1.0 respectively.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA GPU recommended (CPU works but slow)
