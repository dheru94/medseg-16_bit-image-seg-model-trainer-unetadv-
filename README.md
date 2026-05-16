# MedSeg — Medical & Satellite Image Segmentation

A production-ready PyTorch framework for semantic segmentation of 16-bit RGBA images. Designed for medical scans (X-ray, CT), satellite imagery, and industrial inspection images.

---

## What This Project Does

MedSeg takes input images (grayscale, RGB, or RGBA with 8-bit or 16-bit depth) and produces pixel-level segmentation masks. It's built to handle:

- **16-bit depth images** — Preserves full dynamic range from medical sensors
- **RGBA format** — Alpha channel can encode segmentation labels directly
- **Multi-class segmentation** — Binary (defect vs background) or multi-class
- **Small datasets** — Works with limited training data through data augmentation and deep supervision

---

## Features

### Model Architectures
- **UNet** — Classic encoder-decoder with skip connections
- **UNet++** — Nested dense skip connections for better feature fusion
- **DeepLabv3+** — ASPP module captures multi-scale context
- **UNetAdv** — Variable Attention UNet++ with SK blocks and Attention Gates

### Data Pipeline
- Loads 16-bit PNG images preserving full bit depth
- Configurable mask channel (default: alpha channel = channel 3)
- Custom dataset loader with smart ID matching
- Augmentation: random horizontal/vertical flips

### Training
- Mixed precision (AMP) for faster training and lower VRAM
- Gradient clipping for stability
- Learning rate scheduling (ReduceLROnPlateau)
- Class-weighted loss for imbalanced datasets
- Focal loss option for hard-to-learn pixels
- Checkpointing: saves latest, best, and per-epoch models
- Resume training from any checkpoint

### Evaluation
- Dice coefficient
- Intersection over Union (IoU)
- Precision, Recall, F1-score
- Per-class metrics for multi-class segmentation

---

## Data Format

Expected directory structure:

```
data/
├── images/    # Input images (PNG, 8-bit or 16-bit)
│   ├── image1.png
│   └── image2.png
└── masks/     # Segmentation masks (RGBA, same size as images)
    ├── image1.png
    └── image2.png
```

- **Images**: 8-bit or 16-bit, grayscale or RGBA
- **Masks**: RGBA format, mask encoded in one channel (configurable, default: alpha)
- **Multi-class**: Different pixel values in mask channel represent different classes

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train UNetAdv (recommended for defect detection)
python train.py --model unetadv --epochs 50 --batch-size 2 --lr 1e-4 --amp

# Or train with config file
python train.py --config configs/default.yaml

# Run inference on new images
python predict.py --model unetadv --checkpoint checkpoints/best.pth \
                  --input data/images/ --output predictions/ --threshold 0.5

# Evaluate on test dataset
python evaluate.py --model unetadv --checkpoint checkpoints/best.pth \
                   --img-dir data/images --mask-dir data/masks
```

---

## Configuration

All settings in `configs/default.yaml`. CLI args override config values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `unet` | Model: unet, unetpp, deeplab, unetadv |
| `n_channels` | `4` | Input channels: 1 (gray), 3 (RGB), 4 (RGBA) |
| `n_classes` | `1` | 1 = binary, >1 = multi-class |
| `epochs` | `50` | Training epochs |
| `batch_size` | `2` | Batch size |
| `lr` | `1e-5` | Learning rate |
| `scale` | `0.5` | Input resize scale (0.5 = half size) |
| `amp` | `true` | Mixed precision training |
| `dropout` | `0.2` | Bottleneck dropout probability |
| `deep_supervision` | `true` | Multi-output training (UNet++, UNetAdv) |
| `mask_channel` | `3` | Which channel contains mask (0-3 for RGBA) |

---

## Project Structure

```
medseg/
├── models/
│   ├── __init__.py      # Model factory
│   ├── unet.py          # Classic UNet
│   ├── unetpp.py        # Nested UNet++
│   ├── deeplab.py       # DeepLabv3+
│   └── unetadv.py       # Variable Attention UNet++
├── utils/
│   ├── __init__.py
│   ├── data_loading.py  # Dataset loader, image/mask preprocessing
│   └── dice_score.py   # Dice coefficient and loss
├── configs/
│   └── default.yaml    # Default hyperparameters
├── scripts/
│   └── verify_dataset.py # Validate dataset before training
├── train.py            # Training script
├── predict.py          # Inference script
├── evaluate.py         # Evaluation script
└── requirements.txt    # Python dependencies
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA GPU (recommended for training)
- Dependencies: torch, torchvision, numpy, opencv-python, pyyaml, tqdm, matplotlib, pillow

---

## Notes

- All models accept 4-channel RGBA input natively
- For class imbalance (tiny defects in large image), increase `pos_weight` in config
- If out of memory, reduce `batch_size` or lower `scale` to 0.25
- Deep supervision enables model pruning — use fewer outputs during inference for speed

---

## References

1. **UNet** — Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", *MICCAI* 2015

2. **UNet++** — Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", *DLMIA* 2018

3. **DeepLabv3+** — Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation", *ECCV* 2018

4. **UNetAdv** — Liu & Kim, "A Variable Attention Nested UNet++ Network-Based NDT X-ray Image Defect Segmentation Method", *Coatings* 2022