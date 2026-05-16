# scripts/verify_dataset.py
"""
Run this BEFORE training to catch data problems early.

Usage:
    python scripts/verify_dataset.py --img-dir data/images --mask-dir data/masks
"""
import argparse
import logging
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def verify_dataset(img_dir: str, mask_dir: str,
                   mask_suffix: str = '', mask_channel: int = 3):
    img_dir  = Path(img_dir)
    mask_dir = Path(mask_dir)

    img_files  = sorted([f for f in img_dir.iterdir()
                          if not f.name.startswith('.')])
    mask_files = sorted([f for f in mask_dir.iterdir()
                          if not f.name.startswith('.')])

    print(f'\n{"═"*50}')
    print(f'  MedSeg Dataset Verification')
    print(f'{"═"*50}')
    print(f'  Images dir : {img_dir}')
    print(f'  Masks dir  : {mask_dir}')
    print(f'  Images found: {len(img_files)}')
    print(f'  Masks found : {len(mask_files)}')

    errors   = []
    warnings = []
    img_shapes  = set()
    mask_uniq   = set()
    bit_depths  = set()

    for img_path in tqdm(img_files, desc='Checking'):
        stem = img_path.stem
        mask_matches = list(mask_dir.glob(stem + mask_suffix + '.*'))

        # Check mask exists
        if not mask_matches:
            errors.append(f'No mask for image: {img_path.name}')
            continue
        mask_path = mask_matches[0]

        # Load both
        img  = cv2.imread(str(img_path),  cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            errors.append(f'Cannot load image: {img_path.name}')
            continue
        if mask is None:
            errors.append(f'Cannot load mask: {mask_path.name}')
            continue

        # Shape check
        if img.shape[:2] != mask.shape[:2]:
            errors.append(f'Shape mismatch {img_path.name}: '
                          f'img={img.shape[:2]}, mask={mask.shape[:2]}')

        img_shapes.add(img.shape[:2])

        # Bit depth
        bit_depths.add(img.dtype)

        # Channel check
        if img.ndim != 3 or img.shape[2] != 4:
            warnings.append(f'Image is not RGBA: {img_path.name} shape={img.shape}')

        # Mask unique values
        if mask.ndim == 3 and mask.shape[2] >= mask_channel + 1:
            seg = mask[:, :, mask_channel]
        else:
            seg = mask.squeeze()
        mask_uniq.update(np.unique(seg).tolist())

        # Check for NaN/Inf in image
        img_f = img.astype(np.float32)
        if np.isnan(img_f).any() or np.isinf(img_f).any():
            errors.append(f'NaN or Inf pixels in: {img_path.name}')

        # Check mask is binary (for binary segmentation)
        if len(np.unique(seg)) > 10:
            warnings.append(f'Mask has many unique values ({len(np.unique(seg))}): '
                            f'{mask_path.name} — expected binary')

    # Print results
    print(f'\n  Image shapes   : {img_shapes}')
    print(f'  Image dtypes   : {bit_depths}')
    print(f'  Mask unique values (ch {mask_channel}): {sorted(mask_uniq)}')
    print(f'  Expected norm divisor: '
          f'{"65535.0 (16-bit ✓)" if np.uint16 in bit_depths else "255.0 (8-bit)"}')

    print(f'\n  Errors   : {len(errors)}')
    print(f'  Warnings : {len(warnings)}')

    if errors:
        print('\n  ❌ ERRORS (must fix before training):')
        for e in errors:
            print(f'     • {e}')
    if warnings:
        print('\n  ⚠️  WARNINGS:')
        for w in warnings:
            print(f'     • {w}')

    if not errors:
        print('\n  ✅ Dataset looks good — ready to train!')
    print(f'{"═"*50}\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Verify dataset before training')
    p.add_argument('--img-dir',      type=str, required=True)
    p.add_argument('--mask-dir',     type=str, required=True)
    p.add_argument('--mask-suffix',  type=str, default='')
    p.add_argument('--mask-channel', type=int, default=3,
                   help='Which channel in RGBA mask to use (default 3 = alpha)')
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    verify_dataset(args.img_dir, args.mask_dir, args.mask_suffix, args.mask_channel)
