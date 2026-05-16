# predict.py
"""
Run inference on new images using a trained checkpoint.

Usage:
    # Single image
    python predict.py --model unet --checkpoint checkpoints/best.pth \
                      --input image.png --output pred.png

    # Folder of images
    python predict.py --model unet --checkpoint checkpoints/best.pth \
                      --input data/images/ --output predictions/
"""
import argparse
import logging
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

from models import build_model
from utils.data_loading import load_image


def preprocess(img: np.ndarray, scale: float, device: torch.device) -> torch.Tensor:
    """Resize + normalize + convert to tensor."""
    h, w   = img.shape[:2]
    newW   = int(scale * w)
    newH   = int(scale * h)
    img    = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_CUBIC)

    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))

    img = img.astype(np.float32)
    img = img / 65535.0 if img.max() > 255 else img / 255.0

    return torch.from_numpy(img).unsqueeze(0).to(device)   # (1, C, H, W)


@torch.inference_mode()
def predict_single(model, img_path: Path, scale: float,
                   threshold: float, device: torch.device) -> np.ndarray:
    """Run inference on a single image. Returns binary mask as uint8 (0/255)."""
    img    = load_image(str(img_path))
    orig_h, orig_w = img.shape[:2]

    tensor = preprocess(img, scale, device)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True):
        logits = model(tensor)

    if model.n_classes == 1:
        prob = torch.sigmoid(logits.squeeze()).cpu().numpy()
        mask = (prob > threshold).astype(np.uint8) * 255
    else:
        mask = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        mask = (mask * (255 // max(model.n_classes - 1, 1)))

    # Resize back to original resolution
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask


def save_mask(mask: np.ndarray, out_path: Path):
    """Save predicted mask as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='MedSeg — Run inference')
    p.add_argument('--model',      type=str, required=True, help='unet | unetpp | deeplab')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--input',      type=str, required=True, help='Image file or folder')
    p.add_argument('--output',     type=str, required=True, help='Output file or folder')
    p.add_argument('--scale',      type=float, default=0.5)
    p.add_argument('--threshold',  type=float, default=0.5,
                   help='Sigmoid threshold for binary masks (default 0.5)')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Load model from checkpoint
    ckpt       = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_channels = ckpt.get('n_channels', 4)
    n_classes  = ckpt.get('n_classes', 1)

    model = build_model(args.model, n_channels=n_channels, n_classes=n_classes)
    model.load_state_dict(ckpt['model_state'])
    model.to(memory_format=torch.channels_last).to(device).eval()
    logging.info(f'Model loaded: {args.model} | channels={n_channels} | classes={n_classes}')

    input_path  = Path(args.input)
    output_path = Path(args.output)

    # Collect images
    if input_path.is_dir():
        exts   = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in exts]
        logging.info(f'Found {len(images)} images in {input_path}')
    else:
        images = [input_path]

    for img_path in tqdm(images, desc='Predicting'):
        mask = predict_single(model, img_path, args.scale, args.threshold, device)

        if input_path.is_dir():
            out = output_path / (img_path.stem + '_pred.png')
        else:
            out = output_path if output_path.suffix else output_path / (img_path.stem + '_pred.png')

        save_mask(mask, out)

    logging.info(f'Done. Predictions saved to: {output_path}')
