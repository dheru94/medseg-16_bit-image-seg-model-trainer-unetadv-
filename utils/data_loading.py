# utils/data_loading.py
"""
Dataset loader for 16-bit RGBA PNG images.
- Uses cv2 (not PIL) to preserve 16-bit depth
- Supports any number of input channels
- Mask segmentation channel is configurable (default: alpha = channel 3)
- Fixed output size via img_size=(W, H) to ensure consistent batching
"""
import logging
import numpy as np
import cv2
import torch
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filepath: str) -> np.ndarray:
    """
    Load an image preserving its original bit depth and channels.
    Returns numpy array (H, W, C) or (H, W) for grayscale.
    """
    ext = splitext(filepath)[1].lower()

    if ext == '.npy':
        return np.load(filepath)
    elif ext in ['.pt', '.pth']:
        return torch.load(filepath).numpy()
    else:
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f'Cannot load image: {filepath}')
        # cv2 loads as BGR/BGRA — convert to RGB/RGBA
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def _unique_mask_values(idx: str, mask_dir: Path,
                        mask_suffix: str, mask_channel: int) -> np.ndarray:
    """Scan a single mask and return its unique pixel values."""
    files = list(mask_dir.glob(idx + mask_suffix + '.*'))
    if not files:
        raise FileNotFoundError(f'No mask found for ID: {idx}')
    mask = load_image(files[0])
    if mask.ndim == 3:
        mask = mask[:, :, mask_channel]
    return np.unique(mask)


class SegmentationDataset(Dataset):
    """
    Dataset for binary or multi-class segmentation.

    Supports:
      - 16-bit RGBA PNG (medical / satellite)
      - Any number of channels
      - Configurable mask channel (default: alpha = 3)
      - Fixed output size via img_size to avoid batch collation errors
    """

    def __init__(self,
                 images_dir: str,
                 mask_dir: str,
                 scale: float = 1.0,
                 mask_suffix: str = '',
                 mask_channel: int = 3,
                 augment: bool = False,
                 img_size: tuple = (512, 512)):
        """
        Args:
            images_dir:   Path to input images folder.
            mask_dir:     Path to mask images folder.
            scale:        Proportional resize scale (used only if img_size is None).
            mask_suffix:  Optional suffix appended to image ID when looking up masks.
            mask_channel: Which channel of the mask image is the segmentation label (default 3 = alpha).
            augment:      Enable random flip augmentation.
            img_size:     Fixed output (W, H) for all images and masks.
                          Overrides scale. Set to None to use scale instead.
        """
        self.images_dir   = Path(images_dir)
        self.mask_dir     = Path(mask_dir)
        self.scale        = scale
        self.mask_suffix  = mask_suffix
        self.mask_channel = mask_channel
        self.augment      = augment
        self.img_size     = img_size  # (W, H) fixed output size

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [
            splitext(f)[0] for f in listdir(images_dir)
            if isfile(join(images_dir, f)) and not f.startswith('.')
        ]
        if not self.ids:
            raise RuntimeError(f'No images found in {images_dir}')

        logging.info(f'Dataset: {len(self.ids)} images found')
        logging.info(f'Scanning masks for unique values (channel {mask_channel})...')

        fn = partial(_unique_mask_values, mask_dir=self.mask_dir,
                     mask_suffix=self.mask_suffix, mask_channel=self.mask_channel)

        # Use Pool only on non-Windows or fallback to single process
        try:
            with Pool() as p:
                unique = list(tqdm(p.imap(fn, self.ids), total=len(self.ids)))
        except Exception:
            logging.warning('Multiprocessing pool failed — falling back to single process scan.')
            unique = [fn(i) for i in tqdm(self.ids)]

        self.mask_values = list(sorted(np.unique(np.concatenate(unique)).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

        if img_size:
            logging.info(f'Fixed output size: W={img_size[0]}, H={img_size[1]}')
        else:
            logging.info(f'Proportional scale: {scale}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    # def _normalize(img: np.ndarray) -> np.ndarray:
    #     """Normalize image to [0, 1] based on bit depth."""
    #     img = img.astype(np.float32)
    #     max_val = img.max()
    #     if max_val > 255:
    #         return img / 65535.0   # 16-bit
    #     elif max_val > 1:
    #         return img / 255.0     # 8-bit
    #     return img                 # already normalized
    
    # def _normalize(img: np.ndarray) -> np.ndarray:
    #     img = img.astype(np.float32)

    #     # Remove extreme outliers
    #     p1, p99 = np.percentile(img, (1, 99))

    #     img = np.clip(img, p1, p99)

    #     # Normalize to [0,1]
    #     img = (img - p1) / (p99 - p1 + 1e-8)

    #     return img
    def _normalize(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)

        # Detect dead pixels (sensor artifacts common in 16-bit medical images)
        dead_black = img <= 5
        dead_white = img >= 65530
        valid_mask = ~(dead_black | dead_white)

        valid_pixels = img[valid_mask]

        # FIX 1: fallback if no valid pixels exist (fully corrupted image)
        if valid_pixels.size == 0:
            img_norm = img / 65535.0   # plain normalization as fallback
            return img_norm.clip(0.0, 1.0)

        # Percentile normalization on valid pixels only
        p1, p99 = np.percentile(valid_pixels, (1, 99))

        img_norm = img.copy()

        # Clip and normalize valid region to [0, 1]
        img_norm[valid_mask] = np.clip(img_norm[valid_mask], p1, p99)
        img_norm[valid_mask] = (img_norm[valid_mask] - p1) / (p99 - p1 + 1e-8)

        # Dead pixels get fixed values
        img_norm[dead_black] = 0.0
        img_norm[dead_white] = 1.0

        # FIX 2: return the normalized array, not the original
        return img_norm



    @staticmethod
    def _preprocess_image(img: np.ndarray, scale: float,
                          img_size: tuple = None) -> np.ndarray:
        """
        Resize + normalize image.
        Output: (C, H, W) float32 in [0, 1]

        Args:
            img:      Raw numpy array (H, W, C) or (H, W).
            scale:    Proportional scale — used only when img_size is None.
            img_size: Fixed (W, H) output size — takes priority over scale.
        """
        h, w = img.shape[:2]

        if img_size is not None:
            newW, newH = img_size           # fixed size — all images same shape
        else:
            newW = int(scale * w)
            newH = int(scale * h)

        assert newW > 0 and newH > 0, \
            f'Computed size is zero: ({newW}, {newH}). Check scale or img_size.'

        img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_CUBIC)

        if img.ndim == 2:
            img = img[np.newaxis, ...]        # (1, H, W)
        else:
            img = img.transpose((2, 0, 1))   # (C, H, W)

        return SegmentationDataset._normalize(img)

    @staticmethod
    def _preprocess_mask(mask_values: list, mask: np.ndarray,
                         scale: float, mask_channel: int,
                         img_size: tuple = None) -> np.ndarray:
        """
        Extract mask channel, resize, map pixels to class indices.
        Output: (H, W) float32 — values 0.0, 1.0, 2.0 ...

        Args:
            mask_values:  List of unique pixel values (from dataset scan).
            mask:         Raw numpy array (H, W, C) or (H, W).
            scale:        Proportional scale — used only when img_size is None.
            mask_channel: Which channel index to extract as the segmentation label.
            img_size:     Fixed (W, H) output size — takes priority over scale.
        """
        if mask.ndim == 3:
            seg = mask[:, :, mask_channel]
        else:
            seg = mask

        h, w = seg.shape

        if img_size is not None:
            newW, newH = img_size           # fixed size — must match image resize
        else:
            newW = int(scale * w)
            newH = int(scale * h)

        assert newW > 0 and newH > 0, \
            f'Computed mask size is zero: ({newW}, {newH}). Check scale or img_size.'

        # INTER_NEAREST to preserve integer label values
        seg = cv2.resize(seg, (newW, newH), interpolation=cv2.INTER_NEAREST)

        out = np.zeros((newH, newW), dtype=np.float32)
        for i, v in enumerate(mask_values):
            out[seg == v] = float(i)

        return out

    @staticmethod
    def _augment(img: np.ndarray, mask: np.ndarray):
        """Basic augmentation: random horizontal and vertical flip."""
        if np.random.rand() > 0.5:
            img  = np.flip(img,  axis=2).copy()   # flip W axis of (C, H, W)
            mask = np.flip(mask, axis=1).copy()   # flip W axis of (H, W)
        if np.random.rand() > 0.5:
            img  = np.flip(img,  axis=1).copy()   # flip H axis of (C, H, W)
            mask = np.flip(mask, axis=0).copy()   # flip H axis of (H, W)
        return img, mask

    def __getitem__(self, idx: int):
        name       = self.ids[idx]
        img_files  = list(self.images_dir.glob(name + '.*'))
        mask_files = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))

        assert len(img_files)  == 1, \
            f'Expected exactly 1 image for "{name}", found: {img_files}'
        assert len(mask_files) == 1, \
            f'Expected exactly 1 mask for "{name}", found: {mask_files}'

        img  = load_image(img_files[0])
        mask = load_image(mask_files[0])

        assert img.shape[:2] == mask.shape[:2], (
            f'Spatial size mismatch for "{name}": '
            f'image={img.shape}, mask={mask.shape}'
        )

        img  = self._preprocess_image(img, self.scale, self.img_size)
        mask = self._preprocess_mask(self.mask_values, mask, self.scale,
                                     self.mask_channel, self.img_size)

        if self.augment:
            img, mask = self._augment(img, mask)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask':  torch.as_tensor(mask.copy()).float().contiguous(),
        }
