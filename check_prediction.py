import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from models import build_model
from utils.data_loading import (
    load_image,
    SegmentationDataset
)


# ─────────────────────────────────────────────────────────────
# CLASS COLORS
# ─────────────────────────────────────────────────────────────
CLASS_COLORS = {
    0: [0, 0, 0],       # background
    1: [255, 0, 0],     # void
    2: [0, 255, 0],     # hda
    3: [0, 0, 255],     # crack
}


# ─────────────────────────────────────────────────────────────
# CLASS NAMES
# 0 = background (ignored)
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = {
    1: "void",
    2: "hda",
    3: "crack",
}


# ─────────────────────────────────────────────────────────────
# DISPLAY NORMALIZATION ONLY
# For visualization in matplotlib
# NOT used for inference
# ─────────────────────────────────────────────────────────────
def normalize_for_display(img):

    img = img.astype(np.float32)

    # grayscale extraction if RGB/RGBA
    if img.ndim == 3:
        img = img[..., 0]

    # dead pixels
    dead_black = img <= 5
    dead_white = img >= 65530

    valid_mask = ~(dead_black | dead_white)

    valid_pixels = img[valid_mask]

    if valid_pixels.size == 0:
        img = img / 65535.0
        return np.clip(img, 0.0, 1.0)

    p1, p99 = np.percentile(valid_pixels, (1, 99))

    img = np.clip(img, p1, p99)

    img = (img - p1) / (p99 - p1 + 1e-8)

    # keep dead pixels fixed
    img[dead_black] = 0.0
    img[dead_white] = 1.0

    return np.clip(img, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# LOAD + PREPROCESS
# EXACT SAME preprocessing as training
# ─────────────────────────────────────────────────────────────
def load_and_preprocess(img_path, img_size):

    img = load_image(str(img_path))

    original = img.copy()

    # EXACT SAME preprocessing as training
    img = SegmentationDataset._preprocess_image(
        img=img,
        scale=1.0,
        img_size=tuple(img_size)
    )

    tensor = torch.from_numpy(img).unsqueeze(0)

    return tensor, original


# ─────────────────────────────────────────────────────────────
# PREDICT
# Handles deep supervision automatically.
# Pixels whose max softmax probability is below conf_threshold
# are remapped to class 0 (background).
# ─────────────────────────────────────────────────────────────
def predict(model, image_tensor, device, conf_threshold=0.5):

    image_tensor = image_tensor.to(
        device=device,
        dtype=torch.float32
    )

    with torch.no_grad():

        output = model(image_tensor)

        # UNetAdv deep supervision → use last output
        if isinstance(output, (list, tuple)):
            output = output[-1]

        probs = F.softmax(output, dim=1)          # [1, C, H, W]

        # max probability and argmax class per pixel
        max_probs, pred_mask = probs.max(dim=1)   # both [1, H, W]

        pred_mask = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)
        max_probs  = max_probs.squeeze(0).cpu().numpy()   # float32 [H, W]

    # ── confidence gate ──────────────────────────────────────
    # pixels below threshold → background (class 0)
    pred_mask[max_probs < conf_threshold] = 0

    return pred_mask


# ─────────────────────────────────────────────────────────────
# OVERLAY + LEGEND
# ─────────────────────────────────────────────────────────────
def make_overlay(image, mask):
    """
    Overlay segmentation mask on grayscale image
    with legend box in corner.
    """

    h, w = image.shape

    # grayscale → RGB
    overlay = np.stack([image, image, image], axis=-1)
    overlay = (overlay * 255).astype(np.uint8)

    # colored mask
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in CLASS_COLORS.items():
        color_mask[mask == cls] = color

    # blend
    alpha = 0.45

    blended = cv2.addWeighted(
        overlay,
        1.0,
        color_mask,
        alpha,
        0
    )

    # -------------------------------------------------
    # LEGEND BOX
    # -------------------------------------------------

    legend_items = [
        ("void",  CLASS_COLORS[1]),
        ("hda",   CLASS_COLORS[2]),
        ("crack", CLASS_COLORS[3]),
    ]

    box_x = 15
    box_y = 15

    line_h = 35
    box_w = 180
    box_h = 20 + line_h * len(legend_items)

    # semi-transparent dark box
    overlay_copy = blended.copy()

    cv2.rectangle(
        overlay_copy,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (20, 20, 20),
        -1
    )

    blended = cv2.addWeighted(
        overlay_copy,
        0.45,
        blended,
        0.55,
        0
    )

    # draw legend entries
    for i, (name, color) in enumerate(legend_items):

        yy = box_y + 20 + i * line_h

        # color square
        cv2.rectangle(
            blended,
            (box_x + 10, yy - 10),
            (box_x + 30, yy + 10),
            color,
            -1
        )

        # class text
        cv2.putText(
            blended,
            name,
            (box_x + 45, yy + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return blended


# ─────────────────────────────────────────────────────────────
# VISUALIZE
# ─────────────────────────────────────────────────────────────
def visualize(
    model,
    image_dir,
    device,
    img_size,
    conf_threshold=0.5,
    save_dir=None
):

    image_paths = sorted([
        p for p in Path(image_dir).glob("*")
        if p.suffix.lower() in [
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff"
        ]
    ])

    if len(image_paths) == 0:
        print("No images found.")
        return

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:

        print(f"\nProcessing: {img_path.name}")

        # EXACT SAME preprocessing as training
        image_tensor, original = load_and_preprocess(
            img_path,
            img_size
        )

        pred_mask = predict(
            model,
            image_tensor,
            device,
            conf_threshold=conf_threshold
        )

        # visualization normalization only
        display_img = normalize_for_display(original)

        # resize prediction to original size
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            (
                display_img.shape[1],
                display_img.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

        overlay = make_overlay(
            display_img,
            pred_mask
        )

        # ─────────────────────────────────────────────────────
        # SHOW
        # ─────────────────────────────────────────────────────
        plt.figure(figsize=(14, 7))

        plt.subplot(1, 2, 1)
        plt.imshow(display_img, cmap='gray')
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f"Prediction Overlay  (conf ≥ {conf_threshold:.2f})")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        # ─────────────────────────────────────────────────────
        # SAVE
        # ─────────────────────────────────────────────────────
        if save_dir:

            out_path = save_dir / f"{img_path.stem}_overlay.png"

            cv2.imwrite(
                str(out_path),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )

            print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="unet | unetpp | unetadv | deeplab"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="directory containing images"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="optional directory to save overlays"
    )

    parser.add_argument(
        "--n-channels",
        type=int,
        default=1
    )

    parser.add_argument(
        "--classes",
        type=int,
        default=4
    )

    parser.add_argument(
        "--img-size",
        nargs=2,
        type=int,
        default=[640, 640],
        metavar=("W", "H")
    )

    parser.add_argument(
        "--bilinear",
        action="store_true"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum softmax confidence to accept a prediction (0.0–1.0). "
            "Pixels below this threshold are remapped to background (class 0). "
            "Default: 0.5"
        )
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    args = get_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Confidence threshold: {args.conf_threshold}")

    # build model
    model = build_model(
        name=args.model,
        n_channels=args.n_channels,
        n_classes=args.classes,
        bilinear=args.bilinear,
        dropout=args.dropout,
    )

    # load checkpoint
    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(ckpt["model_state"])

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    visualize(
        model=model,
        image_dir=args.input,
        device=device,
        img_size=args.img_size,
        conf_threshold=args.conf_threshold,
        save_dir=args.save_dir
    )































# import argparse
# from pathlib import Path

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# import torch
# import torch.nn.functional as F

# from models import build_model
# from utils.data_loading import (
#     load_image,
#     SegmentationDataset
# )


# # ─────────────────────────────────────────────────────────────
# # CLASS COLORS
# # ─────────────────────────────────────────────────────────────
# CLASS_COLORS = {
#     0: [0, 0, 0],       # background
#     1: [255, 0, 0],     # void
#     2: [0, 255, 0],     # hda
#     3: [0, 0, 255],     # crack
# }


# # ─────────────────────────────────────────────────────────────
# # CLASS NAMES
# # 0 = background (ignored)
# # ─────────────────────────────────────────────────────────────
# CLASS_NAMES = {
#     1: "void",
#     2: "hda",
#     3: "crack",
# }


# # ─────────────────────────────────────────────────────────────
# # DISPLAY NORMALIZATION ONLY
# # For visualization in matplotlib
# # NOT used for inference
# # ─────────────────────────────────────────────────────────────
# def normalize_for_display(img):

#     img = img.astype(np.float32)

#     # grayscale extraction if RGB/RGBA
#     if img.ndim == 3:
#         img = img[..., 0]

#     # dead pixels
#     dead_black = img <= 5
#     dead_white = img >= 65530

#     valid_mask = ~(dead_black | dead_white)

#     valid_pixels = img[valid_mask]

#     if valid_pixels.size == 0:
#         img = img / 65535.0
#         return np.clip(img, 0.0, 1.0)

#     p1, p99 = np.percentile(valid_pixels, (1, 99))

#     img = np.clip(img, p1, p99)

#     img = (img - p1) / (p99 - p1 + 1e-8)

#     # keep dead pixels fixed
#     img[dead_black] = 0.0
#     img[dead_white] = 1.0

#     return np.clip(img, 0.0, 1.0)


# # ─────────────────────────────────────────────────────────────
# # LOAD + PREPROCESS
# # EXACT SAME preprocessing as training
# # ─────────────────────────────────────────────────────────────
# def load_and_preprocess(img_path, img_size):

#     img = load_image(str(img_path))

#     original = img.copy()

#     # EXACT SAME preprocessing as training
#     img = SegmentationDataset._preprocess_image(
#         img=img,
#         scale=1.0,
#         img_size=tuple(img_size)
#     )

#     tensor = torch.from_numpy(img).unsqueeze(0)

#     return tensor, original


# # ─────────────────────────────────────────────────────────────
# # PREDICT
# # Handles deep supervision automatically
# # ─────────────────────────────────────────────────────────────
# def predict(model, image_tensor, device):

#     image_tensor = image_tensor.to(
#         device=device,
#         dtype=torch.float32
#     )

#     with torch.no_grad():

#         output = model(image_tensor)

#         # UNetAdv deep supervision
#         if isinstance(output, (list, tuple)):
#             output = output[-1]

#         probs = F.softmax(output, dim=1)

#         pred_mask = probs.argmax(dim=1)

#         pred_mask = pred_mask.squeeze(0).cpu().numpy()

#     return pred_mask.astype(np.uint8)


# # ─────────────────────────────────────────────────────────────
# # OVERLAY + LABELS
# # ─────────────────────────────────────────────────────────────
# # def make_overlay(image, mask):

# #     h, w = image.shape

# #     overlay = np.stack([image, image, image], axis=-1)
# #     overlay = (overlay * 255).astype(np.uint8)

# #     color_mask = np.zeros((h, w, 3), dtype=np.uint8)

# #     # draw segmentation colors
# #     for cls, color in CLASS_COLORS.items():
# #         color_mask[mask == cls] = color

# #     alpha = 0.45

# #     blended = cv2.addWeighted(
# #         overlay,
# #         1.0,
# #         color_mask,
# #         alpha,
# #         0
# #     )

# #     # ─────────────────────────────────────────────────────────
# #     # WRITE LABELS ON DEFECTS
# #     # ─────────────────────────────────────────────────────────
# #     for cls, name in CLASS_NAMES.items():

# #         binary = (mask == cls).astype(np.uint8)

# #         # skip absent classes
# #         if binary.sum() == 0:
# #             continue

# #         # connected components
# #         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
# #             binary,
# #             connectivity=8
# #         )

# #         for i in range(1, num_labels):

# #             area = stats[i, cv2.CC_STAT_AREA]

# #             # ignore tiny blobs/noise
# #             if area < 40:
# #                 continue

# #             x = stats[i, cv2.CC_STAT_LEFT]
# #             y = stats[i, cv2.CC_STAT_TOP]
# #             w_box = stats[i, cv2.CC_STAT_WIDTH]
# #             h_box = stats[i, cv2.CC_STAT_HEIGHT]

# #             cx = x + w_box // 2
# #             cy = y + h_box // 2

# #             color = CLASS_COLORS[cls]

# #             # text background
# #             cv2.rectangle(
# #                 blended,
# #                 (cx - 40, cy - 18),
# #                 (cx + 40, cy + 6),
# #                 color,
# #                 -1
# #             )

# #             # label text
# #             cv2.putText(
# #                 blended,
# #                 name,
# #                 (cx - 35, cy),
# #                 cv2.FONT_HERSHEY_SIMPLEX,
# #                 0.5,
# #                 (255, 255, 255),
# #                 1,
# #                 cv2.LINE_AA
# #             )

# #     return blended

# def make_overlay(image, mask):
#     """
#     Overlay segmentation mask on grayscale image
#     with legend box in corner.
#     """

#     h, w = image.shape

#     # grayscale → RGB
#     overlay = np.stack([image, image, image], axis=-1)
#     overlay = (overlay * 255).astype(np.uint8)

#     # colored mask
#     color_mask = np.zeros((h, w, 3), dtype=np.uint8)

#     for cls, color in CLASS_COLORS.items():
#         color_mask[mask == cls] = color

#     # blend
#     alpha = 0.45

#     blended = cv2.addWeighted(
#         overlay,
#         1.0,
#         color_mask,
#         alpha,
#         0
#     )

#     # -------------------------------------------------
#     # LEGEND BOX
#     # -------------------------------------------------

#     legend_items = [
#         ("void",  CLASS_COLORS[1]),
#         ("hda",   CLASS_COLORS[2]),
#         ("crack", CLASS_COLORS[3]),
#     ]

#     box_x = 15
#     box_y = 15

#     line_h = 35
#     box_w = 180
#     box_h = 20 + line_h * len(legend_items)

#     # semi-transparent dark box
#     overlay_copy = blended.copy()

#     cv2.rectangle(
#         overlay_copy,
#         (box_x, box_y),
#         (box_x + box_w, box_y + box_h),
#         (20, 20, 20),
#         -1
#     )

#     blended = cv2.addWeighted(
#         overlay_copy,
#         0.45,
#         blended,
#         0.55,
#         0
#     )

#     # draw legend entries
#     for i, (name, color) in enumerate(legend_items):

#         yy = box_y + 20 + i * line_h

#         # color square
#         cv2.rectangle(
#             blended,
#             (box_x + 10, yy - 10),
#             (box_x + 30, yy + 10),
#             color,
#             -1
#         )

#         # class text
#         cv2.putText(
#             blended,
#             name,
#             (box_x + 45, yy + 7),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2,
#             cv2.LINE_AA
#         )

#     return blended



# # ─────────────────────────────────────────────────────────────
# # VISUALIZE
# # ─────────────────────────────────────────────────────────────
# def visualize(
#     model,
#     image_dir,
#     device,
#     img_size,
#     save_dir=None
# ):

#     image_paths = sorted([
#         p for p in Path(image_dir).glob("*")
#         if p.suffix.lower() in [
#             ".png",
#             ".jpg",
#             ".jpeg",
#             ".bmp",
#             ".tif",
#             ".tiff"
#         ]
#     ])

#     if len(image_paths) == 0:
#         print("No images found.")
#         return

#     if save_dir:
#         save_dir = Path(save_dir)
#         save_dir.mkdir(parents=True, exist_ok=True)

#     for img_path in image_paths:

#         print(f"\nProcessing: {img_path.name}")

#         # EXACT SAME preprocessing as training
#         image_tensor, original = load_and_preprocess(
#             img_path,
#             img_size
#         )

#         pred_mask = predict(
#             model,
#             image_tensor,
#             device
#         )

#         # visualization normalization only
#         display_img = normalize_for_display(original)

#         # resize prediction to original size
#         pred_mask = cv2.resize(
#             pred_mask.astype(np.uint8),
#             (
#                 display_img.shape[1],
#                 display_img.shape[0]
#             ),
#             interpolation=cv2.INTER_NEAREST
#         )

#         overlay = make_overlay(
#             display_img,
#             pred_mask
#         )

#         # ─────────────────────────────────────────────────────
#         # SHOW
#         # ─────────────────────────────────────────────────────
#         plt.figure(figsize=(14, 7))

#         plt.subplot(1, 2, 1)
#         plt.imshow(display_img, cmap='gray')
#         plt.title("Original Image")
#         plt.axis("off")

#         plt.subplot(1, 2, 2)
#         plt.imshow(overlay)
#         plt.title("Prediction Overlay")
#         plt.axis("off")

#         plt.tight_layout()
#         plt.show()

#         # ─────────────────────────────────────────────────────
#         # SAVE
#         # ─────────────────────────────────────────────────────
#         if save_dir:

#             out_path = save_dir / f"{img_path.stem}_overlay.png"

#             cv2.imwrite(
#                 str(out_path),
#                 cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
#             )

#             print(f"Saved: {out_path}")


# # ─────────────────────────────────────────────────────────────
# # CLI
# # ─────────────────────────────────────────────────────────────
# def get_args():

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--model",
#         type=str,
#         required=True,
#         help="unet | unetpp | unetadv | deeplab"
#     )

#     parser.add_argument(
#         "--checkpoint",
#         type=str,
#         required=True
#     )

#     parser.add_argument(
#         "--input",
#         type=str,
#         required=True,
#         help="directory containing images"
#     )

#     parser.add_argument(
#         "--save-dir",
#         type=str,
#         default=None,
#         help="optional directory to save overlays"
#     )

#     parser.add_argument(
#         "--n-channels",
#         type=int,
#         default=1
#     )

#     parser.add_argument(
#         "--classes",
#         type=int,
#         default=4
#     )

#     parser.add_argument(
#         "--img-size",
#         nargs=2,
#         type=int,
#         default=[640, 640],
#         metavar=("W", "H")
#     )

#     parser.add_argument(
#         "--bilinear",
#         action="store_true"
#     )

#     parser.add_argument(
#         "--dropout",
#         type=float,
#         default=0.0
#     )

#     return parser.parse_args()


# # ─────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────
# if __name__ == "__main__":

#     args = get_args()

#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     print(f"Using device: {device}")

#     # build model
#     model = build_model(
#         name=args.model,
#         n_channels=args.n_channels,
#         n_classes=args.classes,
#         bilinear=args.bilinear,
#         dropout=args.dropout,
#     )

#     # load checkpoint
#     ckpt = torch.load(
#         args.checkpoint,
#         map_location=device,
#         weights_only=False
#     )

#     model.load_state_dict(ckpt["model_state"])

#     model.to(device)
#     model.eval()

#     print("Model loaded successfully.")

#     visualize(
#         model=model,
#         image_dir=args.input,
#         device=device,
#         img_size=args.img_size,
#         save_dir=args.save_dir
#     )







































# # visualization.py
# #
# # Visualize predictions from UNet / UNet++ / DeepLab / UNetAdv
# # Shows:
# #   1. Original image
# #   2. Original + predicted segmentation overlay
# #
# # Works with:
# #   - Deep supervision models (UNetAdv)
# #   - Multiclass segmentation
# #   - Grayscale images
# #
# # Usage:
# # python visualization.py ^
# #   --model unetadv ^
# #   --checkpoint D:\itarsi\checkpoints\best.pth ^
# #   --input D:\itarsi\mask_data\images ^
# #   --n-channels 1 ^
# #   --classes 4 ^
# #   --img-size 512 512

# import argparse
# from pathlib import Path

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# import torch
# import torch.nn.functional as F

# from models import build_model


# # ─────────────────────────────────────────────────────────────
# # COLORS FOR CLASSES
# # background = transparent
# # class1 = red
# # class2 = green
# # class3 = blue
# # ─────────────────────────────────────────────────────────────
# CLASS_COLORS = {
#     0: [0, 0, 0],
#     1: [255, 0, 0],
#     2: [0, 255, 0],
#     3: [0, 0, 255],
# }


# def load_image(img_path, img_size):
#     """
#     Load grayscale image.
#     """

#     img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

#     if img is None:
#         raise ValueError(f"Could not read image: {img_path}")

#     # normalize
#     img = img.astype(np.float32)

#     if img.max() > 0:
#         img = img / img.max()

#     original = img.copy()

#     # resize for model
#     img_resized = cv2.resize(
#         img,
#         tuple(img_size),
#         interpolation=cv2.INTER_AREA
#     )

#     tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0)

#     return tensor, original


# def make_overlay(image, mask):
#     """
#     Overlay segmentation mask on grayscale image.
#     """

#     h, w = image.shape

#     overlay = np.stack([image, image, image], axis=-1)
#     overlay = (overlay * 255).astype(np.uint8)

#     color_mask = np.zeros((h, w, 3), dtype=np.uint8)

#     for cls, color in CLASS_COLORS.items():
#         color_mask[mask == cls] = color

#     alpha = 0.45

#     blended = cv2.addWeighted(
#         overlay,
#         1.0,
#         color_mask,
#         alpha,
#         0
#     )

#     return blended


# def predict(model, image_tensor, device):
#     """
#     Predict segmentation mask.
#     Handles deep supervision automatically.
#     """

#     image_tensor = image_tensor.to(device=device, dtype=torch.float32)

#     with torch.no_grad():

#         output = model(image_tensor)

#         # UNetAdv deep supervision
#         if isinstance(output, (list, tuple)):
#             output = output[-1]

#         probs = F.softmax(output, dim=1)

#         pred_mask = probs.argmax(dim=1).squeeze(0).cpu().numpy()

#     return pred_mask.astype(np.uint8)


# def visualize(model, image_dir, device, img_size):

#     image_paths = sorted([
#         p for p in Path(image_dir).glob("*")
#         if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
#     ])

#     if len(image_paths) == 0:
#         print("No images found.")
#         return

#     for img_path in image_paths:

#         print(f"Processing: {img_path.name}")

#         image_tensor, original = load_image(img_path, img_size)

#         pred_mask = predict(model, image_tensor, device)

#         # resize prediction back to original size
#         pred_mask = cv2.resize(
#             pred_mask.astype(np.uint8),
#             (original.shape[1], original.shape[0]),
#             interpolation=cv2.INTER_NEAREST
#         )

#         overlay = make_overlay(original, pred_mask)

#         # show
#         plt.figure(figsize=(14, 7))

#         plt.subplot(1, 2, 1)
#         plt.imshow(original, cmap='gray')
#         plt.title("Original Image")
#         plt.axis("off")

#         plt.subplot(1, 2, 2)
#         plt.imshow(overlay)
#         plt.title("Prediction Overlay")
#         plt.axis("off")

#         plt.tight_layout()
#         plt.show()


# def get_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--model",
#         type=str,
#         required=True,
#         help="unet | unetpp | unetadv | deeplab"
#     )

#     parser.add_argument(
#         "--checkpoint",
#         type=str,
#         required=True
#     )

#     parser.add_argument(
#         "--input",
#         type=str,
#         required=True,
#         help="directory containing images"
#     )

#     parser.add_argument(
#         "--n-channels",
#         type=int,
#         default=1
#     )

#     parser.add_argument(
#         "--classes",
#         type=int,
#         default=4
#     )

#     parser.add_argument(
#         "--img-size",
#         nargs=2,
#         type=int,
#         default=[512, 512],
#         metavar=("W", "H")
#     )

#     parser.add_argument(
#         "--bilinear",
#         action="store_true"
#     )

#     parser.add_argument(
#         "--dropout",
#         type=float,
#         default=0.0
#     )

#     return parser.parse_args()


# if __name__ == "__main__":

#     args = get_args()

#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     print(f"Using device: {device}")

#     # build model
#     model = build_model(
#         name=args.model,
#         n_channels=args.n_channels,
#         n_classes=args.classes,
#         bilinear=args.bilinear,
#         dropout=args.dropout,
#     )

#     # load checkpoint
#     ckpt = torch.load(
#         args.checkpoint,
#         map_location=device,
#         weights_only=False
#     )

#     model.load_state_dict(ckpt["model_state"])

#     model.to(device)
#     model.eval()

#     print("Model loaded successfully.")

#     visualize(
#         model=model,
#         image_dir=args.input,
#         device=device,
#         img_size=args.img_size
#     )