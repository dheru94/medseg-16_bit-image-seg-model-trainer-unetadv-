import os
import cv2
import albumentations as A

# -------------------
# Paths
# -------------------
IMAGE_DIR = r"D:\16_bit_training_system\medseg\data\images"
MASK_DIR = r"D:\16_bit_training_system\medseg\data\masks"

OUT_IMAGE_DIR = "output/images"
OUT_MASK_DIR = "output/masks"

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# -------------------
# Augmentation pipeline
# -------------------
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=30,
        border_mode=cv2.BORDER_REFLECT,
        mask_interpolation=cv2.INTER_NEAREST,
        p=0.7
    ),
])

# -------------------
# Loop dataset
# -------------------
image_files = sorted(os.listdir(IMAGE_DIR))

aug_id = 0

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, img_name)

    if not os.path.exists(mask_path):
        print(f"Skipping {img_name} (mask not found)")
        continue

    # -------------------
    # Read PNG (keeps 16-bit if present)
    # -------------------
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if image is None or mask is None:
        print(f"Error reading {img_name}")
        continue

    # Ensure mask is single channel if needed
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # -------------------
    # Augment multiple times
    # -------------------
    for i in range(6):  # number of augmentations per image
        augmented = transform(image=image, mask=mask)

        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        base_name = os.path.splitext(img_name)[0]

        out_img_path = os.path.join(
            OUT_IMAGE_DIR, f"{base_name}_aug{aug_id}.png"
        )
        out_mask_path = os.path.join(
            OUT_MASK_DIR, f"{base_name}_aug{aug_id}.png"
        )

        cv2.imwrite(out_img_path, aug_image)
        cv2.imwrite(out_mask_path, aug_mask)

        aug_id += 1

print("Done! Augmented dataset created.")