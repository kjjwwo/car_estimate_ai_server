import cv2
import numpy as np
from pathlib import Path
from uuid import uuid4

DAMAGE_COLORS = {
    "Scratched": (0, 180, 255),
    "Separated": (0, 60, 255),
    "Crushed": (0, 30, 255),
    "Breakage": (0, 0, 255),
}

# PART_COLORS = {
#     1: (255, 0, 0),       # Blue
#     2: (255, 128, 0),     # SkyBlue
#     3: (255, 255, 0),     # Cyan
#     4: (255, 0, 255),     # Purple
#     5: (128, 0, 255),     # Violet
# }

def resize_mask_to_image(mask, image_bgr):
    h, w = image_bgr.shape[:2]
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def overlay_damage_masks(image_bgr, damage_masks, alpha=0.45):
    overlay = image_bgr.copy()

    for damage_name, mask in damage_masks.items():
        mask = resize_mask_to_image(mask, image_bgr)
        color = DAMAGE_COLORS.get(damage_name, (0, 0, 255))

        color_layer = np.zeros_like(image_bgr)
        color_layer[mask > 0] = color

        overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)

    return overlay

# def overlay_part_mask(image_bgr, part_mask, alpha=0.35):
#     part_mask = resize_mask_to_image(part_mask, image_bgr)
#     overlay = image_bgr.copy()

#     color_layer = np.zeros_like(image_bgr)

#     for part_id, color in PART_COLORS.items():
#         color_layer[part_mask == part_id] = color

#     overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)
#     return overlay

# def overlay_damage_and_part(image_bgr, damage_masks, part_mask):
#     part_overlay = overlay_part_mask(image_bgr, part_mask, alpha=0.25)
#     final_overlay = overlay_damage_masks(part_overlay, damage_masks, alpha=0.55)
#     return final_overlay


def save_overlay_image(image_bgr, damage_masks, part_mask, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = overlay_damage_masks(
        image_bgr=image_bgr,
        damage_masks=damage_masks,
        alpha=0.55
    )

    filename = f"{uuid4().hex}.jpg"
    save_path = output_dir / filename

    cv2.imwrite(str(save_path), overlay)

    return filename, str(save_path)
