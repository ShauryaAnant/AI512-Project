"""Visualization utilities and Cityscapes color mapping."""

import numpy as np
from PIL import Image

# Standard Cityscapes 19-class color palette
CITYSCAPES_PALETTE = np.array(
    [
        [128, 64, 128],   # 0: road
        [244, 35, 232],   # 1: sidewalk
        [70, 70, 70],     # 2: building
        [102, 102, 156],  # 3: wall
        [190, 153, 153],  # 4: fence
        [153, 153, 153],  # 5: pole
        [250, 170, 30],   # 6: traffic light
        [220, 220, 0],    # 7: traffic sign
        [107, 142, 35],   # 8: vegetation
        [152, 251, 152],  # 9: terrain
        [70, 130, 180],   # 10: sky
        [220, 20, 60],    # 11: person
        [255, 0, 0],      # 12: rider
        [0, 0, 142],      # 13: car
        [0, 0, 70],       # 14: truck
        [0, 60, 100],     # 15: bus
        [0, 80, 100],     # 16: train
        [0, 0, 230],      # 17: motorcycle
        [119, 11, 32],    # 18: bicycle
    ],
    dtype=np.uint8,
)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Colorize a 2D integer mask (H, W) in [0..18, 255] into an RGB PIL Image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for c in range(19):
        color_mask[mask == c] = CITYSCAPES_PALETTE[c]

    # Ignore index 255 left as black [0, 0, 0]
    return Image.fromarray(color_mask)
