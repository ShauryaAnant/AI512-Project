"""Joint and image transforms for nighttime semantic segmentation.

Ensures strict paired spatial augmentations between RGB images and masks,
using nearest-neighbor interpolation for labelTrainIds masks (preserving 255 ignore index)
and bilinear interpolation for RGB images.
"""

from typing import List, Optional, Tuple, Union
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance


class Compose:
    """Chains multiple joint transforms together."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self, img: Image.Image, mask: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomHorizontalFlip:
    """Synchronous random horizontal flip for image and mask."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, img: Image.Image, mask: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomScaleCrop:
    """Randomly scales and crops image and mask.

    Scales image using BILINEAR and mask using NEAREST.
    Pads if scaled dimensions are smaller than crop size.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int] = (512, 512),
        scale_range: Tuple[float, float] = (0.75, 1.25),
        ignore_index: int = 255,
    ):
        self.crop_h, self.crop_w = crop_size
        self.scale_range = scale_range
        self.ignore_index = ignore_index

    def __call__(
        self, img: Image.Image, mask: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        w, h = img.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        new_w = max(int(w * scale), self.crop_w)
        new_h = max(int(h * scale), self.crop_h)

        img = img.resize((new_w, new_h), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Random crop
        left = random.randint(0, new_w - self.crop_w)
        top = random.randint(0, new_h - self.crop_h)
        right = left + self.crop_w
        bottom = top + self.crop_h

        img = img.crop((left, top, right, bottom))
        if mask is not None:
            mask = mask.crop((left, top, right, bottom))

        return img, mask


class Resize:
    """Resize image and mask to target dimensions (must be divisible by 32)."""

    def __init__(
        self,
        size: Tuple[int, int] = (512, 1024),
    ):
        # size is (height, width)
        self.target_h, self.target_w = size

    def __call__(
        self, img: Image.Image, mask: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        img = img.resize((self.target_w, self.target_h), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((self.target_w, self.target_h), Image.NEAREST)
        return img, mask


class ColorJitter:
    """Mild photometric adjustments applied only to RGB image."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(
        self, img: Image.Image, mask: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        if self.brightness > 0 and random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)

        if self.contrast > 0 and random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)

        if self.saturation > 0 and random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            img = ImageEnhance.Color(img).enhance(factor)

        return img, mask


class ToTensorNormalize:
    """Converts PIL Images to PyTorch Tensors and normalizes RGB."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = mean
        self.std = std

    def __call__(
        self, img: Image.Image, mask: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Convert RGB image: (H, W, 3) uint8 [0..255] -> (3, H, W) float [0..1]
        img_np = np.array(img, dtype=np.float32) / 255.0
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        # Normalize
        for c in range(3):
            img_np[:, :, c] = (img_np[:, :, c] - self.mean[c]) / self.std[c]
        # (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).contiguous()

        mask_tensor = None
        if mask is not None:
            mask_np = np.array(mask, dtype=np.int64)
            mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor


def get_train_transforms(
    crop_size: Tuple[int, int] = (512, 512),
    scale_range: Tuple[float, float] = (0.75, 1.25),
    color_jitter: bool = True,
) -> Compose:
    """Build train augmentation pipeline."""
    transforms = [
        RandomScaleCrop(crop_size=crop_size, scale_range=scale_range),
        RandomHorizontalFlip(p=0.5),
    ]
    if color_jitter:
        transforms.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    transforms.append(ToTensorNormalize())
    return Compose(transforms)


def get_test_transforms(
    eval_size: Tuple[int, int] = (512, 1024),
) -> Compose:
    """Build evaluation transform pipeline (deterministic, divisible by 32)."""
    transforms = [
        Resize(size=eval_size),
        ToTensorNormalize(),
    ]
    return Compose(transforms)
