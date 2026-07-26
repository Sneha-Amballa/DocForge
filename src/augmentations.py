import random
import io
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.config import DatasetConfig
from src.logger import get_logger

logger = get_logger("DocForge.Augmentations")

# Attempt to import albumentations
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
    logger.info("Albumentations library detected. Using Albumentations for augmentation pipeline.")
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.info("Albumentations not found. Using custom Pillow/NumPy fallback pipeline.")


class DocForgeAugmentations:
    """Configurable augmentation pipeline for DocForge.

    Supports Albumentations with matched transformations for images and masks,
    and a custom Pillow/NumPy fallback implementation if Albumentations is unavailable.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the pipeline from the dataset configuration.

        Args:
            config: DatasetConfig instance containing 'aug_config' parameters.
        """
        self.config = config
        self.enabled = config.aug_enabled
        self.aug_config = config.aug_config

        if self.enabled and ALBUMENTATIONS_AVAILABLE:
            self.transform = self._build_albumentations_pipeline()

    def _build_albumentations_pipeline(self) -> "A.Compose":
        """Construct the Albumentations Compose pipeline based on settings."""
        transforms = []

        # 1. Horizontal Flip
        if self.aug_config.get("horizontal_flip", False):
            transforms.append(A.HorizontalFlip(p=0.5))

        # 2. Small Rotations
        rot_limit = self.aug_config.get("rotation_limit", 10)
        if rot_limit > 0:
            transforms.append(A.Rotate(limit=rot_limit, p=0.5, border_mode=0))

        # 3. Brightness/Contrast
        bright_lim = self.aug_config.get("brightness_limit", 0.15)
        cont_lim = self.aug_config.get("contrast_limit", 0.15)
        if bright_lim > 0 or cont_lim > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=bright_lim,
                    contrast_limit=cont_lim,
                    p=0.5
                )
            )

        # 4. Gaussian Noise
        noise_var = self.aug_config.get("gaussian_noise_var", 15.0)
        if noise_var > 0:
            # Albumentations uses var_limit for GaussNoise
            transforms.append(A.GaussNoise(var_limit=(10.0, max(15.0, noise_var)), p=0.4))

        # 5. JPEG Compression simulation
        quality_min = self.aug_config.get("jpeg_compression_quality_min", 60)
        quality_max = self.aug_config.get("jpeg_compression_quality_max", 95)
        if quality_min < 100:
            transforms.append(
                A.ImageCompression(
                    quality_range=(quality_min, quality_max),
                    p=0.5
                )
            )

        # 6. Blurs
        g_blur = self.aug_config.get("gaussian_blur_limit", 3)
        m_blur = self.aug_config.get("motion_blur_limit", 3)
        if g_blur > 0:
            transforms.append(A.GaussianBlur(blur_limit=(3, max(3, g_blur)), p=0.3))
        if m_blur > 0:
            transforms.append(A.MotionBlur(blur_limit=(3, max(3, m_blur)), p=0.3))

        # 7. Perspective Transformation
        persp_limit = self.aug_config.get("perspective_limit", 0.04)
        if persp_limit > 0:
            transforms.append(A.Perspective(scale=(0.01, persp_limit), p=0.4))

        # 8. Random Crop
        if self.aug_config.get("random_crop_enabled", False):
            crop_h = self.aug_config.get("random_crop_height", 480)
            crop_w = self.aug_config.get("random_crop_width", 480)
            transforms.append(A.RandomCrop(height=crop_h, width=crop_w, p=1.0))

        return A.Compose(transforms)

    def apply(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Apply augmentations to the image (and mask if provided) in spatial synchronization.

        Args:
            image: Original PIL Image.
            mask: Optional binary tampering mask PIL Image.

        Returns:
            Tuple[Image.Image, Optional[Image.Image]]: Augmented image and mask.
        """
        if not self.enabled:
            return image, mask

        # If Albumentations is installed
        if ALBUMENTATIONS_AVAILABLE:
            img_np = np.array(image.convert("RGB"))
            if mask is not None:
                msk_np = np.array(mask.convert("L"))
                augmented = self.transform(image=img_np, mask=msk_np)
                return Image.fromarray(augmented["image"]), Image.fromarray(augmented["mask"])
            else:
                augmented = self.transform(image=img_np)
                return Image.fromarray(augmented["image"]), None

        # Fallback Pillow/NumPy Pipeline
        return self._apply_fallback(image, mask)

    def _apply_fallback(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Custom augmentation logic using Pillow and NumPy."""
        img = image.convert("RGB")
        msk = mask.convert("L") if mask is not None else None

        # 1. Horizontal Flip (only with probability 0.5)
        if self.aug_config.get("horizontal_flip", False) and random.random() < 0.5:
            img = ImageOps.mirror(img)
            if msk is not None:
                msk = ImageOps.mirror(msk)

        # 2. Small Rotation (probability 0.5)
        rot_limit = self.aug_config.get("rotation_limit", 10)
        if rot_limit > 0 and random.random() < 0.5:
            angle = random.uniform(-rot_limit, rot_limit)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
            if msk is not None:
                # Use NEAREST for mask to preserve binary values
                msk = msk.rotate(angle, resample=Image.Resampling.NEAREST, expand=False)

        # 3. Brightness & Contrast (probability 0.5)
        bright_lim = self.aug_config.get("brightness_limit", 0.15)
        cont_lim = self.aug_config.get("contrast_limit", 0.15)
        if (bright_lim > 0 or cont_lim > 0) and random.random() < 0.5:
            if bright_lim > 0:
                factor = random.uniform(1.0 - bright_lim, 1.0 + bright_lim)
                img = ImageEnhance.Brightness(img).enhance(factor)
            if cont_lim > 0:
                factor = random.uniform(1.0 - cont_lim, 1.0 + cont_lim)
                img = ImageEnhance.Contrast(img).enhance(factor)

        # 4. Blur (probability 0.3)
        g_blur = self.aug_config.get("gaussian_blur_limit", 3)
        if g_blur > 0 and random.random() < 0.3:
            # Pillow uses standard radius (approx 1 to 2)
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # 5. JPEG Compression simulation (probability 0.5)
        quality_min = self.aug_config.get("jpeg_compression_quality_min", 60)
        quality_max = self.aug_config.get("jpeg_compression_quality_max", 95)
        if quality_min < 100 and random.random() < 0.5:
            quality = random.randint(quality_min, quality_max)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf)
            img.load()

        # 6. Gaussian Noise (probability 0.4)
        noise_var = self.aug_config.get("gaussian_noise_var", 15.0)
        if noise_var > 0 and random.random() < 0.4:
            img_np = np.array(img).astype(np.float32)
            # Add Gaussian noise
            noise = np.random.normal(0.0, noise_var, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)

        # 7. Random Crop
        if self.aug_config.get("random_crop_enabled", False):
            crop_h = self.aug_config.get("random_crop_height", 480)
            crop_w = self.aug_config.get("random_crop_width", 480)
            w, h = img.size
            if w > crop_w and h > crop_h:
                x1 = random.randint(0, w - crop_w)
                y1 = random.randint(0, h - crop_h)
                x2 = x1 + crop_w
                y2 = y1 + crop_h
                img = img.crop((x1, y1, x2, y2))
                if msk is not None:
                    msk = msk.crop((x1, y1, x2, y2))

        return img, msk
