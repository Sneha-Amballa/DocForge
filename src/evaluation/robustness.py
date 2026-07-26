import io
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Dict, Any, List, Tuple

from src.logger import get_logger

logger = get_logger("DocForge.EvalRobustness")

def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Degrade image quality using JPEG compression.

    Args:
        image: Pillow image.
        quality: JPEG compression quality (1 to 95).

    Returns:
        Image.Image: Compressed Pillow image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur to blur text and shapes.

    Args:
        image: Pillow image.
        radius: Blur radius.

    Returns:
        Image.Image: Blurred Pillow image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_gaussian_noise(image: Image.Image, std_dev: float) -> Image.Image:
    """Add Gaussian sensor noise to simulate scanner/camera noise.

    Args:
        image: Pillow image.
        std_dev: Standard deviation of noise (0 to 255).

    Returns:
        Image.Image: Noisy Pillow image.
    """
    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, std_dev, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)

def apply_brightness_change(image: Image.Image, factor: float) -> Image.Image:
    """Adjust image brightness.

    Args:
        image: Pillow image.
        factor: Enhancement factor (e.g. 0.5 is darker, 1.5 is brighter).

    Returns:
        Image.Image: Adjusted Pillow image.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_contrast_change(image: Image.Image, factor: float) -> Image.Image:
    """Adjust image contrast.

    Args:
        image: Pillow image.
        factor: Enhancement factor.

    Returns:
        Image.Image: Adjusted Pillow image.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def apply_random_occlusion(image: Image.Image, fill_val: int = 128) -> Image.Image:
    """Simulate occlusion by stamping a gray rectangle.

    Args:
        image: Pillow image.
        fill_val: Grayscale intensity of block.

    Returns:
        Image.Image: Occluded Pillow image.
    """
    w, h = image.size
    # occlude 10% to 20% box size
    box_w = random.randint(int(w * 0.1), int(w * 0.2))
    box_h = random.randint(int(h * 0.1), int(h * 0.2))
    
    xmin = random.randint(0, w - box_w)
    ymin = random.randint(0, h - box_h)
    
    occluded_image = image.copy()
    # Draw gray block
    from PIL import ImageDraw
    draw = ImageDraw.Draw(occluded_image)
    draw.rectangle([xmin, ymin, xmin + box_w, ymin + box_h], fill=(fill_val, fill_val, fill_val))
    return occluded_image

def apply_downsampling(image: Image.Image, scale_factor: float) -> Image.Image:
    """Degrade image resolution by resizing down and back up.

    Args:
        image: Pillow image.
        scale_factor: Scaling factor (e.g. 0.5 reduces size to half).

    Returns:
        Image.Image: Downsampled Pillow image.
    """
    w, h = image.size
    temp_w = max(16, int(w * scale_factor))
    temp_h = max(16, int(h * scale_factor))
    
    down_img = image.resize((temp_w, temp_h), resample=Image.Resampling.BILINEAR)
    return down_img.resize((w, h), resample=Image.Resampling.NEAREST)

def apply_degradation(
    image: Image.Image,
    degradation_type: str,
    severity: float
) -> Image.Image:
    """Apply a named document degradation at a specific severity multiplier.

    Args:
        image: Pillow image.
        degradation_type: Name of perturbation ('jpeg', 'blur', 'noise', 'brightness', 'contrast', 'occlusion', 'downsample').
        severity: Strength value.

    Returns:
        Image.Image: Perturbed Pillow image.
    """
    dtype = degradation_type.lower().strip()
    
    if dtype == "jpeg":
        # Severity maps 1.0 (no degradation, quality 95) to 5.0 (quality 10)
        quality = int(max(5, min(95, 95 - (severity * 17))))
        return apply_jpeg_compression(image, quality)
        
    elif dtype == "blur":
        # Severity maps to radius
        radius = severity * 1.5
        return apply_gaussian_blur(image, radius)
        
    elif dtype == "noise":
        # Severity maps to std dev
        std_dev = severity * 15.0
        return apply_gaussian_noise(image, std_dev)
        
    elif dtype == "brightness":
        # Brightness changes (0.5 is darker, 1.5 is brighter)
        factor = 1.0 - (severity * 0.15) if random.random() < 0.5 else 1.0 + (severity * 0.15)
        return apply_brightness_change(image, factor)
        
    elif dtype == "contrast":
        factor = 1.0 - (severity * 0.15) if random.random() < 0.5 else 1.0 + (severity * 0.15)
        return apply_contrast_change(image, factor)
        
    elif dtype == "occlusion":
        return apply_random_occlusion(image)
        
    elif dtype == "downsample":
        # Downsample scale factor
        scale = max(0.1, 1.0 - (severity * 0.18))
        return apply_downsampling(image, scale)
        
    else:
        logger.warning(f"Unknown degradation type '{degradation_type}'. Returning original.")
        return image
