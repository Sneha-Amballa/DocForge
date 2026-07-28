import json
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image

from src.logger import get_logger

logger = get_logger("DocForge.BboxGenerator")

def generate_bboxes_from_mask(mask: Image.Image) -> List[List[int]]:
    """Automatically extract bounding boxes for all tampered regions in the mask.

    Finds connected components in the mask and returns bounding boxes.
    Tries OpenCV, SciPy, and falls back to standard NumPy overall box if needed.

    Args:
        mask: Grayscale binary mask PIL Image where value > 0 indicates tampering.

    Returns:
        List[List[int]]: List of bounding boxes, each formatted as [xmin, ymin, xmax, ymax]
            representing absolute pixel coordinates.
    """
    mask_arr = np.array(mask.convert("L"))
    
    # Threshold mask array to absolute binary (0 or 1)
    binary_arr = (mask_arr > 0).astype(np.uint8)
    
    if np.sum(binary_arr) == 0:
        return []

    # Method 1: Try OpenCV (connectedComponentsWithStats)
    try:
        # pyrefly: ignore [missing-import]
        import cv2
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_arr)
        bboxes = []
        for i in range(1, num_labels):  # Skip background index 0
            left = int(stats[i, cv2.CC_STAT_LEFT])
            top = int(stats[i, cv2.CC_STAT_TOP])
            width = int(stats[i, cv2.CC_STAT_WIDTH])
            height = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            
            # Filter out tiny pixel noise if any
            if area > 4:
                bboxes.append([left, top, left + width, top + height])
        logger.debug(f"Extracted {len(bboxes)} bounding boxes using OpenCV.")
        return bboxes
    except ImportError:
        pass

    # Method 2: Try SciPy (ndimage.label and find_objects)
    try:
        # pyrefly: ignore [missing-import]
        import scipy.ndimage as ndimage
        labeled_arr, num_features = ndimage.label(binary_arr)
        slices = ndimage.find_objects(labeled_arr)
        bboxes = []
        for slc in slices:
            if slc is not None:
                slice_y, slice_x = slc
                xmin, xmax = int(slice_x.start), int(slice_x.stop)
                ymin, ymax = int(slice_y.start), int(slice_y.stop)
                bboxes.append([xmin, ymin, xmax, ymax])
        logger.debug(f"Extracted {len(bboxes)} bounding boxes using SciPy.")
        return bboxes
    except ImportError:
        pass

    # Method 3: Fallback overall enclosing box using standard NumPy
    logger.debug("Falling back to NumPy single-region enclosing box.")
    ys, xs = np.where(binary_arr > 0)
    if len(xs) > 0:
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        return [[xmin, ymin, xmax, ymax]]
    
    return []

def normalize_bbox_for_qwen2_vl(
    bbox: List[int],
    img_width: int,
    img_height: int
) -> List[int]:
    """Convert absolute [xmin, ymin, xmax, ymax] bounding box to Qwen2-VL normalized format.

    Qwen2-VL expects boxes formatted as [ymin, xmin, ymax, xmax],
    normalized to the range [0, 1000].

    Args:
        bbox: Absolute bounding box [xmin, ymin, xmax, ymax].
        img_width: Original image width.
        img_height: Original image height.

    Returns:
        List[int]: Normalized bounding box [ymin, xmin, ymax, xmax] scaled to 1000.
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Scale and round coordinates
    ymin_norm = int(round((ymin / img_height) * 1000))
    xmin_norm = int(round((xmin / img_width) * 1000))
    ymax_norm = int(round((ymax / img_height) * 1000))
    xmax_norm = int(round((xmax / img_width) * 1000))
    
    # Clip coordinates to safe ranges
    ymin_norm = max(0, min(1000, ymin_norm))
    xmin_norm = max(0, min(1000, xmin_norm))
    ymax_norm = max(0, min(1000, ymax_norm))
    xmax_norm = max(0, min(1000, xmax_norm))
    
    return [ymin_norm, xmin_norm, ymax_norm, xmax_norm]

def save_bboxes_to_json(
    bboxes_dict: Dict[str, List[List[int]]],
    save_path: Union[str, Path]
) -> None:
    """Save extracted bounding boxes dictionary to a JSON file.

    Args:
        bboxes_dict: Dictionary mapping sample keys to lists of boxes.
        save_path: Destination JSON filepath.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(bboxes_dict, f, indent=4)
    logger.info(f"Saved {len(bboxes_dict)} sample bounding boxes to {save_path}")
