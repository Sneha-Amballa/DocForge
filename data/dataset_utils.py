import os
import glob
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from PIL import Image
import logging
import lmdb
import struct

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def extract_bboxes_from_mask(mask: np.ndarray, min_area: int = 10) -> List[List[int]]:
    """
    Extract bounding boxes from a binary mask.
    mask: numpy array (H, W) where forged regions > 0
    Returns a list of [x_min, y_min, x_max, y_max]
    """
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    bboxes = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_area:
            bboxes.append([x, y, x + w, y + h])
            
    return bboxes

# ================= LMDB UTILS =================

def get_lmdb_envs(root_dir: str) -> List[Path]:
    """Finds all LMDB directory paths in the given root."""
    root_path = Path(root_dir)
    lmdbs = []
    for path in root_path.iterdir():
        if path.is_dir():
            data_file = path / 'data.mdb'
            if data_file.exists():
                lmdbs.append(path)
    return lmdbs

def read_image_from_lmdb(txn, key: bytes, grayscale=False) -> Optional[np.ndarray]:
    """Reads and decodes an image or mask from LMDB transaction."""
    val = txn.get(key)
    if val is None:
        return None
    
    img_array = np.frombuffer(val, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(img_array, flag)
    return img

def get_jpeg_size(val: bytes) -> Tuple[int, int]:
    """Fast JPEG header parser to get size without decoding full image."""
    try:
        # Try OpenCV decode if structure is complex, but for speed just decode normally
        # OpenCV decode is fast enough for sizing if we do IMREAD_IGNORE_ORIENTATION (or similar)
        # We will just do a standard imdecode.
        img_array = np.frombuffer(val, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if img is not None:
            return (img.shape[1], img.shape[0])
    except Exception:
        pass
    return (-1, -1)

def get_png_size(val: bytes) -> Tuple[int, int]:
    """Fast PNG header parser"""
    try:
        if val.startswith(b'\x89PNG\r\n\x1a\n') and len(val) > 24:
            w, h = struct.unpack('>II', val[16:24])
            return (w, h)
    except Exception:
        pass
    return (-1, -1)
