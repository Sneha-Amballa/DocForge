import os
from pathlib import Path
import json
import logging
from typing import List, Dict, Set, Tuple

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

def find_files_by_extensions(root_dir: str, extensions: Set[str]) -> List[Path]:
    """Recursively find all files in root_dir matching any of the extensions."""
    root_path = Path(root_dir)
    found_files = []
    exts = {ext.lower() for ext in extensions}
    
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            found_files.append(path)
            
    return found_files

def parse_sroie_ocr(txt_path: Path) -> Tuple[List[List[int]], List[str]]:
    """
    Parses a standard SROIE OCR text file.
    Format is typically: x1,y1,x2,y2,x3,y3,x4,y4,text
    Returns: (bounding_boxes, texts) where bounding_boxes are [x_min, y_min, x_max, y_max]
    """
    bboxes = []
    texts = []
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',', 8)
                if len(parts) == 9:
                    # Convert to x_min, y_min, x_max, y_max bounding box
                    try:
                        coords = [int(x) for x in parts[:8]]
                        x_min = min(coords[0], coords[2], coords[4], coords[6])
                        x_max = max(coords[0], coords[2], coords[4], coords[6])
                        y_min = min(coords[1], coords[3], coords[5], coords[7])
                        y_max = max(coords[1], coords[3], coords[5], coords[7])
                        bboxes.append([x_min, y_min, x_max, y_max])
                        texts.append(parts[8])
                    except ValueError:
                        continue
    except Exception as e:
        pass
        
    return bboxes, texts

def match_images_and_annotations(image_files: List[Path], ann_files: List[Path]) -> Dict[Path, Path]:
    """
    Matches images to their OCR annotations based on file stems.
    """
    ann_map = {ann.stem: ann for ann in ann_files}
    matched = {}
    for img in image_files:
        stem = img.stem
        if stem in ann_map:
            matched[img] = ann_map[stem]
    return matched
