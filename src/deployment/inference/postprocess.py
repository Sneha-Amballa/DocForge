import re
from typing import List, Dict, Any, Tuple

from src.logger import get_logger

logger = get_logger("DocForge.DeployPostprocess")

def postprocess_forgery_response(
    generated_text: str,
    img_width: int,
    img_height: int
) -> Tuple[bool, str, List[Dict[str, int]], str]:
    """Parse VLM output text, classification flag, and bounding boxes.

    Args:
        generated_text: Raw generative string returned by the VLM.
        img_width: Original image width in pixels.
        img_height: Original image height in pixels.

    Returns:
        Tuple containing:
            - tampered: bool indicating if forgery was detected.
            - forgery_type: str category of detected forgery (e.g. "Text Replacement").
            - bounding_boxes: List of parsed top-left bounding box dicts {"x", "y", "width", "height"}.
            - explanation: str explanation of prediction.
    """
    text_lower = generated_text.lower().strip()
    
    # 1. Determine tampering status
    tampered = False
    if "tampered" in text_lower or "forge" in text_lower:
        tampered = True
        
    # 2. Extract forgery type category
    # Common DocTamper categories: "Copy-Move", "Splicing", "Text Replacement", "Erasure"
    forgery_type = "Authentic"
    if tampered:
        if "copy-move" in text_lower or "copy_move" in text_lower:
            forgery_type = "Copy-Move"
        elif "splice" in text_lower or "splicing" in text_lower:
            forgery_type = "Splicing"
        elif "replace" in text_lower or "text replacement" in text_lower or "replacement" in text_lower:
            forgery_type = "Text Replacement"
        elif "erasure" in text_lower or "erase" in text_lower or "deletion" in text_lower:
            forgery_type = "Erasure"
        else:
            forgery_type = "Tampering Area"

    # 3. Parse coordinates bounding boxes [ymin, xmin, ymax, xmax] (normally 0-1000 scale)
    bounding_boxes = []
    # Find bracket coordinates
    matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', generated_text)
    
    for m in matches:
        try:
            ymin, xmin, ymax, xmax = map(int, m)
            
            # Scale coordinates back to absolute image coordinates
            xmin_abs = int(round(xmin * img_width / 1000.0))
            ymin_abs = int(round(ymin * img_height / 1000.0))
            xmax_abs = int(round(xmax * img_width / 1000.0))
            ymax_abs = int(round(ymax * img_height / 1000.0))
            
            # Bound coordinates to image dimensions
            xmin_abs = max(0, min(img_width - 1, xmin_abs))
            ymin_abs = max(0, min(img_height - 1, ymin_abs))
            xmax_abs = max(0, min(img_width, xmax_abs))
            ymax_abs = max(0, min(img_height, ymax_abs))
            
            w = max(1, xmax_abs - xmin_abs)
            h = max(1, ymax_abs - ymin_abs)
            
            bounding_boxes.append({
                "x": xmin_abs,
                "y": ymin_abs,
                "width": w,
                "height": h
            })
        except Exception as e:
            logger.warning(f"Error parsing bounding box coordinates match {m}: {e}")

    # 4. Formulate clean explanation
    # Extract sentence before or containing the coordinates, or return clean summary
    explanation = generated_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    
    return tampered, forgery_type, bounding_boxes, explanation


